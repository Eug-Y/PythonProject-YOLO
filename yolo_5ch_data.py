"""Ultralytics dataset, trainer, and validator for normalized five-channel NPY inputs."""

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from multispectral import DEFAULT_MULTISPECTRAL_DIR, find_5ch_path, load_5ch


def _load_five_channel_image(dataset, index: int, search_dirs: Iterable[str | Path]):
    image_file = dataset.im_files[index]
    npy_path = find_5ch_path(image_file, search_dirs)
    if npy_path is None:
        raise FileNotFoundError(
            f"Missing five-channel NPY for {image_file}. "
            "Run: python prepare_5ch.py --repair"
        )
    if dataset.augment and dataset.ims[index] is not None:
        return dataset.ims[index], dataset.im_hw0[index], dataset.im_hw[index]

    image = load_5ch(npy_path, strict=True)
    h0, w0 = image.shape[:2]
    imgsz = dataset.imgsz if isinstance(dataset.imgsz, int) else max(dataset.imgsz)
    ratio = imgsz / max(h0, w0)
    if ratio != 1.0:
        size = (min(int(np.ceil(w0 * ratio)), imgsz), min(int(np.ceil(h0 * ratio)), imgsz))
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    h, w = image.shape[:2]
    image_u8 = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)

    if dataset.augment:
        dataset.ims[index], dataset.im_hw0[index], dataset.im_hw[index] = image_u8, (h0, w0), (h, w)
        dataset.buffer.append(index)
        if 1 < len(dataset.buffer) >= dataset.max_buffer_length:
            old_index = dataset.buffer.pop(0)
            if dataset.cache != "ram":
                dataset.ims[old_index], dataset.im_hw0[old_index], dataset.im_hw[old_index] = None, None, None
    return image_u8, (h0, w0), (h, w)


def _filter_five_channel_files(image_files: list[str], search_dirs: Iterable[str | Path], source: object) -> list[str]:
    filtered = [file for file in image_files if find_5ch_path(file, search_dirs) is not None]
    missing = len(image_files) - len(filtered)
    print(f"[5ch data] {len(filtered)}/{len(image_files)} images ready; missing={missing}")
    if not filtered:
        raise FileNotFoundError(f"No five-channel data found for {source}. Run: python prepare_5ch.py --repair")
    return filtered


def build_five_channel_dataset(
    cfg,
    img_path,
    batch,
    data,
    mode="train",
    rect=False,
    stride=32,
    *,
    photometric_augment: bool = False,
    photometric_kwargs: dict[str, Any] | None = None,
):
    return FiveChannelYOLODataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",
        hyp=cfg,
        rect=cfg.rect or rect,
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=stride,
        pad=0.0 if mode == "train" else 0.5,
        prefix=f"{mode}: ",
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == "train" else 1.0,
        five_channel_dirs=(DEFAULT_MULTISPECTRAL_DIR,),
        photometric_augment=photometric_augment and mode == "train",
        photometric_kwargs=photometric_kwargs,
    )


def assert_dataloader_channels(dataloader, expected: int = 5, name: str = "dataloader") -> None:
    # Inspect a dataset item instead of consuming the persistent
    # InfiniteDataLoader iterator before the first training epoch.
    sample = dataloader.dataset[0]
    image = sample["img"]
    channels = int(image.shape[0])
    if channels != expected:
        raise RuntimeError(f"{name} produced {channels} channels; expected {expected}")
    print(f"[5ch data] {name} verified: sample shape={tuple(image.shape)}")


from ultralytics.data.base import BaseDataset
from ultralytics.data.augment import BaseTransform
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.segment import SegmentationTrainer, SegmentationValidator
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils.torch_utils import unwrap_model

_ORIGINAL_GET_IMG_FILES = BaseDataset.get_img_files


class FiveChannelPhotometric(BaseTransform):
    """Band-safe photometric augmentation for [R, G, B, NDVI, NDRE].

    RGB receives HSV-style color jitter. The two vegetation indices receive
    only small gain, bias, and noise perturbations. Spatial operations stay in
    Ultralytics so images, boxes, and segmentation polygons remain aligned.
    """

    def __init__(
        self,
        *,
        rgb_probability: float = 0.8,
        hue_gain: float = 0.01,
        saturation_gain: float = 0.25,
        value_gain: float = 0.20,
        spectral_probability: float = 0.5,
        spectral_gain: float = 0.03,
        spectral_bias: float = 0.01,
        spectral_noise: float = 0.005,
    ) -> None:
        for name, value in (
            ("rgb_probability", rgb_probability),
            ("spectral_probability", spectral_probability),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        for name, value in (
            ("hue_gain", hue_gain),
            ("saturation_gain", saturation_gain),
            ("value_gain", value_gain),
            ("spectral_gain", spectral_gain),
            ("spectral_bias", spectral_bias),
            ("spectral_noise", spectral_noise),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative, got {value}")

        self.rgb_probability = float(rgb_probability)
        self.hue_gain = float(hue_gain)
        self.saturation_gain = float(saturation_gain)
        self.value_gain = float(value_gain)
        self.spectral_probability = float(spectral_probability)
        self.spectral_gain = float(spectral_gain)
        self.spectral_bias = float(spectral_bias)
        self.spectral_noise = float(spectral_noise)

    @staticmethod
    def _check_image(image: np.ndarray) -> None:
        if image.ndim != 3 or image.shape[-1] != 5:
            raise ValueError(f"FiveChannelPhotometric expected HxWx5, got {image.shape}")
        if image.dtype != np.uint8:
            raise ValueError(f"FiveChannelPhotometric expected uint8, got {image.dtype}")

    def _augment_rgb(self, rgb: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        hue_shift = np.random.uniform(-self.hue_gain, self.hue_gain) * 180.0
        saturation_scale = 1.0 + np.random.uniform(-self.saturation_gain, self.saturation_gain)
        value_scale = 1.0 + np.random.uniform(-self.value_gain, self.value_gain)
        hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180.0
        hsv[..., 1] = np.clip(hsv[..., 1] * saturation_scale, 0.0, 255.0)
        hsv[..., 2] = np.clip(hsv[..., 2] * value_scale, 0.0, 255.0)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    def _augment_spectral(self, spectral: np.ndarray) -> np.ndarray:
        values = spectral.astype(np.float32) / 255.0
        gains = 1.0 + np.random.uniform(-self.spectral_gain, self.spectral_gain, size=(1, 1, 2))
        biases = np.random.uniform(-self.spectral_bias, self.spectral_bias, size=(1, 1, 2))
        if self.spectral_noise:
            noise = np.random.normal(0.0, self.spectral_noise, size=values.shape)
        else:
            noise = 0.0
        values = np.clip(values * gains + biases + noise, 0.0, 1.0)
        return np.rint(values * 255.0).astype(np.uint8)

    def apply_image(self, labels: dict[str, Any], params=None) -> dict[str, Any]:
        image = labels["img"]
        self._check_image(image)
        output = image.copy()
        if np.random.random() < self.rgb_probability:
            output[..., :3] = self._augment_rgb(output[..., :3])
        if np.random.random() < self.spectral_probability:
            output[..., 3:] = self._augment_spectral(output[..., 3:])
        labels["img"] = np.ascontiguousarray(output)
        return labels


class FiveChannelYOLODataset(YOLODataset):
    """YOLODataset whose image source is a strict HxWx5 normalized NPY file."""

    def __init__(
        self,
        *args,
        five_channel_dirs: Iterable[str | Path] = (),
        photometric_augment: bool = False,
        photometric_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        self.five_channel_dirs = tuple(Path(path) for path in five_channel_dirs) or (DEFAULT_MULTISPECTRAL_DIR,)
        self.photometric_augment = bool(photometric_augment)
        self.photometric_kwargs = dict(photometric_kwargs or {})
        super().__init__(*args, **kwargs)

    def get_img_files(self, img_path):
        files = _ORIGINAL_GET_IMG_FILES(self, img_path)
        return _filter_five_channel_files(files, self.five_channel_dirs, img_path)

    def load_image(self, index, rect_mode=True):
        return _load_five_channel_image(self, index, self.five_channel_dirs)

    def build_transforms(self, hyp=None):
        transforms = super().build_transforms(hyp)
        if self.augment and self.photometric_augment:
            # The final transform is Format (HWC uint8 -> CHW tensor). Insert
            # immediately before it so the five-channel contract is explicit.
            transforms.insert(len(transforms.transforms) - 1, FiveChannelPhotometric(**self.photometric_kwargs))
        return transforms


class FiveChannelSegmentationValidator(SegmentationValidator):
    def build_dataset(self, img_path: str, mode: str = "val", batch: int | None = None):
        return build_five_channel_dataset(self.args, img_path, batch, self.data, mode=mode, stride=self.stride)


class FiveChannelSegmentationTrainer(SegmentationTrainer):
    def __init__(
        self,
        *args,
        five_channel_photometric: bool = False,
        five_channel_photometric_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        self.five_channel_photometric = bool(five_channel_photometric)
        self.five_channel_photometric_kwargs = dict(five_channel_photometric_kwargs or {})
        super().__init__(*args, **kwargs)

    def setup_model(self):
        """Restore the serialized custom 5-channel architecture when resuming."""
        if self.resume and str(self.model).endswith(".pt"):
            self.model, checkpoint = load_checkpoint(self.model, device="cpu")
            # Checkpoints may serialize a loss object whose cached tensors remain on CPU.
            # Rebuild it lazily after the trainer moves the restored model to CUDA.
            self.model.criterion = None
            return checkpoint
        return super().setup_model()

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        stride = max(int(unwrap_model(self.model).stride.max()), 32)
        return build_five_channel_dataset(
            self.args,
            img_path,
            batch,
            self.data,
            mode=mode,
            rect=mode == "val",
            stride=stride,
            photometric_augment=self.five_channel_photometric,
            photometric_kwargs=self.five_channel_photometric_kwargs,
        )

    def get_validator(self):
        self.loss_names = "box_loss", "seg_loss", "cls_loss", "dfl_loss", "sem_loss"
        return FiveChannelSegmentationValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )

    def _build_train_pipeline(self):
        super()._build_train_pipeline()
        assert_dataloader_channels(self.train_loader, name="train loader")
        assert_dataloader_channels(self.test_loader, name="validation loader")


_PATCHED = False
_PATCH_SEARCH_DIRS: list[Path] = [DEFAULT_MULTISPECTRAL_DIR]


def patch_ultralytics_5ch(search_dirs: Iterable[str | Path] = ()) -> None:
    """Compatibility patch for older scripts; prefer the dataset/trainer classes above."""
    global _PATCHED
    for directory in search_dirs:
        path = Path(directory)
        if path not in _PATCH_SEARCH_DIRS:
            _PATCH_SEARCH_DIRS.append(path)
    if _PATCHED:
        return

    def load_image_5ch(self, index, rect_mode=True):
        return _load_five_channel_image(self, index, _PATCH_SEARCH_DIRS)

    def get_img_files_5ch(self, img_path):
        files = _ORIGINAL_GET_IMG_FILES(self, img_path)
        return _filter_five_channel_files(files, _PATCH_SEARCH_DIRS, img_path)

    BaseDataset.load_image = load_image_5ch
    BaseDataset.get_img_files = get_img_files_5ch
    _PATCHED = True
