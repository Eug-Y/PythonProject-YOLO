"""Ultralytics segmentation data pipeline for configurable multichannel data.

The candidate arrays are stored as uint8 HxWxC files.  This module selects a
manifest-ordered subset without changing source images or labels.  All
selected layouts must start with R, G, B so the pretrained RGB stem keeps a
stable semantic contract.
"""

from __future__ import annotations

import os
from copy import copy
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler, distributed

from ultralytics.data.augment import BaseTransform
from ultralytics.data.base import BaseDataset
from ultralytics.data.build import ContiguousDistributedSampler, InfiniteDataLoader, seed_worker
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.segment import SegmentationTrainer, SegmentationValidator
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.ops import segments2boxes
from ultralytics.utils.torch_utils import torch_distributed_zero_first, unwrap_model


_ORIGINAL_GET_IMG_FILES = BaseDataset.get_img_files


def build_seeded_dataloader(
    dataset,
    batch: int,
    workers: int,
    *,
    seed: int,
    shuffle: bool = True,
    rank: int = -1,
    drop_last: bool = False,
    pin_memory: bool = True,
    sample_weights: Sequence[float] | None = None,
) -> InfiniteDataLoader:
    """Build an Ultralytics loader whose order and worker RNG follow the run seed.

    Ultralytics 8.4.51 seeds every loader with a package-level constant. That
    makes separately requested experiment seeds consume identical shuffled
    batches and identical worker-side augmentations. Keep seed 0 compatible
    with the upstream sequence while giving every other run an independent,
    reproducible generator.
    """
    batch = min(batch, len(dataset))
    device_count = torch.cuda.device_count()
    worker_count = min((os.cpu_count() or 1) // max(device_count, 1), workers)
    if sample_weights is not None:
        if rank != -1:
            raise NotImplementedError("Weighted train sampling is currently supported only for single-GPU runs")
        weights = torch.as_tensor(sample_weights, dtype=torch.double)
        if weights.ndim != 1 or len(weights) != len(dataset):
            raise ValueError(f"Expected {len(dataset)} sample weights, got shape={tuple(weights.shape)}")
        if not torch.isfinite(weights).all() or torch.any(weights <= 0):
            raise ValueError("Sample weights must be finite and strictly positive")
        sampling_generator = torch.Generator()
        sampling_generator.manual_seed(6148914691236518641 + int(seed))
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(dataset),
            replacement=True,
            generator=sampling_generator,
        )
    else:
        sampler = (
            None
            if rank == -1
            else distributed.DistributedSampler(dataset, shuffle=shuffle, seed=int(seed))
            if shuffle
            else ContiguousDistributedSampler(dataset)
        )
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK + int(seed))
    return InfiniteDataLoader(
        dataset=dataset,
        batch_size=batch,
        shuffle=shuffle and sampler is None,
        num_workers=worker_count,
        sampler=sampler,
        prefetch_factor=4 if worker_count > 0 else None,
        pin_memory=device_count > 0 and pin_memory,
        collate_fn=getattr(dataset, "collate_fn", None),
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=drop_last and len(dataset) % batch != 0,
    )


def validate_channel_selection(all_channels: Sequence[str], selected_channels: Sequence[str]) -> tuple[int, ...]:
    """Validate a selected layout and return its positions in the candidate array."""
    available = tuple(str(name) for name in all_channels)
    selected = tuple(str(name) for name in selected_channels)
    if not selected:
        raise ValueError("At least one channel must be selected")
    if len(selected) != len(set(selected)):
        raise ValueError(f"Selected channels contain duplicates: {selected}")
    if selected[:3] != ("R", "G", "B"):
        raise ValueError(f"Selected channels must start with ('R', 'G', 'B'), got {selected[:3]}")
    missing = [name for name in selected if name not in available]
    if missing:
        raise ValueError(f"Selected channels are absent from the feature manifest: {missing}")
    return tuple(available.index(name) for name in selected)


def feature_path_for_image(image_file: str | Path, feature_root: str | Path) -> Path:
    """Map source/{split}/images/name.JPG to features/{split}/features/name.npy."""
    image_path = Path(image_file)
    try:
        split = image_path.parent.parent.name
    except IndexError as exc:  # pragma: no cover - Path parents are normally available
        raise ValueError(f"Cannot infer split from image path: {image_path}") from exc
    if split not in {"train", "valid", "test"}:
        raise ValueError(f"Expected train/valid/test image path, got split={split!r}: {image_path}")
    return Path(feature_root) / split / "features" / f"{image_path.stem}.npy"


def _load_selected_array(
    path: str | Path,
    channel_indices: Sequence[int],
    expected_source_channels: int,
) -> np.ndarray:
    source = np.load(path, mmap_mode="r", allow_pickle=False)
    if source.ndim != 3 or source.shape[2] != expected_source_channels:
        raise ValueError(
            f"Expected HxWx{expected_source_channels} feature array at {path}, got {source.shape}"
        )
    if source.dtype != np.uint8:
        raise ValueError(f"Expected uint8 feature array at {path}, got {source.dtype}")
    # np.take makes a writable, contiguous copy. Ultralytics augmentations may
    # edit images in-place, so returning a read-only mmap view is unsafe.
    return np.ascontiguousarray(np.take(source, channel_indices, axis=2))


def _load_multichannel_image(dataset, index: int):
    image_file = dataset.im_files[index]
    feature_path = feature_path_for_image(image_file, dataset.feature_root)
    if not feature_path.is_file():
        raise FileNotFoundError(f"Missing multichannel feature array for {image_file}: {feature_path}")
    if dataset.augment and dataset.ims[index] is not None:
        return dataset.ims[index], dataset.im_hw0[index], dataset.im_hw[index]

    image = _load_selected_array(feature_path, dataset.channel_indices, len(dataset.all_channels))
    h0, w0 = image.shape[:2]
    imgsz = dataset.imgsz if isinstance(dataset.imgsz, int) else max(dataset.imgsz)
    ratio = imgsz / max(h0, w0)
    if ratio != 1.0:
        size = (min(int(np.ceil(w0 * ratio)), imgsz), min(int(np.ceil(h0 * ratio)), imgsz))
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        if image.ndim == 2:  # defensive for a one-channel future layout
            image = image[..., None]
    image = np.ascontiguousarray(image)
    h, w = image.shape[:2]

    if dataset.augment:
        dataset.ims[index], dataset.im_hw0[index], dataset.im_hw[index] = image, (h0, w0), (h, w)
        dataset.buffer.append(index)
        if 1 < len(dataset.buffer) >= dataset.max_buffer_length:
            old_index = dataset.buffer.pop(0)
            if dataset.cache != "ram":
                dataset.ims[old_index], dataset.im_hw0[old_index], dataset.im_hw[old_index] = None, None, None
    return image, (h0, w0), (h, w)


def _filter_feature_files(image_files: list[str], feature_root: Path, source: object) -> list[str]:
    filtered = [file for file in image_files if feature_path_for_image(file, feature_root).is_file()]
    missing = len(image_files) - len(filtered)
    print(f"[multichannel] {len(filtered)}/{len(image_files)} feature arrays ready; missing={missing}")
    if missing:
        raise FileNotFoundError(f"Feature store is incomplete for {source}: {missing} arrays are missing")
    if not filtered:
        raise FileNotFoundError(f"No multichannel feature arrays found for {source}")
    return filtered


def _restore_distinct_segments_with_shared_boxes(labels: list[dict[str, Any]]) -> int:
    """Restore distinct polygons that Ultralytics drops when their xywh boxes match.

    Ultralytics' generic verifier deduplicates segmentation rows by class and
    bounding box, not by the full polygon. Distinct reviewed instances may
    legitimately share an exact outer box, so reconstruct those entries from
    the already-validated source label file while still removing exact polygon
    duplicates.
    """
    restored = 0
    for label in labels:
        if not label.get("segments"):
            continue
        image_path = Path(label["im_file"])
        label_path = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
        rows = [row.split() for row in label_path.read_text(encoding="utf-8-sig").splitlines() if row.strip()]
        if not rows or not all(len(row) > 6 for row in rows):
            continue
        unique: list[tuple[np.float32, np.ndarray]] = []
        seen: set[tuple[bytes, tuple[int, int], bytes]] = set()
        for row in rows:
            class_id = np.float32(row[0])
            segment = np.asarray(row[1:], dtype=np.float32).reshape(-1, 2)
            key = (class_id.tobytes(), segment.shape, segment.tobytes())
            if key in seen:
                continue
            seen.add(key)
            unique.append((class_id, segment))
        if len(unique) <= len(label["segments"]):
            continue
        original_count = len(label["segments"])
        segments = [segment for _, segment in unique]
        label["cls"] = np.asarray([class_id for class_id, _ in unique], dtype=np.float32).reshape(-1, 1)
        label["bboxes"] = segments2boxes(segments).astype(np.float32, copy=False)
        label["segments"] = segments
        restored += len(unique) - original_count
        if not (len(label["cls"]) == len(label["bboxes"]) == len(label["segments"])):
            raise RuntimeError(f"Restored segmentation count mismatch for {label_path}")
    return restored


class MultichannelPhotometric(BaseTransform):
    """Apply RGB color jitter and extra-channel perturbations independently."""

    def __init__(
        self,
        channel_names: Sequence[str],
        *,
        rgb_probability: float = 0.8,
        hue_gain: float = 0.01,
        saturation_gain: float = 0.25,
        value_gain: float = 0.20,
        extra_probability: float = 0.5,
        extra_gain: float = 0.03,
        extra_bias: float = 0.01,
        extra_noise: float = 0.005,
    ) -> None:
        self.channel_names = tuple(channel_names)
        validate_channel_selection(self.channel_names, self.channel_names)
        for name, value in (("rgb_probability", rgb_probability), ("extra_probability", extra_probability)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        for name, value in (
            ("hue_gain", hue_gain),
            ("saturation_gain", saturation_gain),
            ("value_gain", value_gain),
            ("extra_gain", extra_gain),
            ("extra_bias", extra_bias),
            ("extra_noise", extra_noise),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        self.rgb_probability = float(rgb_probability)
        self.hue_gain = float(hue_gain)
        self.saturation_gain = float(saturation_gain)
        self.value_gain = float(value_gain)
        self.extra_probability = float(extra_probability)
        self.extra_gain = float(extra_gain)
        self.extra_bias = float(extra_bias)
        self.extra_noise = float(extra_noise)

    def _check_image(self, image: np.ndarray) -> None:
        expected = len(self.channel_names)
        if image.ndim != 3 or image.shape[-1] != expected:
            raise ValueError(f"MultichannelPhotometric expected HxWx{expected}, got {image.shape}")
        if image.dtype != np.uint8:
            raise ValueError(f"MultichannelPhotometric expected uint8, got {image.dtype}")

    def _augment_rgb(self, rgb: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + np.random.uniform(-self.hue_gain, self.hue_gain) * 180.0) % 180.0
        hsv[..., 1] = np.clip(
            hsv[..., 1] * (1.0 + np.random.uniform(-self.saturation_gain, self.saturation_gain)), 0.0, 255.0
        )
        hsv[..., 2] = np.clip(
            hsv[..., 2] * (1.0 + np.random.uniform(-self.value_gain, self.value_gain)), 0.0, 255.0
        )
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    def _augment_extra(self, extra: np.ndarray) -> np.ndarray:
        count = extra.shape[2]
        values = extra.astype(np.float32) / 255.0
        gains = 1.0 + np.random.uniform(-self.extra_gain, self.extra_gain, size=(1, 1, count))
        biases = np.random.uniform(-self.extra_bias, self.extra_bias, size=(1, 1, count))
        noise: float | np.ndarray = 0.0
        if self.extra_noise:
            noise = np.random.normal(0.0, self.extra_noise, size=values.shape)
        return np.rint(np.clip(values * gains + biases + noise, 0.0, 1.0) * 255.0).astype(np.uint8)

    def apply_image(self, labels: dict[str, Any], params=None) -> dict[str, Any]:
        image = labels["img"]
        self._check_image(image)
        output = image.copy()
        if np.random.random() < self.rgb_probability:
            output[..., :3] = self._augment_rgb(output[..., :3])
        if output.shape[2] > 3 and np.random.random() < self.extra_probability:
            output[..., 3:] = self._augment_extra(output[..., 3:])
        labels["img"] = np.ascontiguousarray(output)
        return labels


def _preserve_manifest_rgb_order(transforms) -> None:
    """Disable Format's BGR->RGB reversal because feature arrays store RGB."""
    formatter = transforms.transforms[-1]
    if not hasattr(formatter, "bgr"):
        raise RuntimeError(f"Unexpected final Ultralytics transform: {type(formatter).__name__}")
    formatter.bgr = 1.0


class MultichannelYOLODataset(YOLODataset):
    def __init__(
        self,
        *args,
        feature_root: str | Path,
        all_channels: Sequence[str],
        selected_channels: Sequence[str],
        photometric_augment: bool = False,
        photometric_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        self.feature_root = Path(feature_root)
        self.all_channels = tuple(all_channels)
        self.selected_channels = tuple(selected_channels)
        self.channel_indices = validate_channel_selection(self.all_channels, self.selected_channels)
        self.photometric_augment = bool(photometric_augment)
        self.photometric_kwargs = dict(photometric_kwargs or {})
        super().__init__(*args, **kwargs)

    def get_img_files(self, img_path):
        files = _ORIGINAL_GET_IMG_FILES(self, img_path)
        return _filter_feature_files(files, self.feature_root, img_path)

    def get_labels(self):
        labels = super().get_labels()
        restored = _restore_distinct_segments_with_shared_boxes(labels)
        if restored:
            print(f"[multichannel] restored {restored} distinct segment(s) sharing an exact bounding box")
        return labels

    def load_image(self, index, rect_mode=True):
        return _load_multichannel_image(self, index)

    def build_transforms(self, hyp=None):
        transforms = super().build_transforms(hyp)
        _preserve_manifest_rgb_order(transforms)
        if self.augment and self.photometric_augment:
            transforms.insert(
                len(transforms.transforms) - 1,
                MultichannelPhotometric(self.selected_channels, **self.photometric_kwargs),
            )
        return transforms


def build_multichannel_dataset(
    cfg,
    img_path,
    batch,
    data,
    mode="train",
    rect=False,
    stride=32,
    *,
    feature_root: str | Path,
    all_channels: Sequence[str],
    selected_channels: Sequence[str],
    photometric_augment: bool = False,
    photometric_kwargs: dict[str, Any] | None = None,
):
    return MultichannelYOLODataset(
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
        feature_root=feature_root,
        all_channels=all_channels,
        selected_channels=selected_channels,
        photometric_augment=photometric_augment and mode == "train",
        photometric_kwargs=photometric_kwargs,
    )


def assert_dataloader_channels(dataloader, expected: int, name: str) -> None:
    sample = dataloader.dataset[0]
    image = sample["img"]
    actual = int(image.shape[0])
    if actual != expected:
        raise RuntimeError(f"{name} produced {actual} channels; expected {expected}")
    print(f"[multichannel] {name} verified: sample shape={tuple(image.shape)}")


class MultichannelSegmentationValidator(SegmentationValidator):
    def __init__(
        self,
        *args,
        feature_root: str | Path,
        all_channels: Sequence[str],
        selected_channels: Sequence[str],
        **kwargs,
    ) -> None:
        self.feature_root = Path(feature_root)
        self.all_channels = tuple(all_channels)
        self.selected_channels = tuple(selected_channels)
        super().__init__(*args, **kwargs)

    def build_dataset(self, img_path: str, mode: str = "val", batch: int | None = None):
        return build_multichannel_dataset(
            self.args,
            img_path,
            batch,
            self.data,
            mode=mode,
            stride=self.stride,
            feature_root=self.feature_root,
            all_channels=self.all_channels,
            selected_channels=self.selected_channels,
        )


class MultichannelSegmentationTrainer(SegmentationTrainer):
    def __init__(
        self,
        *args,
        feature_root: str | Path,
        all_channels: Sequence[str],
        selected_channels: Sequence[str],
        train_sampling_weights: Mapping[str, float] | None = None,
        photometric_augment: bool = False,
        photometric_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        self.feature_root = Path(feature_root)
        self.all_channels = tuple(all_channels)
        self.selected_channels = tuple(selected_channels)
        self.channel_indices = validate_channel_selection(self.all_channels, self.selected_channels)
        self.train_sampling_weights = (
            {str(name): float(weight) for name, weight in train_sampling_weights.items()}
            if train_sampling_weights is not None
            else None
        )
        self.photometric_augment = bool(photometric_augment)
        self.photometric_kwargs = dict(photometric_kwargs or {})
        super().__init__(*args, **kwargs)

    def setup_model(self):
        if self.resume and str(self.model).endswith(".pt"):
            self.model, checkpoint = load_checkpoint(self.model, device="cpu")
            self.model.criterion = None
            return checkpoint
        return super().setup_model()

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        stride = max(int(unwrap_model(self.model).stride.max()), 32)
        return build_multichannel_dataset(
            self.args,
            img_path,
            batch,
            self.data,
            mode=mode,
            rect=mode == "val",
            stride=stride,
            feature_root=self.feature_root,
            all_channels=self.all_channels,
            selected_channels=self.selected_channels,
            photometric_augment=self.photometric_augment,
            photometric_kwargs=self.photometric_kwargs,
        )

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """Build a loader with run-seed-specific shuffle and worker RNG state."""
        if mode not in {"train", "val"}:
            raise ValueError(f"mode must be 'train' or 'val', got {mode!r}")
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle and not np.all(dataset.batch_shapes == dataset.batch_shapes[0]):
            LOGGER.warning("'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        sample_weights = None
        if mode == "train" and self.train_sampling_weights is not None:
            image_names = [Path(path).name for path in dataset.im_files]
            if len(image_names) != len(set(image_names)):
                raise ValueError("Weighted sampling requires unique train image file names")
            missing = sorted(set(image_names) - self.train_sampling_weights.keys())
            if missing:
                raise ValueError(f"Sampling weights are missing {len(missing)} train images, e.g. {missing[:5]}")
            sample_weights = [self.train_sampling_weights[name] for name in image_names]
            LOGGER.info(
                "Using train-only weighted sampling: "
                f"images={len(sample_weights)}, min={min(sample_weights):.4f}, "
                f"mean={float(np.mean(sample_weights)):.4f}, max={max(sample_weights):.4f}"
            )
        return build_seeded_dataloader(
            dataset,
            batch=batch_size,
            workers=self.args.workers if mode == "train" else self.args.workers * 2,
            seed=int(self.args.seed),
            shuffle=shuffle,
            rank=rank,
            drop_last=self.args.compile and mode == "train",
            sample_weights=sample_weights,
        )

    def get_validator(self):
        self.loss_names = "box_loss", "seg_loss", "cls_loss", "dfl_loss", "sem_loss"
        return MultichannelSegmentationValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
            feature_root=self.feature_root,
            all_channels=self.all_channels,
            selected_channels=self.selected_channels,
        )

    def _build_train_pipeline(self):
        super()._build_train_pipeline()
        expected = len(self.selected_channels)
        assert_dataloader_channels(self.train_loader, expected, "train loader")
        assert_dataloader_channels(self.test_loader, expected, "validation loader")
