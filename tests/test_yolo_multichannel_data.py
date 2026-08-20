from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from ultralytics.data.augment import Format

from train_feature_ablation import adapt_first_conv_channels
from yolo_multichannel_data import (
    MultichannelPhotometric,
    _load_selected_array,
    _preserve_manifest_rgb_order,
    _restore_distinct_segments_with_shared_boxes,
    build_seeded_dataloader,
    feature_path_for_image,
    validate_channel_selection,
)


ALL_CHANNELS = ("R", "G", "B", "NIR", "RedEdge", "NDVI", "NDRE", "EVI2")


class ChannelSelectionTest(unittest.TestCase):
    def test_distinct_segments_with_same_box_are_restored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "train" / "images" / "sample.JPG"
            label_path = root / "train" / "labels" / "sample.txt"
            image_path.parent.mkdir(parents=True)
            label_path.parent.mkdir(parents=True)
            image_path.touch()
            rectangle = "0 0.1 0.1 0.4 0.1 0.4 0.4 0.1 0.4"
            diamond = "0 0.1 0.2 0.2 0.1 0.4 0.2 0.2 0.4"
            label_path.write_text(f"{rectangle}\n{diamond}\n", encoding="utf-8")
            first_segment = np.asarray(rectangle.split()[1:], dtype=np.float32).reshape(-1, 2)
            labels = [{
                "im_file": str(image_path),
                "cls": np.asarray([[0.0]], dtype=np.float32),
                "bboxes": np.asarray([[0.25, 0.25, 0.3, 0.3]], dtype=np.float32),
                "segments": [first_segment],
            }]
            restored = _restore_distinct_segments_with_shared_boxes(labels)
            self.assertEqual(restored, 1)
            self.assertEqual(len(labels[0]["segments"]), 2)
            np.testing.assert_array_equal(labels[0]["bboxes"][0], labels[0]["bboxes"][1])

    def test_selection_preserves_requested_order(self) -> None:
        selected = ("R", "G", "B", "EVI2", "NIR")
        self.assertEqual(validate_channel_selection(ALL_CHANNELS, selected), (0, 1, 2, 7, 3))

    def test_requires_rgb_prefix(self) -> None:
        with self.assertRaisesRegex(ValueError, "must start"):
            validate_channel_selection(ALL_CHANNELS, ("B", "G", "R"))

    def test_feature_path_uses_locked_split_and_stem(self) -> None:
        result = feature_path_for_image(Path("dataset/train/images/DJI_0130.JPG"), Path("features"))
        self.assertEqual(result, Path("features/train/features/DJI_0130.npy"))

    def test_array_loader_selects_exact_values(self) -> None:
        source = np.zeros((4, 5, len(ALL_CHANNELS)), dtype=np.uint8)
        for index in range(len(ALL_CHANNELS)):
            source[..., index] = index * 10
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.npy"
            np.save(path, source, allow_pickle=False)
            result = _load_selected_array(path, (0, 1, 2, 7, 3), len(ALL_CHANNELS))
        self.assertEqual(result.shape, (4, 5, 5))
        np.testing.assert_array_equal(result[0, 0], np.asarray([0, 10, 20, 70, 30], dtype=np.uint8))
        self.assertTrue(result.flags.c_contiguous)


class RGBOrderTest(unittest.TestCase):
    def test_format_does_not_reverse_manifest_rgb(self) -> None:
        formatter = Format(bgr=0.0)

        class _Transforms:
            transforms = [formatter]

        _preserve_manifest_rgb_order(_Transforms())
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        image[..., 0] = 11
        image[..., 1] = 22
        image[..., 2] = 33
        tensor = formatter._format_img(image)
        torch.testing.assert_close(tensor[:, 0, 0], torch.tensor([11, 22, 33], dtype=torch.uint8))


class SeededDataLoaderTest(unittest.TestCase):
    class _IndexDataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return 24

        def __getitem__(self, index: int) -> int:
            return index

    def _order(self, seed: int) -> list[int]:
        loader = build_seeded_dataloader(self._IndexDataset(), batch=4, workers=0, seed=seed)
        return torch.cat(list(loader)).tolist()

    def test_run_seed_is_reproducible_and_changes_shuffle_order(self) -> None:
        self.assertEqual(self._order(1), self._order(1))
        self.assertNotEqual(self._order(0), self._order(1))


class PhotometricTest(unittest.TestCase):
    def test_rgb_and_extra_are_independently_controlled(self) -> None:
        y, x = np.mgrid[:10, :12]
        image = np.empty((10, 12, 6), dtype=np.uint8)
        image[..., 0] = (x * 13 + y) % 255
        image[..., 1] = (x * 3 + y * 9) % 255
        image[..., 2] = (x * 7 + y * 5) % 255
        image[..., 3:] = 120
        channels = ("R", "G", "B", "EVI2", "GNDVI", "RE_BSI_PROXY")

        rgb_only = MultichannelPhotometric(channels, rgb_probability=1.0, extra_probability=0.0)
        np.random.seed(5)
        rgb_result = rgb_only({"img": image.copy()})["img"]
        self.assertFalse(np.array_equal(rgb_result[..., :3], image[..., :3]))
        np.testing.assert_array_equal(rgb_result[..., 3:], image[..., 3:])

        extra_only = MultichannelPhotometric(
            channels,
            rgb_probability=0.0,
            extra_probability=1.0,
            extra_gain=0.2,
            extra_bias=0.1,
            extra_noise=0.02,
        )
        np.random.seed(7)
        extra_result = extra_only({"img": image.copy()})["img"]
        np.testing.assert_array_equal(extra_result[..., :3], image[..., :3])
        self.assertFalse(np.array_equal(extra_result[..., 3:], image[..., 3:]))


class _FirstBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1, bias=True)


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.ModuleList([_FirstBlock()])


class FirstConvTest(unittest.TestCase):
    def test_preserves_rgb_and_initializes_all_extra_channels(self) -> None:
        torch.manual_seed(31)
        model = _TinyModel()
        old = model.model[0].conv.weight.detach().clone()
        adapt_first_conv_channels(model, 6, "mean")
        new = model.model[0].conv.weight
        torch.testing.assert_close(new[:, :3], old)
        torch.testing.assert_close(new[:, 3:], old.mean(1, keepdim=True).expand_as(new[:, 3:]))

    def test_three_channel_model_is_unchanged(self) -> None:
        model = _TinyModel()
        original = model.model[0].conv
        adapt_first_conv_channels(model, 3, "mean")
        self.assertIs(model.model[0].conv, original)

    def test_five_channel_checkpoint_can_expand_for_new_indices(self) -> None:
        torch.manual_seed(47)
        model = _TinyModel()
        model.model[0].conv = nn.Conv2d(5, 8, 3, padding=1, bias=True)
        old = model.model[0].conv.weight.detach().clone()
        adapt_first_conv_channels(model, 7, "mean")
        new = model.model[0].conv.weight
        torch.testing.assert_close(new[:, :5], old)
        torch.testing.assert_close(new[:, 5:], old.mean(1, keepdim=True).expand_as(new[:, 5:]))

    def test_channel_expansion_never_silently_drops_existing_kernels(self) -> None:
        model = _TinyModel()
        model.model[0].conv = nn.Conv2d(5, 8, 3, padding=1)
        with self.assertRaisesRegex(ValueError, "cannot shrink"):
            adapt_first_conv_channels(model, 4, "mean")


if __name__ == "__main__":
    unittest.main()
