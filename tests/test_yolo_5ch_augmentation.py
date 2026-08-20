from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import unittest

from train_ca_5ch import adapt_first_conv_5ch
from yolo_5ch_data import FiveChannelPhotometric


def _sample_image() -> np.ndarray:
    y, x = np.mgrid[:12, :16]
    image = np.empty((12, 16, 5), dtype=np.uint8)
    image[..., 0] = (x * 11 + y * 3) % 256
    image[..., 1] = (x * 5 + y * 13) % 256
    image[..., 2] = (x * 17 + y * 7) % 256
    image[..., 3] = 80 + (x + y) % 80
    image[..., 4] = 100 + (x * 2 + y) % 80
    return image


def _apply(transform: FiveChannelPhotometric, image: np.ndarray) -> np.ndarray:
    return transform({"img": image.copy()})["img"]


class FiveChannelPhotometricTest(unittest.TestCase):
    def test_rgb_only_preserves_spectral_channels(self) -> None:
        image = _sample_image()
        transform = FiveChannelPhotometric(
            rgb_probability=1.0,
            hue_gain=0.2,
            saturation_gain=0.5,
            value_gain=0.4,
            spectral_probability=0.0,
        )

        np.random.seed(3)
        result = _apply(transform, image)

        self.assertFalse(np.array_equal(result[..., :3], image[..., :3]))
        np.testing.assert_array_equal(result[..., 3:], image[..., 3:])

    def test_spectral_only_preserves_rgb_channels(self) -> None:
        image = _sample_image()
        transform = FiveChannelPhotometric(
            rgb_probability=0.0,
            spectral_probability=1.0,
            spectral_gain=0.2,
            spectral_bias=0.1,
            spectral_noise=0.02,
        )

        np.random.seed(4)
        result = _apply(transform, image)

        np.testing.assert_array_equal(result[..., :3], image[..., :3])
        self.assertFalse(np.array_equal(result[..., 3:], image[..., 3:]))

    def test_output_contract_and_seed_reproducibility(self) -> None:
        image = _sample_image()
        transform = FiveChannelPhotometric(rgb_probability=1.0, spectral_probability=1.0)

        np.random.seed(17)
        first = _apply(transform, image)
        np.random.seed(17)
        second = _apply(transform, image)

        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, image.shape)
        self.assertEqual(first.dtype, np.uint8)
        self.assertTrue(first.flags.c_contiguous)
        self.assertGreaterEqual(int(first.min()), 0)
        self.assertLessEqual(int(first.max()), 255)

    def test_invalid_image_contract_raises(self) -> None:
        cases = [
            (np.zeros((8, 8, 3), dtype=np.uint8), "HxWx5"),
            (np.zeros((8, 8, 5), dtype=np.float32), "uint8"),
        ]
        for bad_image, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                _apply(FiveChannelPhotometric(), bad_image)


class _FirstBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, stride=2, padding=1, bias=True)


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.ModuleList([_FirstBlock()])


class FirstConvInitializationTest(unittest.TestCase):
    def test_initialization_preserves_rgb_weights(self) -> None:
        for mode in ("mean", "zero", "kaiming"):
            with self.subTest(mode=mode):
                torch.manual_seed(23)
                model = _TinyModel()
                old_weight = model.model[0].conv.weight.detach().clone()
                old_bias = model.model[0].conv.bias.detach().clone()

                torch.manual_seed(29)
                adapt_first_conv_5ch(model, spectral_init=mode)
                new_conv = model.model[0].conv

                self.assertEqual(new_conv.in_channels, 5)
                torch.testing.assert_close(new_conv.weight[:, :3], old_weight)
                torch.testing.assert_close(new_conv.bias, old_bias)

                spectral = new_conv.weight[:, 3:]
                if mode == "mean":
                    expected = old_weight.mean(dim=1, keepdim=True).expand_as(spectral)
                    torch.testing.assert_close(spectral, expected)
                elif mode == "zero":
                    torch.testing.assert_close(spectral, torch.zeros_like(spectral))
                else:
                    self.assertTrue(bool(torch.isfinite(spectral).all()))
                    self.assertGreater(int(torch.count_nonzero(spectral)), 0)

    def test_rejects_unknown_initialization(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported spectral initialization"):
            adapt_first_conv_5ch(_TinyModel(), spectral_init="unknown")


if __name__ == "__main__":
    unittest.main()
