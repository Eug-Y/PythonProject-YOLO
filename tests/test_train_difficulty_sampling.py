from __future__ import annotations

import json
import unittest
from pathlib import Path

import torch
from torch.utils.data import Dataset

from yolo_multichannel_data import build_seeded_dataloader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = PROJECT_ROOT / "configs" / "generated" / "data_v8_train_difficulty_weights_h1.json"


class IndexDataset(Dataset):
    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> int:
        return index


def one_epoch(seed: int, weights: list[float]) -> list[int]:
    loader = build_seeded_dataloader(
        IndexDataset(len(weights)),
        batch=8,
        workers=0,
        seed=seed,
        rank=-1,
        sample_weights=weights,
        pin_memory=False,
    )
    return torch.cat([batch for batch in loader]).tolist()


class WeightedSamplerTests(unittest.TestCase):
    def test_same_seed_repeats_exact_order(self) -> None:
        weights = [1.0] * 31 + [2.0]
        self.assertEqual(one_epoch(7, weights), one_epoch(7, weights))

    def test_different_seed_changes_order(self) -> None:
        weights = [1.0] * 31 + [2.0]
        self.assertNotEqual(one_epoch(7, weights), one_epoch(8, weights))

    def test_epoch_size_is_unchanged(self) -> None:
        weights = [1.0] * 31 + [2.0]
        self.assertEqual(len(one_epoch(7, weights)), len(weights))

    def test_invalid_weights_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            one_epoch(7, [1.0, 0.0])

    def test_generated_contract_is_train_only_and_complete(self) -> None:
        payload = json.loads(WEIGHTS_PATH.read_text(encoding="utf-8"))
        self.assertEqual(payload["source_split"], "train")
        self.assertFalse(payload["validation_split_used"])
        self.assertFalse(payload["held_out_split_used"])
        self.assertEqual(payload["train_images"], 210)
        self.assertEqual(len(payload["weights_by_image"]), 210)
        self.assertEqual(payload["sampling"]["samples_per_epoch"], 210)
        self.assertLessEqual(payload["sampling"]["max_to_min_ratio"], 2.0)


if __name__ == "__main__":
    unittest.main()
