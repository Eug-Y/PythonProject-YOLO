import unittest

import torch

from build_data_v16_model_soup import average_state_dicts, first_tensor
from run_data_v16_model_soup import gate


class V16ModelSoupTests(unittest.TestCase):
    def test_average_state_dicts_averages_float_and_preserves_equal_integer(self) -> None:
        states = [
            {"weight": torch.tensor([value], dtype=torch.float32), "count": torch.tensor(3)}
            for value in (1.0, 2.0, 3.0)
        ]
        averaged, mismatches = average_state_dicts(states)
        self.assertEqual(float(averaged["weight"].item()), 2.0)
        self.assertEqual(int(averaged["count"].item()), 3)
        self.assertEqual(mismatches, [])

    def test_first_tensor_handles_nested_yolo_output(self) -> None:
        value = (None, (torch.zeros(1, 2, 3),))
        self.assertEqual(tuple(first_tensor(value).shape), (1, 2, 3))

    def test_gate_requires_gain_over_strongest_seed(self) -> None:
        def payload(map_value: float, precision: float = 0.7, recall: float = 0.6) -> dict:
            return {"aggregate_metrics": {
                "metrics/mAP50-95(M)": map_value,
                "metrics/mAP50(M)": map_value + 0.2,
                "metrics/precision(M)": precision,
                "metrics/recall(M)": recall,
            }}
        self.assertTrue(gate(payload(0.37), payload(0.373))["passed"])
        self.assertFalse(gate(payload(0.37), payload(0.371))["passed"])


if __name__ == "__main__":
    unittest.main()
