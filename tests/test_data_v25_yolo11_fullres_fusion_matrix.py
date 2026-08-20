import unittest

from run_data_v25_yolo11_fullres_fusion_matrix import configs, protocol


class DataV25FullResolutionMatrixTests(unittest.TestCase):
    def test_matrix_is_locked_and_yolo11_primary_has_priority(self) -> None:
        rows = configs()
        self.assertEqual(len(rows), 24)
        strict = [row for row in rows if row["role"] == "strict_yolo11_primary"]
        self.assertEqual(len(strict), 12)
        self.assertTrue(all(row["required_model_indices"] == [0] for row in strict))
        self.assertEqual(protocol()["selection_split"], "valid")
        self.assertFalse(protocol()["held_out_test_used"])


if __name__ == "__main__":
    unittest.main()
