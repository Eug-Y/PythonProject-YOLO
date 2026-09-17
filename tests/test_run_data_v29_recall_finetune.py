import argparse
import unittest

from run_data_v29_recall_finetune import recall_gate, training_command


class RunDataV29RecallFinetuneTests(unittest.TestCase):
    def test_training_command_is_low_lr_v19_finetune_without_mosaic(self) -> None:
        args = argparse.Namespace(epochs=12, patience=0, imgsz=1024, batch=1, workers=2, device="0")
        command = training_command(args)
        self.assertIn("--init-checkpoint", command)
        self.assertIn("--sampling-weights", command)
        self.assertEqual(command[command.index("--lr0") + 1], "0.0002")
        self.assertEqual(command[command.index("--patience") + 1], "0")
        self.assertEqual(command[command.index("--mosaic") + 1], "0.0")
        self.assertEqual(command[command.index("--mixup") + 1], "0.0")
        self.assertIn("--no-overlap-mask", command)

    def test_recall_gate_requires_positive_recall_gain_and_original_guards(self) -> None:
        baseline = {"f1": 0.64, "precision": 0.64, "recall": 0.64, "mean_matched_iou": 0.84}
        candidate = {"f1": 0.655, "precision": 0.64, "recall": 0.65, "mean_matched_iou": 0.835}
        self.assertTrue(recall_gate(candidate, baseline)["passed"])
        candidate["recall"] = 0.645
        self.assertFalse(recall_gate(candidate, baseline)["passed"])


if __name__ == "__main__":
    unittest.main()
