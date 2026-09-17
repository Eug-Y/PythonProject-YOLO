import unittest

from build_data_v29_recall_sampling import build_weights, effective_sample_size, select_recall_scenes


class DataV29RecallSamplingTests(unittest.TestCase):
    def test_selection_prefers_group_diversity_before_repeats(self) -> None:
        rows = [
            {"source_image": "a.JPG", "source_group": "g1", "fn": 10, "fn_rate": 0.5},
            {"source_image": "b.JPG", "source_group": "g1", "fn": 9, "fn_rate": 0.6},
            {"source_image": "c.JPG", "source_group": "g2", "fn": 8, "fn_rate": 0.7},
        ]
        selected = select_recall_scenes(rows, 2)
        self.assertEqual([row["source_image"] for row in selected], ["a.JPG", "c.JPG"])

    def test_weights_only_boost_selected_positive_scenes(self) -> None:
        selected = [
            {"source_image": "a.JPG"},
            {"source_image": "c.JPG"},
        ]
        weights = build_weights(["a.JPG", "b.JPG", "c.JPG"], selected, minimum_boost=1.1, maximum_boost=1.4)
        self.assertEqual(weights, {"a.JPG": 1.4, "b.JPG": 1.0, "c.JPG": 1.1})
        self.assertGreater(effective_sample_size(list(weights.values())), 2.9)


if __name__ == "__main__":
    unittest.main()
