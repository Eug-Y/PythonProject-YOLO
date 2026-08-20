import unittest

from build_data_v22_dense_replay_dataset import flight_groups, select_balanced_replays


class DataV22DenseReplayDatasetTests(unittest.TestCase):
    def test_flight_groups_split_large_numeric_gaps(self) -> None:
        groups = flight_groups(["DJI_0010.JPG", "DJI_0020.JPG", "DJI_0100.JPG"])
        self.assertEqual(groups["DJI_0010.JPG"], groups["DJI_0020.JPG"])
        self.assertNotEqual(groups["DJI_0020.JPG"], groups["DJI_0100.JPG"])

    def test_selection_uses_top_quartile_and_group_diversity(self) -> None:
        rows = [
            {
                "source_image": f"DJI_{index:04d}.JPG",
                "source_group": f"g{index % 4}",
                "overlap_foreground_fraction": index / 20.0,
                "overlap_instances": index + 1,
            }
            for index in range(20)
        ]
        selected, threshold = select_balanced_replays(rows, 4)
        self.assertEqual(len(selected), 4)
        self.assertEqual(len({row["source_group"] for row in selected}), 4)
        self.assertTrue(all(float(row["overlap_foreground_fraction"]) >= threshold for row in selected))


if __name__ == "__main__":
    unittest.main()
