import unittest

from build_release_blind_holdout import contiguous_blocks, select_largest_complete_blocks


class ReleaseBlindHoldoutTests(unittest.TestCase):
    def test_contiguous_blocks_use_locked_step(self) -> None:
        self.assertEqual(contiguous_blocks([10, 20, 40, 50, 60]), [[10, 20], [40, 50, 60]])

    def test_selects_largest_complete_blocks_without_slicing(self) -> None:
        blocks = [list(range(0, 16)), list(range(20, 28)), list(range(30, 36)), [50, 51, 52]]
        selected = select_largest_complete_blocks(blocks, 30)
        self.assertEqual([len(block) for block in selected], [16, 8, 6])
        self.assertEqual(sum(map(len, selected)), 30)


if __name__ == "__main__":
    unittest.main()
