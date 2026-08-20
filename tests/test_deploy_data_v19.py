import unittest

from deploy_data_v19 import release_name


class DeployDataV19Tests(unittest.TestCase):
    def test_release_name_is_v19_multichannel_release(self) -> None:
        self.assertEqual(release_name(0), "prod_v19_5ch_boundary_nooverlap_seed0_v1")


if __name__ == "__main__":
    unittest.main()
