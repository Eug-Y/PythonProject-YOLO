import unittest

from deploy_data_v20 import release_name


class DeployDataV20Tests(unittest.TestCase):
    def test_release_name_is_production_and_multichannel_discoverable(self) -> None:
        name = release_name(2)
        self.assertTrue(name.startswith("prod_v20_"))
        self.assertIn("5ch", name)
        self.assertIn("seed2", name)


if __name__ == "__main__":
    unittest.main()
