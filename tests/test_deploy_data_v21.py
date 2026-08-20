import unittest

from deploy_data_v21 import release_name


class DeployDataV21Tests(unittest.TestCase):
    def test_release_name_is_production_and_multichannel_discoverable(self) -> None:
        name = release_name(1)
        self.assertTrue(name.startswith("prod_v21_"))
        self.assertIn("5ch", name)
        self.assertIn("seed1", name)


if __name__ == "__main__":
    unittest.main()
