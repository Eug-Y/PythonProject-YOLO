import unittest

from verify_data_v21_deployment import validate_deployment_config


class VerifyDataV21DeploymentTests(unittest.TestCase):
    def test_rejects_non_v21_release_before_checkpoint_access(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "not the V21"):
            validate_deployment_config({"release_version": "incumbent_pre_v20"})


if __name__ == "__main__":
    unittest.main()
