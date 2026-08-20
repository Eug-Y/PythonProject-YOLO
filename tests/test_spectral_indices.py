from __future__ import annotations

import unittest

import numpy as np

from multispectral import MultichannelDataError, decode_multichannel, encode_multichannel_uint8, validate_multichannel
from spectral_indices import INDEX_REGISTRY, compute_index, compute_indices, normalize_dn, required_bands, safe_ratio


class SpectralIndicesTest(unittest.TestCase):
    def setUp(self) -> None:
        shape = (3, 4)
        self.bands = {
            "blue": np.full(shape, 0.10, dtype=np.float32),
            "green": np.full(shape, 0.30, dtype=np.float32),
            "red": np.full(shape, 0.20, dtype=np.float32),
            "rededge": np.full(shape, 0.40, dtype=np.float32),
            "nir": np.full(shape, 0.80, dtype=np.float32),
        }

    def test_ndvi_raw_and_encoded(self) -> None:
        raw = compute_index("NDVI", self.bands, encoded=False)
        encoded = compute_index("NDVI", self.bands)
        np.testing.assert_allclose(raw, 0.6, atol=2e-6)
        np.testing.assert_allclose(encoded, 0.8, atol=2e-6)

    def test_all_indices_are_finite_and_encoded(self) -> None:
        outputs = compute_indices(list(INDEX_REGISTRY), self.bands)
        self.assertEqual(set(outputs), set(INDEX_REGISTRY))
        for name, values in outputs.items():
            self.assertEqual(values.dtype, np.float32, name)
            self.assertTrue(np.isfinite(values).all(), name)
            self.assertGreaterEqual(float(values.min()), 0.0, name)
            self.assertLessEqual(float(values.max()), 1.0, name)

    def test_zero_bands_do_not_create_nan_or_inf(self) -> None:
        zeros = {name: np.zeros((2, 2), dtype=np.float32) for name in self.bands}
        for name in INDEX_REGISTRY:
            self.assertTrue(np.isfinite(compute_index(name, zeros)).all(), name)

    def test_dn_normalization_is_fixed_and_clipped(self) -> None:
        values = np.asarray([-1, 0, 32767.5, 65535, 70000], dtype=np.float32)
        np.testing.assert_allclose(normalize_dn(values), [0.0, 0.0, 0.5, 1.0, 1.0], atol=1e-6)

    def test_registry_marks_rededge_soil_proxy_nonstandard(self) -> None:
        self.assertFalse(INDEX_REGISTRY["RE_BSI_PROXY"].standard)
        self.assertEqual(required_bands(["NDVI", "NDRE"]), ("nir", "red", "rededge"))

    def test_new_v5_index_formulas(self) -> None:
        np.testing.assert_allclose(compute_index("VARI", self.bands, encoded=False), 0.25, atol=2e-6)
        np.testing.assert_allclose(compute_index("ARVI", self.bands, encoded=False), 5.0 / 11.0, atol=2e-6)
        np.testing.assert_allclose(compute_index("SIPI", self.bands, encoded=False), 7.0 / 6.0, atol=2e-6)
        np.testing.assert_allclose(compute_index("PSRI", self.bands, encoded=False), -0.25, atol=2e-6)
        np.testing.assert_allclose(compute_index("MCARI", self.bands, encoded=False), 0.36, atol=2e-6)
        np.testing.assert_allclose(compute_index("MTCI", self.bands, encoded=False), 2.0, atol=2e-5)

    def test_spatial_features_are_registered_as_nonstandard(self) -> None:
        for name in ("NDVI_GRAD", "NDVI_STD5"):
            self.assertFalse(INDEX_REGISTRY[name].standard)
            self.assertEqual(INDEX_REGISTRY[name].category, "spatial_feature")
        np.testing.assert_allclose(compute_index("NDVI_GRAD", self.bands, encoded=False), 0.0, atol=1e-7)
        np.testing.assert_allclose(compute_index("NDVI_STD5", self.bands, encoded=False), 0.0, atol=1e-4)

    def test_safe_ratio_preserves_denominator_sign(self) -> None:
        numerator = np.asarray([1.0, 1.0], dtype=np.float32)
        denominator = np.asarray([0.0, -0.0], dtype=np.float32)
        result = safe_ratio(numerator, denominator)
        self.assertTrue(np.isfinite(result).all())

    def test_missing_band_is_rejected(self) -> None:
        with self.assertRaisesRegex(KeyError, "missing bands"):
            compute_index("NDVI", {"nir": self.bands["nir"]})

    def test_uint8_multichannel_round_trip(self) -> None:
        names = ("R", "G", "NDVI")
        array = np.dstack((self.bands["red"], self.bands["green"], compute_index("NDVI", self.bands)))
        encoded = encode_multichannel_uint8(array, names)
        decoded = decode_multichannel(encoded, names)
        self.assertEqual(encoded.dtype, np.uint8)
        self.assertEqual(decoded.dtype, np.float32)
        np.testing.assert_allclose(decoded, array, atol=1 / 255)

    def test_multichannel_manifest_rejects_shape_or_range(self) -> None:
        with self.assertRaisesRegex(MultichannelDataError, "Expected HxWx2"):
            validate_multichannel(np.zeros((2, 2, 3), dtype=np.float32), ("R", "G"))
        with self.assertRaisesRegex(MultichannelDataError, "outside"):
            validate_multichannel(np.full((2, 2, 1), 1.5, dtype=np.float32), ("R",))


if __name__ == "__main__":
    unittest.main()
