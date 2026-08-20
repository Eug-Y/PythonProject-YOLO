"""Numerically stable spectral-index registry for DJI P4 Multispectral bands.

Inputs are reflectance-like Blue/Green/Red/RedEdge/NIR arrays normalized to
[0, 1]. Every encoded output is float32 in [0, 1], suitable for storage or a
neural-network input channel. Raw scientific values remain available for
feature analysis through ``compute_index(..., encoded=False)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
import cv2


EPSILON = 1e-6
BandMap = Mapping[str, np.ndarray]
Formula = Callable[[BandMap], np.ndarray]


@dataclass(frozen=True)
class IndexSpec:
    name: str
    category: str
    required_bands: tuple[str, ...]
    raw_range: tuple[float, float]
    formula: Formula
    description: str
    standard: bool = True
    caveat: str | None = None


def normalize_dn(array: np.ndarray, maximum: float = 65535.0) -> np.ndarray:
    """Convert uint16-like DN values to a fixed [0, 1] scale."""
    values = np.asarray(array, dtype=np.float32)
    return np.clip(values / maximum, 0.0, 1.0).astype(np.float32, copy=False)


def normalized_difference(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    high = np.asarray(high, dtype=np.float32)
    low = np.asarray(low, dtype=np.float32)
    return (high - low) / (high + low + EPSILON)


def safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Divide with a sign-preserving epsilon for ratios that can cross zero."""
    numerator = np.asarray(numerator, dtype=np.float32)
    denominator = np.asarray(denominator, dtype=np.float32)
    stabilized = np.where(
        np.abs(denominator) < EPSILON,
        np.where(denominator < 0.0, -EPSILON, EPSILON),
        denominator,
    )
    return numerator / stabilized


def _ndvi(bands: BandMap) -> np.ndarray:
    return normalized_difference(bands["nir"], bands["red"])


def _ndre(bands: BandMap) -> np.ndarray:
    return normalized_difference(bands["nir"], bands["rededge"])


def _gndvi(bands: BandMap) -> np.ndarray:
    return normalized_difference(bands["nir"], bands["green"])


def _savi(bands: BandMap) -> np.ndarray:
    nir, red = bands["nir"], bands["red"]
    return 1.5 * (nir - red) / (nir + red + 0.5 + EPSILON)


def _osavi(bands: BandMap) -> np.ndarray:
    nir, red = bands["nir"], bands["red"]
    return 1.16 * (nir - red) / (nir + red + 0.16 + EPSILON)


def _msavi2(bands: BandMap) -> np.ndarray:
    nir, red = bands["nir"], bands["red"]
    leading = 2.0 * nir + 1.0
    discriminant = np.maximum(leading * leading - 8.0 * (nir - red), 0.0)
    return 0.5 * (leading - np.sqrt(discriminant))


def _evi2(bands: BandMap) -> np.ndarray:
    nir, red = bands["nir"], bands["red"]
    return 2.5 * (nir - red) / (nir + 2.4 * red + 1.0 + EPSILON)


def _reci(bands: BandMap) -> np.ndarray:
    return bands["nir"] / (bands["rededge"] + EPSILON) - 1.0


def _brightness_index(bands: BandMap) -> np.ndarray:
    red, green, blue = bands["red"], bands["green"], bands["blue"]
    return np.sqrt((red * red + green * green + blue * blue) / 3.0)


def _soil_colour_index(bands: BandMap) -> np.ndarray:
    red, green = bands["red"], bands["green"]
    return (red - green) / (red + green + EPSILON)


def _excess_green(bands: BandMap) -> np.ndarray:
    return 2.0 * bands["green"] - bands["red"] - bands["blue"]


def _rededge_bare_soil_proxy(bands: BandMap) -> np.ndarray:
    """Experimental P4M proxy; RedEdge is not presented as SWIR/standard BSI."""
    rededge, red, nir, blue = bands["rededge"], bands["red"], bands["nir"], bands["blue"]
    numerator = rededge + red - nir - blue
    denominator = rededge + red + nir + blue + EPSILON
    return numerator / denominator


def _vari(bands: BandMap) -> np.ndarray:
    """Visible atmospherically resistant index."""
    green, red, blue = bands["green"], bands["red"], bands["blue"]
    return safe_ratio(green - red, green + red - blue)


def _arvi(bands: BandMap) -> np.ndarray:
    """Atmospherically resistant vegetation index with gamma=1."""
    nir, red, blue = bands["nir"], bands["red"], bands["blue"]
    corrected_red = 2.0 * red - blue
    return safe_ratio(nir - corrected_red, nir + corrected_red)


def _sipi(bands: BandMap) -> np.ndarray:
    """Structure-insensitive pigment index."""
    nir, blue, red = bands["nir"], bands["blue"], bands["red"]
    return safe_ratio(nir - blue, nir - red)


def _psri(bands: BandMap) -> np.ndarray:
    """Plant senescence reflectance index using P4M RedEdge as the denominator."""
    return safe_ratio(bands["red"] - bands["green"], bands["rededge"])


def _mcari(bands: BandMap) -> np.ndarray:
    """Modified chlorophyll absorption ratio index mapped to P4M bands."""
    rededge, red, green = bands["rededge"], bands["red"], bands["green"]
    return ((rededge - red) - 0.2 * (rededge - green)) * safe_ratio(rededge, red)


def _mtci(bands: BandMap) -> np.ndarray:
    """MERIS terrestrial chlorophyll index mapped to P4M bands."""
    nir, rededge, red = bands["nir"], bands["rededge"], bands["red"]
    return safe_ratio(nir - rededge, rededge - red)


def _ndvi_gradient(bands: BandMap) -> np.ndarray:
    """Spatial NDVI boundary magnitude, not a standard spectral index."""
    ndvi = _ndvi(bands)
    grad_y, grad_x = np.gradient(ndvi)
    return np.hypot(grad_x, grad_y)


def _ndvi_local_std(bands: BandMap) -> np.ndarray:
    """Five-pixel local NDVI standard deviation, a spatial heterogeneity feature."""
    ndvi = _ndvi(bands).astype(np.float32, copy=False)
    mean = cv2.boxFilter(ndvi, -1, (5, 5), normalize=True, borderType=cv2.BORDER_REFLECT101)
    mean_square = cv2.boxFilter(ndvi * ndvi, -1, (5, 5), normalize=True, borderType=cv2.BORDER_REFLECT101)
    return np.sqrt(np.maximum(mean_square - mean * mean, 0.0))


INDEX_REGISTRY: dict[str, IndexSpec] = {
    spec.name: spec
    for spec in (
        IndexSpec("NDVI", "vegetation", ("nir", "red"), (-1.0, 1.0), _ndvi, "Normalized difference vegetation index."),
        IndexSpec("NDRE", "vegetation", ("nir", "rededge"), (-1.0, 1.0), _ndre, "Normalized difference red-edge index."),
        IndexSpec("GNDVI", "vegetation", ("nir", "green"), (-1.0, 1.0), _gndvi, "Green normalized difference vegetation index."),
        IndexSpec("SAVI", "soil_adjusted_vegetation", ("nir", "red"), (-1.0, 1.0), _savi, "Soil-adjusted vegetation index with L=0.5."),
        IndexSpec("OSAVI", "soil_adjusted_vegetation", ("nir", "red"), (-1.0, 1.0), _osavi, "Optimized soil-adjusted vegetation index."),
        IndexSpec("MSAVI2", "soil_adjusted_vegetation", ("nir", "red"), (-1.0, 1.0), _msavi2, "Second modified soil-adjusted vegetation index."),
        IndexSpec("EVI2", "vegetation", ("nir", "red"), (-1.0, 1.25), _evi2, "Two-band enhanced vegetation index."),
        IndexSpec("RECI", "vegetation", ("nir", "rededge"), (-1.0, 4.0), _reci, "Red-edge chlorophyll index."),
        IndexSpec("BI", "soil_proxy", ("red", "green", "blue"), (0.0, 1.0), _brightness_index, "Visible-band brightness index; illumination-sensitive."),
        IndexSpec("SCI", "soil_proxy", ("red", "green"), (-1.0, 1.0), _soil_colour_index, "Red-green soil colour contrast."),
        IndexSpec("ExG", "vegetation_visible", ("red", "green", "blue"), (-2.0, 2.0), _excess_green, "Visible excess-green index."),
        IndexSpec(
            "RE_BSI_PROXY",
            "soil_proxy",
            ("rededge", "red", "nir", "blue"),
            (-1.0, 1.0),
            _rededge_bare_soil_proxy,
            "Experimental RedEdge bare-soil proxy; not the standard SWIR-based BSI.",
            standard=False,
        ),
        IndexSpec(
            "VARI",
            "vegetation_visible",
            ("green", "red", "blue"),
            (-1.0, 1.0),
            _vari,
            "Visible atmospherically resistant index; candidate for illumination-robust vegetation contrast.",
        ),
        IndexSpec(
            "ARVI",
            "vegetation",
            ("nir", "red", "blue"),
            (-1.0, 1.0),
            _arvi,
            "Atmospherically resistant vegetation index with gamma=1.",
        ),
        IndexSpec(
            "SIPI",
            "pigment",
            ("nir", "blue", "red"),
            (0.0, 2.0),
            _sipi,
            "Structure-insensitive pigment index; candidate for pigment and stress contrast.",
        ),
        IndexSpec(
            "PSRI",
            "senescence",
            ("red", "green", "rededge"),
            (-1.0, 1.0),
            _psri,
            "Plant senescence reflectance index using the P4M RedEdge band.",
            caveat="P4M RedEdge is used as the closest available band to the original PSRI reference wavelength.",
        ),
        IndexSpec(
            "MCARI",
            "rededge_chlorophyll",
            ("rededge", "red", "green"),
            (-0.5, 1.5),
            _mcari,
            "Modified chlorophyll absorption ratio index mapped to P4M Green/Red/RedEdge bands.",
            caveat="Band-adapted to DJI P4M center wavelengths; validate empirically before deployment.",
        ),
        IndexSpec(
            "MTCI",
            "rededge_chlorophyll",
            ("nir", "rededge", "red"),
            (-2.0, 10.0),
            _mtci,
            "Terrestrial chlorophyll index mapped to P4M Red/RedEdge/NIR bands.",
            caveat="Band-adapted from MTCI; near-zero RedEdge-Red differences are clipped by the encoded range.",
        ),
        IndexSpec(
            "NDVI_GRAD",
            "spatial_feature",
            ("nir", "red"),
            (0.0, float(np.sqrt(8.0))),
            _ndvi_gradient,
            "NDVI spatial-gradient magnitude for vegetation-boundary contrast.",
            standard=False,
            caveat="Project spatial feature, not a standard spectral index.",
        ),
        IndexSpec(
            "NDVI_STD5",
            "spatial_feature",
            ("nir", "red"),
            (0.0, 1.0),
            _ndvi_local_std,
            "Five-pixel local NDVI standard deviation for spectral heterogeneity.",
            standard=False,
            caveat="Project spatial feature, not a standard spectral index.",
        ),
    )
}


def encode_index(values: np.ndarray, raw_range: tuple[float, float]) -> np.ndarray:
    minimum, maximum = raw_range
    encoded = (np.asarray(values, dtype=np.float32) - minimum) / (maximum - minimum)
    encoded = np.nan_to_num(encoded, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(encoded, 0.0, 1.0).astype(np.float32, copy=False)


def compute_index(name: str, bands: BandMap, *, encoded: bool = True) -> np.ndarray:
    if name not in INDEX_REGISTRY:
        raise KeyError(f"Unknown spectral index {name!r}; expected one of {sorted(INDEX_REGISTRY)}")
    spec = INDEX_REGISTRY[name]
    missing = [band for band in spec.required_bands if band not in bands]
    if missing:
        raise KeyError(f"{name} requires missing bands: {missing}")
    raw = np.asarray(spec.formula(bands), dtype=np.float32)
    if not np.isfinite(raw).all():
        raw = np.nan_to_num(raw, nan=0.0, posinf=spec.raw_range[1], neginf=spec.raw_range[0])
    return encode_index(raw, spec.raw_range) if encoded else raw


def compute_indices(names: tuple[str, ...] | list[str], bands: BandMap, *, encoded: bool = True) -> dict[str, np.ndarray]:
    return {name: compute_index(name, bands, encoded=encoded) for name in names}


def required_bands(index_names: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    required: set[str] = set()
    for name in index_names:
        if name not in INDEX_REGISTRY:
            raise KeyError(f"Unknown spectral index {name!r}")
        required.update(INDEX_REGISTRY[name].required_bands)
    return tuple(sorted(required))


def registry_manifest() -> dict[str, dict[str, object]]:
    return {
        name: {
            "category": spec.category,
            "required_bands": list(spec.required_bands),
            "raw_range": list(spec.raw_range),
            "description": spec.description,
            "standard": spec.standard,
            "caveat": spec.caveat,
        }
        for name, spec in INDEX_REGISTRY.items()
    }
