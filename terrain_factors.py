"""Terrain factors derived from a pixel-aligned elevation raster.

The RGB/multispectral photographs in this project do not contain elevation.
Callers must supply a DEM/DSM/DTM aligned to each image.  Every returned
channel is float32 in [0, 1] so it can be appended to the existing candidate
feature arrays without changing the multichannel loader contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import cv2
import numpy as np


TERRAIN_CHANNELS = (
    "SLOPE",
    "ASPECT_SIN",
    "ASPECT_COS",
    "TRI",
    "TPI",
    "RELIEF5",
)


class TerrainDataError(ValueError):
    """Raised when an elevation raster cannot produce trustworthy factors."""


def load_elevation(path: str | Path, target_hw: tuple[int, int] | None = None) -> np.ndarray:
    """Load a single-band NPY/TIF elevation raster and optionally align its size."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".npy":
        elevation = np.load(path, allow_pickle=False)
    elif path.suffix.lower() in {".tif", ".tiff"}:
        raw = np.fromfile(str(path), dtype=np.uint8)
        elevation = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
        if elevation is None:
            raise TerrainDataError(f"Cannot decode elevation raster: {path}")
    else:
        raise TerrainDataError(f"Elevation raster must be .npy, .tif or .tiff: {path}")

    elevation = np.asarray(elevation).squeeze()
    if elevation.ndim != 2:
        raise TerrainDataError(f"Expected a single-band elevation raster, got {elevation.shape}: {path}")
    elevation = elevation.astype(np.float32, copy=False)
    finite = np.isfinite(elevation)
    if not finite.any():
        raise TerrainDataError(f"Elevation raster contains no finite pixels: {path}")
    if not finite.all():
        elevation = elevation.copy()
        elevation[~finite] = float(np.median(elevation[finite]))
    if target_hw is not None and elevation.shape != tuple(target_hw):
        target_h, target_w = target_hw
        elevation = cv2.resize(elevation, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(elevation, dtype=np.float32)


def _validate_scale(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise TerrainDataError(f"{name} must be a positive finite value, got {value}")
    return value


def compute_terrain_factors(
    elevation: np.ndarray,
    *,
    pixel_size_m: float | tuple[float, float],
    tri_scale_m: float = 2.0,
    tpi_scale_m: float = 2.0,
    relief_scale_m: float = 4.0,
) -> dict[str, np.ndarray]:
    """Compute encoded slope, aspect and local-relief factors.

    ``pixel_size_m`` is either one square-pixel ground sampling distance or
    ``(y_size, x_size)``. Aspect is represented by sine/cosine channels to
    avoid the discontinuity between 359 and 0 degrees. Flat pixels use the
    neutral encoded value 0.5 for both aspect channels.
    """
    z = np.asarray(elevation, dtype=np.float32)
    if z.ndim != 2 or min(z.shape) < 3:
        raise TerrainDataError(f"Elevation must be a 2D raster of at least 3x3, got {z.shape}")
    if not np.isfinite(z).all():
        raise TerrainDataError("Elevation contains NaN or infinite values")
    if isinstance(pixel_size_m, (int, float)):
        dy = dx = _validate_scale("pixel_size_m", pixel_size_m)
    else:
        if len(pixel_size_m) != 2:
            raise TerrainDataError("pixel_size_m must be a scalar or (y_size, x_size)")
        dy = _validate_scale("pixel_size_m[0]", pixel_size_m[0])
        dx = _validate_scale("pixel_size_m[1]", pixel_size_m[1])
    tri_scale_m = _validate_scale("tri_scale_m", tri_scale_m)
    tpi_scale_m = _validate_scale("tpi_scale_m", tpi_scale_m)
    relief_scale_m = _validate_scale("relief_scale_m", relief_scale_m)

    dz_dy, dz_dx = np.gradient(z, dy, dx)
    gradient = np.hypot(dz_dx, dz_dy)
    slope = np.arctan(gradient) / (np.pi / 2.0)

    aspect = np.arctan2(dz_dy, -dz_dx)
    aspect_sin = (np.sin(aspect) + 1.0) * 0.5
    aspect_cos = (np.cos(aspect) + 1.0) * 0.5
    flat = gradient <= 1e-6
    aspect_sin[flat] = 0.5
    aspect_cos[flat] = 0.5

    padded = cv2.copyMakeBorder(z, 1, 1, 1, 1, cv2.BORDER_REFLECT101)
    neighbor_difference = np.zeros_like(z, dtype=np.float32)
    for row_offset in range(3):
        for column_offset in range(3):
            if row_offset == 1 and column_offset == 1:
                continue
            neighbor = padded[row_offset : row_offset + z.shape[0], column_offset : column_offset + z.shape[1]]
            neighbor_difference += np.abs(z - neighbor)
    tri = neighbor_difference / 8.0 / tri_scale_m

    local_mean = cv2.boxFilter(z, -1, (5, 5), normalize=True, borderType=cv2.BORDER_REFLECT101)
    tpi_raw = z - local_mean
    tpi = (np.clip(tpi_raw, -tpi_scale_m, tpi_scale_m) / tpi_scale_m + 1.0) * 0.5

    kernel = np.ones((5, 5), dtype=np.uint8)
    local_max = cv2.dilate(z, kernel, borderType=cv2.BORDER_REFLECT101)
    local_min = cv2.erode(z, kernel, borderType=cv2.BORDER_REFLECT101)
    relief = (local_max - local_min) / relief_scale_m

    factors = {
        "SLOPE": slope,
        "ASPECT_SIN": aspect_sin,
        "ASPECT_COS": aspect_cos,
        "TRI": tri,
        "TPI": tpi,
        "RELIEF5": relief,
    }
    return {
        name: np.ascontiguousarray(np.clip(values, 0.0, 1.0), dtype=np.float32)
        for name, values in factors.items()
    }


def stack_terrain_factors(factors: Mapping[str, np.ndarray], channel_names: tuple[str, ...]) -> np.ndarray:
    """Stack selected terrain factors after validating names and dimensions."""
    unknown = [name for name in channel_names if name not in TERRAIN_CHANNELS]
    if unknown:
        raise TerrainDataError(f"Unknown terrain channels: {unknown}")
    if not channel_names:
        raise TerrainDataError("At least one terrain channel is required")
    arrays = [np.asarray(factors[name], dtype=np.float32) for name in channel_names]
    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays):
        raise TerrainDataError("Terrain factor shapes do not match")
    return np.dstack(arrays).astype(np.float32, copy=False)

