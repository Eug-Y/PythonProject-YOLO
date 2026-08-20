"""Shared multispectral preprocessing for fixed 5ch and configurable N-channel inputs."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from spectral_indices import INDEX_REGISTRY, compute_index, normalize_dn


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MULTISPECTRAL_DIR = PROJECT_ROOT / "多光谱图片"
BAND_OFFSETS = {"blue": 1, "green": 2, "red": 3, "rededge": 4, "nir": 5}
CHANNEL_NAMES = ("R", "G", "B", "NDVI", "NDRE")
RAW_CHANNEL_BANDS = {"NIR": "nir", "RedEdge": "rededge", "BlueMS": "blue", "GreenMS": "green", "RedMS": "red"}


class FiveChannelDataError(ValueError):
    """Raised when a five-channel array does not follow the project contract."""


class MultichannelDataError(ValueError):
    """Raised when a configurable multichannel array violates its manifest."""


def read_image(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Read an encoded image while supporting non-ASCII Windows paths."""
    path = Path(path)
    raw = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(raw, flags)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def parse_dji_number(path: str | Path) -> int:
    match = re.search(r"DJI_(\d{4})", Path(path).stem, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot parse DJI image number from: {path}")
    return int(match.group(1))


def load_band(
    tif_dir: str | Path,
    image_number: int,
    band: str,
    target_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Load a P4M band, optionally resizing it to (width, height)."""
    if band not in BAND_OFFSETS:
        raise ValueError(f"Unknown band {band!r}; expected one of {sorted(BAND_OFFSETS)}")
    tif_path = Path(tif_dir) / f"DJI_{image_number + BAND_OFFSETS[band]:04d}.TIF"
    if not tif_path.exists():
        raise FileNotFoundError(f"Missing {band} band: {tif_path}")
    array = read_image(tif_path, cv2.IMREAD_UNCHANGED).astype(np.float32).squeeze()
    if array.ndim != 2:
        raise ValueError(f"Expected a single-band TIF, got shape {array.shape}: {tif_path}")
    if target_size is not None and array.shape[::-1] != target_size:
        array = cv2.resize(array, target_size, interpolation=cv2.INTER_LINEAR)
    return array


def normalized_difference(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    index = (high - low) / (high + low + 1e-6)
    return np.clip((index + 1.0) * 0.5, 0.0, 1.0).astype(np.float32)


def build_5ch(jpg_path: str | Path, tif_dir: str | Path = DEFAULT_MULTISPECTRAL_DIR) -> np.ndarray:
    """Build [R, G, B, NDVI, NDRE] float32 data with every channel in [0, 1]."""
    jpg_path = Path(jpg_path)
    image_number = parse_dji_number(jpg_path)
    bgr = read_image(jpg_path, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    height, width = rgb.shape[:2]
    size = (width, height)

    red = load_band(tif_dir, image_number, "red", size)
    rededge = load_band(tif_dir, image_number, "rededge", size)
    nir = load_band(tif_dir, image_number, "nir", size)
    ndvi = normalized_difference(nir, red)
    ndre = normalized_difference(nir, rededge)

    array = np.dstack((rgb, ndvi, ndre)).astype(np.float32)
    validate_5ch(array, expected_hw=(height, width))
    return array


def build_multichannel(
    jpg_path: str | Path,
    channel_names: Iterable[str],
    tif_dir: str | Path = DEFAULT_MULTISPECTRAL_DIR,
) -> np.ndarray:
    """Build a manifest-ordered float32 HxWxC array with channels in [0, 1]."""
    names = tuple(channel_names)
    if not names:
        raise MultichannelDataError("At least one channel is required")
    if len(names) != len(set(names)):
        raise MultichannelDataError(f"Duplicate channel names: {names}")
    supported = {"R", "G", "B", *RAW_CHANNEL_BANDS, *INDEX_REGISTRY}
    unknown = sorted(set(names) - supported)
    if unknown:
        raise MultichannelDataError(f"Unknown channels: {unknown}")

    jpg_path = Path(jpg_path)
    image_number = parse_dji_number(jpg_path)
    bgr = read_image(jpg_path, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    height, width = rgb.shape[:2]
    size = (width, height)

    required: set[str] = set()
    for name in names:
        if name in RAW_CHANNEL_BANDS:
            required.add(RAW_CHANNEL_BANDS[name])
        elif name in INDEX_REGISTRY:
            required.update(INDEX_REGISTRY[name].required_bands)
    bands = {
        band: normalize_dn(load_band(tif_dir, image_number, band, size))
        for band in sorted(required)
    }
    rgb_channels = {"R": rgb[..., 0], "G": rgb[..., 1], "B": rgb[..., 2]}
    output: list[np.ndarray] = []
    for name in names:
        if name in rgb_channels:
            output.append(rgb_channels[name])
        elif name in RAW_CHANNEL_BANDS:
            output.append(bands[RAW_CHANNEL_BANDS[name]])
        else:
            output.append(compute_index(name, bands, encoded=True))

    array = np.dstack(output).astype(np.float32)
    validate_multichannel(array, names, expected_hw=(height, width))
    return array


def validate_multichannel(
    array: np.ndarray,
    channel_names: Iterable[str],
    expected_hw: tuple[int, int] | None = None,
    tolerance: float = 1e-4,
) -> dict[str, object]:
    """Validate a decoded float multichannel array against an ordered manifest."""
    names = tuple(channel_names)
    issues: list[str] = []
    if array.ndim != 3 or array.shape[-1] != len(names):
        raise MultichannelDataError(f"Expected HxWx{len(names)} array, got shape {array.shape}")
    if len(names) != len(set(names)):
        issues.append("channel names are not unique")
    if expected_hw is not None and tuple(array.shape[:2]) != tuple(expected_hw):
        issues.append(f"shape {array.shape[:2]} does not match expected {expected_hw}")
    if not np.issubdtype(array.dtype, np.number):
        issues.append(f"non-numeric dtype {array.dtype}")
    if not np.isfinite(array).all():
        issues.append("contains NaN or infinite values")

    channel_stats: list[dict[str, float | str]] = []
    for index, name in enumerate(names):
        channel = array[..., index]
        minimum = float(np.nanmin(channel))
        maximum = float(np.nanmax(channel))
        mean = float(np.nanmean(channel))
        channel_stats.append({"name": name, "min": minimum, "max": maximum, "mean": mean})
        if minimum < -tolerance or maximum > 1.0 + tolerance:
            issues.append(f"{name} range [{minimum:.6f}, {maximum:.6f}] is outside [0, 1]")
    if issues:
        raise MultichannelDataError("; ".join(issues))
    return {"shape": list(array.shape), "dtype": str(array.dtype), "channels": channel_stats}


def encode_multichannel_uint8(array: np.ndarray, channel_names: Iterable[str]) -> np.ndarray:
    """Encode a validated float [0, 1] array as compact uint8 storage."""
    validate_multichannel(array, channel_names)
    return np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)


def decode_multichannel(array: np.ndarray, channel_names: Iterable[str]) -> np.ndarray:
    """Decode uint8 storage (or accept float storage) into float32 [0, 1]."""
    if array.dtype == np.uint8:
        decoded = array.astype(np.float32) / 255.0
    else:
        decoded = np.asarray(array, dtype=np.float32)
    validate_multichannel(decoded, channel_names)
    return decoded


def save_multichannel_uint8(path: str | Path, array: np.ndarray, channel_names: Iterable[str]) -> Path:
    """Atomically save a compact manifest-ordered multichannel array."""
    path = Path(path)
    encoded = encode_multichannel_uint8(array, channel_names)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npy")
    np.save(temporary, encoded, allow_pickle=False)
    os.replace(temporary, path)
    return path


def load_multichannel(path: str | Path, channel_names: Iterable[str]) -> np.ndarray:
    array = np.load(Path(path), allow_pickle=False)
    return decode_multichannel(array, channel_names)


def validate_5ch(
    array: np.ndarray,
    expected_hw: tuple[int, int] | None = None,
    tolerance: float = 1e-4,
) -> dict[str, object]:
    """Validate the project five-channel contract and return channel statistics."""
    issues: list[str] = []
    if array.ndim != 3 or array.shape[-1] != 5:
        raise FiveChannelDataError(f"Expected HxWx5 array, got shape {array.shape}")
    if expected_hw is not None and tuple(array.shape[:2]) != tuple(expected_hw):
        issues.append(f"shape {array.shape[:2]} does not match expected {expected_hw}")
    if not np.issubdtype(array.dtype, np.number):
        issues.append(f"non-numeric dtype {array.dtype}")

    finite = bool(np.isfinite(array).all())
    if not finite:
        issues.append("contains NaN or infinite values")

    channel_stats: list[dict[str, float | str]] = []
    for index, name in enumerate(CHANNEL_NAMES):
        channel = array[..., index]
        minimum = float(np.nanmin(channel))
        maximum = float(np.nanmax(channel))
        mean = float(np.nanmean(channel))
        channel_stats.append({"name": name, "min": minimum, "max": maximum, "mean": mean})
        if minimum < -tolerance or maximum > 1.0 + tolerance:
            issues.append(f"{name} range [{minimum:.6f}, {maximum:.6f}] is outside [0, 1]")

    if issues:
        raise FiveChannelDataError("; ".join(issues))
    return {"shape": list(array.shape), "dtype": str(array.dtype), "channels": channel_stats}


def validate_5ch_file(path: str | Path, expected_hw: tuple[int, int] | None = None) -> dict[str, object]:
    path = Path(path)
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except Exception as exc:
        raise FiveChannelDataError(f"Cannot load {path}: {exc}") from exc
    return validate_5ch(array, expected_hw=expected_hw)


def save_5ch(path: str | Path, array: np.ndarray) -> Path:
    """Validate and atomically save a five-channel array."""
    path = Path(path)
    validate_5ch(array)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npy")
    np.save(temporary, np.asarray(array, dtype=np.float32), allow_pickle=False)
    os.replace(temporary, path)
    return path


def ensure_5ch(
    jpg_path: str | Path,
    tif_dir: str | Path = DEFAULT_MULTISPECTRAL_DIR,
    output_path: str | Path | None = None,
    *,
    force: bool = False,
    repair: bool = True,
) -> tuple[Path, str]:
    """Ensure a normalized NPY exists; return (path, status)."""
    jpg_path = Path(jpg_path)
    output_path = Path(output_path) if output_path else jpg_path.with_name(f"{jpg_path.stem}_5ch.npy")

    existed = output_path.exists()
    if existed and not force:
        if not repair:
            return output_path, "skipped"
        try:
            bgr = read_image(jpg_path, cv2.IMREAD_COLOR)
            validate_5ch_file(output_path, expected_hw=bgr.shape[:2])
            return output_path, "valid"
        except (FiveChannelDataError, ValueError):
            pass

    array = build_5ch(jpg_path, tif_dir)
    save_5ch(output_path, array)
    return output_path, "repaired" if existed else "generated"


def find_5ch_path(image_path: str | Path, search_dirs: Iterable[str | Path] = ()) -> Path | None:
    image_path = Path(image_path)
    candidates = [Path(directory) / f"{image_path.stem}_5ch.npy" for directory in search_dirs]
    candidates.append(image_path.with_name(f"{image_path.stem}_5ch.npy"))
    return next((candidate for candidate in candidates if candidate.exists()), None)


def load_5ch(path: str | Path, *, strict: bool = True) -> np.ndarray:
    array = np.load(Path(path), allow_pickle=False).astype(np.float32, copy=False)
    if strict:
        validate_5ch(array)
    return array
