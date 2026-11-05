"""Image loading / saving and the on-disk frame cache.

Internally every image is a float32 RGB array in *linear* light, range ~[0, 1].
RAW files are decoded linearly; JPEG/PNG/TIFF are converted from sRGB to linear
so that calibration and averaging are physically meaningful.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

import cv2
import numpy as np

RAW_EXTS = {".cr2", ".cr3", ".crw", ".nef", ".nrw", ".arw", ".srf", ".sr2", ".raf",
            ".orf", ".rw2", ".dng", ".pef", ".srw", ".x3f", ".3fr", ".iiq", ".rwl", ".erf"}
FITS_EXTS = {".fit", ".fits", ".fts"}
STD_EXTS = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}
ALL_EXTS = RAW_EXTS | FITS_EXTS | STD_EXTS


def is_supported(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in ALL_EXTS


# ----------------------------------------------------------------------------- colour
def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img.astype(np.float32)
    return (0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]).astype(np.float32)


# ----------------------------------------------------------------------------- loading
def load_image(path: str, half_size: bool = False) -> np.ndarray:
    """Load any supported image as linear float32 RGB (H, W, 3)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in RAW_EXTS:
        import rawpy
        with rawpy.imread(path) as raw:
            rgb = raw.postprocess(gamma=(1, 1), no_auto_bright=True, output_bps=16,
                                  use_camera_wb=True, half_size=half_size,
                                  output_color=rawpy.ColorSpace.sRGB)
        img = rgb.astype(np.float32) / 65535.0
    elif ext in FITS_EXTS:
        from astropy.io import fits
        with fits.open(path, memmap=False) as hdul:
            data = next(h.data for h in hdul if h.data is not None)
        data = np.asarray(data, dtype=np.float32)
        if data.ndim == 3 and data.shape[0] in (1, 3):
            data = np.moveaxis(data, 0, -1)
        if data.ndim == 2:
            data = np.repeat(data[..., None], 3, axis=2)
        lo = float(np.nanmin(data))
        hi = float(np.nanmax(data))
        if hi > 1.5:  # integer-valued FITS
            scale = 65535.0 if hi <= 65535 else hi
            data = data / scale
        elif lo < 0:
            data = data - lo
        img = np.nan_to_num(data).astype(np.float32)
        img = img[::-1].copy()  # FITS origin is bottom-left
        if half_size:
            img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA)
    else:
        raw = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        if raw is None:
            raise IOError(f"Could not read image: {path}")
        if raw.dtype == np.uint8:
            img = raw.astype(np.float32) / 255.0
        elif raw.dtype == np.uint16:
            img = raw.astype(np.float32) / 65535.0
        else:
            img = np.clip(raw.astype(np.float32), 0, None)
            if img.max() > 1.5:
                img /= img.max()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if raw.dtype != np.float32:  # integer files are sRGB encoded
            img = srgb_to_linear(img)
        if half_size:
            img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(img, dtype=np.float32)


def load_preview(path: str, max_side: int = 1600) -> np.ndarray:
    """Fast, small linear preview (uses the embedded-size RAW decode)."""
    img = load_image(path, half_size=os.path.splitext(path)[1].lower() in RAW_EXTS)
    return resize_max(img, max_side)


def resize_max(img: np.ndarray, max_side: int) -> np.ndarray:
    if img.dtype == np.float16:
        img = img.astype(np.float32)
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (max(1, int(round(w * s))), max(1, int(round(h * s)))),
                      interpolation=cv2.INTER_AREA)


# ----------------------------------------------------------------------------- saving
def save_image(path: str, rgb: np.ndarray, bits: int = 16, quality: int = 95) -> None:
    """Save display-ready RGB data in [0, 1]."""
    ext = os.path.splitext(path)[1].lower()
    rgb = np.clip(np.nan_to_num(rgb), 0.0, 1.0)
    if ext in (".tif", ".tiff"):
        import tifffile
        if bits == 32:
            data = rgb.astype(np.float32)
        elif bits == 8:
            data = (rgb * 255.0 + 0.5).astype(np.uint8)
        else:
            data = (rgb * 65535.0 + 0.5).astype(np.uint16)
        if data.ndim == 3:
            tifffile.imwrite(path, data, photometric="rgb", compression="zlib")
        else:
            tifffile.imwrite(path, data, photometric="minisblack", compression="zlib")
    elif ext == ".png":
        data = (rgb * (65535.0 if bits >= 16 else 255.0) + 0.5).astype(np.uint16 if bits >= 16 else np.uint8)
        if data.ndim == 3:
            data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(path, data):
            raise IOError(f"Could not write {path}")
    elif ext in (".jpg", ".jpeg"):
        data = (rgb * 255.0 + 0.5).astype(np.uint8)
        if data.ndim == 3:
            data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(path, data, [cv2.IMWRITE_JPEG_QUALITY, int(quality)]):
            raise IOError(f"Could not write {path}")
    elif ext in FITS_EXTS:
        from astropy.io import fits
        data = rgb[::-1]
        if data.ndim == 3:
            data = np.moveaxis(data, -1, 0)
        fits.PrimaryHDU(data.astype(np.float32)).writeto(path, overwrite=True)
    else:
        raise ValueError(f"Unsupported output format: {ext}")


# ----------------------------------------------------------------------------- cache
def default_cache_root() -> str:
    if sys.platform == "darwin":
        root = os.path.expanduser("~/Library/Caches/NightStack")
    else:
        root = os.path.join(tempfile.gettempdir(), "nightstack-cache")
    os.makedirs(root, exist_ok=True)
    return root


class FrameCache:
    """A stack of frames stored on disk as float16 so large RAW sets fit in RAM-limited Macs."""

    def __init__(self, directory: str, name: str, n: int, h: int, w: int, channels: int = 3,
                 dtype=np.float16):
        os.makedirs(directory, exist_ok=True)
        self.path = os.path.join(directory, f"{name}.dat")
        shape = (n, h, w, channels) if channels > 1 else (n, h, w)
        self.shape = shape
        self.mm = np.memmap(self.path, dtype=dtype, mode="w+", shape=shape)

    @property
    def n(self) -> int:
        return self.shape[0]

    def write(self, i: int, arr: np.ndarray) -> None:
        self.mm[i] = arr.astype(self.mm.dtype, copy=False)

    def write_rows(self, i: int, r0: int, r1: int, arr: np.ndarray) -> None:
        self.mm[i, r0:r1] = arr.astype(self.mm.dtype, copy=False)

    def read(self, i: int) -> np.ndarray:
        return np.asarray(self.mm[i], dtype=np.float32)

    def rows(self, r0: int, r1: int, indices=None) -> np.ndarray:
        if indices is None:
            return np.asarray(self.mm[:, r0:r1], dtype=np.float32)
        return np.asarray(self.mm[list(indices), r0:r1], dtype=np.float32)

    def flush(self) -> None:
        self.mm.flush()

    def close(self) -> None:
        try:
            del self.mm
        except AttributeError:
            pass
        try:
            os.remove(self.path)
        except OSError:
            pass


def new_cache_dir() -> str:
    return tempfile.mkdtemp(prefix="session-", dir=default_cache_root())


def remove_cache_dir(d: str) -> None:
    shutil.rmtree(d, ignore_errors=True)
