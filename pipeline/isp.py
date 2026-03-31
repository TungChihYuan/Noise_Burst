"""
isp.py
──────
Full ISP post-processing pipeline (matches Single-Frame Camera Pipeline diagram).

Stage order after multi-frame denoising:
  1.  Raw pre-processing      — black level subtract, normalise
  2.  Bayer demosaicing        — bilinear | edge-aware | VNG
  3.  White balance            — grey-world auto or manual gains
  4.  Color space transform    — linear RGB → CIE XYZ → sRGB or ProPhoto
  5.  Color manipulation       — saturation / hue in HSV (photo-finishing)
  6.  Tone mapping             — Reinhard global operator
  7.  Noise reduction          — single-frame Gaussian / bilateral polish
  8.  Sharpening               — unsharp mask
  9.  Output color space       — sRGB piecewise OETF or ProPhoto γ1.8
  10. Image resizing           — bilinear resize + digital zoom crop
  11. JPEG compression + save  — OpenCV imwrite with quality param
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Literal

DemosaicMethod  = Literal["bilinear", "edgeaware", "vng"]
BayerPattern    = Literal["RGGB", "BGGR", "GRBG", "GBRG", "auto"]

# ── Bayer pattern → OpenCV colour conversion codes ────────────────────────────
# OpenCV naming: BayerXY where X=top-left, Y=top-right pixel
# OpenCV BayerRG = RGGB, BayerBG = BGGR, BayerGR = GRBG, BayerGB = GBRG
# Some manufacturers use non-standard naming for Bayer patterns.
# Map these aliases to the canonical 4-pattern set before lookup.
_BAYER_ALIASES = {
    "RGBG": "RGGB",  # Canon CR3: G2 labelled as B, same physical layout as RGGB
    "BGRG": "BGGR",  # Corresponding Canon alias for BGGR
    "GRБG": "GRBG",  # Typo-safe: just in case
    "GBGR": "GBRG",
}

_BAYER_CODES = {
    # (pattern, method) -> cv2 code
    ("RGGB", "bilinear"):  cv2.COLOR_BayerRG2RGB,
    ("RGGB", "edgeaware"): cv2.COLOR_BayerRG2RGB_EA,
    ("RGGB", "vng"):       cv2.COLOR_BayerRG2RGB_VNG,
    ("BGGR", "bilinear"):  cv2.COLOR_BayerBG2RGB,
    ("BGGR", "edgeaware"): cv2.COLOR_BayerBG2RGB_EA,
    ("BGGR", "vng"):       cv2.COLOR_BayerBG2RGB_VNG,
    ("GRBG", "bilinear"):  cv2.COLOR_BayerGR2RGB,
    ("GRBG", "edgeaware"): cv2.COLOR_BayerGR2RGB_EA,
    ("GRBG", "vng"):       cv2.COLOR_BayerGR2RGB_VNG,
    ("GBRG", "bilinear"):  cv2.COLOR_BayerGB2RGB,
    ("GBRG", "edgeaware"): cv2.COLOR_BayerGB2RGB_EA,
    ("GBRG", "vng"):       cv2.COLOR_BayerGB2RGB_VNG,
}
ColorSpace      = Literal["srgb", "prophoto"]
NoiseReduceMode = Literal["gaussian", "bilateral", "none"]

# ── Color matrices ────────────────────────────────────────────────────────────
_MAT_RGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=np.float32)

_MAT_XYZ_TO_SRGB = np.array([
    [ 3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [ 0.0556434, -0.2040259,  1.0572252],
], dtype=np.float32)

_MAT_XYZ_TO_PROPHOTO = np.array([
    [ 1.3459433, -0.2556075, -0.0511118],
    [-0.5445989,  1.5081673,  0.0205351],
    [ 0.0000000,  0.0000000,  1.2118128],
], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Top-level runner
# ─────────────────────────────────────────────────────────────────────────────

def run_isp(
    bayer: np.ndarray,
    demosaic_method: DemosaicMethod  = "edgeaware",
    bayer_pattern:   str             = "auto",
    white_balance:   tuple | None    = None,
    color_space:     ColorSpace      = "srgb",
    saturation:      float           = 1.15,
    hue_shift:       float           = 0.0,
    tone_map:        bool            = True,
    noise_reduce:    NoiseReduceMode = "gaussian",
    sharpen:         bool            = True,
    sharpen_amount:  float           = 0.6,
    output_size:     tuple | None    = None,
    zoom:            float           = 1.0,
    jpeg_quality:    int             = 92,
    output_path:     str | None      = None,
    save_stages:     bool            = False,
    stages_dir:      str | Path | None = None,
    scene_name:      str             = "out",
) -> np.ndarray:
    """
    Run the full ISP chain on a denoised Bayer mosaic.
    Returns uint8 RGB array, shape (H, W, 3).
    """
    stager = _Stager(save_stages, stages_dir, scene_name)

    # 1. Raw pre-processing
    bayer = raw_preprocess(bayer)
    stager.save(1, "raw_preprocess", _bayer_preview(bayer))

    # 2. Bayer demosaicing
    _pattern = detect_bayer_pattern(bayer) if bayer_pattern == "auto" else _BAYER_ALIASES.get(bayer_pattern.upper(), bayer_pattern.upper())
    if bayer_pattern == "auto":
        print(f"  [isp] Bayer pattern auto-detected: {_pattern}")
    rgb = demosaic(bayer, method=demosaic_method, pattern=_pattern)
    stager.save(2, "demosaic", rgb)

    # 3. White balance
    rgb = apply_white_balance(rgb, white_balance)
    stager.save(3, "white_balance", rgb)

    # 4. Color space transform
    rgb = color_space_transform(rgb, target=color_space)
    stager.save(4, f"colorspace_{color_space}", rgb)

    # 5. Color manipulation
    rgb = color_manipulation(rgb, saturation=saturation, hue_shift=hue_shift)
    stager.save(5, "color_manipulation", rgb)

    # 6. Tone mapping
    if tone_map:
        rgb = reinhard_tone_map(rgb)
    stager.save(6, "tone_mapping", rgb)

    # 7. Single-frame noise reduction
    if noise_reduce != "none":
        rgb = single_frame_denoise(rgb, mode=noise_reduce)
    stager.save(7, "noise_reduction", rgb)

    # 8. Sharpening
    if sharpen:
        rgb = unsharp_mask(rgb, amount=sharpen_amount)
    stager.save(8, "sharpening", rgb)

    # 9. Output gamma encode
    rgb = apply_gamma(rgb, space=color_space)
    stager.save(9, "gamma_encode", rgb)

    # 10. Resize + digital zoom
    rgb = resize_and_zoom(rgb, output_size=output_size, zoom=zoom)
    stager.save(10, "resize_zoom", rgb)

    # 11/12. JPEG compress + save
    rgb_u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    if output_path:
        save_jpeg(rgb_u8, output_path, quality=jpeg_quality)
    stager.save(11, "final_jpeg", rgb_u8.astype(np.float32) / 255.0)

    return rgb_u8


# ─────────────────────────────────────────────────────────────────────────────
# Individual stage functions
# ─────────────────────────────────────────────────────────────────────────────

def raw_preprocess(bayer: np.ndarray, black_level: float = 0.0, white_level: float = 1.0) -> np.ndarray:
    """
    Optional additional black/white level correction inside the ISP.
    When loading via rawpy, black level is already subtracted in raw_loader.py.
    Use this only if you need a second-pass correction (e.g. dcraw fallback).
    Default values (0.0, 1.0) are a no-op.
    """
    if black_level == 0.0 and white_level == 1.0:
        return np.clip(bayer.astype(np.float32), 0.0, 1.0)
    out = (bayer.astype(np.float32) - black_level) / (white_level - black_level + 1e-8)
    return np.clip(out, 0.0, 1.0)


def detect_bayer_pattern(bayer: np.ndarray) -> str:
    """
    Heuristic Bayer pattern detection using channel variance.

    In a correctly-ordered RGGB mosaic the two green sub-pixels (Gr, Gb)
    should have higher mean intensity than R and B under typical illumination.
    We try all four patterns, demosaic with bilinear, then pick the one whose
    resulting green channel has the highest mean — green dominates natural scenes.

    Returns one of: "RGGB", "BGGR", "GRBG", "GBRG".
    """
    u16 = (np.clip(bayer, 0, 1) * 65535).astype(np.uint16)
    best_pattern = "RGGB"
    best_score = -1.0
    for pattern in ("RGGB", "BGGR", "GRBG", "GBRG"):
        code = _BAYER_CODES[(pattern, "bilinear")]
        rgb = cv2.cvtColor(u16, code).astype(np.float32)
        # Score = green mean / (red mean + blue mean): highest for correct pattern
        g = rgb[..., 1].mean()
        rb = rgb[..., 0].mean() + rgb[..., 2].mean() + 1e-6
        score = g / rb
        if score > best_score:
            best_score = score
            best_pattern = pattern
    return best_pattern


def demosaic(
    bayer: np.ndarray,
    method: DemosaicMethod = "edgeaware",
    pattern: str = "RGGB",
) -> np.ndarray:
    """
    Demosaic a Bayer mosaic to float32 RGB [0, 1].

    Parameters
    ----------
    bayer   : float32 array (H, W).
    method  : "bilinear" | "edgeaware" | "vng"
    pattern : "RGGB" | "BGGR" | "GRBG" | "GBRG"
              Also accepts manufacturer aliases such as Canon "RGBG" (= RGGB).
    """
    # Resolve manufacturer-specific aliases (e.g. Canon RGBG -> RGGB)
    pattern = _BAYER_ALIASES.get(pattern.upper(), pattern.upper())
    key = (pattern, method)
    if key not in _BAYER_CODES:
        raise ValueError(f"Unknown Bayer pattern/method combo: {key}. "
                         f"Pattern must be one of RGGB/BGGR/GRBG/GBRG "
                         f"(or a known alias like RGBG), "
                         f"method one of bilinear/edgeaware/vng.")
    code = _BAYER_CODES[key]
    if method == "vng":
        # VNG requires uint8
        u8 = (np.clip(bayer, 0, 1) * 255).astype(np.uint8)
        return cv2.cvtColor(u8, code).astype(np.float32) / 255.0
    u16 = (np.clip(bayer, 0, 1) * 65535).astype(np.uint16)
    return cv2.cvtColor(u16, code).astype(np.float32) / 65535.0


def apply_white_balance(rgb: np.ndarray, gains: tuple | None = None) -> np.ndarray:
    if gains is None:
        gains = _grey_world_gains(rgb)
    out = rgb.copy()
    out[..., 0] *= gains[0]
    out[..., 1] *= gains[1]
    out[..., 2] *= gains[2]
    return np.clip(out, 0.0, 1.0)


def color_space_transform(rgb: np.ndarray, target: ColorSpace = "srgb") -> np.ndarray:
    H, W, _ = rgb.shape
    flat = rgb.reshape(-1, 3)
    xyz = flat @ _MAT_RGB_TO_XYZ.T
    mat = _MAT_XYZ_TO_SRGB if target == "srgb" else _MAT_XYZ_TO_PROPHOTO
    out = xyz @ mat.T
    return np.clip(out.reshape(H, W, 3), 0.0, 1.0).astype(np.float32)


def color_manipulation(rgb: np.ndarray, saturation: float = 1.15, hue_shift: float = 0.0) -> np.ndarray:
    if saturation == 1.0 and hue_shift == 0.0:
        return rgb
    u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    hsv = cv2.cvtColor(u8, cv2.COLOR_RGB2HSV).astype(np.float32)
    if hue_shift != 0.0:
        hsv[..., 0] = (hsv[..., 0] + hue_shift / 2.0) % 180.0
    if saturation != 1.0:
        hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0, 255)
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0


def reinhard_tone_map(rgb: np.ndarray, key: float = 0.18) -> np.ndarray:
    lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    lum_mean = np.exp(np.mean(np.log(lum + 1e-6)))
    lum_s = lum * (key / (lum_mean + 1e-6))
    lum_d = lum_s / (1.0 + lum_s)
    scale = np.where(lum > 1e-6, lum_d / (lum + 1e-6), 1.0)[..., np.newaxis]
    return np.clip(rgb * scale, 0.0, 1.0)


def single_frame_denoise(rgb: np.ndarray, mode: NoiseReduceMode = "gaussian", strength: float = 0.5) -> np.ndarray:
    u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    if mode == "gaussian":
        ksize = 3 if strength < 0.5 else 5
        out = cv2.GaussianBlur(u8, (ksize, ksize), sigmaX=strength * 1.5)
    elif mode == "bilateral":
        d = int(5 + strength * 5)
        sigma = strength * 50
        out = cv2.bilateralFilter(u8, d, sigma, sigma)
    else:
        return rgb
    return out.astype(np.float32) / 255.0


def unsharp_mask(rgb: np.ndarray, amount: float = 0.6, radius: float = 1.0, threshold: int = 3) -> np.ndarray:
    u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    ksize = int(2 * round(2 * radius) + 1)
    blurred = cv2.GaussianBlur(u8, (ksize, ksize), sigmaX=radius)
    diff = u8.astype(np.int16) - blurred.astype(np.int16)
    mask = np.abs(diff) > threshold
    sharpened = u8.astype(np.float32) + amount * diff * mask
    return np.clip(sharpened / 255.0, 0.0, 1.0).astype(np.float32)


def apply_gamma(rgb: np.ndarray, space: ColorSpace = "srgb") -> np.ndarray:
    rgb = np.clip(rgb, 0.0, 1.0)
    if space == "srgb":
        return np.where(
            rgb <= 0.0031308,
            rgb * 12.92,
            1.055 * np.power(np.clip(rgb, 1e-8, 1.0), 1.0 / 2.4) - 0.055,
        ).astype(np.float32)
    return np.power(np.clip(rgb, 1e-8, 1.0), 1.0 / 1.8).astype(np.float32)


def resize_and_zoom(rgb: np.ndarray, output_size: tuple | None = None, zoom: float = 1.0) -> np.ndarray:
    if zoom > 1.0:
        H, W = rgb.shape[:2]
        nh, nw = int(H / zoom), int(W / zoom)
        y0, x0 = (H - nh) // 2, (W - nw) // 2
        rgb = rgb[y0:y0 + nh, x0:x0 + nw]
    if output_size is not None:
        rgb = cv2.resize(rgb, output_size, interpolation=cv2.INTER_LINEAR)
    return rgb.astype(np.float32)


def save_jpeg(rgb_u8: np.ndarray, path: str | Path, quality: int = 92) -> None:
    cv2.imwrite(str(path), cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, quality])


def save_image(rgb_u8: np.ndarray, path: str | Path) -> None:
    cv2.imwrite(str(path), cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))


# ─────────────────────────────────────────────────────────────────────────────
# _Stager — saves each intermediate stage as PNG
# ─────────────────────────────────────────────────────────────────────────────

class _Stager:
    def __init__(self, enabled: bool, stages_dir, scene_name: str):
        self.enabled = enabled
        self.scene_name = scene_name
        if enabled:
            self.dir = Path(stages_dir)
            self.dir.mkdir(parents=True, exist_ok=True)

    def save(self, step: int, name: str, rgb_f32: np.ndarray) -> None:
        if not self.enabled:
            return
        u8 = (np.clip(rgb_f32, 0, 1) * 255).astype(np.uint8)
        fname = self.dir / f"{self.scene_name}_stage{step:02d}_{name}.png"
        img = u8 if u8.ndim == 2 else cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(fname), img)
        print(f"  [stage {step:02d}] → {fname.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _grey_world_gains(rgb: np.ndarray) -> tuple:
    means = rgb.mean(axis=(0, 1))
    overall = means.mean()
    g = overall / (means + 1e-6)
    return float(g[0]), float(g[1]), float(g[2])


def _bayer_preview(bayer: np.ndarray) -> np.ndarray:
    u8 = (np.clip(bayer, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(u8, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
