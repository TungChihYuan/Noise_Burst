"""
raw_loader.py
─────────────
Load Canon CR3 (or any RAW) files and extract the Bayer RGGB mosaic.

Dependency chain:
  rawpy  →  reads CR3/CR2/NEF/… directly
  dcraw  →  fallback: shell-out via subprocess (must be installed separately)
  numpy  →  array operations
"""

import numpy as np
import subprocess
import tempfile
import os
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_raw_bayer(path: str | Path) -> np.ndarray:
    """
    Load a RAW file and return the 2-D Bayer mosaic as a float32 array
    normalised to [0, 1].

    Parameters
    ----------
    path : str | Path
        Path to a Canon CR3/CR2 or any other RAW file supported by rawpy/dcraw.

    Returns
    -------
    bayer : np.ndarray, shape (H, W), dtype float32
        Raw sensor values normalised to [0, 1].
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"RAW file not found: {path}")

    # Try rawpy first (fastest, no external binary needed)
    try:
        return _load_via_rawpy(path)
    except ImportError:
        pass  # rawpy not installed
    except Exception as e:
        print(f"[raw_loader] rawpy failed ({e}), falling back to dcraw …")

    # Fallback: dcraw → TIFF → numpy
    return _load_via_dcraw(path)


def bayer_to_rggb_planes(bayer: np.ndarray) -> dict[str, np.ndarray]:
    """
    Split a Bayer RGGB mosaic into four separate colour-channel planes.

    The RGGB pattern is:
        R  Gr
        Gb  B

    Parameters
    ----------
    bayer : np.ndarray, shape (H, W)
        Full-resolution Bayer mosaic.

    Returns
    -------
    dict with keys 'R', 'Gr', 'Gb', 'B', each shape (H/2, W/2).
    """
    H, W = bayer.shape
    assert H % 2 == 0 and W % 2 == 0, "Bayer image dimensions must be even."
    return {
        "R":  bayer[0::2, 0::2],
        "Gr": bayer[0::2, 1::2],
        "Gb": bayer[1::2, 0::2],
        "B":  bayer[1::2, 1::2],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _crop_black_border(bayer: np.ndarray, threshold: float = 0.02) -> np.ndarray:
    """
    Remove rows/cols from each edge whose mean is below threshold (near-black).
    Crop is always aligned to 2-pixel Bayer grid boundaries.
    """
    H, W = bayer.shape
    row_means = bayer.mean(axis=1)
    col_means = bayer.mean(axis=0)

    # Find first/last row above threshold
    top = 0
    while top < H and row_means[top] < threshold:
        top += 1
    bottom = H
    while bottom > top and row_means[bottom - 1] < threshold:
        bottom -= 1

    # Find first/last col above threshold
    left = 0
    while left < W and col_means[left] < threshold:
        left += 1
    right = W
    while right > left and col_means[right - 1] < threshold:
        right -= 1

    # Align to even Bayer boundary
    top    = top    - (top    % 2)
    left   = left   - (left   % 2)
    bottom = bottom - (bottom % 2)
    right  = right  - (right  % 2)

    cropped = bayer[top:bottom, left:right]
    if cropped.shape != bayer.shape:
        print(f"  [raw_loader] auto-cropped black border: {bayer.shape} -> {cropped.shape}")
    return cropped


def _load_via_rawpy(path: Path) -> tuple[np.ndarray, str]:
    """Returns (bayer_f32, pattern_string) e.g. ('RGGB')."""
    import rawpy  # type: ignore
    with rawpy.imread(str(path)) as raw:
        # ── Active image area via raw.sizes ───────────────────────────────
        # raw.sizes gives the precise active pixel rectangle, which is more
        # reliable than raw_image_visible on Canon bodies where the two can
        # have the same shape but still include masked border pixels.
        sz = raw.sizes
        top  = sz.top_margin
        left = sz.left_margin
        # Use raw_image (full sensor) and crop to the active area manually,
        # ensuring the crop is aligned to an even row/col (Bayer grid requires it).
        top  = top  + (top  % 2)   # round up to even
        left = left + (left % 2)
        height = sz.iheight - (sz.iheight % 2)
        width  = sz.iwidth  - (sz.iwidth  % 2)

        bayer_u16 = raw.raw_image[top:top+height, left:left+width].copy()
        white_level = raw.white_level if raw.white_level else int(bayer_u16.max())

        # Black level per channel [R, Gr, Gb, B] — used for per-channel subtraction
        black_levels = raw.black_level_per_channel   # [R, Gr, Gb, B]
        if black_levels is None:
            black_levels = [0, 0, 0, 0]
        black_level = int(min(black_levels))  # scalar fallback

        print(f"  [raw_loader] active area: top={top} left={left} h={height} w={width}")
        print(f"  [raw_loader] black_levels per channel: {black_levels}")

        # ── Bayer pattern ─────────────────────────────────────────────────
        raw_pattern = raw.color_desc.decode(errors="replace").strip(chr(0))[:4]
        _ALIASES = {"RGBG": "RGGB", "BGRG": "BGGR", "GBGR": "GBRG"}
        pattern = _ALIASES.get(raw_pattern.upper(), raw_pattern.upper())
        if pattern != raw_pattern.upper():
            print(f"  [raw_loader] Bayer alias {raw_pattern!r} -> {pattern}")

        # ── Camera white balance gains ────────────────────────────────────
        wb_gains = raw.camera_whitebalance   # [R, G1, B, G2]
        g1 = wb_gains[1] if wb_gains[1] > 0 else 1.0
        wb_norm = [wb_gains[0] / g1, 1.0, wb_gains[2] / g1, 1.0]
        print(f"  [raw_loader] Camera WB gains  R={wb_norm[0]:.3f}  G=1.000  B={wb_norm[2]:.3f}")

    print(f"  [raw_loader] black={black_level}  white={white_level}  pattern={pattern}")

    # ── Per-channel black-level subtract + normalise ─────────────────────
    # Subtracting per-channel avoids residual dark gradients at sensor edges
    # that arise when channels have slightly different black levels.
    bayer_f32 = bayer_u16.astype(np.float32)
    bl = black_levels  # [R, Gr, Gb, B] maps to Bayer tile positions
    bayer_f32[0::2, 0::2] = (bayer_f32[0::2, 0::2] - bl[0]) / (white_level - bl[0])  # R
    bayer_f32[0::2, 1::2] = (bayer_f32[0::2, 1::2] - bl[1]) / (white_level - bl[1])  # Gr
    bayer_f32[1::2, 0::2] = (bayer_f32[1::2, 0::2] - bl[2]) / (white_level - bl[2])  # Gb
    bayer_f32[1::2, 1::2] = (bayer_f32[1::2, 1::2] - bl[3]) / (white_level - bl[3])  # B
    bayer_f32 = np.clip(bayer_f32, 0.0, 1.0)

    # ── Auto-crop residual black border rows/cols ─────────────────────────
    # Some Canon bodies have a few near-black rows at the edges even after
    # active-area cropping. Detect and remove them with a threshold scan.
    bayer_f32 = _crop_black_border(bayer_f32)

    # ── Apply camera WB in Bayer domain ───────────────────────────────────
    bayer_f32[0::2, 0::2] *= wb_norm[0]   # R
    bayer_f32[1::2, 1::2] *= wb_norm[2]   # B
    bayer_f32 = np.clip(bayer_f32, 0.0, 1.0)

    return bayer_f32, pattern, True


def _load_via_dcraw(path: Path) -> np.ndarray:
    """
    Shell out to dcraw:
      dcraw -D -4 -T <file>
      -D  : raw, no demosaicing
      -4  : 16-bit linear output
      -T  : TIFF output
    """
    try:
        import cv2  # type: ignore
    except ImportError:
        raise RuntimeError("Neither rawpy nor (dcraw + opencv) is available.")

    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = ["dcraw", "-D", "-4", "-T", str(path)]
        result = subprocess.run(
            cmd, capture_output=True, cwd=tmpdir
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"dcraw failed: {result.stderr.decode(errors='replace')}"
            )
        # dcraw writes <stem>.tiff next to the input file by default
        tiff_path = path.with_suffix(".tiff")
        if not tiff_path.exists():
            # Some versions put it in cwd
            tiff_path = Path(tmpdir) / (path.stem + ".tiff")
        if not tiff_path.exists():
            raise RuntimeError("dcraw did not produce a TIFF file.")
        img_u16 = cv2.imread(str(tiff_path), cv2.IMREAD_UNCHANGED)

    if img_u16 is None:
        raise RuntimeError("Could not read dcraw output TIFF.")

    # Grayscale Bayer mosaic
    if img_u16.ndim == 3:
        img_u16 = img_u16[:, :, 0]

    return (img_u16.astype(np.float32) / 65535.0)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic test helper (for unit tests / demos without a real camera)
# ─────────────────────────────────────────────────────────────────────────────

def make_synthetic_burst(
    height: int = 512,
    width: int = 512,
    n_frames: int = 8,
    noise_sigma: float = 0.08,
    shift_px: float = 2.0,
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """
    Generate a synthetic burst of noisy Bayer images for testing.

    Returns
    -------
    list of n_frames float32 arrays, shape (height, width), in [0, 1].
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Create a colourful scene with distinct R, G, B regions
    yy, xx = np.mgrid[0:height, 0:width]

    # Base luminance gradient
    lum = (xx / width * 0.15 + yy / height * 0.10 + 0.05).astype(np.float32)

    # Build separate R, G, B channels with distinct coloured patches
    R = lum.copy()
    G = lum.copy()
    B = lum.copy()

    # Warm orange patch (top-left quadrant)  — high R, mid G, low B
    R[height//8 : height//3,   width//8 : width//3]  += 0.80
    G[height//8 : height//3,   width//8 : width//3]  += 0.35
    B[height//8 : height//3,   width//8 : width//3]  += 0.02

    # Cool blue patch (top-right quadrant)   — low R, low G, high B
    R[height//8 : height//3,   width//2 : 3*width//4] += 0.03
    G[height//8 : height//3,   width//2 : 3*width//4] += 0.12
    B[height//8 : height//3,   width//2 : 3*width//4] += 0.90

    # Green patch (bottom-left)             — low R, high G, low B
    R[height//2 : 3*height//4, width//8 : width//3]  += 0.04
    G[height//2 : 3*height//4, width//8 : width//3]  += 0.85
    B[height//2 : 3*height//4, width//8 : width//3]  += 0.04

    # Magenta patch (bottom-right)          — high R, low G, high B
    R[height//2 : 3*height//4, width//2 : 3*width//4] += 0.85
    G[height//2 : 3*height//4, width//2 : 3*width//4] += 0.04
    B[height//2 : 3*height//4, width//2 : 3*width//4] += 0.80

    R = np.clip(R, 0.0, 1.0)
    G = np.clip(G, 0.0, 1.0)
    B = np.clip(B, 0.0, 1.0)

    # Pack into Bayer RGGB mosaic:
    #   even rows, even cols = R
    #   even rows, odd cols  = Gr
    #   odd  rows, even cols = Gb
    #   odd  rows, odd cols  = B
    bayer_scene = np.zeros((height, width), dtype=np.float32)
    bayer_scene[0::2, 0::2] = R[0::2, 0::2]     # R
    bayer_scene[0::2, 1::2] = G[0::2, 1::2]     # Gr
    bayer_scene[1::2, 0::2] = G[1::2, 0::2]     # Gb
    bayer_scene[1::2, 1::2] = B[1::2, 1::2]     # B

    frames = []
    for i in range(n_frames):
        dx = rng.uniform(-shift_px, shift_px)
        dy = rng.uniform(-shift_px, shift_px)
        import cv2  # type: ignore
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(bayer_scene, M, (width, height),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT_101)
        noise = rng.normal(0, noise_sigma, shifted.shape).astype(np.float32)
        noisy = np.clip(shifted + noise, 0.0, 1.0)
        frames.append(noisy)

    return frames
