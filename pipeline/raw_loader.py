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


def _load_via_rawpy(path: Path) -> tuple[np.ndarray, str, bool, np.ndarray | None]:
    """
    Load CR3/RAW via rawpy. Returns Bayer mosaic + derived color matrix.

    Color matrix derivation follows the DNG spec:
      1. cam_to_XYZ = inv(rgb_xyz_matrix[:3,:3])      # rawpy gives XYZ->cam
      2. AsShotNeutral = 1/wb (normalised to G)
      3. xyz_white = cam_to_XYZ @ AsShotNeutral
      4. adapt = diag(D65_white / xyz_white)           # scene white -> D65
      5. M = M_xyz_sRGB @ adapt @ cam_to_XYZ @ diag(1/wb_n)
    This maps WB-scaled camera RGB directly to linear sRGB (D65).
    """
    import rawpy  # type: ignore

    with rawpy.imread(str(path)) as raw:
        # ── Bayer mosaic (active area only) ───────────────────────────────
        # raw_image_visible already crops out optical black borders.
        bayer_u16 = raw.raw_image_visible.copy()
        # Align to 2x2 Bayer grid
        hv, wv = bayer_u16.shape
        bayer_u16 = bayer_u16[:hv-(hv%2), :wv-(wv%2)]
        wl = raw.white_level if raw.white_level else int(bayer_u16.max())
        bl = raw.black_level_per_channel or [0, 0, 0, 0]

        # ── Bayer pattern from raw_pattern ────────────────────────────────
        # raw.raw_pattern is a 2x2 array of channel indices:
        #   0=R, 1=Gr, 2=B, 3=Gb  (matches color_desc RGBG order)
        # Map to standard 4-char pattern name by reading top-left 2x2 tile.
        pat = raw.raw_pattern  # e.g. [[0,1],[3,2]] for RGGB
        _IDX_TO_CHAR = {0: 'R', 1: 'G', 2: 'B', 3: 'G'}
        pattern_4 = (_IDX_TO_CHAR[pat[0,0]] + _IDX_TO_CHAR[pat[0,1]] +
                     _IDX_TO_CHAR[pat[1,0]] + _IDX_TO_CHAR[pat[1,1]])
        # Collapse RGGB/RGGB variants -> canonical name
        _CANON = {'RGGB':'RGGB','BGGR':'BGGR','GRBG':'GRBG','GBRG':'GBRG',
                  'RGBG':'RGGB','BGRG':'BGGR'}
        pattern = _CANON.get(pattern_4, pattern_4)
        print(f'  [raw_loader] raw_pattern={pat.tolist()} -> {pattern_4} -> {pattern}')

        # ── Camera white balance ──────────────────────────────────────────
        # camera_whitebalance: [R, Gr, B, Gb] multipliers as shot.
        # Normalise to Gr=1 so green channel is unchanged.
        wb_raw = np.array(raw.camera_whitebalance[:3], dtype=np.float64)
        g = wb_raw[1] if wb_raw[1] > 0 else 1.0
        wb_n = wb_raw / g   # [R_gain, 1.0, B_gain]
        print(f'  [raw_loader] camera_whitebalance  R={wb_n[0]:.3f}  G=1.000  B={wb_n[2]:.3f}')

        # ── Color matrix from rgb_xyz_matrix ──────────────────────────────
        # rgb_xyz_matrix is XYZ->cam (the DNG ColorMatrix tag).
        # We need cam->XYZ, so we invert it.
        xyz_to_cam = raw.rgb_xyz_matrix[:3, :3].astype(np.float64)

    h, w = bayer_u16.shape
    print(f'  [raw_loader] raw_image_visible: h={h} w={w}  black={min(bl)}  white={wl}')

    # cam -> XYZ (D50, calibration illuminant)
    cam_to_xyz = np.linalg.inv(xyz_to_cam)

    # AsShotNeutral: what the camera sensor reads for a neutral surface
    asn = 1.0 / wb_n
    asn = asn / asn[1]               # normalise to G=1

    # Scene white point in XYZ (D50)
    xyz_white = cam_to_xyz @ asn

    # Chromatic adaptation: scene white -> D65 white
    D65 = np.array([0.95047, 1.0, 1.08883])
    adapt_scale = D65 / (xyz_white + 1e-10)
    M_adapt = np.diag(adapt_scale)

    # XYZ D65 -> linear sRGB
    M_xyz_srgb = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252],
    ], dtype=np.float64)

    # Full matrix: cam_raw_RGB -> linear sRGB
    # Pipeline: cam_raw -> [cam_to_xyz] -> XYZ_D50 -> [adapt] -> XYZ_D65 -> [M_xyz_srgb] -> sRGB
    # No explicit WB term needed: the white point adaptation (adapt) derived from
    # AsShotNeutral already accounts for the scene illuminant.
    color_matrix = (M_xyz_srgb @ M_adapt @ cam_to_xyz).astype(np.float32)
    print(f'  [raw_loader] Color matrix (cam_wb_RGB -> linear sRGB):')
    for row in color_matrix:
        print(f'              {row.round(4)}')

    # ── Per-channel black_level_per_channel subtract ─────────────────────
    # bl = [R, Gr, Gb, B] matching the Bayer tile positions:
    #   even row / even col = R  -> bl[0]
    #   even row / odd  col = Gr -> bl[1]
    #   odd  row / even col = Gb -> bl[2]
    #   odd  row / odd  col = B  -> bl[3]
    bayer_f32 = bayer_u16.astype(np.float32)
    bayer_f32[0::2, 0::2] = (bayer_f32[0::2, 0::2] - bl[0]) / (wl - bl[0])  # R
    bayer_f32[0::2, 1::2] = (bayer_f32[0::2, 1::2] - bl[1]) / (wl - bl[1])  # Gr
    bayer_f32[1::2, 0::2] = (bayer_f32[1::2, 0::2] - bl[2]) / (wl - bl[2])  # Gb
    bayer_f32[1::2, 1::2] = (bayer_f32[1::2, 1::2] - bl[3]) / (wl - bl[3])  # B
    bayer_f32 = np.clip(bayer_f32, 0.0, 1.0)

    # NOTE: WB is NOT applied in Bayer domain to avoid clipping highlights.
    # The color_matrix already encodes WB (via the diag(1/wb_n) term),
    # so applying WB after demosaic inside color_space_transform is sufficient.

    return bayer_f32, pattern, True, color_matrix


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
