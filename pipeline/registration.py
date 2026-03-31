"""
registration.py
───────────────
Align a burst of Bayer images to a reference frame.

Three backends are provided:
  'ecc'     – OpenCV Enhanced Correlation Coefficient (sub-pixel, translation)
  'feature' – ORB/SIFT keypoint matching + homography (handles rotation/zoom)
  'phase'   – FFT phase correlation (fast, translation-only)

All functions operate on the raw Bayer mosaics (grayscale float32).
Demosaicing before registration would introduce colour artefacts at edges, so
we treat the Bayer image as a luminance proxy.
"""

import numpy as np
import cv2
from typing import Literal


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

RegistrationMethod = Literal["ecc", "feature", "phase"]


def register_burst(
    frames: list[np.ndarray],
    reference_idx: int = 0,
    method: RegistrationMethod = "ecc",
    ecc_iterations: int = 200,
    ecc_eps: float = 1e-6,
    max_features: int = 500,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Align all frames in a burst to a single reference frame.

    Parameters
    ----------
    frames       : list of float32 Bayer arrays, all (H, W).
    reference_idx: index of the reference frame (default: 0, sharpest frame).
    method       : 'ecc' | 'feature' | 'phase'
    ecc_iterations / ecc_eps : ECC termination criteria.
    max_features : number of ORB keypoints for 'feature' method.

    Returns
    -------
    aligned : list of aligned float32 arrays, same shapes as input.
    transforms : list of 2×3 or 3×3 warp matrices (one per frame).
    """
    ref = frames[reference_idx]
    H, W = ref.shape

    aligned = []
    transforms = []

    for i, frame in enumerate(frames):
        if i == reference_idx:
            aligned.append(frame.copy())
            transforms.append(np.eye(2, 3, dtype=np.float64))
            continue

        try:
            if method == "ecc":
                M = _ecc_align(ref, frame, ecc_iterations, ecc_eps)
            elif method == "feature":
                M = _feature_align(ref, frame, max_features)
            elif method == "phase":
                M = _phase_align(ref, frame)
            else:
                raise ValueError(f"Unknown registration method: {method}")
        except Exception as e:
            print(f"[registration] Frame {i}: alignment failed ({e}). "
                  f"Using identity transform.")
            M = np.eye(2, 3, dtype=np.float64)

        warped = _apply_warp(frame, M, (H, W))
        aligned.append(warped)
        transforms.append(M)

    return aligned, transforms


def estimate_sharpness(frame: np.ndarray) -> float:
    """
    Variance-of-Laplacian focus measure — higher = sharper.
    Useful for selecting the best reference frame.
    """
    frame_u8 = (frame * 255).astype(np.uint8)
    lap = cv2.Laplacian(frame_u8, cv2.CV_64F)
    return float(lap.var())


def select_reference_frame(frames: list[np.ndarray]) -> int:
    """Return the index of the sharpest frame (best reference candidate)."""
    scores = [estimate_sharpness(f) for f in frames]
    return int(np.argmax(scores))


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_u8(img: np.ndarray) -> np.ndarray:
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def _ecc_align(
    ref: np.ndarray,
    src: np.ndarray,
    n_iter: int,
    eps: float,
) -> np.ndarray:
    """
    Enhanced Correlation Coefficient alignment (translation model).
    Returns a 2×3 affine matrix.
    """
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, n_iter, eps)
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    _, M = cv2.findTransformECC(
        _to_u8(ref).astype(np.float32),
        _to_u8(src).astype(np.float32),
        warp_matrix,
        cv2.MOTION_TRANSLATION,
        criteria,
        None,
        5,   # gaussFiltSize for gradient computation
    )
    return M.astype(np.float64)


def _feature_align(
    ref: np.ndarray,
    src: np.ndarray,
    max_features: int,
) -> np.ndarray:
    """
    ORB keypoint matching → homography → affine approximation.
    Falls back to SIFT if ORB produces too few inliers.
    Returns a 2×3 partial affine matrix.
    """
    ref_u8 = _to_u8(ref)
    src_u8 = _to_u8(src)

    # Try ORB
    detector = cv2.ORB_create(max_features)
    kp1, des1 = detector.detectAndCompute(ref_u8, None)
    kp2, des2 = detector.detectAndCompute(src_u8, None)

    M = _match_and_estimate(kp1, des1, kp2, des2, norm=cv2.NORM_HAMMING)

    if M is None:
        # Fallback: SIFT (requires opencv-contrib or opencv>=4.4)
        try:
            detector = cv2.SIFT_create(max_features)
            kp1, des1 = detector.detectAndCompute(ref_u8, None)
            kp2, des2 = detector.detectAndCompute(src_u8, None)
            M = _match_and_estimate(kp1, des1, kp2, des2, norm=cv2.NORM_L2)
        except cv2.error:
            pass

    return M if M is not None else np.eye(2, 3, dtype=np.float64)


def _match_and_estimate(kp1, des1, kp2, des2, norm) -> np.ndarray | None:
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None
    bf = cv2.BFMatcher(norm, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    # Lowe ratio test
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    if len(good) < 4:
        return None
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    M, _ = cv2.estimateAffinePartial2D(pts2, pts1, method=cv2.RANSAC,
                                        ransacReprojThreshold=3.0)
    return M.astype(np.float64) if M is not None else None


def _phase_align(ref: np.ndarray, src: np.ndarray) -> np.ndarray:
    """
    FFT phase correlation — returns translation-only 2×3 matrix.
    Fast O(N log N), good for sub-pixel translation estimation.
    """
    shift, _ = cv2.phaseCorrelate(ref.astype(np.float64),
                                   src.astype(np.float64))
    dx, dy = shift
    M = np.array([[1.0, 0.0, dx],
                  [0.0, 1.0, dy]], dtype=np.float64)
    return M


def _apply_warp(
    src: np.ndarray,
    M: np.ndarray,
    size: tuple[int, int],
) -> np.ndarray:
    """
    Warp a Bayer mosaic without cross-channel contamination.

    Direct warpAffine on a Bayer array mixes adjacent pixels from different
    colour channels (R, Gr, Gb, B).  Fix: warp each half-resolution colour
    plane independently, then reassemble.

    For correct colour separation, the integer part of the translation must
    be even (so that R pixels stay on even columns, Gr on odd columns, etc.).
    We round the integer translation components to the nearest even value and
    keep only the sub-pixel remainder for interpolation.
    """
    H, W = size
    Hh, Wh = H // 2, W // 2
    m = M[:2].astype(np.float64)
    a, b, tx = m[0]
    c, d, ty = m[1]

    # Round integer translation to nearest even number, keep sub-pixel residual.
    tx_int = round(tx / 2) * 2   # nearest even integer
    ty_int = round(ty / 2) * 2
    tx_sub = tx - tx_int          # sub-pixel residual  (|tx_sub| <= 1)
    ty_sub = ty - ty_int

    warped = np.empty_like(src)
    for dr, dc in ((0,0),(0,1),(1,0),(1,1)):
        plane = src[dr::2, dc::2].astype(np.float32)

        # Translate the integer shift to plane coordinates (halve it).
        # The sub-pixel residual is also halved for the plane warp.
        tx_p = tx_int / 2.0 + tx_sub / 2.0  # = tx / 2  (same as before)
        ty_p = ty_int / 2.0 + ty_sub / 2.0

        M_plane = np.array([[a, b, tx_p],
                            [c, d, ty_p]], dtype=np.float64)
        w = cv2.warpAffine(
            plane, M_plane, (Wh, Hh),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        warped[dr::2, dc::2] = w

    return warped.astype(np.float32)
