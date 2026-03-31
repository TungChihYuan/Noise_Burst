"""
denoising.py
────────────
Post-registration multi-frame denoising for a burst of Bayer images.

Four selectable methods:
  'mean'      – Simple per-pixel temporal average (fastest baseline)
  'weighted'  – Variance-based weighted average  (noise-adaptive)
  'frequency' – Frequency-domain Wiener filter   (preserves textures)
  'nlm'       – Non-Local Means via OpenCV        (state-of-art quality)

All methods accept aligned float32 Bayer arrays (shape H×W) and return a
single float32 array of the same shape.
"""

import numpy as np
import cv2
from typing import Literal
from scipy.fft import fft2, ifft2, fftshift, ifftshift  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

DenoiseMethod = Literal["mean", "weighted", "frequency", "nlm"]
ALL_METHODS: list[DenoiseMethod] = ["mean", "weighted", "frequency", "nlm"]


def denoise_burst(
    aligned_frames: list[np.ndarray],
    method: DenoiseMethod = "weighted",
    **kwargs,
) -> np.ndarray:
    """
    Fuse a burst of aligned frames into one clean image.

    Parameters
    ----------
    aligned_frames : list of float32 arrays, shape (H, W).
    method         : one of 'mean' | 'weighted' | 'frequency' | 'nlm'.
    **kwargs       : method-specific parameters (see individual functions).

    Returns
    -------
    fused : float32 array, shape (H, W), in [0, 1].
    """
    methods = {
        "mean":      _mean_fuse,
        "weighted":  _weighted_fuse,
        "frequency": _frequency_fuse,
        "nlm":       _nlm_fuse,
    }
    if method not in methods:
        raise ValueError(f"Unknown denoising method '{method}'. "
                         f"Choose from {list(methods)}")
    return methods[method](aligned_frames, **kwargs)


def estimate_noise_sigma(frame: np.ndarray, patch_size: int = 64) -> float:
    """
    Estimate sensor noise standard deviation using the MAD estimator on a
    flat region (corner patch of the image).
    σ ≈ median(|Laplacian|) / 0.6745
    """
    h, w = frame.shape
    patch = frame[:patch_size, :patch_size]
    lap = cv2.Laplacian((patch * 65535).astype(np.uint16), cv2.CV_64F)
    sigma = np.median(np.abs(lap)) / 0.6745 / 65535.0
    return float(np.clip(sigma, 1e-6, 1.0))


def compute_psnr(clean: np.ndarray, noisy: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio (dB). Higher = better."""
    mse = float(np.mean((clean - noisy) ** 2))
    if mse < 1e-12:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Simplified single-scale SSIM in [−1, 1]; 1 = identical.
    Uses standard constants C1=(0.01)², C2=(0.03)².
    """
    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12   = mu1 * mu2
    sig1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
    sig2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
    sig12   = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu12
    C1, C2  = (0.01 ** 2, 0.03 ** 2)
    num = (2 * mu12 + C1) * (2 * sig12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sig1_sq + sig2_sq + C2)
    return float(np.mean(num / den))


# ─────────────────────────────────────────────────────────────────────────────
# Method 1 – Simple temporal mean
# ─────────────────────────────────────────────────────────────────────────────

def _mean_fuse(frames: list[np.ndarray], **_) -> np.ndarray:
    """
    Arithmetic mean across aligned frames.
    Reduces Gaussian noise by √N where N = number of frames.
    """
    stack = np.stack(frames, axis=0)           # (N, H, W)
    return stack.mean(axis=0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Method 2 – Variance-weighted average
# ─────────────────────────────────────────────────────────────────────────────

def _weighted_fuse(
    frames: list[np.ndarray],
    eps: float = 1e-8,
    **_,
) -> np.ndarray:
    """
    Per-pixel inverse-variance weighting:
        w_i(x,y) = 1 / (σ_i(x,y)² + ε)

    Pixels with high local variance (noise spikes, motion residuals) get
    down-weighted automatically.  σ² is estimated via a local window.
    """
    stack = np.stack(frames, axis=0).astype(np.float64)   # (N, H, W)
    N = len(frames)

    # Local variance per frame using a 5×5 box filter as proxy
    weights = np.zeros_like(stack)
    for i in range(N):
        frame_u8 = (np.clip(stack[i], 0, 1) * 255).astype(np.uint8)
        mean_sq  = cv2.blur(stack[i] ** 2, (5, 5))
        mean_val = cv2.blur(stack[i],      (5, 5))
        local_var = np.maximum(mean_sq - mean_val ** 2, 0.0)
        weights[i] = 1.0 / (local_var + eps)

    total_w = weights.sum(axis=0, keepdims=True)
    fused = (stack * weights).sum(axis=0) / (total_w.squeeze(0) + eps)
    return np.clip(fused, 0.0, 1.0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Method 3 – Frequency-domain (Wiener) filter
# ─────────────────────────────────────────────────────────────────────────────

def _frequency_fuse(
    frames: list[np.ndarray],
    noise_sigma: float | None = None,
    **_,
) -> np.ndarray:
    """
    1. Stack → temporal mean (reduces noise floor)
    2. 2-D DFT → Wiener filter in frequency domain → IDFT

    Wiener transfer function:
        H(u,v) = |S(u,v)|² / (|S(u,v)|² + σ_n²)

    where S is the estimated signal PSD from the averaged stack and σ_n is the
    noise variance estimate.
    """
    mean_img  = _mean_fuse(frames).astype(np.float64)

    if noise_sigma is None:
        noise_sigma = estimate_noise_sigma(frames[0].astype(np.float32))
    sigma_n2 = noise_sigma ** 2

    # Compute PSD of the mean image as signal estimate
    F   = fft2(mean_img)
    PSD = (np.abs(F) ** 2) / mean_img.size

    # Wiener filter in frequency domain
    H   = PSD / (PSD + sigma_n2)
    filtered = np.real(ifft2(F * H))
    return np.clip(filtered, 0.0, 1.0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Method 4 – Non-Local Means (OpenCV fastNlMeansDenoising)
# ─────────────────────────────────────────────────────────────────────────────

def _nlm_fuse(
    frames: list[np.ndarray],
    h_param: float | None = None,
    template_window: int = 7,
    search_window: int = 21,
    **_,
) -> np.ndarray:
    """
    Fuse using temporal mean first, then apply OpenCV NLM on the result.

    h_param controls the filter strength; auto-estimated from noise if None.
    """
    mean_img = _mean_fuse(frames)

    if h_param is None:
        sigma = estimate_noise_sigma(frames[0])
        h_param = max(3.0, float(sigma * 255 * 1.5))

    # OpenCV NLM expects uint8
    mean_u8 = (np.clip(mean_img, 0, 1) * 255).astype(np.uint8)
    denoised_u8 = cv2.fastNlMeansDenoising(
        mean_u8,
        h=h_param,
        templateWindowSize=template_window,
        searchWindowSize=search_window,
    )
    return (denoised_u8.astype(np.float32) / 255.0)
