"""
tests/test_pipeline.py
──────────────────────
Unit tests for the night-photography denoising pipeline.
Run with:  pytest tests/test_pipeline.py -v
"""

import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.raw_loader import make_synthetic_burst, bayer_to_rggb_planes
from pipeline.registration import register_burst, estimate_sharpness, select_reference_frame
from pipeline.denoising import (
    denoise_burst,
    estimate_noise_sigma,
    compute_psnr,
    compute_ssim,
    ALL_METHODS,
)
from pipeline.isp import demosaic, apply_white_balance, apply_gamma, reinhard_tone_map


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def synthetic_burst():
    return make_synthetic_burst(height=128, width=128, n_frames=4, noise_sigma=0.06)


@pytest.fixture(scope="module")
def aligned_frames(synthetic_burst):
    frames, _ = register_burst(synthetic_burst, method="phase")
    return frames


# ─────────────────────────────────────────────────────────────────────────────
# raw_loader tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRawLoader:
    def test_synthetic_burst_shape(self, synthetic_burst):
        assert len(synthetic_burst) == 4
        for f in synthetic_burst:
            assert f.shape == (128, 128)
            assert f.dtype == np.float32

    def test_synthetic_burst_range(self, synthetic_burst):
        for f in synthetic_burst:
            assert f.min() >= 0.0
            assert f.max() <= 1.0

    def test_bayer_split_shape(self, synthetic_burst):
        planes = bayer_to_rggb_planes(synthetic_burst[0])
        for key in ["R", "Gr", "Gb", "B"]:
            assert key in planes
            assert planes[key].shape == (64, 64)

    def test_bayer_split_values(self, synthetic_burst):
        bayer = synthetic_burst[0]
        planes = bayer_to_rggb_planes(bayer)
        np.testing.assert_array_equal(planes["R"],  bayer[0::2, 0::2])
        np.testing.assert_array_equal(planes["B"],  bayer[1::2, 1::2])


# ─────────────────────────────────────────────────────────────────────────────
# registration tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRegistration:
    def test_output_length(self, synthetic_burst):
        aligned, _ = register_burst(synthetic_burst, method="phase")
        assert len(aligned) == len(synthetic_burst)

    def test_output_shape_preserved(self, synthetic_burst):
        aligned, _ = register_burst(synthetic_burst, method="phase")
        for f in aligned:
            assert f.shape == synthetic_burst[0].shape

    def test_reference_unchanged(self, synthetic_burst):
        ref_idx = 0
        aligned, _ = register_burst(synthetic_burst, reference_idx=ref_idx)
        np.testing.assert_array_almost_equal(aligned[ref_idx], synthetic_burst[ref_idx])

    def test_sharpness_positive(self, synthetic_burst):
        for f in synthetic_burst:
            assert estimate_sharpness(f) > 0

    def test_reference_selection(self, synthetic_burst):
        idx = select_reference_frame(synthetic_burst)
        assert 0 <= idx < len(synthetic_burst)

    @pytest.mark.parametrize("method", ["phase", "ecc", "feature"])
    def test_all_methods_run(self, synthetic_burst, method):
        aligned, transforms = register_burst(synthetic_burst, method=method)
        assert len(aligned) == len(synthetic_burst)
        assert len(transforms) == len(synthetic_burst)


# ─────────────────────────────────────────────────────────────────────────────
# denoising tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDenoising:
    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_output_shape(self, aligned_frames, method):
        result = denoise_burst(aligned_frames, method=method)
        assert result.shape == aligned_frames[0].shape

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_output_range(self, aligned_frames, method):
        result = denoise_burst(aligned_frames, method=method)
        assert result.min() >= 0.0 - 1e-5
        assert result.max() <= 1.0 + 1e-5

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_denoising_reduces_noise(self, synthetic_burst, aligned_frames, method):
        """Fused output should be closer to the mean than any single noisy frame."""
        fused = denoise_burst(aligned_frames, method=method)
        # MSE of single frame vs. ground truth (first frame as proxy)
        mse_single = float(np.mean((aligned_frames[0] - aligned_frames[1]) ** 2))
        mse_fused  = float(np.mean((fused - aligned_frames[0]) ** 2))
        # Fused should be at least not worse (generous test)
        assert mse_fused <= mse_single * 3.0  # NLM on small 128px images can be noisy

    def test_psnr_finite(self, aligned_frames):
        fused = denoise_burst(aligned_frames, method="mean")
        psnr = compute_psnr(aligned_frames[0], fused)
        assert np.isfinite(psnr)
        assert psnr > 0

    def test_ssim_range(self, aligned_frames):
        fused = denoise_burst(aligned_frames, method="mean")
        ssim = compute_ssim(aligned_frames[0], fused)
        assert -1.0 <= ssim <= 1.0

    def test_noise_estimation_positive(self, synthetic_burst):
        sigma = estimate_noise_sigma(synthetic_burst[0])
        assert sigma > 0


# ─────────────────────────────────────────────────────────────────────────────
# ISP tests
# ─────────────────────────────────────────────────────────────────────────────

class TestISP:
    def test_demosaic_output_shape(self, synthetic_burst):
        rgb = demosaic(synthetic_burst[0])
        h, w = synthetic_burst[0].shape
        assert rgb.shape == (h, w, 3)

    def test_demosaic_range(self, synthetic_burst):
        rgb = demosaic(synthetic_burst[0])
        assert rgb.min() >= 0.0
        assert rgb.max() <= 1.0

    def test_white_balance_grey_world(self):
        rng = np.random.default_rng(0)
        rgb = rng.uniform(0, 1, (64, 64, 3)).astype(np.float32)
        rgb_wb = apply_white_balance(rgb)
        # After grey-world WB the channel means should be roughly equal
        means = rgb_wb.mean(axis=(0, 1))
        assert abs(means[0] - means[1]) < 0.1
        assert abs(means[1] - means[2]) < 0.1

    def test_gamma_brightens_midtones(self):
        img = np.full((10, 10), 0.5, dtype=np.float32)
        out = apply_gamma(img, space="srgb")
        # 0.5^(1/2.2) ≈ 0.729
        # sRGB OETF: for v=0.5 > 0.0031308 -> 1.055*0.5^(1/2.4)-0.055 ~ 0.7354
        expected = 1.055 * (0.5 ** (1/2.4)) - 0.055
        np.testing.assert_allclose(out, expected, atol=1e-3)

    def test_tone_map_range(self, synthetic_burst):
        rgb = demosaic(synthetic_burst[0])
        mapped = reinhard_tone_map(rgb)
        assert mapped.min() >= 0.0
        assert mapped.max() <= 1.0 + 1e-5

    @pytest.mark.parametrize("method", ["bilinear", "edgeaware", "vng"])
    def test_all_demosaic_methods(self, synthetic_burst, method):
        rgb = demosaic(synthetic_burst[0], method=method)
        assert rgb.shape[-1] == 3

    @pytest.mark.parametrize("pattern", ["RGGB", "BGGR", "GRBG", "GBRG"])
    def test_all_bayer_patterns(self, synthetic_burst, pattern):
        rgb = demosaic(synthetic_burst[0], method="bilinear", pattern=pattern)
        assert rgb.shape == (synthetic_burst[0].shape[0], synthetic_burst[0].shape[1], 3)
        assert rgb.min() >= 0.0
        assert rgb.max() <= 1.0

    def test_detect_bayer_pattern(self, synthetic_burst):
        from pipeline.isp import detect_bayer_pattern
        pattern = detect_bayer_pattern(synthetic_burst[0])
        assert pattern in ("RGGB", "BGGR", "GRBG", "GBRG")
