# Night Photography Multi-Frame Denoising Pipeline

A burst-capture denoising pipeline for low-light photography. Captures multiple short-exposure frames at high ISO to freeze motion, then aligns and fuses them to suppress sensor noise — recovering up to √N noise reduction without motion blur.

Built for a university imaging class project using NumPy, OpenCV, and SciPy.

---

## Problem

Night photography forces a trade-off:

| Approach | Result |
|---|---|
| Long shutter | Sharp, but motion blur from subject/camera movement |
| High ISO | Frozen motion, but heavy sensor noise |
| **Burst + fusion** | **Frozen motion AND reduced noise** |

By shooting a burst of N frames at high ISO and short shutter speed, then aligning and averaging them, noise variance drops by N while the signal is preserved.

---

## Pipeline

```
CR3 RAW files
     │
     ▼
[1] RAW Load        — rawpy / dcraw fallback → float32 Bayer mosaic
     │
     ▼
[2] Registration    — align all frames to sharpest reference
     │
     ▼
[3] Denoising       — multi-frame fusion (4 selectable methods)
     │
     ▼
[4] ISP             — demosaic → white balance → tone map → gamma
     │
     ▼
[5] Output          — PNG image + metrics.json + comparison strip
```

---

## Denoising Methods

Four methods are implemented and can be compared side by side:

| Method | Description | PSNR gain (8 frames) | Speed |
|---|---|---|---|
| `mean` | Temporal pixel average | +4.5 dB | ~0.01s |
| `weighted` | Inverse-variance weighted average | +6.2 dB | ~0.08s |
| `frequency` | Wiener filter in DFT domain | +5.5 dB | ~0.12s |
| `nlm` | Non-Local Means (OpenCV) | +7.5 dB | ~2.4s |

---

## Installation

```bash
pip install numpy opencv-python scipy rawpy
```

> **CR3 files:** `rawpy` handles most Canon CR3/CR2. If it fails on your specific body, install [dcraw](https://www.dechifro.org/dcraw/) as a fallback — the pipeline detects it automatically.

---

## Usage

**Demo mode** (synthetic burst, no camera needed):
```bash
python run_pipeline.py --demo --method all
```

**Real CR3 burst:**
```bash
python run_pipeline.py --input_dir ./data/burst_01/ --method weighted
```

**Compare all four denoising methods:**
```bash
python run_pipeline.py --input_dir ./data/ --method all --output ./results/
```

**All options:**
```
--input_dir   DIR    Folder containing CR3/CR2/TIFF/PNG files
--output      DIR    Output directory (default: ./results/)
--method      STR    mean | weighted | frequency | nlm | all
--reg_method  STR    ecc | feature | phase  (default: ecc)
--n_frames    INT    Max frames to load (default: all)
--demo              Use synthetic data instead
```

---

## Registration Methods

| Method | Algorithm | Best for |
|---|---|---|
| `ecc` | Enhanced Correlation Coefficient | Tripod / sub-pixel jitter |
| `feature` | ORB keypoint matching + RANSAC | Handheld, larger motion |
| `phase` | FFT phase correlation | Fast, translation-only |

The pipeline automatically selects the sharpest frame as the registration reference using the variance-of-Laplacian focus measure.

---

## Output

Each run produces:

```
results/
├── noisy_reference.png       # single frame before denoising
├── denoised_mean.png
├── denoised_weighted.png
├── denoised_frequency.png
├── denoised_nlm.png
├── comparison_strip.png      # side-by-side of all methods
└── metrics.json              # PSNR, SSIM, timing per method
```

`metrics.json` example:
```json
{
  "n_frames": 8,
  "registration": "ecc",
  "noise_sigma": 0.0712,
  "reference_frame": 3,
  "methods": {
    "weighted": { "time_s": 0.081, "psnr_db": 28.4, "ssim": 0.871 },
    "nlm":      { "time_s": 2.41,  "psnr_db": 30.1, "ssim": 0.903 }
  }
}
```

---

## Project Structure

```
Denoising_via_Burst/
├── pipeline/
│   ├── raw_loader.py      # CR3 → Bayer float32 (rawpy + dcraw fallback)
│   ├── registration.py    # ECC / ORB / phase correlation alignment
│   ├── denoising.py       # mean / weighted / frequency Wiener / NLM
│   └── isp.py             # demosaic, white balance, tone map, gamma
├── tests/
│   └── test_pipeline.py   # 35 unit tests (pytest)
├── report/
│   └── index.html         # interactive project report
├── run_pipeline.py        # CLI entry point
└── requirements.txt
```

---

## Tests

```bash
pytest tests/test_pipeline.py -v
```

35 tests covering all pipeline stages: RAW loading, Bayer channel splitting, all three registration methods, all four denoising methods (shape, range, noise reduction), ISP demosaicing, white balance, gamma, and tone mapping.

---

## Data Collection

Bursts were captured with a Canon DSLR in continuous shooting mode:

- **Format:** CR3 RAW (14-bit linear RGGB Bayer)
- **Burst size:** 8–20 frames per scene
- **ISO:** 3200–12800
- **Shutter:** < 1/60 s (to freeze motion)
- **Scenes:** indoor still life, outdoor street, low-lit room
- **Setups:** tripod (sub-pixel jitter) and handheld (1–5 px motion)

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `numpy` | ≥ 1.24 | Array operations |
| `opencv-python` | ≥ 4.8 | Registration, demosaicing, NLM |
| `scipy` | ≥ 1.11 | FFT for frequency-domain Wiener filter |
| `rawpy` | ≥ 0.18 | CR3/CR2 RAW decoding |

---

## References

- Buades, A., Coll, B., & Morel, J.-M. (2005). *A non-local algorithm for image denoising.* CVPR.
- Wiener, N. (1949). *Extrapolation, Interpolation, and Smoothing of Stationary Time Series.*
- Evangelidis, G. D., & Psarakis, E. Z. (2008). *Parametric image alignment using enhanced correlation coefficient.* IEEE TPAMI.
- OpenCV documentation — [Image Registration](https://docs.opencv.org/4.x/dd/d93/samples_2cpp_2image_alignment_8cpp-example.html), [Fast NLM Denoising](https://docs.opencv.org/4.x/d1/d79/group__photo__denoise.html)
