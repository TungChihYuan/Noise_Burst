"""
run_pipeline.py
───────────────
CLI entry-point for Denoising via Burst.

Usage
─────
  # Demo mode (synthetic data, no camera needed)
  python run_pipeline.py --demo --method all --save_stages

  # Single scene
  python run_pipeline.py --input_dir ./data/scene_01/ --method weighted

  # Multiple scenes (one subdir per scene)
  python run_pipeline.py --input_dir ./data/ --method all --save_stages

Options
-------
  --input_dir    DIR    Folder with CR3/CR2/TIFF/PNG files (or scene subdirs)
  --output       DIR    Output directory (default: ./results/)
  --method       STR    mean | weighted | frequency | nlm | all
  --reg_method   STR    ecc | feature | phase  (default: ecc)
  --color_space  STR    srgb | prophoto  (default: srgb)
  --zoom         FLOAT  Digital zoom factor, e.g. 2.0 (default: 1.0)
  --jpeg_quality INT    JPEG quality 0-100 (default: 92)
  --n_frames     INT    Max frames to load (default: all)
  --save_stages        Save a PNG after each ISP stage
  --demo               Use synthetic data instead of real files
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import cv2

from pipeline.raw_loader import load_raw_bayer, make_synthetic_burst
from pipeline.registration import register_burst, select_reference_frame
from pipeline.denoising import (
    denoise_burst,
    estimate_noise_sigma,
    compute_psnr,
    compute_ssim,
    ALL_METHODS,
)
from pipeline.isp import run_isp, save_image


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

RAW_EXTENSIONS = {".cr3", ".cr2", ".nef", ".arw", ".dng", ".tiff", ".tif", ".png"}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_burst_from_dir(input_dir: Path, max_frames: int | None) -> list[np.ndarray]:
    files = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in RAW_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(f"No RAW files found in {input_dir}")
    if max_frames:
        files = files[:max_frames]
    print(f"[pipeline] Loading {len(files)} frames from '{input_dir.name}' ...")
    frames = []
    detected_pattern = None
    camera_wb_applied = False
    for f in files:
        bayer, pattern, wb = load_raw_bayer(f)
        frames.append(bayer)
        if detected_pattern is None and pattern not in ("unknown", ""):
            detected_pattern = pattern
        if wb:
            camera_wb_applied = True
        print(f"  {f.name}  [pattern: {pattern}  camera_wb: {wb}]")
    if detected_pattern:
        print(f"  => Bayer pattern: {detected_pattern}  camera WB pre-applied: {camera_wb_applied}")
    return frames, detected_pattern or "auto", camera_wb_applied


def find_scene_dirs(data_dir: Path) -> list[Path]:
    """
    Return subdirectories of data_dir that contain RAW files (one per scene).
    Falls back to data_dir itself if no scene subdirs are found.
    """
    subdirs = sorted(
        d for d in data_dir.iterdir()
        if d.is_dir() and any(
            f.suffix.lower() in RAW_EXTENSIONS for f in d.iterdir() if f.is_file()
        )
    )
    return subdirs if subdirs else [data_dir]


# ─────────────────────────────────────────────────────────────────────────────
# Comparison strip
# ─────────────────────────────────────────────────────────────────────────────

def _save_comparison_png(
    noisy_rgb: np.ndarray,
    results: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    panels = [noisy_rgb] + list(results.values())
    labels = ["Noisy (single frame)"] + [m.upper() for m in results]
    h, w = panels[0].shape[:2]
    strip = np.zeros((h + 30, w * len(panels), 3), dtype=np.uint8)
    for i, (panel, label) in enumerate(zip(panels, labels)):
        strip[:h, i * w:(i + 1) * w] = panel
        cv2.putText(strip, label, (i * w + 5, h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    save_image(strip, str(output_dir / "comparison_strip.png"))
    print(f"[pipeline] Saved comparison strip -> {output_dir / 'comparison_strip.png'}")


# ─────────────────────────────────────────────────────────────────────────────
# Core pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run(
    frames: list[np.ndarray],
    methods: list[str],
    reg_method: str,
    output_dir: Path,
    bayer_pattern: str = "auto",
    white_balance: tuple | None = None,
    color_space: str = "srgb",
    zoom: float = 1.0,
    jpeg_quality: int = 92,
    save_stages: bool = False,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict = {"n_frames": len(frames), "registration": reg_method}

    # Stage 1: Noise estimation
    print("\n[Stage 1] Estimating noise ...")
    noise_sigma = estimate_noise_sigma(frames[0])
    metrics["noise_sigma"] = round(noise_sigma, 5)
    print(f"  sigma_noise ~ {noise_sigma:.4f}")

    # Stage 2: Registration
    print(f"\n[Stage 2] Registering burst ({reg_method}) ...")
    ref_idx = select_reference_frame(frames)
    metrics["reference_frame"] = ref_idx
    t0 = time.perf_counter()
    aligned, transforms = register_burst(frames, reference_idx=ref_idx,
                                         method=reg_method)  # type: ignore
    reg_time = time.perf_counter() - t0
    metrics["registration_time_s"] = round(reg_time, 3)
    print(f"  Reference frame: {ref_idx}  |  Time: {reg_time:.2f}s")

    # Save noisy reference (no stage saving for the single-frame reference)
    noisy_rgb = run_isp(frames[ref_idx], bayer_pattern=bayer_pattern, white_balance=white_balance, color_space=color_space, zoom=zoom)
    save_image(noisy_rgb, str(output_dir / "noisy_reference.png"))

    # Stage 3: Denoising + full ISP per method
    print("\n[Stage 3] Denoising ...")
    denoised_results: dict[str, np.ndarray] = {}
    method_metrics: dict[str, dict] = {}

    for method in methods:
        print(f"  [{method}] ...", end=" ", flush=True)
        t0 = time.perf_counter()
        denoised_bayer = denoise_burst(aligned, method=method)  # type: ignore
        elapsed = time.perf_counter() - t0

        stages_dir = output_dir / "stages" if save_stages else None
        rgb = run_isp(
            denoised_bayer,
            bayer_pattern=bayer_pattern,
            white_balance=white_balance,
            color_space=color_space,
            zoom=zoom,
            jpeg_quality=jpeg_quality,
            output_path=str(output_dir / f"denoised_{method}.jpg"),
            save_stages=save_stages,
            stages_dir=stages_dir,
            scene_name=method,
        )
        denoised_results[method] = rgb
        save_image(rgb, str(output_dir / f"denoised_{method}.png"))

        psnr = compute_psnr(aligned[ref_idx], denoised_bayer)
        ssim = compute_ssim(aligned[ref_idx], denoised_bayer)
        method_metrics[method] = {
            "time_s": round(elapsed, 3),
            "psnr_db": round(psnr, 2),
            "ssim": round(ssim, 4),
        }
        print(f"done  [{elapsed:.2f}s | PSNR {psnr:.1f} dB | SSIM {ssim:.3f}]")

    metrics["methods"] = method_metrics

    _save_comparison_png(noisy_rgb, denoised_results, output_dir)
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"\n[pipeline] Metrics -> {metrics_path}")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Denoising via Burst Pipeline")
    p.add_argument("--input_dir",    type=Path,  default=None)
    p.add_argument("--output",       type=Path,  default=Path("results"))
    p.add_argument("--method",       type=str,   default="weighted",
                   help="mean|weighted|frequency|nlm|all")
    p.add_argument("--reg_method",   type=str,   default="ecc",
                   help="ecc|feature|phase")
    p.add_argument("--bayer_pattern", type=str,   default="auto",
                   help="RGGB|BGGR|GRBG|GBRG|auto  (default: auto-detect)")
    p.add_argument("--color_space",  type=str,   default="srgb",
                   help="srgb|prophoto")
    p.add_argument("--zoom",         type=float, default=1.0,
                   help="Digital zoom factor (1.0 = no zoom)")
    p.add_argument("--jpeg_quality", type=int,   default=92,
                   help="JPEG output quality 0-100")
    p.add_argument("--n_frames",     type=int,   default=None)
    p.add_argument("--save_stages",  action="store_true",
                   help="Save a PNG after each ISP stage")
    p.add_argument("--demo",         action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve denoising methods first (needed by all code paths)
    if args.method == "all":
        methods = ALL_METHODS
    elif args.method in ALL_METHODS:
        methods = [args.method]
    else:
        print(f"[pipeline] Unknown method '{args.method}'. Choose from {ALL_METHODS}.")
        return

    isp_kwargs = dict(
        bayer_pattern=args.bayer_pattern,
        color_space=args.color_space,
        zoom=args.zoom,
        jpeg_quality=args.jpeg_quality,
        save_stages=args.save_stages,
    )

    # Demo mode
    if args.demo:
        print("[pipeline] Demo mode: generating synthetic burst ...")
        frames = make_synthetic_burst(height=512, width=512, n_frames=8, noise_sigma=0.07)
        run(frames=frames, methods=methods, reg_method=args.reg_method,
            output_dir=args.output, **isp_kwargs)
        print(f"\n[pipeline] Done. Results in: {args.output}")
        return

    if not args.input_dir:
        print("[pipeline] No --input_dir specified. Use --demo for a test run.")
        return

    # Multi-scene subdirectory support
    scene_dirs = find_scene_dirs(args.input_dir)

    if len(scene_dirs) == 1 and scene_dirs[0] == args.input_dir:
        # Single scene — no subdirs found
        frames, detected_pattern, cam_wb = load_burst_from_dir(args.input_dir, args.n_frames)
        if args.bayer_pattern == "auto" and detected_pattern != "auto":
            isp_kwargs["bayer_pattern"] = detected_pattern
        if cam_wb:
            isp_kwargs["white_balance"] = (1.0, 1.0, 1.0)  # camera WB already applied
        run(frames=frames, methods=methods, reg_method=args.reg_method,
            output_dir=args.output, **isp_kwargs)
        print(f"\n[pipeline] Done. Results in: {args.output}")
    else:
        # Multiple scenes
        print(f"[pipeline] Found {len(scene_dirs)} scene(s): {[d.name for d in scene_dirs]}")
        for scene_dir in scene_dirs:
            sep = "=" * 60
            print(f"\n{sep}")
            print(f"[pipeline] Scene: {scene_dir.name}")
            print(sep)
            try:
                frames, detected_pattern, cam_wb = load_burst_from_dir(scene_dir, args.n_frames)
                scene_kwargs = dict(isp_kwargs)
                if args.bayer_pattern == "auto" and detected_pattern != "auto":
                    scene_kwargs["bayer_pattern"] = detected_pattern
                if cam_wb:
                    scene_kwargs["white_balance"] = (1.0, 1.0, 1.0)
                scene_output = args.output / scene_dir.name
                run(frames=frames, methods=methods, reg_method=args.reg_method,
                    output_dir=scene_output, **scene_kwargs)
                print(f"[pipeline] Scene '{scene_dir.name}' done -> {scene_output}")
            except Exception as e:
                print(f"[pipeline] Scene '{scene_dir.name}' failed: {e}")
                continue
        print(f"\n[pipeline] All scenes done. Results in: {args.output}/")


if __name__ == "__main__":
    main()
