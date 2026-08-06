#!/usr/bin/env python3
"""
Image Dataset Analyzer for Qwen-VL Training
============================================

Analyzes images in dataset/images/ to determine optimal max_pixels setting
for config files. Provides statistics on dimensions, aspect ratios, and
resize behavior at different thresholds.

Usage:
    python analyze_images.py
    python analyze_images.py --image-dir dataset/images --extension .jpg
"""

import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image
from tqdm import tqdm


def analyze_images(image_dir: Path, extension: str = ".jpg"):
    """
    Analyze all images in directory and return statistics.

    Returns:
        dict: Statistics including dimensions, pixels, aspect ratios
    """
    image_files = list(image_dir.glob(f"*{extension}"))

    if not image_files:
        raise FileNotFoundError(f"No {extension} files found in {image_dir}")

    print(f"Found {len(image_files)} images\n")

    stats = {
        'widths': [],
        'heights': [],
        'pixels': [],
        'aspect_ratios': [],
        'orientations': defaultdict(int),
    }

    for img_path in tqdm(image_files, desc="Analyzing images"):
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                pixels = w * h
                aspect_ratio = w / h

                stats['widths'].append(w)
                stats['heights'].append(h)
                stats['pixels'].append(pixels)
                stats['aspect_ratios'].append(aspect_ratio)

                # Classify orientation
                if w > h:
                    stats['orientations']['landscape'] += 1
                elif h > w:
                    stats['orientations']['portrait'] += 1
                else:
                    stats['orientations']['square'] += 1

        except Exception as e:
            print(f"Warning: Failed to process {img_path.name}: {e}")
            continue

    # Convert to numpy arrays for statistics
    for key in ['widths', 'heights', 'pixels', 'aspect_ratios']:
        stats[key] = np.array(stats[key])

    return stats, len(image_files)


def print_statistics(stats: dict, n_images: int):
    """Print detailed statistics about the image dataset."""

    print("=" * 70)
    print("IMAGE DATASET STATISTICS")
    print("=" * 70)

    # Dimensions
    print("\n📐 DIMENSIONS")
    print(f"  Width:  {stats['widths'].min():,} - {stats['widths'].max():,} px")
    print(f"          Mean: {stats['widths'].mean():.0f} px, Median: {np.median(stats['widths']):.0f} px")
    print(f"  Height: {stats['heights'].min():,} - {stats['heights'].max():,} px")
    print(f"          Mean: {stats['heights'].mean():.0f} px, Median: {np.median(stats['heights']):.0f} px")

    # Pixels
    print("\n🔢 PIXEL COUNTS")
    pixels_min = stats['pixels'].min()
    pixels_max = stats['pixels'].max()
    pixels_mean = stats['pixels'].mean()
    pixels_median = np.median(stats['pixels'])

    print(f"  Min:    {pixels_min:,} ({format_pixels(pixels_min)})")
    print(f"  Max:    {pixels_max:,} ({format_pixels(pixels_max)})")
    print(f"  Mean:   {pixels_mean:,.0f} ({format_pixels(pixels_mean)})")
    print(f"  Median: {pixels_median:,.0f} ({format_pixels(pixels_median)})")

    # Percentiles
    print("\n📊 PIXEL COUNT PERCENTILES")
    percentiles = [25, 50, 75, 90, 95, 99]
    for p in percentiles:
        val = np.percentile(stats['pixels'], p)
        print(f"  {p:2d}th: {val:>12,.0f} ({format_pixels(val)})")

    # Aspect ratios
    print("\n📏 ASPECT RATIOS")
    print(f"  Min:    {stats['aspect_ratios'].min():.3f} (tallest portrait)")
    print(f"  Max:    {stats['aspect_ratios'].max():.3f} (widest landscape)")
    print(f"  Mean:   {stats['aspect_ratios'].mean():.3f}")
    print(f"  Median: {np.median(stats['aspect_ratios']):.3f}")

    # Orientation
    print("\n🔄 ORIENTATION")
    total = sum(stats['orientations'].values())
    for orient, count in sorted(stats['orientations'].items()):
        pct = 100 * count / total
        print(f"  {orient.capitalize():12s}: {count:4d} ({pct:5.1f}%)")


def format_pixels(pixels: float) -> str:
    """Format pixel count as approximate dimensions."""
    if pixels < 1e6:
        return f"~{int(pixels**0.5)}x{int(pixels**0.5)}"
    else:
        # Approximate square dimensions
        side = int(pixels ** 0.5)
        return f"~{side}x{side}"


def analyze_resize_impact(stats: dict, thresholds: list):
    """Analyze how many images would be resized at different max_pixels thresholds."""

    print("\n" + "=" * 70)
    print("RESIZE IMPACT ANALYSIS")
    print("=" * 70)
    print("\nImpact of different max_pixels settings:\n")

    print(f"{'max_pixels':<15} {'Approx Dims':<15} {'Resized':<15} {'Avg Scale':<15}")
    print("-" * 70)

    for threshold in thresholds:
        n_resized = np.sum(stats['pixels'] > threshold)
        pct_resized = 100 * n_resized / len(stats['pixels'])

        # Calculate average scale factor for resized images
        resized_mask = stats['pixels'] > threshold
        if resized_mask.any():
            scale_factors = np.sqrt(threshold / stats['pixels'][resized_mask])
            avg_scale = scale_factors.mean()
        else:
            avg_scale = 1.0

        dims = format_pixels(threshold)
        resized_str = f"{n_resized} ({pct_resized:.1f}%)"
        scale_str = f"{avg_scale:.2%}" if n_resized > 0 else "N/A"

        print(f"{threshold:>12,}   {dims:<15} {resized_str:<15} {scale_str:<15}")


def recommend_max_pixels(stats: dict):
    """Recommend optimal max_pixels setting based on dataset characteristics."""

    print("\n" + "=" * 70)
    print("📌 RECOMMENDATIONS")
    print("=" * 70)

    pixels = stats['pixels']
    p90 = np.percentile(pixels, 90)
    p95 = np.percentile(pixels, 95)
    p99 = np.percentile(pixels, 99)
    median = np.median(pixels)

    print("\nBased on your dataset:\n")

    # Conservative (median)
    print(f"1. 🐢 CONSERVATIVE (fast training, may lose detail)")
    print(f"   max_pixels: {int(median):,}")
    print(f"   - Keeps ~50% of images at original resolution")
    print(f"   - Good for: Initial experiments, limited VRAM")
    print(f"   - Dimensions: {format_pixels(median)}")

    # Balanced (90th percentile)
    print(f"\n2. ⚖️  BALANCED (recommended for most cases)")
    print(f"   max_pixels: {int(p90):,}")
    print(f"   - Keeps ~90% of images at original resolution")
    print(f"   - Good for: Production training, balanced speed/quality")
    print(f"   - Dimensions: {format_pixels(p90)}")

    # High quality (95th percentile)
    print(f"\n3. 🔍 HIGH QUALITY (best for degraded historical docs)")
    print(f"   max_pixels: {int(p95):,}")
    print(f"   - Keeps ~95% of images at original resolution")
    print(f"   - Good for: Final model, A100 80GB")
    print(f"   - Dimensions: {format_pixels(p95)}")

    # Maximum (99th percentile)
    print(f"\n4. 🚀 MAXIMUM (highest detail, slowest)")
    print(f"   max_pixels: {int(p99):,}")
    print(f"   - Keeps ~99% of images at original resolution")
    print(f"   - Good for: Maximum quality, ample VRAM")
    print(f"   - Dimensions: {format_pixels(p99)}")

    # Current config comparison
    print("\n" + "-" * 70)
    print("CURRENT CONFIG VALUES:")
    print("-" * 70)

    current_configs = {
        "config.yaml (Qwen2-7B)": 2016000,
        "config_qwen3_2b.yaml (Qwen3-2B)": 2016000,
    }

    for name, max_pix in current_configs.items():
        n_resized = np.sum(pixels > max_pix)
        pct_kept = 100 * (1 - n_resized / len(pixels))
        print(f"  {name}")
        print(f"    max_pixels: {max_pix:,} ({format_pixels(max_pix)})")
        print(f"    Keeps {pct_kept:.1f}% at original resolution\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze image dataset for optimal Qwen-VL training settings"
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("dataset/images"),
        help="Path to image directory (default: dataset/images)"
    )
    parser.add_argument(
        "--extension",
        default=".jpg",
        help="Image file extension (default: .jpg)"
    )
    args = parser.parse_args()

    # Validate image directory
    if not args.image_dir.exists():
        print(f"Error: Image directory not found: {args.image_dir}")
        print(f"Current working directory: {Path.cwd()}")
        return 1

    # Analyze images
    stats, n_images = analyze_images(args.image_dir, args.extension)

    # Print statistics
    print_statistics(stats, n_images)

    # Analyze resize impact at different thresholds
    thresholds = [
        1_000_000,   # ~1000x1000
        1_500_000,   # ~1225x1225
        2_016_000,   # ~1420x1420 (current default)
        2_500_000,   # ~1581x1581
        3_000_000,   # ~1732x1732
        4_000_000,   # ~2000x2000
        5_000_000,   # ~2236x2236
    ]
    analyze_resize_impact(stats, thresholds)

    # Recommendations
    recommend_max_pixels(stats)

    print("\n" + "=" * 70)
    print("✅ Analysis complete!")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())
