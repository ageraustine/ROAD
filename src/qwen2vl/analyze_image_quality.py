"""
Image Quality Analysis for Historical Documents

Identifies "hard" images with quality issues:
- Dark patches (uneven illumination)
- Low contrast (faded ink)
- High noise
- Scan artifacts
- Overall degradation score

Usage:
    python analyze_image_quality.py --config config_qwen3_8b_full.yaml

Output:
    dataset/image_quality.csv (ID, quality metrics, difficulty_score)
"""

import argparse
import warnings
from pathlib import Path

import yaml
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent


def compute_dark_patch_score(img_gray):
    """
    Detect dark patches (uneven illumination, shadows, scan artifacts).

    Returns:
        score: 0-100 (higher = more dark patches)
    """
    # Divide image into 16x16 grid
    h, w = img_gray.shape
    grid_h, grid_w = h // 16, w // 16

    if grid_h < 10 or grid_w < 10:
        # Image too small, use 4x4 grid
        grid_h, grid_w = h // 4, w // 4

    patch_means = []
    for i in range(0, h - grid_h, grid_h):
        for j in range(0, w - grid_w, grid_w):
            patch = img_gray[i:i+grid_h, j:j+grid_w]
            patch_means.append(patch.mean())

    if len(patch_means) == 0:
        return 0.0

    # High variance in patch brightness = uneven illumination
    brightness_std = np.std(patch_means)

    # Count patches that are significantly darker than median
    median_brightness = np.median(patch_means)
    dark_threshold = median_brightness - brightness_std
    dark_patch_ratio = sum(1 for m in patch_means if m < dark_threshold) / len(patch_means)

    # Normalize to 0-100 score
    score = min(100, brightness_std * 0.5 + dark_patch_ratio * 100)
    return score


def compute_contrast_score(img_gray):
    """
    Measure overall contrast (faded ink = low contrast).

    Returns:
        score: 0-100 (higher = better contrast, lower = more faded)
    """
    # Use Michelson contrast
    min_val = img_gray.min()
    max_val = img_gray.max()

    if max_val + min_val == 0:
        return 0.0

    contrast = (max_val - min_val) / (max_val + min_val)

    # Also check histogram spread
    hist, _ = np.histogram(img_gray, bins=256, range=(0, 256))
    hist = hist / hist.sum()  # Normalize

    # Entropy as measure of histogram spread
    hist = hist[hist > 0]  # Remove zeros
    entropy = -np.sum(hist * np.log2(hist))

    # High contrast + high entropy = good
    # Low contrast + low entropy = faded/washed out
    contrast_score = contrast * 50 + (entropy / 8.0) * 50  # entropy max ~8 for uniform

    return min(100, contrast_score * 100)


def compute_noise_score(img_gray):
    """
    Estimate noise level (grain, scan artifacts, JPEG compression).

    Returns:
        score: 0-100 (higher = more noise)
    """
    # Use Laplacian variance (edges + noise)
    laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
    laplacian_var = laplacian.var()

    # High-frequency noise estimation
    # Apply median filter, compare to original
    median_filtered = cv2.medianBlur(img_gray, 5)
    noise = np.abs(img_gray.astype(float) - median_filtered.astype(float))
    noise_level = noise.mean()

    # Normalize to 0-100
    # Typical laplacian_var: 100-1000 for clean, 1000+ for noisy
    # Typical noise_level: 2-5 for clean, 10+ for noisy
    score = min(100, (laplacian_var / 50.0) + noise_level * 5)

    return score


def compute_ink_fade_score(img_gray):
    """
    Specific to historical documents: measure ink fade.

    Faded ink = small difference between text and background.

    Returns:
        score: 0-100 (higher = more faded)
    """
    # Otsu thresholding to separate text from background
    threshold_val = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]

    # Split into text (dark) and background (light) regions
    text_mask = img_gray < threshold_val
    bg_mask = img_gray >= threshold_val

    if text_mask.sum() == 0 or bg_mask.sum() == 0:
        # Can't separate text from background - severely faded
        return 100.0

    text_mean = img_gray[text_mask].mean()
    bg_mean = img_gray[bg_mask].mean()

    # Ink fade = small difference
    ink_contrast = bg_mean - text_mean

    # Normalize: typical good contrast is 80-120, faded is <40
    fade_score = max(0, 100 - ink_contrast * 0.8)

    return min(100, fade_score)


def compute_blur_score(img_gray):
    """
    Detect motion blur or out-of-focus blur.

    Returns:
        score: 0-100 (higher = more blurred)
    """
    # Variance of Laplacian (low variance = blurred)
    laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
    laplacian_var = laplacian.var()

    # Typical sharp image: 500+, blurred: <200
    blur_score = max(0, 100 - laplacian_var / 10.0)

    return min(100, blur_score)


def compute_skew_score(img_gray):
    """
    Detect page skew (rotation).

    Returns:
        score: 0-100 (higher = more skewed)
    """
    # Use Hough line transform to detect text lines
    edges = cv2.Canny(img_gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

    if lines is None or len(lines) == 0:
        return 0.0

    # Extract angles
    angles = []
    for line in lines[:50]:  # Use top 50 lines
        rho, theta = line[0]
        angle = np.degrees(theta) - 90  # Convert to rotation angle
        angles.append(angle)

    # Median angle = page skew
    median_angle = np.median(angles)
    skew_score = min(100, abs(median_angle) * 10)

    return skew_score


def analyze_image(image_path):
    """
    Comprehensive quality analysis for a single image.

    Returns:
        dict of quality metrics
    """
    try:
        img = Image.open(image_path).convert("RGB")
        img_np = np.array(img)

        # Convert to grayscale for analysis
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        metrics = {
            "dark_patches": compute_dark_patch_score(img_gray),
            "contrast": compute_contrast_score(img_gray),  # Higher is better
            "noise": compute_noise_score(img_gray),
            "ink_fade": compute_ink_fade_score(img_gray),
            "blur": compute_blur_score(img_gray),
            "skew": compute_skew_score(img_gray),
        }

        # Composite difficulty score (0-100, higher = harder)
        # Invert contrast (high contrast = easy)
        difficulty = (
            metrics["dark_patches"] * 0.25 +
            (100 - metrics["contrast"]) * 0.30 +  # Inverted
            metrics["noise"] * 0.15 +
            metrics["ink_fade"] * 0.20 +
            metrics["blur"] * 0.05 +
            metrics["skew"] * 0.05
        )

        metrics["difficulty_score"] = difficulty
        metrics["success"] = True

        return metrics

    except Exception as e:
        return {
            "dark_patches": 0,
            "contrast": 0,
            "noise": 0,
            "ink_fade": 0,
            "blur": 0,
            "skew": 0,
            "difficulty_score": 0,
            "success": False,
            "error": str(e)
        }


def main():
    parser = argparse.ArgumentParser(description="Analyze image quality for dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="config_qwen3_8b_full.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: dataset/image_quality.csv)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="train",
        choices=["train", "test", "both"],
        help="Which dataset to analyze"
    )
    args = parser.parse_args()

    # Load config
    config_path = SCRIPT_DIR / args.config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]

    # Resolve paths
    train_csv = REPO_ROOT / data_cfg["train_csv"]
    test_csv = REPO_ROOT / data_cfg.get("test_csv", "dataset/Test.csv")
    image_dir = REPO_ROOT / data_cfg["image_dir"]
    image_ext = data_cfg.get("image_ext", ".jpg")

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = REPO_ROOT / "dataset" / "image_quality.csv"

    # Determine which datasets to analyze
    datasets = []
    if args.dataset in ["train", "both"]:
        df_train = pd.read_csv(train_csv)
        df_train["split"] = "train"
        datasets.append(df_train)

    if args.dataset in ["test", "both"]:
        if test_csv.exists():
            df_test = pd.read_csv(test_csv)
            df_test["split"] = "test"
            datasets.append(df_test)

    df = pd.concat(datasets, ignore_index=True)
    print(f"Analyzing {len(df)} images ({args.dataset})...")

    # Analyze each image
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing images"):
        image_path = image_dir / f"{row['ID']}{image_ext}"

        if not image_path.exists():
            metrics = {
                "ID": row["ID"],
                "split": row.get("split", "unknown"),
                "dark_patches": 0,
                "contrast": 0,
                "noise": 0,
                "ink_fade": 0,
                "blur": 0,
                "skew": 0,
                "difficulty_score": 0,
                "success": False,
                "error": "Image not found"
            }
        else:
            metrics = analyze_image(image_path)
            metrics["ID"] = row["ID"]
            metrics["split"] = row.get("split", "unknown")

        results.append(metrics)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Sort by difficulty (hardest first)
    results_df = results_df.sort_values("difficulty_score", ascending=False)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    print(f"\n✓ Saved quality analysis to: {output_path}")

    # Print statistics
    print("\n" + "="*70)
    print("Image Quality Statistics")
    print("="*70)

    successful = results_df[results_df["success"] == True]

    if len(successful) > 0:
        print(f"\nSuccessfully analyzed: {len(successful)}/{len(results_df)} images")
        print(f"\nDifficulty Score Distribution (0=easy, 100=hard):")
        print(f"  Mean: {successful['difficulty_score'].mean():.1f}")
        print(f"  Median: {successful['difficulty_score'].median():.1f}")
        print(f"  Std: {successful['difficulty_score'].std():.1f}")
        print(f"  Min: {successful['difficulty_score'].min():.1f}")
        print(f"  Max: {successful['difficulty_score'].max():.1f}")

        print(f"\nQuality Metrics (0-100):")
        for metric in ["dark_patches", "contrast", "noise", "ink_fade", "blur", "skew"]:
            mean_val = successful[metric].mean()
            print(f"  {metric:15s}: {mean_val:5.1f} (±{successful[metric].std():.1f})")

        # Identify hardest images
        print(f"\n🔴 Top 10 Hardest Images:")
        print("="*70)
        hardest = successful.nlargest(10, "difficulty_score")
        for idx, row in hardest.iterrows():
            print(f"{row['ID']:20s} | Difficulty: {row['difficulty_score']:5.1f} | "
                  f"Contrast: {row['contrast']:4.1f} | Ink Fade: {row['ink_fade']:4.1f} | "
                  f"Dark Patches: {row['dark_patches']:4.1f}")

        # Identify easiest images
        print(f"\n🟢 Top 10 Easiest Images:")
        print("="*70)
        easiest = successful.nsmallest(10, "difficulty_score")
        for idx, row in easiest.iterrows():
            print(f"{row['ID']:20s} | Difficulty: {row['difficulty_score']:5.1f} | "
                  f"Contrast: {row['contrast']:4.1f} | Ink Fade: {row['ink_fade']:4.1f} | "
                  f"Dark Patches: {row['dark_patches']:4.1f}")

        # Check if difficulty correlates with split
        if "split" in successful.columns and successful["split"].nunique() > 1:
            print(f"\n📊 Difficulty by Split:")
            print("="*70)
            for split in successful["split"].unique():
                split_df = successful[successful["split"] == split]
                print(f"{split:10s}: Mean difficulty = {split_df['difficulty_score'].mean():.1f} "
                      f"(±{split_df['difficulty_score'].std():.1f}), n={len(split_df)}")

    else:
        print("\n⚠️  No images successfully analyzed!")

    failed = results_df[results_df["success"] == False]
    if len(failed) > 0:
        print(f"\n⚠️  Failed to analyze {len(failed)} images")
        print(f"First 5 failures:")
        for idx, row in failed.head(5).iterrows():
            print(f"  {row['ID']}: {row.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
