"""
Document Physical Condition Analysis

Analyzes physical document degradation that affects OCR:
- Paper color consistency (brown/cream uniformity)
- Physical damage (burnt parts, fire damage)
- Tears and holes (white openings)
- Background texture degradation

This captures visual artifacts that standard quality metrics miss.

Usage:
    python analyze_document_condition.py --config config_qwen3_8b_full.yaml
"""

import argparse
from pathlib import Path

import yaml
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent


def detect_paper_color_variance(img_gray, img_rgb):
    """
    Detect paper color inconsistency (brown/cream mixing, staining).

    Historical documents should have uniform paper color.
    High variance = damaged/stained paper = harder to process.

    Returns:
        score: 0-100 (higher = more color variance = worse)
    """
    # Convert to LAB color space (better for color perception)
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)

    # Analyze background (non-text) regions
    # Use Otsu to separate text from background
    threshold_val = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    background_mask = img_gray >= threshold_val

    if background_mask.sum() == 0:
        return 0.0

    # Get background pixels only
    bg_L = img_lab[:,:,0][background_mask]  # Lightness
    bg_A = img_lab[:,:,1][background_mask]  # Green-Red
    bg_B = img_lab[:,:,2][background_mask]  # Blue-Yellow

    # Paper color variance (standard deviation of background)
    # Uniform brown/cream paper = low std
    # Mixed colors, stains = high std
    color_variance = np.std(bg_L) + np.std(bg_A) + np.std(bg_B)

    # Normalize to 0-100 (typical range: 5-40)
    score = min(100, color_variance / 0.5)

    return score


def detect_burnt_damage(img_gray, img_rgb):
    """
    Detect burnt/fire damaged sections (dark brown/black edges).

    Burnt paper shows up as very dark regions that aren't ink.

    Returns:
        score: 0-100 (higher = more burnt damage = worse)
    """
    # Convert to HSV for better color detection
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    # Burnt paper characteristics in HSV:
    # - Very low V (value/brightness)
    # - Brown/orange hue (10-30 in HSV) or very dark (near black)

    # Create mask for very dark regions (potential burnt areas)
    very_dark_mask = img_hsv[:,:,2] < 60  # V channel < 60

    # Also check for brown/orange dark regions (burnt edges)
    brown_hue_mask = (img_hsv[:,:,0] >= 5) & (img_hsv[:,:,0] <= 30)  # Brown/orange hue
    brown_dark_mask = brown_hue_mask & (img_hsv[:,:,2] < 100)  # Dark brown

    # Combine masks
    burnt_mask = very_dark_mask | brown_dark_mask

    # Calculate percentage of image that looks burnt
    burnt_ratio = burnt_mask.sum() / burnt_mask.size

    # Edge damage is more common - check if burnt areas are at edges
    h, w = img_gray.shape
    edge_margin = min(h, w) // 20  # 5% margin

    # Create edge mask
    edge_mask = np.zeros_like(burnt_mask)
    edge_mask[:edge_margin, :] = True  # Top edge
    edge_mask[-edge_margin:, :] = True  # Bottom edge
    edge_mask[:, :edge_margin] = True  # Left edge
    edge_mask[:, -edge_margin:] = True  # Right edge

    # Check if burnt areas are concentrated at edges
    edge_burnt = burnt_mask & edge_mask
    edge_burnt_ratio = edge_burnt.sum() / edge_mask.sum() if edge_mask.sum() > 0 else 0

    # Higher score if burnt areas exist, especially at edges
    score = min(100, burnt_ratio * 300 + edge_burnt_ratio * 200)

    return score


def detect_tears_and_holes(img_gray, img_rgb):
    """
    Detect torn sections and holes (white/bright openings).

    Tears show up as very bright regions (exposed white backing or gaps).

    Returns:
        score: 0-100 (higher = more tears/holes = worse)
    """
    # Tears and holes are very bright (near white)
    # They're brighter than normal paper background

    # Find very bright regions
    very_bright_mask = img_gray > 240  # Very bright pixels

    # Remove small noise (use morphological opening)
    kernel = np.ones((3,3), np.uint8)
    very_bright_mask = cv2.morphologyEx(very_bright_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)

    # Calculate percentage of image with tears/holes
    tear_ratio = very_bright_mask.sum() / very_bright_mask.size

    # Find connected components (separate holes)
    num_holes, labels, stats, _ = cv2.connectedComponentsWithStats(very_bright_mask, connectivity=8)

    # Ignore background (label 0) and tiny specks (<100 pixels)
    significant_holes = sum(1 for i in range(1, num_holes) if stats[i, cv2.CC_STAT_AREA] > 100)

    # Score based on both percentage and number of holes
    score = min(100, tear_ratio * 500 + significant_holes * 5)

    return score


def detect_background_texture_degradation(img_gray):
    """
    Detect background texture degradation (rough, uneven paper).

    Degraded paper has high-frequency noise in background regions.

    Returns:
        score: 0-100 (higher = more degradation = worse)
    """
    # Separate text from background
    threshold_val = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    background_mask = img_gray >= threshold_val

    if background_mask.sum() == 0:
        return 0.0

    # Apply high-pass filter to detect texture noise
    # Gaussian blur to get low-frequency component
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # High-frequency component (texture)
    high_freq = cv2.absdiff(img_gray, blurred)

    # Measure texture in background only
    bg_texture = high_freq[background_mask]
    texture_level = bg_texture.mean()

    # Normalize to 0-100 (typical: 2-15)
    score = min(100, texture_level * 5)

    return score


def detect_stains_and_watermarks(img_gray, img_rgb):
    """
    Detect stains, watermarks, and irregular discoloration.

    Returns:
        score: 0-100 (higher = more staining = worse)
    """
    # Stains show up as localized discoloration
    # Use variance in local regions

    # Divide image into grid
    h, w = img_gray.shape
    grid_h, grid_w = h // 16, w // 16

    if grid_h < 10 or grid_w < 10:
        return 0.0

    # Calculate mean intensity for each grid cell
    patch_means = []
    for i in range(0, h - grid_h, grid_h):
        for j in range(0, w - grid_w, grid_w):
            patch = img_gray[i:i+grid_h, j:j+grid_w]
            patch_means.append(patch.mean())

    if len(patch_means) == 0:
        return 0.0

    # High variance in patch means = uneven discoloration = stains
    stain_variance = np.std(patch_means)

    # Also check for specific stain patterns (dark spots on light background)
    # Use morphological operations
    # Close operation to fill in text, then look for dark regions
    kernel = np.ones((15,15), np.uint8)
    closed = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, kernel)

    # Find dark patches in the closed image (potential stains)
    median_brightness = np.median(closed)
    dark_stains = closed < (median_brightness - 30)
    stain_ratio = dark_stains.sum() / dark_stains.size

    # Combine variance and stain detection
    score = min(100, stain_variance * 2 + stain_ratio * 300)

    return score


def analyze_document_condition(image_path):
    """
    Comprehensive document condition analysis.

    Returns:
        dict of condition metrics
    """
    try:
        img = Image.open(image_path).convert("RGB")
        img_np = np.array(img)
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        metrics = {
            "paper_color_variance": detect_paper_color_variance(img_gray, img_np),
            "burnt_damage": detect_burnt_damage(img_gray, img_np),
            "tears_and_holes": detect_tears_and_holes(img_gray, img_np),
            "texture_degradation": detect_background_texture_degradation(img_gray),
            "stains": detect_stains_and_watermarks(img_gray, img_np),
        }

        # Composite condition score (0-100, higher = worse condition)
        condition_score = (
            metrics["paper_color_variance"] * 0.30 +  # Most important
            metrics["burnt_damage"] * 0.25 +
            metrics["tears_and_holes"] * 0.20 +
            metrics["texture_degradation"] * 0.15 +
            metrics["stains"] * 0.10
        )

        metrics["condition_score"] = condition_score
        metrics["success"] = True

        return metrics

    except Exception as e:
        return {
            "paper_color_variance": 0,
            "burnt_damage": 0,
            "tears_and_holes": 0,
            "texture_degradation": 0,
            "stains": 0,
            "condition_score": 0,
            "success": False,
            "error": str(e)
        }


def main():
    parser = argparse.ArgumentParser(description="Analyze document physical condition")
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
        help="Output CSV path (default: dataset/document_condition.csv)"
    )
    args = parser.parse_args()

    # Load config
    config_path = SCRIPT_DIR / args.config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]

    # Load training data
    train_csv = REPO_ROOT / data_cfg["train_csv"]
    df = pd.read_csv(train_csv)

    image_dir = REPO_ROOT / data_cfg["image_dir"]
    image_ext = data_cfg.get("image_ext", ".jpg")

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = REPO_ROOT / "dataset" / "document_condition.csv"

    print(f"Analyzing document condition for {len(df)} images...")
    print("Detecting: paper color, burnt damage, tears/holes, texture, stains")

    # Analyze each image
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing documents"):
        image_path = image_dir / f"{row['ID']}{image_ext}"

        if not image_path.exists():
            metrics = {
                "ID": row["ID"],
                "paper_color_variance": 0,
                "burnt_damage": 0,
                "tears_and_holes": 0,
                "texture_degradation": 0,
                "stains": 0,
                "condition_score": 0,
                "success": False,
                "error": "Image not found"
            }
        else:
            metrics = analyze_document_condition(image_path)
            metrics["ID"] = row["ID"]

        results.append(metrics)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("condition_score", ascending=False)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    print(f"\n✓ Saved document condition analysis to: {output_path}")

    # Print statistics
    successful = results_df[results_df["success"] == True]

    if len(successful) > 0:
        print("\n" + "="*80)
        print("Document Condition Statistics")
        print("="*80)

        print(f"\nCondition Score Distribution (0=pristine, 100=heavily damaged):")
        print(f"  Mean: {successful['condition_score'].mean():.1f}")
        print(f"  Median: {successful['condition_score'].median():.1f}")
        print(f"  Std: {successful['condition_score'].std():.1f}")
        print(f"  Min: {successful['condition_score'].min():.1f}")
        print(f"  Max: {successful['condition_score'].max():.1f}")

        print(f"\nCondition Metrics (0-100):")
        for metric in ["paper_color_variance", "burnt_damage", "tears_and_holes",
                       "texture_degradation", "stains"]:
            mean_val = successful[metric].mean()
            print(f"  {metric:25s}: {mean_val:5.1f} (±{successful[metric].std():.1f})")

        # Most damaged documents
        print(f"\n🔴 Top 10 Most Damaged Documents:")
        print("="*80)
        worst = successful.nlargest(10, "condition_score")
        for _, row in worst.iterrows():
            print(f"\n{row['ID']}")
            print(f"  Condition: {row['condition_score']:.1f}")
            print(f"  Paper color variance: {row['paper_color_variance']:.1f} | "
                  f"Burnt: {row['burnt_damage']:.1f} | "
                  f"Tears: {row['tears_and_holes']:.1f}")
            print(f"  Texture degradation: {row['texture_degradation']:.1f} | "
                  f"Stains: {row['stains']:.1f}")

        # Best condition documents
        print(f"\n🟢 Top 10 Best Condition Documents:")
        print("="*80)
        best = successful.nsmallest(10, "condition_score")
        for _, row in best.iterrows():
            print(f"\n{row['ID']}")
            print(f"  Condition: {row['condition_score']:.1f}")
            print(f"  Paper color variance: {row['paper_color_variance']:.1f} | "
                  f"Burnt: {row['burnt_damage']:.1f} | "
                  f"Tears: {row['tears_and_holes']:.1f}")


if __name__ == "__main__":
    main()
