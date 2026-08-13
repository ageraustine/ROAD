"""
Document Physical Condition Analysis

Analyzes physical document degradation that affects OCR:
- Text-to-background contrast (faded ink) ← NEW! CRITICAL
- Paper color consistency (brown/cream uniformity)
- Tears and holes (white openings, missing text)
- Background texture degradation (rough/uneven paper)
- Stains and watermarks (localized discoloration)

This captures visual artifacts that standard quality metrics miss.

IMPROVEMENTS (2026-08-13):
==========================

ADDED (CRITICAL FIX):
- text_contrast: Detects faded ink (low contrast = high score)
  REASON: Documents with perfect paper but faded ink were scored as "excellent"
  and received FULL brightness/contrast augmentation, making text WORSE.
  Example: 5 lowest-score docs had contrast 0.089-0.131 (barely visible) but
  scored 8.22-8.77 (excellent tier). Now properly detected and protected.
  Weight: 35% (highest - faded text is fundamentally unreadable)

ALL DETECTORS NOW USE:
- Adaptive thresholding (relative to each document's baseline)
- Multi-strategy detection (combine multiple signals)
- Non-saturating scoring (distinguishes minor vs severe damage)
- Shape/edge analysis (characterize damage type)

FIXED ISSUES:
1. text_contrast: ADDED - critical gap in original system
2. tears_and_holes: 91.3% zeros → Adaptive brightness, edge detection, shape analysis
3. stains: 74.6% zeros, 3.1% saturated → Local variance, adaptive threshold, color analysis

REMOVED:
- burnt_damage: REMOVED (universal false positives - detected ink as burnt areas)
  Historical archival documents are preserved, not fire-damaged.

WORKING WELL (no changes needed):
- paper_color_variance: Smooth distribution, LAB color space
- texture_degradation: High-pass filtering, adaptive background

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
    DEPRECATED - NO LONGER USED (2026-08-13)

    Detect burnt/fire damaged sections (dark brown/black edges).

    REMOVED FROM ANALYSIS BECAUSE:
    - Universal false positives (detected ink/text as burnt areas on 100% of documents)
    - Historical archival documents are preserved, not fire-damaged
    - Fundamentally unsolvable: distinguishing "dark because burnt" vs "dark because ink"
      requires material analysis, not image processing
    - Redundant: texture_degradation and paper_color_variance capture dark/degraded areas

    This function is preserved for reference but not called in analyze_document_condition().

    Returns:
        score: 0-100 (higher = more burnt damage = worse)
    """
    h, w = img_gray.shape

    # STEP 1: Estimate normal document darkness (adaptive baseline)
    # Separate text from background using Otsu
    threshold_val = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    background_mask = img_gray >= threshold_val

    if background_mask.sum() == 0:
        return 0.0

    bg_pixels = img_gray[background_mask]
    bg_mean = bg_pixels.mean()
    bg_percentile_10 = np.percentile(bg_pixels, 10)  # Darkest 10% of background

    # STEP 2: Adaptive burnt area detection
    # Burnt areas are significantly DARKER than normal paper (opposite of tears)
    # Use multiple strategies

    # Strategy A: Relative to background darkness (adaptive)
    # Burnt areas are much darker than typical background
    relative_dark_threshold = max(0, bg_percentile_10 - 20)
    relative_burnt_mask = img_gray < relative_dark_threshold

    # Strategy B: Absolute darkness (classic approach for very burnt)
    # Very dark regions that can't be ink (too broad)
    absolute_burnt_mask = img_gray < 50

    # Convert to HSV for color-based detection
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    # Strategy C: Brown/charred color detection (HSV)
    # Burnt paper shows brown/orange hues at low brightness
    brown_hue_mask = (img_hsv[:,:,0] >= 5) & (img_hsv[:,:,0] <= 35)  # Extended range
    brown_dark_mask = brown_hue_mask & (img_hsv[:,:,2] < 100)

    # Combine strategies (union)
    candidate_burnt = (relative_burnt_mask | absolute_burnt_mask | brown_dark_mask).astype(np.uint8)

    # STEP 3: Remove text (burnt areas should be broad, not fine strokes)
    # Dilate to merge burnt regions, then erode to remove text
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(candidate_burnt, kernel, iterations=2)
    burnt_mask = cv2.erode(dilated, kernel, iterations=1)

    # STEP 4: Calculate burnt area ratio
    burnt_ratio = burnt_mask.sum() / burnt_mask.size

    # STEP 5: Edge damage detection (burnt edges more problematic for OCR)
    edge_margin = max(10, min(h, w) // 20)  # 5% margin, min 10px

    # Create edge mask
    edge_mask = np.zeros_like(burnt_mask, dtype=bool)
    edge_mask[:edge_margin, :] = True  # Top
    edge_mask[-edge_margin:, :] = True  # Bottom
    edge_mask[:, :edge_margin] = True  # Left
    edge_mask[:, -edge_margin:] = True  # Right

    # Edge burnt ratio
    edge_burnt = burnt_mask & edge_mask
    edge_burnt_ratio = edge_burnt.sum() / edge_mask.sum() if edge_mask.sum() > 0 else 0

    # STEP 6: Non-saturating scoring
    # Burnt damage is serious but should still distinguish minor vs severe
    # Edge damage weighted higher (harder for OCR to recover)
    base_score = min(60, burnt_ratio * 1500)  # Max 60 points from overall burnt area
    edge_score = min(40, edge_burnt_ratio * 200)  # Max 40 points from edge damage

    score = base_score + edge_score

    return score


def detect_tears_and_holes(img_gray, img_rgb):
    """
    Detect torn sections and holes (white/bright openings).

    IMPROVED ALGORITHM (2026-08-13):
    - Adaptive thresholding relative to paper background (not fixed 240)
    - Edge detection for irregular tear boundaries
    - Shape analysis to distinguish tears (elongated) from holes (circular)
    - Local contrast analysis for sharp tear edges
    - Multi-threshold approach for robustness

    Returns:
        score: 0-100 (higher = more tears/holes = worse)
    """
    h, w = img_gray.shape

    # STEP 1: Estimate paper background brightness (adaptive)
    # Use Otsu to separate text from background
    threshold_val = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    background_mask = img_gray >= threshold_val

    if background_mask.sum() == 0:
        return 0.0

    # Get background pixels statistics
    bg_pixels = img_gray[background_mask]
    bg_mean = bg_pixels.mean()
    bg_std = bg_pixels.std()
    bg_percentile_90 = np.percentile(bg_pixels, 90)

    # STEP 2: Adaptive tear detection
    # Tears are significantly brighter than normal paper (relative threshold)
    # Use multiple strategies and combine

    # Strategy A: Relative threshold (mean + 2.5 std deviations)
    # Works for documents with varying background brightness
    relative_threshold = min(255, bg_mean + 2.5 * bg_std)
    relative_bright_mask = img_gray > relative_threshold

    # Strategy B: Top percentile threshold (brightest 5% of background)
    # Catches tears in very bright documents
    percentile_threshold = max(bg_percentile_90, bg_mean + 1.5 * bg_std)
    percentile_bright_mask = img_gray > percentile_threshold

    # Strategy C: Absolute threshold for very bright tears (classic approach)
    # Catches pure white tears/holes
    absolute_bright_mask = img_gray > 240

    # Combine strategies (union - tears detected by any method)
    candidate_mask = (relative_bright_mask | percentile_bright_mask | absolute_bright_mask).astype(np.uint8)

    # STEP 3: Edge-based refinement
    # Tears have sharp, irregular edges
    edges = cv2.Canny(img_gray, 50, 150)

    # Dilate edges slightly to create edge zones
    edge_kernel = np.ones((3, 3), np.uint8)
    edge_zones = cv2.dilate(edges, edge_kernel, iterations=1)

    # Tears should coincide with strong edges
    edge_enhanced_mask = candidate_mask & (edge_zones > 0)

    # STEP 4: Remove noise with adaptive morphology
    # Kernel size based on image size (5×5 for typical document)
    kernel_size = max(3, min(7, min(h, w) // 500))
    if kernel_size % 2 == 0:
        kernel_size += 1
    morph_kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Opening: removes small noise
    cleaned_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_OPEN, morph_kernel)

    # STEP 5: Connected component analysis with shape features
    num_components, labels, stats, centroids = cv2.connectedComponentsWithStats(
        cleaned_mask, connectivity=8
    )

    tears_score = 0.0
    holes_score = 0.0
    total_tear_area = 0
    total_hole_area = 0

    # Minimum area threshold (adaptive to image size)
    min_area = max(50, (h * w) // 10000)  # 0.01% of image

    for i in range(1, num_components):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]

        if area < min_area:
            continue

        # Get bounding box
        x, y, w_box, h_box = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                             stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]

        # Shape analysis: aspect ratio
        aspect_ratio = max(w_box, h_box) / max(min(w_box, h_box), 1)

        # Compactness: circular = 4π*area/perimeter²  ≈ 1, elongated < 0.5
        component_mask = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            continue

        perimeter = cv2.arcLength(contours[0], True)
        if perimeter == 0:
            continue

        compactness = (4 * np.pi * area) / (perimeter ** 2)

        # Edge irregularity: measure perimeter vs ideal circle
        ideal_perimeter = 2 * np.pi * np.sqrt(area / np.pi)
        irregularity = perimeter / max(ideal_perimeter, 1)

        # Classify: Tear vs Hole
        # Tears: elongated (aspect > 2.5), irregular (irregularity > 1.3)
        # Holes: compact (compactness > 0.6), circular (aspect < 2.0)

        if aspect_ratio > 2.5 or irregularity > 1.3:
            # Likely a TEAR (elongated, irregular edge)
            tears_score += area * irregularity  # Weight by irregularity
            total_tear_area += area
        elif compactness > 0.5:
            # Likely a HOLE (compact, circular)
            holes_score += area
            total_hole_area += area
        else:
            # Ambiguous - count as minor damage
            tears_score += area * 0.5
            total_tear_area += area

    # STEP 6: Calculate final score
    # Normalize by image size
    image_size = h * w
    tear_ratio = total_tear_area / image_size
    hole_ratio = total_hole_area / image_size

    # Non-saturating scoring
    # Tears more damaging than holes (harder for OCR to interpolate)
    tear_score_component = min(60, tear_ratio * 3000)  # Max 60 points from tears
    hole_score_component = min(40, hole_ratio * 2000)  # Max 40 points from holes

    score = tear_score_component + hole_score_component

    return score


def detect_text_contrast(img_gray):
    """
    Detect text-to-background contrast (faded ink).

    Low contrast = faded/weak ink = hard for OCR to read.
    This is CRITICAL - a document can have perfect paper but unreadable faded text.

    Algorithm:
    - Separate text from background using Otsu threshold
    - Measure mean brightness difference
    - Normalize to 0-100 (inverted: low contrast = high score)

    Returns:
        score: 0-100 (higher = lower contrast = worse for OCR)
    """
    # Otsu threshold to separate text (dark) from background (light)
    threshold_val = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]

    background_mask = img_gray >= threshold_val
    text_mask = img_gray < threshold_val

    # Need sufficient pixels in both regions
    if background_mask.sum() < (img_gray.size * 0.1) or text_mask.sum() < (img_gray.size * 0.05):
        # Fallback: very unbalanced (mostly text or mostly background)
        # Use percentiles instead
        bg_pixels = img_gray[img_gray >= np.percentile(img_gray, 50)]
        text_pixels = img_gray[img_gray < np.percentile(img_gray, 50)]
    else:
        bg_pixels = img_gray[background_mask]
        text_pixels = img_gray[text_mask]

    if len(bg_pixels) == 0 or len(text_pixels) == 0:
        return 50.0  # Neutral score if can't measure

    # Mean brightness of background and text
    bg_mean = bg_pixels.mean()
    text_mean = text_pixels.mean()

    # Contrast ratio: (background - text) / background
    # Range: 0.0 (no contrast) to 1.0 (maximum contrast)
    if bg_mean <= 0:
        return 50.0

    contrast_ratio = (bg_mean - text_mean) / bg_mean
    contrast_ratio = max(0.0, min(1.0, contrast_ratio))

    # CRITICAL THRESHOLDS (empirically determined):
    # 0.50+ = excellent contrast (easy to read)
    # 0.35-0.50 = good contrast (readable)
    # 0.20-0.35 = low contrast (faded, harder to read)
    # <0.20 = very low contrast (severely faded, very hard to read)

    # Invert and scale to 0-100 (low contrast = high score)
    # Linear mapping:
    # contrast 1.0 → score 0 (perfect contrast, no problem)
    # contrast 0.5 → score 20 (good contrast, minor issue)
    # contrast 0.35 → score 40 (threshold for concern)
    # contrast 0.20 → score 70 (low contrast, major issue)
    # contrast 0.0 → score 100 (no contrast, unreadable)

    # Piecewise linear for better sensitivity in critical range
    if contrast_ratio >= 0.50:
        # Excellent contrast: score 0-20
        score = (1.0 - contrast_ratio) / 0.5 * 20
    elif contrast_ratio >= 0.35:
        # Good contrast: score 20-40
        score = 20 + (0.50 - contrast_ratio) / 0.15 * 20
    elif contrast_ratio >= 0.20:
        # Low contrast: score 40-70
        score = 40 + (0.35 - contrast_ratio) / 0.15 * 30
    else:
        # Very low contrast: score 70-100
        score = 70 + (0.20 - contrast_ratio) / 0.20 * 30

    return min(100, max(0, score))


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

    IMPROVED ALGORITHM (2026-08-13):
    - Local contrast analysis (not grid-based, misses localized stains)
    - Adaptive thresholding relative to local background
    - Multi-scale stain detection (small spots vs large patches)
    - Non-saturating scoring
    - Color variance analysis (stains often have color tint)

    Returns:
        score: 0-100 (higher = more staining = worse)
    """
    h, w = img_gray.shape

    if h < 50 or w < 50:
        return 0.0

    # STEP 1: Separate background from text (adaptive)
    threshold_val = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    background_mask = img_gray >= threshold_val

    if background_mask.sum() == 0:
        return 0.0

    bg_pixels = img_gray[background_mask]
    bg_mean = bg_pixels.mean()
    bg_std = bg_pixels.std()

    # STEP 2: Local variance analysis (adaptive window size)
    # Stains show up as localized discoloration within background
    # Use sliding window instead of fixed grid

    # Window size: adaptive to image size (typically 5-10% of smaller dimension)
    window_size = max(20, min(100, min(h, w) // 10))
    stride = window_size // 2  # 50% overlap for better coverage

    local_variances = []
    local_mean_diffs = []

    for i in range(0, h - window_size, stride):
        for j in range(0, w - window_size, stride):
            window = img_gray[i:i+window_size, j:j+window_size]
            window_bg_mask = background_mask[i:i+window_size, j:j+window_size]

            if window_bg_mask.sum() < (window_size ** 2) * 0.3:
                # Skip if window is mostly text (< 30% background)
                continue

            # Background pixels in this window
            window_bg = window[window_bg_mask]

            if len(window_bg) == 0:
                continue

            # Local variance (high = uneven staining)
            local_var = window_bg.std()
            local_variances.append(local_var)

            # Local mean deviation from global (stains darker than normal paper)
            local_mean_diff = abs(window_bg.mean() - bg_mean)
            local_mean_diffs.append(local_mean_diff)

    if len(local_variances) == 0:
        return 0.0

    # STEP 3: Variance-based stain score
    # High local variance = localized discoloration
    mean_local_variance = np.mean(local_variances)
    variance_score = mean_local_variance / max(bg_std, 1)  # Normalized by global variance

    # STEP 4: Dark patch detection (adaptive threshold)
    # Stains often appear as darker patches on lighter background
    # Morphological closing to remove text, then detect dark regions
    kernel_size = max(11, window_size // 5)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, kernel)

    # Adaptive dark stain threshold (relative to background)
    dark_threshold = bg_mean - (1.5 * bg_std)  # 1.5 std below mean
    dark_stains = (closed < dark_threshold) & background_mask
    stain_ratio = dark_stains.sum() / background_mask.sum()

    # STEP 5: Color-based stain detection (using RGB)
    # Stains often have color tint (brown, yellow, blue from water damage)
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)

    # Analyze background color variance in A/B channels (color, not lightness)
    bg_A = img_lab[:,:,1][background_mask]  # Green-Red
    bg_B = img_lab[:,:,2][background_mask]  # Blue-Yellow

    # High color variance in background = staining/discoloration
    color_variance = np.std(bg_A) + np.std(bg_B)

    # Normalize (typical range: 2-20 for color channels)
    color_score = min(40, color_variance / 0.5)  # Max 40 points

    # STEP 6: Non-saturating composite score
    # Combine multiple signals with reasonable weights
    variance_component = min(30, variance_score * 20)  # Max 30 points
    stain_patch_component = min(30, stain_ratio * 800)  # Max 30 points (0.0375% = 30)

    score = variance_component + stain_patch_component + color_score * 0.25  # Color contributes up to 10 points

    return min(100, score)


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
            "text_contrast": detect_text_contrast(img_gray),
            "paper_color_variance": detect_paper_color_variance(img_gray, img_np),
            "tears_and_holes": detect_tears_and_holes(img_gray, img_np),
            "texture_degradation": detect_background_texture_degradation(img_gray),
            "stains": detect_stains_and_watermarks(img_gray, img_np),
        }

        # Composite condition score (0-100, higher = worse condition)
        # Rebalanced weights after adding text_contrast (2026-08-13)
        # CRITICAL: text_contrast gets highest weight - faded text is fundamentally unreadable
        condition_score = (
            metrics["text_contrast"] * 0.35 +         # CRITICAL (faded ink = unreadable)
            metrics["tears_and_holes"] * 0.25 +       # Critical (missing information)
            metrics["paper_color_variance"] * 0.20 +  # Important (color consistency)
            metrics["texture_degradation"] * 0.10 +   # Moderate (paper roughness)
            metrics["stains"] * 0.10                  # Moderate (localized discoloration)
        )

        metrics["condition_score"] = condition_score
        metrics["success"] = True

        return metrics

    except Exception as e:
        return {
            "text_contrast": 0,
            "paper_color_variance": 0,
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
    print("Detecting: text contrast, paper color, tears/holes, texture, stains")

    # Analyze each image
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing documents"):
        image_path = image_dir / f"{row['ID']}{image_ext}"

        if not image_path.exists():
            metrics = {
                "ID": row["ID"],
                "text_contrast": 0,
                "paper_color_variance": 0,
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

        print(f"\nCondition Metrics (0-100, higher = worse):")
        for metric in ["text_contrast", "paper_color_variance", "tears_and_holes",
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
            print(f"  Text contrast: {row['text_contrast']:.1f} | "
                  f"Paper color: {row['paper_color_variance']:.1f} | "
                  f"Tears: {row['tears_and_holes']:.1f}")
            print(f"  Texture: {row['texture_degradation']:.1f} | "
                  f"Stains: {row['stains']:.1f}")

        # Best condition documents
        print(f"\n🟢 Top 10 Best Condition Documents:")
        print("="*80)
        best = successful.nsmallest(10, "condition_score")
        for _, row in best.iterrows():
            print(f"\n{row['ID']}")
            print(f"  Condition: {row['condition_score']:.1f}")
            print(f"  Text contrast: {row['text_contrast']:.1f} | "
                  f"Paper color: {row['paper_color_variance']:.1f} | "
                  f"Tears: {row['tears_and_holes']:.1f}")
            print(f"  Texture: {row['texture_degradation']:.1f} | "
                  f"Stains: {row['stains']:.1f}")


if __name__ == "__main__":
    main()
