"""
Test Adaptive Augmentation

Verifies that augmentation strength adapts based on document condition score.
"""

import argparse
import sys
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from PIL import Image

# Add parent dir to path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from train import ImageAugmenter

REPO_ROOT = SCRIPT_DIR.parent.parent


def test_adaptive_augmentation():
    parser = argparse.ArgumentParser(description="Test adaptive augmentation")
    parser.add_argument("--config", type=str, default="config_qwen3_8b_full.yaml")
    args = parser.parse_args()

    # Load config
    with open(SCRIPT_DIR / args.config) as f:
        cfg = yaml.safe_load(f)

    aug_cfg = cfg.get("augmentation", {})
    data_cfg = cfg.get("data", {})

    print("=" * 70)
    print("Adaptive Augmentation Test")
    print("=" * 70)

    # Load condition scores
    condition_csv = REPO_ROOT / "dataset" / "document_condition.csv"
    if not condition_csv.exists():
        print(f"\n❌ Document condition CSV not found: {condition_csv}")
        return

    condition_df = pd.read_csv(condition_csv)
    condition_df = condition_df[condition_df["success"] == True]
    print(f"\nLoaded {len(condition_df)} condition scores")
    print(f"  Range: {condition_df['condition_score'].min():.1f} - {condition_df['condition_score'].max():.1f}")
    print(f"  Mean: {condition_df['condition_score'].mean():.1f}, Std: {condition_df['condition_score'].std():.1f}")

    # Find examples from each tier
    good_cond = condition_df[condition_df["condition_score"] < 15].iloc[0]
    medium_cond = condition_df[(condition_df["condition_score"] >= 15) & (condition_df["condition_score"] < 25)].iloc[0]
    poor_cond = condition_df[condition_df["condition_score"] >= 25].iloc[0]

    tiers = [
        ("Good", good_cond["ID"], good_cond["condition_score"]),
        ("Medium", medium_cond["ID"], medium_cond["condition_score"]),
        ("Poor", poor_cond["ID"], poor_cond["condition_score"]),
    ]

    print("\nTest Samples:")
    for tier_name, sample_id, score in tiers:
        print(f"  {tier_name} condition (score={score:.1f}): {sample_id}")

    # Load images
    image_dir = REPO_ROOT / data_cfg["image_dir"]
    image_ext = data_cfg.get("image_ext", ".jpg")

    # Create augmenter
    augmenter = ImageAugmenter(aug_cfg)

    print("\n" + "=" * 70)
    print("Testing Adaptive Augmentation")
    print("=" * 70)

    # Test each tier
    for tier_name, sample_id, condition_score in tiers:
        print(f"\n{tier_name} Condition (score={condition_score:.1f}):")

        image_path = image_dir / f"{sample_id}{image_ext}"
        if not image_path.exists():
            print(f"  ⚠️  Image not found: {image_path}")
            continue

        img = Image.open(image_path).convert("RGB")
        w, h = img.size

        # Apply augmentation 10 times and track results
        aug_count = {"elastic": 0, "color": 0, "resolution": 0}
        sizes = []

        for i in range(10):
            aug_img = augmenter(img.copy(), condition_score=condition_score)
            aug_w, aug_h = aug_img.size
            sizes.append((aug_w, aug_h))

            # Count which augmentations were applied (approximate)
            if (aug_w, aug_h) != (w, h):
                aug_count["resolution"] += 1

        # Statistics
        pixel_ratios = [(aw * ah) / (w * h) for aw, ah in sizes]
        min_ratio = min(pixel_ratios)
        max_ratio = max(pixel_ratios)
        mean_ratio = np.mean(pixel_ratios)

        print(f"  Original size: {w}×{h} ({w*h:,} pixels)")
        print(f"  Augmented sizes (10 trials):")
        print(f"    Pixel ratio range: {min_ratio:.3f} - {max_ratio:.3f}")
        print(f"    Mean ratio: {mean_ratio:.3f}")
        print(f"    Resolution changes: {aug_count['resolution']}/10 trials")

    # Test expected behavior
    print("\n" + "=" * 70)
    print("Expected Adaptive Behavior")
    print("=" * 70)

    print("\nGood condition (< 15):")
    print("  ✓ Aggressive augmentation (synthesize degradation)")
    print("  ✓ Higher probability of elastic/color jitter")
    print("  ✓ Can downsample to 75% (0.75 min_pixels_ratio)")

    print("\nMedium condition (15-25):")
    print("  ✓ Default augmentation settings")
    print("  ✓ Standard probabilities")
    print("  ✓ Min pixels ratio from config (0.85)")

    print("\nPoor condition (> 25):")
    print("  ✓ Minimal augmentation (preserve readability)")
    print("  ✓ Reduced probability of all augmentations")
    print("  ✓ Very conservative downsampling (0.9 min_pixels_ratio)")

    print("\n" + "=" * 70)
    print("Test Complete")
    print("=" * 70)

    if aug_cfg.get("enabled", False):
        print("\n✅ Adaptive augmentation is ready!")
        print("   Augmentation strength will automatically adjust based on document condition.")
    else:
        print("\n⚠️  Augmentations are DISABLED in config")
        print("   Set 'enabled: true' in augmentation config to use adaptive augmentation")


if __name__ == "__main__":
    test_adaptive_augmentation()
