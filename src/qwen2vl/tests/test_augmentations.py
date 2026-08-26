"""
Test Document-Aware Augmentations

Verifies that elastic, color jitter, and resolution jitter augmentations work correctly.

Usage:
    python test_augmentations.py --config config_qwen3_8b_full.yaml
"""

import argparse
import sys
from pathlib import Path
import yaml
import random
import numpy as np
from PIL import Image

# Add parent dir to path to import from train.py
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from train import ImageAugmenter

REPO_ROOT = SCRIPT_DIR.parent.parent


def test_augmentations():
    parser = argparse.ArgumentParser(description="Test augmentations")
    parser.add_argument("--config", type=str, default="config_qwen3_8b_full.yaml")
    parser.add_argument("--sample_id", type=str, default=None, help="Sample ID to test (random if not specified)")
    args = parser.parse_args()

    # Load config
    with open(SCRIPT_DIR / args.config) as f:
        cfg = yaml.safe_load(f)

    aug_cfg = cfg.get("augmentation", {})
    data_cfg = cfg.get("data", {})

    print("="*70)
    print("Document-Aware Augmentation Test")
    print("="*70)

    # Check enabled augmentations
    print("\nEnabled Augmentations:")
    print(f"  enabled: {aug_cfg.get('enabled', False)}")
    print(f"  p_rotate: {aug_cfg.get('p_rotate', 0.0)}")
    print(f"  p_shear: {aug_cfg.get('p_shear', 0.0)}")
    print(f"  p_elastic: {aug_cfg.get('p_elastic', 0.0)} (NEW)")
    print(f"  p_color_jitter: {aug_cfg.get('p_color_jitter', 0.0)} (NEW)")
    print(f"  p_resolution_jitter: {aug_cfg.get('p_resolution_jitter', 0.0)} (NEW)")

    # Load a sample image
    train_csv = REPO_ROOT / data_cfg["train_csv"]
    import pandas as pd
    df = pd.read_csv(train_csv)

    if args.sample_id:
        sample = df[df["ID"] == args.sample_id].iloc[0]
    else:
        sample = df.sample(1).iloc[0]

    sample_id = sample["ID"]
    sample_text = sample["Target"]

    image_dir = REPO_ROOT / data_cfg["image_dir"]
    image_ext = data_cfg.get("image_ext", ".jpg")
    image_path = image_dir / f"{sample_id}{image_ext}"

    if not image_path.exists():
        print(f"\n❌ Image not found: {image_path}")
        return

    print(f"\nTest Image:")
    print(f"  ID: {sample_id}")
    print(f"  Text: {sample_text[:60]}...")
    print(f"  Path: {image_path}")

    # Load image
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    print(f"  Size: {w} × {h} ({w*h:,} pixels)")

    # Create augmenter
    augmenter = ImageAugmenter(aug_cfg)

    print("\n" + "="*70)
    print("Testing Individual Augmentations")
    print("="*70)

    # Test each augmentation 5 times
    tests = [
        ("Elastic Deformation", "p_elastic"),
        ("Color Jitter (Hue/Sat)", "p_color_jitter"),
        ("Resolution Jitter", "p_resolution_jitter"),
    ]

    for aug_name, cfg_key in tests:
        print(f"\n{aug_name}:")
        p_value = aug_cfg.get(cfg_key, 0.0)

        if p_value == 0.0:
            print(f"  ⚠️  Disabled (p=0.0) - skipping test")
            continue

        # Temporarily set probability to 1.0 for testing
        original_p = getattr(augmenter, cfg_key, 0.0)
        setattr(augmenter, cfg_key, 1.0)

        successes = 0
        failures = []

        for i in range(5):
            try:
                # Apply full augmentation pipeline
                aug_img = augmenter(img.copy())
                aug_w, aug_h = aug_img.size

                # Verify image is still valid
                assert aug_img.mode == "RGB", f"Wrong mode: {aug_img.mode}"
                assert aug_w > 0 and aug_h > 0, f"Invalid size: {aug_w}×{aug_h}"

                # Check if resolution jitter changed size
                if cfg_key == "p_resolution_jitter":
                    ratio = (aug_w * aug_h) / (w * h)
                    print(f"  Trial {i+1}: {aug_w}×{aug_h} (ratio: {ratio:.2f})")
                else:
                    print(f"  Trial {i+1}: ✓ Success ({aug_w}×{aug_h})")

                successes += 1

            except Exception as e:
                failures.append(f"Trial {i+1}: {str(e)}")

        # Restore original probability
        setattr(augmenter, cfg_key, original_p)

        if failures:
            print(f"\n  ❌ {len(failures)}/5 trials failed:")
            for failure in failures:
                print(f"     {failure}")
        else:
            print(f"  ✅ All 5 trials successful")

    # Test full pipeline
    print("\n" + "="*70)
    print("Testing Full Augmentation Pipeline")
    print("="*70)

    print("\nApplying full augmentation pipeline 10 times...")
    sizes = []

    for i in range(10):
        try:
            aug_img = augmenter(img.copy())
            aug_w, aug_h = aug_img.size
            sizes.append((aug_w, aug_h))
            print(f"  Trial {i+1}: {aug_w:4d} × {aug_h:4d} pixels")
        except Exception as e:
            print(f"  Trial {i+1}: ❌ Failed - {e}")

    if sizes:
        widths = [s[0] for s in sizes]
        heights = [s[1] for s in sizes]

        print(f"\nSize Statistics:")
        print(f"  Width:  min={min(widths)}, max={max(widths)}, mean={np.mean(widths):.0f}")
        print(f"  Height: min={min(heights)}, max={max(heights)}, mean={np.mean(heights):.0f}")

        unique_sizes = len(set(sizes))
        print(f"  Unique sizes: {unique_sizes}/10")

        if unique_sizes == 1:
            print(f"  ⚠️  All images have same size - resolution jitter may not be working")
        elif unique_sizes < 5:
            print(f"  🟡 Low variety - check resolution jitter probability")
        else:
            print(f"  ✅ Good variety - augmentations working")

    print("\n" + "="*70)
    print("Test Complete")
    print("="*70)

    if aug_cfg.get("enabled", False):
        print("\n✅ Augmentations are ENABLED in config")
        print("   Ready to train with document-aware augmentations!")
    else:
        print("\n⚠️  Augmentations are DISABLED in config")
        print("   Set 'enabled: true' in augmentation config to use them")


if __name__ == "__main__":
    test_augmentations()
