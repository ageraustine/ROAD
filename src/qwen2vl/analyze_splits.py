"""
Analyze train/val/test distribution mismatch
"""
import pandas as pd
import numpy as np
import cv2
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent

def compute_features(img_path):
    """Extract features from image"""
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    return {
        'variance': float(img.std()),
        'mean_intensity': float(img.mean()),
        'width': img.shape[1],
        'height': img.shape[0],
        'edge_density': float(cv2.Canny(img, 50, 150).mean()),
    }

def analyze():
    train_csv = REPO_ROOT / "dataset/Train.csv"
    test_csv = REPO_ROOT / "dataset/Test.csv"
    image_dir = REPO_ROOT / "dataset/images"

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print()

    # Text length comparison
    train_df['length'] = train_df['Target'].astype(str).str.len()
    print("TEXT LENGTH DISTRIBUTION:")
    print(f"Train: min={train_df['length'].min()}, "
          f"median={train_df['length'].median():.0f}, "
          f"max={train_df['length'].max()}")
    print()

    # Image features comparison
    print("Computing image features (sample 500 from each)...")
    train_sample = train_df.sample(min(500, len(train_df)), random_state=42)
    test_sample = test_df.sample(min(500, len(test_df)), random_state=42)

    train_features = []
    for img_id in train_sample['ID']:
        feat = compute_features(image_dir / f"{img_id}.jpg")
        if feat:
            train_features.append(feat)

    test_features = []
    for img_id in test_sample['ID']:
        feat = compute_features(image_dir / f"{img_id}.jpg")
        if feat:
            test_features.append(feat)

    train_feat_df = pd.DataFrame(train_features)
    test_feat_df = pd.DataFrame(test_features)

    print("\nIMAGE FEATURE COMPARISON:")
    print(f"{'Feature':<20} {'Train Median':<15} {'Test Median':<15} {'Difference'}")
    print("-" * 70)

    for col in ['variance', 'mean_intensity', 'width', 'height', 'edge_density']:
        train_med = train_feat_df[col].median()
        test_med = test_feat_df[col].median()
        diff_pct = ((test_med - train_med) / train_med) * 100
        print(f"{col:<20} {train_med:<15.2f} {test_med:<15.2f} {diff_pct:+.1f}%")

    print("\n" + "="*70)
    print("INTERPRETATION:")
    print("="*70)

    # Check variance
    var_diff = ((test_feat_df['variance'].median() - train_feat_df['variance'].median())
                / train_feat_df['variance'].median()) * 100

    if abs(var_diff) > 10:
        if var_diff < 0:
            print(f"⚠️  Test images are {abs(var_diff):.0f}% MORE DEGRADED (lower variance)")
            print("   → Validation should weight harder (lower variance) samples more")
        else:
            print(f"⚠️  Test images are {abs(var_diff):.0f}% CLEARER (higher variance)")
            print("   → Validation should weight easier (higher variance) samples more")
    else:
        print("✓ Variance distribution is similar")

    # Check intensity
    intensity_diff = ((test_feat_df['mean_intensity'].median() - train_feat_df['mean_intensity'].median())
                      / train_feat_df['mean_intensity'].median()) * 100

    if abs(intensity_diff) > 5:
        print(f"⚠️  Test images have {intensity_diff:+.1f}% different brightness")
    else:
        print("✓ Brightness distribution is similar")

    print("\nRECOMMENDATIONS:")
    if abs(var_diff) > 10 or abs(intensity_diff) > 5:
        print("1. Use adversarial validation to make val set match test")
        print("2. Or increase val_split to 15-20% for better coverage")
    else:
        print("1. Distribution looks similar - mismatch may be from other factors")
        print("2. Try increasing val_split to 15-20%")
        print("3. Check if test has different document types/scribes")

if __name__ == "__main__":
    analyze()
