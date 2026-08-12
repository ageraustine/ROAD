"""
Verify Text Difficulty Stratification

Quick check to ensure train/val splits have balanced difficulty distributions.

Usage:
    python verify_stratification.py --config config_qwen3_8b_full.yaml
"""

import argparse
from pathlib import Path

import yaml
import pandas as pd
import numpy as np

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent


def main():
    parser = argparse.ArgumentParser(description="Verify stratification balance")
    parser.add_argument(
        "--config",
        type=str,
        default="config_qwen3_8b_full.yaml",
        help="Path to config file"
    )
    args = parser.parse_args()

    # Load config
    config_path = SCRIPT_DIR / args.config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]

    # Check text difficulty CSV exists
    text_diff_path = REPO_ROOT / "dataset" / "text_difficulty.csv"
    if not text_diff_path.exists():
        print(f"❌ Text difficulty CSV not found: {text_diff_path}")
        print(f"\nRun this first:")
        print(f"  python analyze_text_difficulty.py --config {args.config}")
        return

    # Load text difficulty
    text_df = pd.read_csv(text_diff_path)
    print(f"✓ Loaded text difficulty scores for {len(text_df)} samples")

    # Load training data
    train_csv = REPO_ROOT / data_cfg["train_csv"]
    df = pd.read_csv(train_csv)

    # Merge
    df = df.merge(text_df[["ID", "difficulty_score"]], on="ID", how="left")

    # Fill missing values (avoid chained assignment warning)
    median_diff = df["difficulty_score"].median()
    df.loc[df["difficulty_score"].isna(), "difficulty_score"] = median_diff

    # Simulate stratification (same as train.py)
    # Create bins
    df["_text_diff_bin"] = pd.qcut(df["difficulty_score"], q=3, labels=["easy", "medium", "hard"], duplicates="drop")
    df["_has_digit"] = df["Target"].str.contains(r"\d", regex=True, na=False)
    df["_has_upper"] = df["Target"].str.contains(r"[A-Z]", regex=True, na=False)
    df["_text_len"] = df["Target"].str.len()

    df["_digit_bin"] = df["_has_digit"].map({True: "has_nums", False: "no_nums"})
    df["_upper_bin"] = df["_has_upper"].map({True: "has_names", False: "no_names"})

    # Simple 12-bin stratification (no length to avoid rare bins)
    df["_bin"] = (df["_text_diff_bin"].astype(str) + "_" +
                  df["_digit_bin"].astype(str) + "_" +
                  df["_upper_bin"].astype(str))

    # Simulate split
    from sklearn.model_selection import train_test_split

    val_split = data_cfg.get("val_split", 0.1)
    seed = data_cfg.get("seed", 42)

    train_df, val_df = train_test_split(
        df,
        test_size=val_split,
        stratify=df["_bin"],
        random_state=seed
    )

    # Analyze distributions
    print("\n" + "="*70)
    print("Stratification Verification")
    print("="*70)

    print(f"\nSplit: {len(train_df)} train ({len(train_df)/len(df)*100:.1f}%), {len(val_df)} val ({len(val_df)/len(df)*100:.1f}%)")

    # Check difficulty balance
    train_diff = train_df["difficulty_score"].describe()
    val_diff = val_df["difficulty_score"].describe()

    print(f"\nText Difficulty Distribution:")
    print(f"  {'Metric':<10} {'Train':>10} {'Val':>10} {'Δ':>8}")
    print(f"  {'-'*40}")
    print(f"  {'Min':<10} {train_diff['min']:>10.1f} {val_diff['min']:>10.1f} {abs(train_diff['min']-val_diff['min']):>8.1f}")
    print(f"  {'25%':<10} {train_diff['25%']:>10.1f} {val_diff['25%']:>10.1f} {abs(train_diff['25%']-val_diff['25%']):>8.1f}")
    print(f"  {'Median':<10} {train_diff['50%']:>10.1f} {val_diff['50%']:>10.1f} {abs(train_diff['50%']-val_diff['50%']):>8.1f}")
    print(f"  {'75%':<10} {train_diff['75%']:>10.1f} {val_diff['75%']:>10.1f} {abs(train_diff['75%']-val_diff['75%']):>8.1f}")
    print(f"  {'Max':<10} {train_diff['max']:>10.1f} {val_diff['max']:>10.1f} {abs(train_diff['max']-val_diff['max']):>8.1f}")
    print(f"  {'Mean':<10} {train_diff['mean']:>10.1f} {val_diff['mean']:>10.1f} {abs(train_diff['mean']-val_diff['mean']):>8.1f}")
    print(f"  {'Std':<10} {train_diff['std']:>10.1f} {val_diff['std']:>10.1f} {abs(train_diff['std']-val_diff['std']):>8.1f}")

    # Check bin distributions
    print(f"\nDifficulty Bin Distribution:")
    train_bins = train_df["_text_diff_bin"].value_counts(normalize=True).sort_index()
    val_bins = val_df["_text_diff_bin"].value_counts(normalize=True).sort_index()

    for bin_name in ["easy", "medium", "hard"]:
        train_pct = train_bins.get(bin_name, 0) * 100
        val_pct = val_bins.get(bin_name, 0) * 100
        print(f"  {bin_name.capitalize():<10} Train: {train_pct:>5.1f}%  Val: {val_pct:>5.1f}%  Δ: {abs(train_pct-val_pct):>4.1f}%")

    # Check digit distribution
    print(f"\nHas Digits Distribution:")
    train_digits = train_df["_has_digit"].value_counts(normalize=True)
    val_digits = val_df["_has_digit"].value_counts(normalize=True)

    train_has_nums = train_digits.get(True, 0) * 100
    val_has_nums = val_digits.get(True, 0) * 100
    print(f"  {'Has numbers':<15} Train: {train_has_nums:>5.1f}%  Val: {val_has_nums:>5.1f}%  Δ: {abs(train_has_nums-val_has_nums):>4.1f}%")

    # Check uppercase distribution
    print(f"\nHas Names (Uppercase) Distribution:")
    train_upper = train_df["_has_upper"].value_counts(normalize=True)
    val_upper = val_df["_has_upper"].value_counts(normalize=True)

    train_has_names = train_upper.get(True, 0) * 100
    val_has_names = val_upper.get(True, 0) * 100
    print(f"  {'Has names':<15} Train: {train_has_names:>5.1f}%  Val: {val_has_names:>5.1f}%  Δ: {abs(train_has_names-val_has_names):>4.1f}%")

    # Check length distribution
    print(f"\nText Length Distribution:")
    train_len = train_df["_text_len"].describe()
    val_len = val_df["_text_len"].describe()

    print(f"  {'Metric':<10} {'Train':>10} {'Val':>10} {'Δ':>8}")
    print(f"  {'-'*40}")
    print(f"  {'Min':<10} {train_len['min']:>10.0f} {val_len['min']:>10.0f} {abs(train_len['min']-val_len['min']):>8.0f}")
    print(f"  {'Median':<10} {train_len['50%']:>10.0f} {val_len['50%']:>10.0f} {abs(train_len['50%']-val_len['50%']):>8.0f}")
    print(f"  {'Max':<10} {train_len['max']:>10.0f} {val_len['max']:>10.0f} {abs(train_len['max']-val_len['max']):>8.0f}")
    print(f"  {'Mean':<10} {train_len['mean']:>10.1f} {val_len['mean']:>10.1f} {abs(train_len['mean']-val_len['mean']):>8.1f}")

    # Overall assessment
    print("\n" + "="*70)
    print("Assessment")
    print("="*70)

    diff_gap = abs(train_diff['mean'] - val_diff['mean'])
    len_gap = abs(train_len['mean'] - val_len['mean'])

    if diff_gap < 1.0 and len_gap < 5.0:
        print("\n✅ EXCELLENT - Stratification is working perfectly!")
        print(f"  Difficulty gap: {diff_gap:.2f} (target: <1.0)")
        print(f"  Length gap: {len_gap:.1f} chars (target: <5.0)")
        print("\n  Train and val have nearly identical difficulty distributions.")
        print("  The model will receive consistent, balanced data.")
    elif diff_gap < 2.0 and len_gap < 10.0:
        print("\n🟢 GOOD - Stratification is working well")
        print(f"  Difficulty gap: {diff_gap:.2f}")
        print(f"  Length gap: {len_gap:.1f} chars")
        print("\n  Minor differences are acceptable and won't impact performance.")
    else:
        print("\n⚠️  WARNING - Stratification imbalance detected")
        print(f"  Difficulty gap: {diff_gap:.2f} (target: <2.0)")
        print(f"  Length gap: {len_gap:.1f} chars (target: <10.0)")
        print("\n  Check that text_difficulty.csv was generated correctly.")

    # Number of bins check
    n_bins = df["_bin"].nunique()
    print(f"\n📊 Stratification bins created: {n_bins}")
    print(f"  Target: 12 bins (difficulty×3 × digits×2 × names×2)")

    if n_bins < 8:
        print(f"  ⚠️  Fewer bins than expected - may indicate data issues")
    elif n_bins < 10:
        print(f"  🟡 Some rare bins missing - acceptable")
    else:
        print(f"  ✅ All bins present - stratification working correctly")

    print("\n" + "="*70)
    print("Ready to train! Run:")
    print("  python train.py --config config_qwen3_8b_full.yaml")
    print("="*70)


if __name__ == "__main__":
    main()
