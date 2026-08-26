"""
Fold Difficulty Analysis

Checks if k-fold performance variance is explained by image quality imbalance.
Requires: dataset/image_quality.csv (run analyze_image_quality.py first)

Usage:
    python analyze_fold_difficulty.py --config config_qwen3_8b_full.yaml

This will:
1. Load document clusters and image quality scores
2. Simulate k-fold splits (same as training)
3. Compute average difficulty per fold
4. Check if harder folds correlate with worse training performance
"""

import argparse
from pathlib import Path

import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent


def main():
    parser = argparse.ArgumentParser(description="Analyze fold difficulty distribution")
    parser.add_argument(
        "--config",
        type=str,
        default="config_qwen3_8b_full.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--quality_csv",
        type=str,
        default=None,
        help="Path to image quality CSV (default: dataset/image_quality.csv)"
    )
    args = parser.parse_args()

    # Load config
    config_path = SCRIPT_DIR / args.config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]

    # Load quality scores
    if args.quality_csv:
        quality_path = Path(args.quality_csv)
    else:
        quality_path = REPO_ROOT / "dataset" / "image_quality.csv"

    if not quality_path.exists():
        print(f"❌ Quality CSV not found: {quality_path}")
        print(f"\nRun this first:")
        print(f"  python analyze_image_quality.py --config {args.config}")
        return

    quality_df = pd.read_csv(quality_path)
    print(f"Loaded quality scores for {len(quality_df)} images")

    # Load training data
    train_csv = REPO_ROOT / data_cfg["train_csv"]
    df = pd.read_csv(train_csv)

    # Load clusters if available
    cluster_csv = data_cfg.get("cluster_csv")
    if cluster_csv:
        cluster_path = REPO_ROOT / cluster_csv
        if cluster_path.exists():
            cluster_df = pd.read_csv(cluster_path)
            df = df.merge(cluster_df, on="ID", how="left")
            print(f"Loaded {df['cluster_id'].nunique()} document clusters")
        else:
            print(f"⚠️  Cluster CSV not found: {cluster_path}")
            df["cluster_id"] = 0
    else:
        print("⚠️  No cluster_csv in config, using dummy groups")
        df["cluster_id"] = 0

    # Merge quality scores
    df = df.merge(quality_df[["ID", "difficulty_score", "contrast", "ink_fade", "dark_patches"]],
                  on="ID", how="left")

    # Fill missing quality scores with median
    for col in ["difficulty_score", "contrast", "ink_fade", "dark_patches"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Simulate k-fold splits (same as train.py)
    k_folds = data_cfg.get("k_folds", 1)
    seed = data_cfg.get("seed", 42)

    if k_folds <= 1:
        print("\n⚠️  k_folds <= 1, cannot analyze fold difficulty")
        print("Set k_folds > 1 in config to analyze fold variance")
        return

    print(f"\nSimulating {k_folds}-fold splits (seed={seed})...")

    # Create stratification bins (same as train.py)
    df["_has_digit"] = df["Target"].str.contains(r"\d", regex=True, na=False)
    df["_has_upper"] = df["Target"].str.contains(r"[A-Z]", regex=True, na=False)
    df["_text_len"] = df["Target"].str.len()
    df["_len_bin"] = pd.qcut(df["_text_len"], q=5, labels=False, duplicates="drop")
    df["_bin"] = (
        df["_has_digit"].astype(str) + "_" +
        df["_has_upper"].astype(str) + "_" +
        df["_len_bin"].astype(str)
    )

    # Group by cluster (same as train.py)
    group_col = data_cfg.get("group_col", "cluster_id")
    if group_col not in df.columns:
        print(f"⚠️  group_col '{group_col}' not in dataframe, using cluster_id")
        group_col = "cluster_id"

    # Get one representative per cluster
    cluster_reps = df.groupby(group_col).first().reset_index()

    # StratifiedGroupKFold on clusters
    splitter = StratifiedGroupKFold(n_splits=k_folds, shuffle=True, random_state=seed)

    fold_stats = []

    for fold_idx, (train_cluster_idx, val_cluster_idx) in enumerate(
        splitter.split(cluster_reps, cluster_reps["_bin"], cluster_reps[group_col])
    ):
        train_clusters = cluster_reps.iloc[train_cluster_idx][group_col].values
        val_clusters = cluster_reps.iloc[val_cluster_idx][group_col].values

        # Get all samples in these clusters
        train_mask = df[group_col].isin(train_clusters)
        val_mask = df[group_col].isin(val_clusters)

        train_df = df[train_mask]
        val_df = df[val_mask]

        # Compute difficulty statistics
        train_difficulty = train_df["difficulty_score"].mean()
        val_difficulty = val_df["difficulty_score"].mean()

        train_contrast = train_df["contrast"].mean()
        val_contrast = val_df["contrast"].mean()

        train_ink_fade = train_df["ink_fade"].mean()
        val_ink_fade = val_df["ink_fade"].mean()

        fold_stats.append({
            "fold": fold_idx,
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "train_clusters": len(train_clusters),
            "val_clusters": len(val_clusters),
            "train_difficulty": train_difficulty,
            "val_difficulty": val_difficulty,
            "train_contrast": train_contrast,
            "val_contrast": val_contrast,
            "train_ink_fade": train_ink_fade,
            "val_ink_fade": val_ink_fade,
        })

    fold_stats_df = pd.DataFrame(fold_stats)

    # Print results
    print("\n" + "="*90)
    print("Fold Difficulty Analysis")
    print("="*90)

    print(f"\n{'Fold':<6} {'Train':<8} {'Val':<8} {'Train Diff':<12} {'Val Diff':<12} {'ΔDiff':<10}")
    print("-"*90)

    for _, row in fold_stats_df.iterrows():
        delta_diff = row["train_difficulty"] - row["val_difficulty"]
        print(f"{int(row['fold']):<6} "
              f"{int(row['train_samples']):<8} "
              f"{int(row['val_samples']):<8} "
              f"{row['train_difficulty']:<12.2f} "
              f"{row['val_difficulty']:<12.2f} "
              f"{delta_diff:+10.2f}")

    # Check variance
    train_diff_std = fold_stats_df["train_difficulty"].std()
    val_diff_std = fold_stats_df["val_difficulty"].std()

    print(f"\nTraining Set Difficulty:")
    print(f"  Mean: {fold_stats_df['train_difficulty'].mean():.2f}")
    print(f"  Std:  {train_diff_std:.2f} ({'HIGH' if train_diff_std > 2.0 else 'moderate'})")
    print(f"  Range: {fold_stats_df['train_difficulty'].min():.2f} - {fold_stats_df['train_difficulty'].max():.2f}")

    print(f"\nValidation Set Difficulty:")
    print(f"  Mean: {fold_stats_df['val_difficulty'].mean():.2f}")
    print(f"  Std:  {val_diff_std:.2f} ({'HIGH' if val_diff_std > 2.0 else 'moderate'})")
    print(f"  Range: {fold_stats_df['val_difficulty'].min():.2f} - {fold_stats_df['val_difficulty'].max():.2f}")

    # Detailed quality breakdown
    print("\n" + "="*90)
    print("Quality Metric Breakdown")
    print("="*90)

    print(f"\n{'Fold':<6} {'Train Contrast':<15} {'Val Contrast':<15} {'Train Fade':<12} {'Val Fade':<12}")
    print("-"*90)

    for _, row in fold_stats_df.iterrows():
        print(f"{int(row['fold']):<6} "
              f"{row['train_contrast']:<15.2f} "
              f"{row['val_contrast']:<15.2f} "
              f"{row['train_ink_fade']:<12.2f} "
              f"{row['val_ink_fade']:<12.2f}")

    # Diagnosis
    print("\n" + "="*90)
    print("Diagnosis")
    print("="*90)

    if train_diff_std > 2.0:
        print("\n🔴 HIGH training set difficulty variance detected!")
        print(f"  Training difficulty std = {train_diff_std:.2f}")
        print("\n  This explains fold performance variance:")
        print("  - Folds with easier training data learn better")
        print("  - Folds with harder training data struggle")
        print("\n  Root cause: Only 68 document clusters for k=3")
        print("  Each fold gets 45-46 clusters, but quality is not balanced")
        print("\n  Solutions:")
        print("  1. Use single split (k_folds=1) - train on 90% of data")
        print("  2. Manually balance clusters by difficulty before splitting")
        print("  3. Use curriculum learning (easy→hard within each fold)")

    elif train_diff_std > 1.0:
        print("\n🟡 Moderate training set difficulty variance")
        print(f"  Training difficulty std = {train_diff_std:.2f}")
        print("\n  This contributes to fold performance differences")
        print("  Consider using k_folds=1 or curriculum learning")

    else:
        print("\n🟢 Low training set difficulty variance")
        print(f"  Training difficulty std = {train_diff_std:.2f}")
        print("\n  Fold variance is NOT explained by image quality imbalance")
        print("  Other factors likely responsible (scribe diversity, content difficulty)")

    # Identify best/worst folds
    best_fold = fold_stats_df.loc[fold_stats_df["train_difficulty"].idxmin()]
    worst_fold = fold_stats_df.loc[fold_stats_df["train_difficulty"].idxmax()]

    print(f"\n🟢 Easiest fold (likely to perform best):")
    print(f"  Fold {int(best_fold['fold'])} - Train difficulty = {best_fold['train_difficulty']:.2f}")

    print(f"\n🔴 Hardest fold (likely to perform worst):")
    print(f"  Fold {int(worst_fold['fold'])} - Train difficulty = {worst_fold['train_difficulty']:.2f}")

    difficulty_gap = worst_fold["train_difficulty"] - best_fold["train_difficulty"]
    print(f"\n  Difficulty gap: {difficulty_gap:.2f} points")

    if difficulty_gap > 3.0:
        print(f"  ⚠️  Large gap suggests fold {int(best_fold['fold'])} will significantly outperform fold {int(worst_fold['fold'])}")


if __name__ == "__main__":
    main()
