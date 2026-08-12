"""
Analyze Cluster Difficulty Distribution

Checks if certain document clusters are significantly harder than others.
This could explain fold variance if hard clusters are unevenly distributed.

Usage:
    python analyze_cluster_difficulty.py --config config_qwen3_8b_full.yaml
"""

import argparse
from pathlib import Path

import yaml
import pandas as pd
import numpy as np

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent


def main():
    parser = argparse.ArgumentParser(description="Analyze cluster difficulty")
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
        help="Path to quality CSV (default: dataset/image_quality.csv)"
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
    quality_df = quality_df[quality_df["success"] == True]
    print(f"Loaded quality scores for {len(quality_df)} images")

    # Load training data
    train_csv = REPO_ROOT / data_cfg["train_csv"]
    train_df = pd.read_csv(train_csv)

    # Merge
    df = quality_df.merge(train_df, on="ID", how="left")

    # Load clusters
    cluster_csv = data_cfg.get("cluster_csv")
    if not cluster_csv:
        print("\n❌ No cluster_csv in config")
        return

    cluster_path = REPO_ROOT / cluster_csv
    if not cluster_path.exists():
        print(f"\n❌ Cluster CSV not found: {cluster_path}")
        return

    cluster_df = pd.read_csv(cluster_path)
    df = df.merge(cluster_df, on="ID", how="left")

    group_col = data_cfg.get("group_col", "cluster_id")
    if group_col not in df.columns:
        print(f"\n❌ group_col '{group_col}' not found in data")
        return

    print(f"Loaded {df[group_col].nunique()} document clusters")

    # Compute cluster-level statistics
    cluster_stats = df.groupby(group_col).agg({
        "difficulty_score": ["mean", "std", "min", "max", "count"],
        "contrast": "mean",
        "ink_fade": "mean",
        "dark_patches": "mean",
        "Target": lambda x: x.str.len().mean()
    }).reset_index()

    cluster_stats.columns = [
        group_col, "difficulty_mean", "difficulty_std", "difficulty_min",
        "difficulty_max", "n_images", "contrast_mean", "ink_fade_mean",
        "dark_patches_mean", "text_len_mean"
    ]

    cluster_stats = cluster_stats.sort_values("difficulty_mean", ascending=False)

    # Print overall stats
    print("\n" + "="*90)
    print("Cluster Difficulty Distribution")
    print("="*90)

    print(f"\nOverall cluster difficulty:")
    print(f"  Mean: {cluster_stats['difficulty_mean'].mean():.2f}")
    print(f"  Std:  {cluster_stats['difficulty_mean'].std():.2f}")
    print(f"  Min:  {cluster_stats['difficulty_mean'].min():.2f}")
    print(f"  Max:  {cluster_stats['difficulty_mean'].max():.2f}")
    print(f"  Range: {cluster_stats['difficulty_mean'].max() - cluster_stats['difficulty_mean'].min():.2f}")

    cluster_variance = cluster_stats['difficulty_mean'].std()

    if cluster_variance > 2.0:
        print(f"\n  🔴 HIGH cluster difficulty variance!")
        print(f"  Some clusters are SIGNIFICANTLY harder than others.")
    elif cluster_variance > 1.0:
        print(f"\n  🟡 Moderate cluster difficulty variance")
    else:
        print(f"\n  🟢 Low cluster difficulty variance")
        print(f"  All clusters have similar difficulty")

    # Top/bottom clusters
    print(f"\n🔴 Top 10 Hardest Clusters:")
    print("="*90)
    print(f"{'Cluster':>8} {'N':>5} {'Difficulty':>12} {'Contrast':>10} {'Ink Fade':>10} {'Dark Patches':>13}")
    print("-"*90)

    for _, row in cluster_stats.head(10).iterrows():
        print(f"{int(row[group_col]):>8} {int(row['n_images']):>5} "
              f"{row['difficulty_mean']:>12.2f} "
              f"{row['contrast_mean']:>10.2f} "
              f"{row['ink_fade_mean']:>10.2f} "
              f"{row['dark_patches_mean']:>13.2f}")

    print(f"\n🟢 Top 10 Easiest Clusters:")
    print("="*90)
    print(f"{'Cluster':>8} {'N':>5} {'Difficulty':>12} {'Contrast':>10} {'Ink Fade':>10} {'Dark Patches':>13}")
    print("-"*90)

    for _, row in cluster_stats.tail(10).iterrows():
        print(f"{int(row[group_col]):>8} {int(row['n_images']):>5} "
              f"{row['difficulty_mean']:>12.2f} "
              f"{row['contrast_mean']:>10.2f} "
              f"{row['ink_fade_mean']:>10.2f} "
              f"{row['dark_patches_mean']:>13.2f}")

    # Check if cluster size correlates with difficulty
    correlation = np.corrcoef(cluster_stats['n_images'], cluster_stats['difficulty_mean'])[0, 1]

    print(f"\n" + "="*90)
    print("Cluster Size vs Difficulty")
    print("="*90)
    print(f"\nCorrelation: {correlation:.3f}")

    if abs(correlation) > 0.3:
        direction = "harder" if correlation > 0 else "easier"
        print(f"  ⚠️  Larger clusters tend to be {direction}")
        print(f"  This creates imbalance in k-fold splits")
    else:
        print(f"  ✓ No strong correlation - size doesn't predict difficulty")

    # Identify outlier clusters
    difficulty_threshold_high = cluster_stats['difficulty_mean'].mean() + cluster_stats['difficulty_mean'].std()
    difficulty_threshold_low = cluster_stats['difficulty_mean'].mean() - cluster_stats['difficulty_mean'].std()

    outlier_hard = cluster_stats[cluster_stats['difficulty_mean'] > difficulty_threshold_high]
    outlier_easy = cluster_stats[cluster_stats['difficulty_mean'] < difficulty_threshold_low]

    print(f"\n" + "="*90)
    print("Outlier Clusters (±1 std)")
    print("="*90)

    print(f"\n🔴 Abnormally hard clusters ({len(outlier_hard)}):")
    if len(outlier_hard) > 0:
        print(f"  Clusters: {outlier_hard[group_col].tolist()}")
        print(f"  Total images: {outlier_hard['n_images'].sum()} ({outlier_hard['n_images'].sum()/len(df)*100:.1f}% of dataset)")
        print(f"  Avg difficulty: {outlier_hard['difficulty_mean'].mean():.2f}")

    print(f"\n🟢 Abnormally easy clusters ({len(outlier_easy)}):")
    if len(outlier_easy) > 0:
        print(f"  Clusters: {outlier_easy[group_col].tolist()}")
        print(f"  Total images: {outlier_easy['n_images'].sum()} ({outlier_easy['n_images'].sum()/len(df)*100:.1f}% of dataset)")
        print(f"  Avg difficulty: {outlier_easy['difficulty_mean'].mean():.2f}")

    # Impact on fold variance
    print(f"\n" + "="*90)
    print("Impact on K-Fold Variance")
    print("="*90)

    k_folds = data_cfg.get("k_folds", 3)

    if cluster_variance > 1.5:
        print(f"\n⚠️  With {len(cluster_stats)} clusters and k={k_folds}:")
        print(f"  Each fold gets ~{len(cluster_stats)//k_folds} clusters")
        print(f"  Cluster difficulty variance = {cluster_variance:.2f}")
        print(f"\n  Random assignment of clusters to folds will create:")
        print(f"  - Folds that get more hard clusters → worse training")
        print(f"  - Folds that get more easy clusters → better training")
        print(f"\n  This explains your fold performance variance!")
        print(f"\n  Solutions:")
        print(f"  1. Use k_folds=1 (single 90/10 split)")
        print(f"  2. Manually balance folds by cluster difficulty")
        print(f"  3. Increase document clustering to 100-150 clusters (better balance)")

    else:
        print(f"\n✓ Cluster difficulty variance is acceptable ({cluster_variance:.2f})")
        print(f"  Fold variance is NOT primarily caused by image quality imbalance")


if __name__ == "__main__":
    main()
