"""
Combined Image + Text Difficulty Analysis

Checks if fold performance is explained by COMBINED image quality + text complexity.
Individual factors might be balanced, but their interaction could cause variance.

Usage:
    python analyze_combined_difficulty.py --config config_qwen3_8b_full.yaml
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
    parser = argparse.ArgumentParser(description="Analyze combined difficulty")
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

    # Load image quality scores
    quality_path = REPO_ROOT / "dataset" / "image_quality.csv"
    if not quality_path.exists():
        print(f"❌ Image quality CSV not found: {quality_path}")
        return

    quality_df = pd.read_csv(quality_path)
    quality_df = quality_df[quality_df["success"] == True]
    print(f"Loaded image quality scores for {len(quality_df)} samples")

    # Load text difficulty scores
    text_path = REPO_ROOT / "dataset" / "text_difficulty.csv"
    if not text_path.exists():
        print(f"❌ Text difficulty CSV not found: {text_path}")
        return

    text_df = pd.read_csv(text_path)
    print(f"Loaded text difficulty scores for {len(text_df)} samples")

    # Load training data
    train_csv = REPO_ROOT / data_cfg["train_csv"]
    df = pd.read_csv(train_csv)

    # Load clusters
    cluster_csv = data_cfg.get("cluster_csv")
    if cluster_csv:
        cluster_path = REPO_ROOT / cluster_csv
        if cluster_path.exists():
            cluster_df = pd.read_csv(cluster_path)
            df = df.merge(cluster_df, on="ID", how="left")
            print(f"Loaded {df['cluster_id'].nunique()} document clusters")
        else:
            df["cluster_id"] = 0
    else:
        df["cluster_id"] = 0

    # Merge quality and text scores
    df = df.merge(quality_df[["ID", "difficulty_score", "contrast", "ink_fade"]],
                  on="ID", how="left", suffixes=("", "_image"))
    df = df.merge(text_df[["ID", "difficulty_score", "rare_word_ratio", "named_entity_score", "number_complexity"]],
                  on="ID", how="left", suffixes=("_image", "_text"))

    # Rename for clarity
    df["image_difficulty"] = df["difficulty_score_image"]
    df["text_difficulty"] = df["difficulty_score_text"]

    # Compute combined difficulty (weighted average)
    # Image quality is often more impactful than text complexity for OCR
    df["combined_difficulty"] = (
        df["image_difficulty"] * 0.6 +  # Image quality more important
        df["text_difficulty"] * 0.4     # Text complexity secondary
    )

    # Simulate k-fold splits
    k_folds = data_cfg.get("k_folds", 1)
    seed = data_cfg.get("seed", 42)

    if k_folds <= 1:
        print("\n⚠️  k_folds <= 1, cannot analyze fold difficulty")
        return

    print(f"\nSimulating {k_folds}-fold splits (seed={seed})...")

    # Create stratification bins
    df["_has_digit"] = df["Target"].str.contains(r"\d", regex=True, na=False)
    df["_has_upper"] = df["Target"].str.contains(r"[A-Z]", regex=True, na=False)
    df["_text_len"] = df["Target"].str.len()
    df["_len_bin"] = pd.qcut(df["_text_len"], q=5, labels=False, duplicates="drop")
    df["_bin"] = (
        df["_has_digit"].astype(str) + "_" +
        df["_has_upper"].astype(str) + "_" +
        df["_len_bin"].astype(str)
    )

    # Group by cluster
    group_col = data_cfg.get("group_col", "cluster_id")
    if group_col not in df.columns:
        group_col = "cluster_id"

    cluster_reps = df.groupby(group_col).first().reset_index()

    # StratifiedGroupKFold
    splitter = StratifiedGroupKFold(n_splits=k_folds, shuffle=True, random_state=seed)

    fold_stats = []

    for fold_idx, (train_cluster_idx, val_cluster_idx) in enumerate(
        splitter.split(cluster_reps, cluster_reps["_bin"], cluster_reps[group_col])
    ):
        train_clusters = cluster_reps.iloc[train_cluster_idx][group_col].values
        val_clusters = cluster_reps.iloc[val_cluster_idx][group_col].values

        train_mask = df[group_col].isin(train_clusters)
        val_mask = df[group_col].isin(val_clusters)

        train_df = df[train_mask]
        val_df = df[val_mask]

        # Compute combined statistics
        fold_stats.append({
            "fold": fold_idx,
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "combined_difficulty": train_df["combined_difficulty"].mean(),
            "image_difficulty": train_df["image_difficulty"].mean(),
            "text_difficulty": train_df["text_difficulty"].mean(),
            "contrast": train_df["contrast"].mean(),
            "ink_fade": train_df["ink_fade"].mean(),
            "rare_words": train_df["rare_word_ratio"].mean(),
            "names": train_df["named_entity_score"].mean(),
            "numbers": train_df["number_complexity"].mean(),
        })

    fold_stats_df = pd.DataFrame(fold_stats)

    # Print results
    print("\n" + "="*90)
    print("Combined Image + Text Difficulty Analysis")
    print("="*90)

    print(f"\n{'Fold':<6} {'Train':<8} {'Combined':<12} {'Image':<12} {'Text':<12}")
    print("-"*90)

    for _, row in fold_stats_df.iterrows():
        print(f"{int(row['fold']):<6} "
              f"{int(row['train_samples']):<8} "
              f"{row['combined_difficulty']:<12.2f} "
              f"{row['image_difficulty']:<12.2f} "
              f"{row['text_difficulty']:<12.2f}")

    # Variance analysis
    combined_std = fold_stats_df["combined_difficulty"].std()
    image_std = fold_stats_df["image_difficulty"].std()
    text_std = fold_stats_df["text_difficulty"].std()

    print(f"\nVariance Analysis:")
    print(f"  Combined difficulty std: {combined_std:.3f}")
    print(f"  Image difficulty std:    {image_std:.3f}")
    print(f"  Text difficulty std:     {text_std:.3f}")

    # Detailed breakdown
    print("\n" + "="*90)
    print("Detailed Quality Breakdown")
    print("="*90)

    print(f"\n{'Fold':<6} {'Contrast':<10} {'Ink Fade':<10} {'Names':<10} {'Numbers':<10}")
    print("-"*90)

    for _, row in fold_stats_df.iterrows():
        print(f"{int(row['fold']):<6} "
              f"{row['contrast']:<10.2f} "
              f"{row['ink_fade']:<10.2f} "
              f"{row['names']:<10.2f} "
              f"{row['numbers']:<10.2f}")

    # Correlation analysis
    print("\n" + "="*90)
    print("Correlation: Image vs Text Difficulty")
    print("="*90)

    correlation = np.corrcoef(df["image_difficulty"], df["text_difficulty"])[0, 1]
    print(f"\nCorrelation coefficient: {correlation:.3f}")

    if abs(correlation) > 0.3:
        print(f"  ⚠️  Strong correlation - difficult images tend to have difficult text")
        print(f"  This compounds the problem")
    elif abs(correlation) > 0.1:
        print(f"  Moderate correlation - slight relationship between image and text difficulty")
    else:
        print(f"  ✓ No correlation - image and text difficulty are independent")

    # Check for hardest samples (image + text both hard)
    hard_image_threshold = df["image_difficulty"].quantile(0.75)
    hard_text_threshold = df["text_difficulty"].quantile(0.75)

    df["doubly_hard"] = (
        (df["image_difficulty"] > hard_image_threshold) &
        (df["text_difficulty"] > hard_text_threshold)
    )

    print(f"\n" + "="*90)
    print("Doubly Hard Samples (Hard Image + Hard Text)")
    print("="*90)

    print(f"\nSamples where BOTH image and text are hard (top 25%):")
    print(f"  Total: {df['doubly_hard'].sum()} / {len(df)} ({df['doubly_hard'].sum()/len(df)*100:.1f}%)")

    # Check if doubly hard samples are evenly distributed
    for fold_idx in range(k_folds):
        train_clusters = cluster_reps.iloc[
            list(splitter.split(cluster_reps, cluster_reps["_bin"], cluster_reps[group_col]))[fold_idx][0]
        ][group_col].values

        train_mask = df[group_col].isin(train_clusters)
        fold_df = df[train_mask]

        doubly_hard_count = fold_df["doubly_hard"].sum()
        doubly_hard_ratio = doubly_hard_count / len(fold_df)

        print(f"  Fold {fold_idx}: {doubly_hard_count} / {len(fold_df)} ({doubly_hard_ratio*100:.1f}%)")

    # Final diagnosis
    print("\n" + "="*90)
    print("Final Diagnosis")
    print("="*90)

    if combined_std > 0.5:
        print("\n🔴 Combined difficulty variance detected!")
        print(f"  Combined std = {combined_std:.3f}")

        easiest_fold = fold_stats_df.loc[fold_stats_df["combined_difficulty"].idxmin()]
        hardest_fold = fold_stats_df.loc[fold_stats_df["combined_difficulty"].idxmax()]

        print(f"\n  🟢 Easiest fold: Fold {int(easiest_fold['fold'])}")
        print(f"     Combined difficulty: {easiest_fold['combined_difficulty']:.2f}")
        print(f"     Image: {easiest_fold['image_difficulty']:.2f} | Text: {easiest_fold['text_difficulty']:.2f}")

        print(f"\n  🔴 Hardest fold: Fold {int(hardest_fold['fold'])}")
        print(f"     Combined difficulty: {hardest_fold['combined_difficulty']:.2f}")
        print(f"     Image: {hardest_fold['image_difficulty']:.2f} | Text: {hardest_fold['text_difficulty']:.2f}")

        difficulty_gap = hardest_fold["combined_difficulty"] - easiest_fold["combined_difficulty"]
        print(f"\n  Gap: {difficulty_gap:.2f} points")

        if difficulty_gap > 1.0:
            print(f"  ⚠️  Fold {int(easiest_fold['fold'])} should significantly outperform Fold {int(hardest_fold['fold'])}")

    else:
        print("\n🟢 No significant combined difficulty variance")
        print(f"  Combined std = {combined_std:.3f}")
        print("\n  Fold performance variance is NOT explained by:")
        print("  ❌ Image quality differences")
        print("  ❌ Text complexity differences")
        print("  ❌ Combined image + text difficulty")
        print("\n  Must be caused by:")
        print("  ✅ Random variation in model optimization")
        print("  ✅ Hyperparameter sensitivity to initial conditions")
        print("  ✅ Subtle dataset artifacts not captured by metrics")
        print("  ✅ Annotation quality differences (not measurable without gold labels)")

    # Ranking
    print("\n" + "="*90)
    print("Performance Prediction (Easiest → Hardest)")
    print("="*90)

    fold_stats_df = fold_stats_df.sort_values("combined_difficulty")

    for rank, (_, row) in enumerate(fold_stats_df.iterrows(), 1):
        marker = "🟢" if rank == 1 else ("🟡" if rank == 2 else "🔴")
        print(f"{marker} Rank {rank}: Fold {int(row['fold'])} - Combined {row['combined_difficulty']:.2f} "
              f"(Image: {row['image_difficulty']:.2f}, Text: {row['text_difficulty']:.2f})")

    print(f"\n💡 Prediction: Fold {int(fold_stats_df.iloc[0]['fold'])} should achieve best leaderboard score")


if __name__ == "__main__":
    main()
