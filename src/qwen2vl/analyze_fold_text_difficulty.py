"""
Fold Text Difficulty Analysis

Checks if k-fold performance variance is explained by text complexity imbalance.
Requires: dataset/text_difficulty.csv (run analyze_text_difficulty.py first)

Usage:
    python analyze_fold_text_difficulty.py --config config_qwen3_8b_full.yaml

This will:
1. Load document clusters and text difficulty scores
2. Simulate k-fold splits (same as training)
3. Compute average text difficulty per fold
4. Check if easier text folds correlate with better performance
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
    parser = argparse.ArgumentParser(description="Analyze fold text difficulty")
    parser.add_argument(
        "--config",
        type=str,
        default="config_qwen3_8b_full.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--text_csv",
        type=str,
        default=None,
        help="Path to text difficulty CSV (default: dataset/text_difficulty.csv)"
    )
    args = parser.parse_args()

    # Load config
    config_path = SCRIPT_DIR / args.config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]

    # Load text difficulty scores
    if args.text_csv:
        text_path = Path(args.text_csv)
    else:
        text_path = REPO_ROOT / "dataset" / "text_difficulty.csv"

    if not text_path.exists():
        print(f"❌ Text difficulty CSV not found: {text_path}")
        print(f"\nRun this first:")
        print(f"  python analyze_text_difficulty.py --config {args.config}")
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
            print(f"⚠️  Cluster CSV not found: {cluster_path}")
            df["cluster_id"] = 0
    else:
        print("⚠️  No cluster_csv in config, using dummy groups")
        df["cluster_id"] = 0

    # Merge text difficulty
    df = df.merge(text_df, on="ID", how="left")

    # Simulate k-fold splits
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

    # Group by cluster
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

        # Compute text difficulty statistics
        train_difficulty = train_df["difficulty_score"].mean()
        val_difficulty = val_df["difficulty_score"].mean()

        train_rare_words = train_df["rare_word_ratio"].mean()
        val_rare_words = val_df["rare_word_ratio"].mean()

        train_names = train_df["named_entity_score"].mean()
        val_names = val_df["named_entity_score"].mean()

        train_numbers = train_df["number_complexity"].mean()
        val_numbers = val_df["number_complexity"].mean()

        train_special = train_df["special_char_complexity"].mean()
        val_special = val_df["special_char_complexity"].mean()

        fold_stats.append({
            "fold": fold_idx,
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "train_clusters": len(train_clusters),
            "val_clusters": len(val_clusters),
            "train_difficulty": train_difficulty,
            "val_difficulty": val_difficulty,
            "train_rare_words": train_rare_words,
            "val_rare_words": val_rare_words,
            "train_names": train_names,
            "val_names": val_names,
            "train_numbers": train_numbers,
            "val_numbers": val_numbers,
            "train_special": train_special,
            "val_special": val_special,
        })

    fold_stats_df = pd.DataFrame(fold_stats)

    # Print results
    print("\n" + "="*90)
    print("Fold Text Difficulty Analysis")
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

    print(f"\nTraining Set Text Difficulty:")
    print(f"  Mean: {fold_stats_df['train_difficulty'].mean():.2f}")
    print(f"  Std:  {train_diff_std:.2f} ({'HIGH' if train_diff_std > 1.0 else 'moderate'})")
    print(f"  Range: {fold_stats_df['train_difficulty'].min():.2f} - {fold_stats_df['train_difficulty'].max():.2f}")

    print(f"\nValidation Set Text Difficulty:")
    print(f"  Mean: {fold_stats_df['val_difficulty'].mean():.2f}")
    print(f"  Std:  {val_diff_std:.2f} ({'HIGH' if val_diff_std > 1.0 else 'moderate'})")
    print(f"  Range: {fold_stats_df['val_difficulty'].min():.2f} - {fold_stats_df['val_difficulty'].max():.2f}")

    # Detailed breakdown
    print("\n" + "="*90)
    print("Text Complexity Breakdown")
    print("="*90)

    print(f"\n{'Fold':<6} {'Rare Words':<12} {'Names':<12} {'Numbers':<12} {'Special Chars':<15}")
    print("-"*90)

    for _, row in fold_stats_df.iterrows():
        print(f"{int(row['fold']):<6} "
              f"{row['train_rare_words']:<12.2f} "
              f"{row['train_names']:<12.2f} "
              f"{row['train_numbers']:<12.2f} "
              f"{row['train_special']:<15.2f}")

    # Diagnosis
    print("\n" + "="*90)
    print("Diagnosis")
    print("="*90)

    if train_diff_std > 1.0:
        print("\n🔴 HIGH training set text difficulty variance detected!")
        print(f"  Training text difficulty std = {train_diff_std:.2f}")
        print("\n  This likely explains fold performance variance:")

        # Identify easiest and hardest folds
        easiest_fold = fold_stats_df.loc[fold_stats_df["train_difficulty"].idxmin()]
        hardest_fold = fold_stats_df.loc[fold_stats_df["train_difficulty"].idxmax()]

        print(f"\n  🟢 Easiest fold: Fold {int(easiest_fold['fold'])}")
        print(f"     Text difficulty: {easiest_fold['train_difficulty']:.2f}")
        print(f"     Rare words: {easiest_fold['train_rare_words']:.2f}")
        print(f"     Names: {easiest_fold['train_names']:.2f}")
        print(f"     Numbers: {easiest_fold['train_numbers']:.2f}")
        print(f"     → More boilerplate, fewer names/numbers = easier to learn")

        print(f"\n  🔴 Hardest fold: Fold {int(hardest_fold['fold'])}")
        print(f"     Text difficulty: {hardest_fold['train_difficulty']:.2f}")
        print(f"     Rare words: {hardest_fold['train_rare_words']:.2f}")
        print(f"     Names: {hardest_fold['train_names']:.2f}")
        print(f"     Numbers: {hardest_fold['train_numbers']:.2f}")
        print(f"     → More names/rare words/numbers = harder to learn")

        difficulty_gap = hardest_fold["train_difficulty"] - easiest_fold["train_difficulty"]
        print(f"\n  Text difficulty gap: {difficulty_gap:.2f} points")

        if difficulty_gap > 2.0:
            print(f"  ⚠️  Large gap suggests Fold {int(easiest_fold['fold'])} will significantly outperform Fold {int(hardest_fold['fold'])}")
            print(f"\n  Expected performance pattern:")
            print(f"    Fold {int(easiest_fold['fold'])}: Best performance (easiest training data)")
            print(f"    Fold {int(hardest_fold['fold'])}: Worst performance (hardest training data)")

        # Check if this matches observed results
        print(f"\n  Compare to your actual results:")
        print(f"    If Fold {int(easiest_fold['fold'])} achieved 0.909 and other folds ~0.87-0.88,")
        print(f"    this confirms text complexity is the root cause of variance!")

    elif train_diff_std > 0.5:
        print("\n🟡 Moderate training set text difficulty variance")
        print(f"  Training text difficulty std = {train_diff_std:.2f}")
        print("\n  This contributes to fold performance differences")

    else:
        print("\n🟢 Low training set text difficulty variance")
        print(f"  Training text difficulty std = {train_diff_std:.2f}")
        print("\n  Fold variance is NOT explained by text complexity imbalance")

    # Identify which fold matches observed best performance
    print("\n" + "="*90)
    print("Prediction: Which Fold Will Perform Best?")
    print("="*90)

    fold_stats_df = fold_stats_df.sort_values("train_difficulty")

    print(f"\nRanked by training text difficulty (easiest to hardest):")
    for rank, (_, row) in enumerate(fold_stats_df.iterrows(), 1):
        marker = "🟢" if rank == 1 else ("🟡" if rank == 2 else "🔴")
        print(f"{marker} Rank {rank}: Fold {int(row['fold'])} - Difficulty {row['train_difficulty']:.2f}")

    print(f"\n💡 Hypothesis: Fold {int(fold_stats_df.iloc[0]['fold'])} should have the BEST leaderboard score")
    print(f"   (It got the easiest training text)")


if __name__ == "__main__":
    main()
