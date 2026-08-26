"""
Check if fold variance is explained by document physical condition.
"""

import argparse
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_qwen3_8b_full.yaml")
    args = parser.parse_args()

    # Load config
    with open(SCRIPT_DIR / args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]

    # Load data
    train_csv = REPO_ROOT / data_cfg["train_csv"]
    df = pd.read_csv(train_csv)

    # Load document condition
    condition_path = REPO_ROOT / "dataset" / "document_condition.csv"
    if not condition_path.exists():
        print(f"❌ Document condition CSV not found: {condition_path}")
        return

    condition_df = pd.read_csv(condition_path)
    condition_df = condition_df[condition_df["success"] == True]

    # Merge
    df = df.merge(condition_df[["ID", "condition_score", "paper_color_variance", "burnt_damage", "tears_and_holes"]],
                  on="ID", how="left")

    # Fill missing
    median_cond = df["condition_score"].median()
    df.loc[df["condition_score"].isna(), "condition_score"] = median_cond

    print(f"Loaded document condition for {len(df)} samples")
    print(f"Condition range: {df['condition_score'].min():.1f} - {df['condition_score'].max():.1f}")

    # Simulate single split (same as current train.py)
    val_split = data_cfg.get("val_split", 0.1)
    seed = data_cfg.get("seed", 42)

    # Create stratification bins (simplified - just for testing)
    df["_has_digit"] = df["Target"].str.contains(r"\d", regex=True, na=False)
    df["_has_upper"] = df["Target"].str.contains(r"[A-Z]", regex=True, na=False)
    df["_bin"] = df["_has_digit"].astype(str) + "_" + df["_has_upper"].astype(str)

    # Split
    train_df, val_df = train_test_split(
        df,
        test_size=val_split,
        stratify=df["_bin"],
        random_state=seed
    )

    print("\n" + "="*70)
    print("Document Condition: Train vs Val")
    print("="*70)

    print(f"\nTrain: {len(train_df)} samples")
    print(f"  Condition mean: {train_df['condition_score'].mean():.2f}")
    print(f"  Paper color variance: {train_df['paper_color_variance'].mean():.1f}")
    print(f"  Burnt damage: {train_df['burnt_damage'].mean():.1f}")
    print(f"  Tears/holes: {train_df['tears_and_holes'].mean():.1f}")

    print(f"\nVal: {len(val_df)} samples")
    print(f"  Condition mean: {val_df['condition_score'].mean():.2f}")
    print(f"  Paper color variance: {val_df['paper_color_variance'].mean():.1f}")
    print(f"  Burnt damage: {val_df['burnt_damage'].mean():.1f}")
    print(f"  Tears/holes: {val_df['tears_and_holes'].mean():.1f}")

    # Check variance
    cond_gap = abs(train_df['condition_score'].mean() - val_df['condition_score'].mean())

    print(f"\n" + "="*70)
    print("Assessment")
    print("="*70)

    print(f"\nCondition gap: {cond_gap:.2f}")

    if cond_gap < 0.5:
        print("  ✅ EXCELLENT - Document condition is balanced")
    elif cond_gap < 1.0:
        print("  🟢 GOOD - Minor condition difference")
    else:
        print("  ⚠️  Document condition imbalance detected")

    # Check std
    cond_std = df['condition_score'].std()
    print(f"\nDataset condition variance: {cond_std:.2f}")

    print(f"\nComparison with other metrics:")
    print(f"  Image quality std:       0.024 (negligible)")
    print(f"  Text difficulty std:     0.109 (low)")
    print(f"  Document condition std:  ~0.0X (TBD based on split)")


if __name__ == "__main__":
    main()
