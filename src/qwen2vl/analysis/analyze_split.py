"""
Analyze train/val split quality for small dataset.

Usage:
    python analyze_split.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).parent.parent.parent
TRAIN_CSV = REPO_ROOT / "dataset" / "Train.csv"

def analyze_split():
    """Analyze and visualize train/val split distribution."""

    df = pd.read_csv(TRAIN_CSV)

    print("=" * 70)
    print("DATASET SPLIT ANALYSIS")
    print("=" * 70)

    # Calculate text statistics
    df['text_length'] = df['Target'].str.len()
    df['word_count'] = df['Target'].str.split().str.len()
    df['char_diversity'] = df['Target'].apply(lambda x: len(set(x)) / len(x) if len(x) > 0 else 0)

    # Create length bins for stratification
    df['length_bin'] = pd.qcut(df['text_length'], q=5, labels=False, duplicates='drop')

    # Random split vs Stratified split
    print("\n[1] RANDOM SPLIT (Current Baseline)")
    print("-" * 70)
    train_random, val_random = train_test_split(df, test_size=0.1, random_state=42)

    print(f"Train samples: {len(train_random)}")
    print(f"Val samples: {len(val_random)}")
    print(f"\nText length distribution:")
    print(f"  Train - Mean: {train_random['text_length'].mean():.1f}, Std: {train_random['text_length'].std():.1f}")
    print(f"  Val   - Mean: {val_random['text_length'].mean():.1f}, Std: {val_random['text_length'].std():.1f}")
    print(f"  Difference: {abs(train_random['text_length'].mean() - val_random['text_length'].mean()):.1f} chars")

    print("\n[2] STRATIFIED SPLIT (By Text Length - Recommended)")
    print("-" * 70)
    train_strat, val_strat = train_test_split(
        df,
        test_size=0.1,
        stratify=df['length_bin'],
        random_state=42
    )

    print(f"Train samples: {len(train_strat)}")
    print(f"Val samples: {len(val_strat)}")
    print(f"\nText length distribution:")
    print(f"  Train - Mean: {train_strat['text_length'].mean():.1f}, Std: {train_strat['text_length'].std():.1f}")
    print(f"  Val   - Mean: {val_strat['text_length'].mean():.1f}, Std: {val_strat['text_length'].std():.1f}")
    print(f"  Difference: {abs(train_strat['text_length'].mean() - val_strat['text_length'].mean()):.1f} chars")

    # Detailed statistics
    print("\n[3] DETAILED STATISTICS")
    print("-" * 70)
    print(f"\nFull Dataset:")
    print(f"  Total samples: {len(df)}")
    print(f"  Text length - Min: {df['text_length'].min()}, Max: {df['text_length'].max()}, Median: {df['text_length'].median():.1f}")
    print(f"  Word count - Min: {df['word_count'].min()}, Max: {df['word_count'].max()}, Median: {df['word_count'].median():.1f}")

    print(f"\nLength distribution by bin (for stratification):")
    for i in range(5):
        bin_data = df[df['length_bin'] == i]
        print(f"  Bin {i}: {len(bin_data)} samples, Length range: {bin_data['text_length'].min()}-{bin_data['text_length'].max()}")

    # Comparison
    print("\n[4] SPLIT COMPARISON")
    print("-" * 70)
    print(f"Mean length difference (lower is better):")
    print(f"  Random split: {abs(train_random['text_length'].mean() - val_random['text_length'].mean()):.2f} chars")
    print(f"  Stratified split: {abs(train_strat['text_length'].mean() - val_strat['text_length'].mean()):.2f} chars")

    # Check if each bin is represented in val
    print(f"\nBin representation in validation set:")
    print(f"  Random split: {val_random['length_bin'].nunique()}/5 bins")
    print(f"  Stratified split: {val_strat['length_bin'].nunique()}/5 bins")

    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    print("✓ Use stratified split by text length (already implemented in train.py)")
    print("  Benefits:")
    print("  - Ensures val set has similar length distribution to train")
    print("  - More reliable WER/CER metrics (longer texts weighted more)")
    print("  - Reduces variance in validation scores")

    print("\nOther stratification options to consider:")
    print("  - By difficulty (character diversity, special chars)")
    print("  - K-fold cross-validation (5x training time but more reliable)")
    print("=" * 70)


if __name__ == "__main__":
    analyze_split()
