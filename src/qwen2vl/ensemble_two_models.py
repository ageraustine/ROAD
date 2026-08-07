"""
Ensemble two independently trained full-dataset models.

Usage:
    python ensemble_two_models.py --config1 config_qwen3_8b_full.yaml --config2 config_qwen3_8b_full_v2.yaml
"""

import argparse
import json
from pathlib import Path
from collections import Counter

import yaml
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent


def ensemble_predictions(pred1: str, pred2: str, strategy: str = "majority") -> str:
    """
    Ensemble two predictions.

    Args:
        pred1: Prediction from model 1
        pred2: Prediction from model 2
        strategy: Ensemble strategy
            - "majority": Pick most common (if tied, pick first)
            - "shortest": Pick shortest prediction
            - "longest": Pick longest prediction
            - "char_voting": Character-level voting

    Returns:
        Ensembled prediction
    """
    if pred1 == pred2:
        return pred1

    if strategy == "majority":
        # With 2 models, just pick first (they're equally weighted)
        return pred1

    elif strategy == "shortest":
        return min([pred1, pred2], key=len)

    elif strategy == "longest":
        return max([pred1, pred2], key=len)

    elif strategy == "char_voting":
        # Pad to same length
        max_len = max(len(pred1), len(pred2))
        p1 = pred1 + ' ' * (max_len - len(pred1))
        p2 = pred2 + ' ' * (max_len - len(pred2))

        # Vote for each position
        result = []
        for i in range(max_len):
            chars = [p1[i], p2[i]]
            most_common = Counter(chars).most_common(1)[0][0]
            result.append(most_common)

        return ''.join(result).rstrip()

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def main():
    parser = argparse.ArgumentParser(
        description="Ensemble two full-dataset models"
    )
    parser.add_argument("--config1", default="config_qwen3_8b_full.yaml",
                        help="Config for model 1")
    parser.add_argument("--config2", default="config_qwen3_8b_full_v2.yaml",
                        help="Config for model 2")
    parser.add_argument("--strategy", default="shortest",
                        choices=["majority", "shortest", "longest", "char_voting"],
                        help="Ensemble strategy")
    parser.add_argument("--output", default="submission_ensemble_2models.csv",
                        help="Output CSV filename")
    args = parser.parse_args()

    # Load configs
    config1_path = SCRIPT_DIR / args.config1
    config2_path = SCRIPT_DIR / args.config2

    with open(config1_path) as f:
        cfg1 = yaml.safe_load(f)
    with open(config2_path) as f:
        cfg2 = yaml.safe_load(f)

    # Get prediction files (must run inference first)
    pred1_csv = REPO_ROOT / cfg1["inference"]["output_csv"]
    pred2_csv = REPO_ROOT / cfg2["inference"]["output_csv"]

    if not pred1_csv.exists():
        print(f"Error: {pred1_csv} not found!")
        print(f"Run: python inference.py --config {args.config1}")
        return

    if not pred2_csv.exists():
        print(f"Error: {pred2_csv} not found!")
        print(f"Run: python inference.py --config {args.config2}")
        return

    # Load predictions
    print(f"Loading predictions from:")
    print(f"  Model 1: {pred1_csv}")
    print(f"  Model 2: {pred2_csv}")

    df1 = pd.read_csv(pred1_csv)
    df2 = pd.read_csv(pred2_csv)

    # Ensure same IDs
    assert len(df1) == len(df2), "Prediction files have different lengths!"
    assert all(df1["ID"] == df2["ID"]), "Prediction files have different IDs!"

    # Ensemble
    print(f"\nEnsembling {len(df1)} predictions (strategy={args.strategy})...")

    results = []
    for _, (row1, row2) in enumerate(zip(df1.itertuples(), df2.itertuples())):
        img_id = row1.ID
        pred1 = row1.Target
        pred2 = row2.Target

        ensembled = ensemble_predictions(pred1, pred2, strategy=args.strategy)
        results.append({"ID": img_id, "Target": ensembled})

    # Analyze agreement
    agreements = sum(1 for r1, r2 in zip(df1.itertuples(), df2.itertuples())
                     if r1.Target == r2.Target)
    agreement_pct = 100 * agreements / len(df1)

    print(f"\nModel agreement: {agreements}/{len(df1)} ({agreement_pct:.1f}%)")

    # Save
    output_csv = REPO_ROOT / args.output
    out_df = pd.DataFrame(results)
    out_df.to_csv(output_csv, index=False)

    print(f"✅ Saved ensemble to {output_csv}")

    # Show examples where models disagree
    print(f"\n{'='*70}")
    print("Examples of disagreement:")
    print(f"{'='*70}")

    disagreements = [(r1, r2) for r1, r2 in zip(df1.itertuples(), df2.itertuples())
                     if r1.Target != r2.Target]

    for i, (r1, r2) in enumerate(disagreements[:5]):
        print(f"\nImage: {r1.ID}")
        print(f"  Model 1: \"{r1.Target[:80]}{'...' if len(r1.Target) > 80 else ''}\"")
        print(f"  Model 2: \"{r2.Target[:80]}{'...' if len(r2.Target) > 80 else ''}\"")

        ensembled = ensemble_predictions(r1.Target, r2.Target, strategy=args.strategy)
        print(f"  Ensemble: \"{ensembled[:80]}{'...' if len(ensembled) > 80 else ''}\"")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
