"""
Analyze individual fold performance to understand ensemble behavior.

Usage:
    python analyze_folds.py --config config_qwen3_8b.yaml
"""

import json
import argparse
from pathlib import Path
import pandas as pd
import yaml
from collections import Counter

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_fold_predictions(output_dir: Path, fold_num: int) -> dict:
    """Load cached predictions for a fold."""
    cache_file = output_dir / "inference_cache" / f"fold_{fold_num}_predictions.json"

    if not cache_file.exists():
        raise FileNotFoundError(f"Fold {fold_num} cache not found: {cache_file}")

    with open(cache_file, 'r') as f:
        return json.load(f)


def analyze_fold_agreement(all_fold_preds: list[dict], test_ids: list[str]) -> pd.DataFrame:
    """Analyze agreement between folds."""

    results = []

    for img_id in test_ids:
        predictions = [fold_preds.get(img_id, "") for fold_preds in all_fold_preds]

        # Count unique predictions
        unique_preds = list(set(predictions))
        num_unique = len(unique_preds)

        # Full agreement?
        full_agreement = num_unique == 1

        # Most common prediction
        pred_counter = Counter(predictions)
        most_common_pred, count = pred_counter.most_common(1)[0]

        results.append({
            "ID": img_id,
            "num_unique": num_unique,
            "agreement": count / len(predictions),  # Fraction agreeing on most common
            "full_agreement": full_agreement,
            "most_common": most_common_pred,
            "all_predictions": predictions,
        })

    return pd.DataFrame(results)


def compare_ensemble_strategies(all_fold_preds: list[dict], test_ids: list[str]) -> dict:
    """Compare different ensemble strategies."""

    strategies = {}

    for img_id in test_ids:
        predictions = [fold_preds.get(img_id, "") for fold_preds in all_fold_preds]

        # Strategy 1: Character-level voting (current)
        char_voted = ensemble_char_voting(predictions)

        # Strategy 2: Simple majority voting (pick most common full prediction)
        majority_voted = Counter(predictions).most_common(1)[0][0]

        # Strategy 3: First fold (baseline)
        first_fold = predictions[0]

        # Strategy 4: Longest prediction
        longest = max(predictions, key=len)

        # Strategy 5: Shortest prediction
        shortest = min(predictions, key=len)

        strategies[img_id] = {
            "char_voting": char_voted,
            "majority_voting": majority_voted,
            "first_fold": first_fold,
            "longest": longest,
            "shortest": shortest,
        }

    return strategies


def ensemble_char_voting(predictions: list[str]) -> str:
    """Current character-level voting (from inference.py)."""
    if len(predictions) == 1:
        return predictions[0]

    max_len = max(len(p) for p in predictions)
    padded = [p + ' ' * (max_len - len(p)) for p in predictions]

    result = []
    for i in range(max_len):
        chars = [p[i] for p in padded]
        most_common = Counter(chars).most_common(1)[0][0]
        result.append(most_common)

    return ''.join(result).rstrip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Config file")
    args = parser.parse_args()

    # Load config
    config_path = REPO_ROOT / "src" / "qwen2vl" / args.config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    output_dir = REPO_ROOT / cfg["training"]["output_dir"]
    test_csv = REPO_ROOT / cfg["data"]["test_csv"]
    k_folds = cfg["data"].get("k_folds", 5)

    print(f"\n{'='*70}")
    print(f"FOLD ANALYSIS")
    print(f"{'='*70}\n")

    # Load all fold predictions
    print("Loading fold predictions...")
    all_fold_preds = []
    for fold_num in range(1, k_folds + 1):
        try:
            fold_preds = load_fold_predictions(output_dir, fold_num)
            all_fold_preds.append(fold_preds)
            print(f"  ✓ Fold {fold_num}: {len(fold_preds)} predictions")
        except FileNotFoundError as e:
            print(f"  ✗ Fold {fold_num}: Not found")

    if len(all_fold_preds) == 0:
        print("\nError: No fold predictions found!")
        return

    # Load test IDs
    test_df = pd.read_csv(test_csv)
    test_ids = test_df["ID"].astype(str).str.strip().tolist()

    print(f"\nAnalyzing {len(test_ids)} test samples across {len(all_fold_preds)} folds...\n")

    # Analyze agreement
    agreement_df = analyze_fold_agreement(all_fold_preds, test_ids)

    print(f"{'='*70}")
    print("AGREEMENT ANALYSIS")
    print(f"{'='*70}")
    print(f"Full agreement (all folds identical): {agreement_df['full_agreement'].sum()} / {len(test_ids)} ({agreement_df['full_agreement'].mean()*100:.1f}%)")
    print(f"\nAgreement distribution:")
    print(f"  Mean agreement: {agreement_df['agreement'].mean()*100:.1f}%")
    print(f"  Median agreement: {agreement_df['agreement'].median()*100:.1f}%")
    print(f"  Min agreement: {agreement_df['agreement'].min()*100:.1f}%")

    print(f"\nUnique predictions per image:")
    print(agreement_df['num_unique'].value_counts().sort_index())

    # Show examples of disagreement
    print(f"\n{'='*70}")
    print("EXAMPLES OF FOLD DISAGREEMENT")
    print(f"{'='*70}")

    disagreements = agreement_df[agreement_df['num_unique'] > 1].head(5)
    for _, row in disagreements.iterrows():
        print(f"\nImage: {row['ID']}")
        print(f"Agreement: {row['agreement']*100:.0f}% ({row['num_unique']} unique predictions)")
        for i, pred in enumerate(row['all_predictions'], 1):
            print(f"  Fold {i}: \"{pred[:80]}{'...' if len(pred) > 80 else ''}\"")

    # Compare ensemble strategies
    print(f"\n{'='*70}")
    print("ENSEMBLE STRATEGY COMPARISON")
    print(f"{'='*70}")

    strategies = compare_ensemble_strategies(all_fold_preds, test_ids)

    # Show examples where strategies differ
    print("\nExamples where strategies produce different results:")
    count = 0
    for img_id, preds in strategies.items():
        if len(set(preds.values())) > 1 and count < 3:
            print(f"\nImage: {img_id}")
            for strategy, pred in preds.items():
                print(f"  {strategy:20s}: \"{pred[:60]}{'...' if len(pred) > 60 else ''}\"")
            count += 1

    # Generate submission CSVs for each strategy
    print(f"\n{'='*70}")
    print("GENERATING STRATEGY SUBMISSIONS")
    print(f"{'='*70}")

    for strategy_name in ["char_voting", "majority_voting", "first_fold", "longest", "shortest"]:
        results = []
        for img_id in test_ids:
            results.append({
                "ID": img_id,
                "Target": strategies[img_id][strategy_name]
            })

        output_file = REPO_ROOT / f"submission_{strategy_name}.csv"
        pd.DataFrame(results).to_csv(output_file, index=False)
        print(f"  ✓ {strategy_name:20s} → {output_file.name}")

    print(f"\n{'='*70}")
    print("RECOMMENDATION")
    print(f"{'='*70}")
    print("\nTest all strategies on the platform:")
    print("  1. submission_first_fold.csv       (your current best: 0.855)")
    print("  2. submission_majority_voting.csv  (most common full prediction)")
    print("  3. submission_char_voting.csv      (current ensemble: 0.850)")
    print("  4. submission_longest.csv          (pick longest prediction)")
    print("  5. submission_shortest.csv         (pick shortest prediction)")
    print("\nIf first_fold is still best, that fold just got lucky with the split.")
    print("Consider training more folds and selecting the best 2-3 for ensemble.\n")


if __name__ == "__main__":
    main()
