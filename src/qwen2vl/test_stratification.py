"""
Standalone stratification tester for the Qwen-VL HTR pipeline.

Runs the exact same data loading + train/val split logic as train.py (via
data_utils.py), but nothing else - no model, no processor, no torch, no
transformers/peft. Use this to check bin population and split quality before
committing to a multi-hour training run.

Usage:
    python test_stratification.py                       # uses config.yaml
    python test_stratification.py --config my_run.yaml
    python test_stratification.py --folds 5              # override k_folds
    python test_stratification.py --val-split 0.15       # override val_split
    python test_stratification.py --seed 123             # try a different seed

What it prints, in order:
    1. Raw bin population BEFORE any split is attempted - the direct answer
       to "will full stratification succeed, or fall back?" Any bin with
       fewer than 2 samples will force sklearn's stratified split to raise
       ValueError, which is exactly what triggers train.py's fallback ladder
       (36 bins -> 12 bins -> 4 bins). Seeing this table tells you which
       fallback will fire and why, before you've waited through the log spam
       of an actual failed attempt.
    2. The actual make_splits() output - same prints you'd see mid-training,
       including duplicate/fuzzy detection, which fallback level succeeded,
       and per-split distribution summaries.
    3. A compact train-vs-val comparison table per numeric feature, so skew
       between the two splits is visible directly instead of having to
       eyeball two separate min/median/max blocks.

Runs in well under a second on datasets in the thousands-of-rows range -
no GPU, no model weights, no image files touched.
"""

import argparse
from pathlib import Path

import yaml
import pandas as pd

import data_utils

SCRIPT_DIR = Path(__file__).parent


def print_bin_population(df: pd.DataFrame, data_cfg: dict) -> None:
    """
    Show the _bin population BEFORE any split is attempted, so a small-bin
    fallback can be diagnosed directly instead of inferred from which log
    message happened to print during an actual (possibly multi-fold) run.
    """
    annotated = data_utils.compute_stratification_bins(df, data_cfg)
    counts = annotated["_bin"].value_counts().sort_values()

    n_bins = len(counts)
    n_small = (counts < 2).sum()  # <2 samples -> breaks sklearn's stratify=

    print(f"\n{'=' * 70}")
    print(f"BIN POPULATION (pre-split diagnostic)")
    print(f"{'=' * 70}")
    print(f"  {n_bins} distinct bins, {len(annotated)} total samples")
    print(f"  Smallest bin: {counts.min()} samples | Largest bin: {counts.max()} samples")

    if n_small > 0:
        print(f"  ⚠️  {n_small}/{n_bins} bins have <2 samples - full stratification WILL fail")
        print(f"      and make_splits() will fall back to a coarser bin scheme.")
    else:
        print(f"  ✓ All bins have >=2 samples - full stratification should succeed")
        print(f"    (this checks raw row counts; duplicate-grouping in make_splits")
        print(f"    can still shrink effective bin sizes further - see below)")

    # Show the smallest handful of bins - these are exactly the ones that
    # decide whether the fallback ladder fires.
    print(f"\n  Smallest 10 bins:")
    for bin_name, count in counts.head(10).items():
        flag = " <-- breaks stratified split" if count < 2 else ""
        print(f"    {count:>4}  {bin_name}{flag}")

    print(f"{'=' * 70}")


def print_split_comparison(train_df: pd.DataFrame, val_df: pd.DataFrame,
                            data_cfg: dict, fold_label: str = "") -> None:
    """Compact train-vs-val comparison, computed fresh on the split output
    (make_splits already dropped the internal _digit_density/etc. helper
    columns, so recompute the lightweight ones for a quick sanity check)."""
    label = f" ({fold_label})" if fold_label else ""
    print(f"\n{'-' * 70}")
    print(f"TRAIN vs VAL COMPARISON{label}")
    print(f"{'-' * 70}")

    rows = []
    for name, series_fn in [
        ("digit_density", data_utils.compute_digit_density),
        ("uppercase_ratio", data_utils.compute_uppercase_ratio),
        ("lexical_diversity", data_utils.compute_lexical_diversity),
    ]:
        t = train_df["Target"].apply(series_fn)
        v = val_df["Target"].apply(series_fn)
        rows.append((name, t.median(), v.median(), t.mean(), v.mean()))

    if "condition_score" in train_df.columns:
        t = train_df["condition_score"]
        v = val_df["condition_score"]
        rows.append(("condition_score", t.median(), v.median(), t.mean(), v.mean()))

    print(f"  {'feature':<20} {'train median':>13} {'val median':>13} {'train mean':>12} {'val mean':>10}")
    for name, tm, vm, ta, va in rows:
        gap_flag = "  <-- check" if tm != 0 and abs(tm - vm) / (abs(tm) + 1e-9) > 0.25 else ""
        print(f"  {name:<20} {tm:>13.3f} {vm:>13.3f} {ta:>12.3f} {va:>10.3f}{gap_flag}")

    print(f"  {'n_samples':<20} {len(train_df):>13} {len(val_df):>13}")
    print(f"{'-' * 70}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml",
                        help="Config filename, resolved as SCRIPT_DIR/configs/<name>")
    parser.add_argument("--folds", type=int, default=None,
                        help="Override data.k_folds")
    parser.add_argument("--val-split", type=float, default=None,
                        help="Override data.val_split")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override data.seed")
    args = parser.parse_args()

    # configs/ is a sibling directory of this script (scripts/train/configs/config.yaml),
    # same convention as train.py, so both scripts resolve config paths identically.
    config_path = SCRIPT_DIR / "configs" / args.config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    if args.folds is not None:
        data_cfg["k_folds"] = args.folds
    if args.val_split is not None:
        data_cfg["val_split"] = args.val_split
    if args.seed is not None:
        data_cfg["seed"] = args.seed

    print(f"Config: {config_path}")
    print(f"  k_folds={data_cfg.get('k_folds', 1)}  val_split={data_cfg.get('val_split')}  "
          f"seed={data_cfg.get('seed', 42)}  fuzzy_duplicate_threshold={data_cfg.get('fuzzy_duplicate_threshold', 0.90)}")

    df = data_utils.load_and_prepare_dataframe(data_cfg)

    # Pre-split diagnostic - answers "will this fall back, and to what?"
    # before make_splits() actually runs (and before any fallback log spam).
    print_bin_population(df, data_cfg)

    # Actual split, using the exact same function train.py calls.
    print(f"\n{'=' * 70}")
    print(f"RUNNING make_splits()")
    print(f"{'=' * 70}")

    n_folds_seen = 0
    for train_df, val_df, fold_num in data_utils.make_splits(df, data_cfg):
        n_folds_seen += 1
        fold_label = f"fold {fold_num}" if fold_num else "single split"
        print_split_comparison(train_df, val_df, data_cfg, fold_label)

    print(f"\n{n_folds_seen} split(s) generated. No model, no images, no training was touched.")


if __name__ == "__main__":
    main()