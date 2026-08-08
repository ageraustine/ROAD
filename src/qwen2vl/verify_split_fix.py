"""
Quick verification that duplicate-aware splitting prevents train/val leakage.
Uses same logic as train.py make_splits().
"""
import pandas as pd
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).parent.parent.parent

# Import feature computation functions from train.py
sys.path.insert(0, str(Path(__file__).parent))
from train import compute_digit_density, compute_uppercase_ratio, compute_lexical_diversity

def verify_split():
    train_csv = REPO_ROOT / "dataset/Train.csv"
    df = pd.read_csv(train_csv)

    print(f"Loaded {len(df)} samples")
    print()

    # STEP 1: Identify duplicates
    print("="*70)
    print("STEP 1: Identify duplicate texts")
    print("="*70)
    df["_text_clean"] = df["Target"].astype(str).str.lower().str.strip()

    text_counts = df["_text_clean"].value_counts()
    duplicate_texts = text_counts[text_counts > 1]

    print(f"Found {len(duplicate_texts)} unique texts with duplicates")
    print(f"Total duplicate samples: {duplicate_texts.sum()}")

    if len(duplicate_texts) == 0:
        print("No duplicates found - standard split will work fine")
        return

    # Assign duplicate group IDs
    df["_dup_group"] = -1
    for group_id, dup_text in enumerate(duplicate_texts.index):
        df.loc[df["_text_clean"] == dup_text, "_dup_group"] = group_id

    print()

    # STEP 2: Compute semantic features
    print("="*70)
    print("STEP 2: Compute semantic features")
    print("="*70)
    df["_digit_density"] = df["Target"].apply(compute_digit_density)
    df["_uppercase_ratio"] = df["Target"].apply(compute_uppercase_ratio)
    df["_lexical_diversity"] = df["Target"].apply(compute_lexical_diversity)

    # Bin features
    try:
        df["_digit_bin"] = pd.qcut(df["_digit_density"], q=2, labels=["no_nums", "has_nums"], duplicates="drop")
    except ValueError:
        df["_digit_bin"] = "has_nums"

    try:
        df["_upper_bin"] = pd.qcut(df["_uppercase_ratio"], q=2, labels=["informal", "formal"], duplicates="drop")
    except ValueError:
        df["_upper_bin"] = "informal"

    try:
        df["_lex_bin"] = pd.qcut(df["_lexical_diversity"], q=3, labels=["repetitive", "moderate", "diverse"], duplicates="drop")
    except ValueError:
        df["_lex_bin"] = "moderate"

    df["_bin"] = (df["_digit_bin"].astype(str) + "_" +
                  df["_upper_bin"].astype(str) + "_" +
                  df["_lex_bin"].astype(str))

    print("Features computed")
    print()

    # STEP 3: Split with duplicate awareness
    print("="*70)
    print("STEP 3: Duplicate-aware stratified split")
    print("="*70)

    df["_orig_idx"] = df.index

    # Separate duplicates from unique samples
    dup_mask = df["_dup_group"] >= 0
    dup_df = df[dup_mask].copy()
    unique_df = df[~dup_mask].copy()

    print(f"Duplicate samples: {len(dup_df)}")
    print(f"Unique samples: {len(unique_df)}")

    # Get one representative per duplicate group
    dup_representatives = dup_df.groupby("_dup_group", as_index=False).first()
    print(f"Duplicate representatives: {len(dup_representatives)}")

    # Combine for splitting
    split_df = pd.concat([unique_df, dup_representatives], ignore_index=True)
    print(f"Total for splitting: {len(split_df)}")

    # Split with stratification
    split_train, split_val = train_test_split(
        split_df, test_size=0.15, stratify=split_df["_bin"], random_state=42
    )

    print(f"Split: {len(split_train)} train, {len(split_val)} val (representatives)")

    # Propagate to all group members
    train_dup_groups = set(split_train[split_train["_dup_group"] >= 0]["_dup_group"])
    val_dup_groups = set(split_val[split_val["_dup_group"] >= 0]["_dup_group"])

    print(f"Train duplicate groups: {len(train_dup_groups)}")
    print(f"Val duplicate groups: {len(val_dup_groups)}")

    # Build final train/val
    train_mask = (df["_dup_group"] == -1) & (df["_orig_idx"].isin(split_train["_orig_idx"]))
    train_mask = train_mask | (df["_dup_group"].isin(train_dup_groups))

    val_mask = (df["_dup_group"] == -1) & (df["_orig_idx"].isin(split_val["_orig_idx"]))
    val_mask = val_mask | (df["_dup_group"].isin(val_dup_groups))

    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()

    print(f"Final: {len(train_df)} train, {len(val_df)} val (expanded)")
    print()

    # STEP 4: Verify no leakage
    print("="*70)
    print("STEP 4: Verify no leakage")
    print("="*70)

    train_texts = set(train_df["_text_clean"])
    val_texts = set(val_df["_text_clean"])
    leaked_texts = train_texts & val_texts

    if leaked_texts:
        print(f"❌ LEAKAGE DETECTED: {len(leaked_texts)} texts appear in BOTH train and val")
        print(f"\nExamples of leaked texts:")
        for text in list(leaked_texts)[:5]:
            train_count = (train_df["_text_clean"] == text).sum()
            val_count = (val_df["_text_clean"] == text).sum()
            print(f"  '{text[:60]}...'")
            print(f"    Train: {train_count} copies, Val: {val_count} copies")
        print(f"\n❌ FIX DID NOT WORK")
    else:
        print(f"✅ NO LEAKAGE: All duplicate texts kept in same split")
        print(f"\nVerifying duplicate groups:")

        # Show some examples
        examples = 0
        for group_id in sorted(train_dup_groups)[:3]:
            group_samples = train_df[train_df["_dup_group"] == group_id]
            text = group_samples.iloc[0]["_text_clean"]
            print(f"  Group {group_id}: {len(group_samples)} copies in TRAIN")
            print(f"    Text: '{text[:60]}...'")
            examples += 1

        for group_id in sorted(val_dup_groups)[:2]:
            group_samples = val_df[val_df["_dup_group"] == group_id]
            text = group_samples.iloc[0]["_text_clean"]
            print(f"  Group {group_id}: {len(group_samples)} copies in VAL")
            print(f"    Text: '{text[:60]}...'")
            examples += 1

        print(f"\n✅ FIX WORKS - All copies of each text kept together")

if __name__ == "__main__":
    verify_split()
