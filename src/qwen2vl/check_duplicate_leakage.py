"""
Check if duplicate texts are leaking across train/val split.
"""
import pandas as pd
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).parent.parent.parent

def check_leakage():
    train_csv = REPO_ROOT / "dataset/Train.csv"
    df = pd.read_csv(train_csv)

    # Clean text
    df['text_clean'] = df['Target'].astype(str).str.lower().str.strip()

    # Find duplicates
    text_counts = df['text_clean'].value_counts()
    duplicate_texts = text_counts[text_counts > 1].index.tolist()

    print(f"Found {len(duplicate_texts)} unique texts that appear multiple times")
    print(f"Total duplicate samples: {text_counts[text_counts > 1].sum()}")

    # Simulate stratified split (same as training code)
    from sklearn.model_selection import train_test_split

    # Compute features (same as make_splits)
    def compute_digit_density(text):
        if not text or pd.isna(text):
            return 0.0
        text = str(text)
        digits = sum(1 for c in text if c.isdigit())
        return digits / max(len(text), 1)

    def compute_uppercase_ratio(text):
        if not text or pd.isna(text):
            return 0.0
        text = str(text)
        letters = [c for c in text if c.isalpha()]
        if not letters:
            return 0.0
        uppercase = sum(1 for c in letters if c.isupper())
        return uppercase / len(letters)

    def compute_lexical_diversity(text):
        if not text or pd.isna(text):
            return 0.0
        words = str(text).lower().split()
        if not words:
            return 0.0
        return len(set(words)) / len(words)

    df['_digit_density'] = df['Target'].apply(compute_digit_density)
    df['_uppercase_ratio'] = df['Target'].apply(compute_uppercase_ratio)
    df['_lexical_diversity'] = df['Target'].apply(compute_lexical_diversity)

    # Bin features
    try:
        df['_digit_bin'] = pd.qcut(df['_digit_density'], q=2, labels=["no_nums", "has_nums"], duplicates="drop")
    except:
        df['_digit_bin'] = "has_nums"

    try:
        df['_upper_bin'] = pd.qcut(df['_uppercase_ratio'], q=2, labels=["informal", "formal"], duplicates="drop")
    except:
        df['_upper_bin'] = "informal"

    try:
        df['_lex_bin'] = pd.qcut(df['_lexical_diversity'], q=3, labels=["repetitive", "moderate", "diverse"], duplicates="drop")
    except:
        df['_lex_bin'] = "moderate"

    df['_bin'] = (df['_digit_bin'].astype(str) + "_" +
                  df['_upper_bin'].astype(str) + "_" +
                  df['_lex_bin'].astype(str))

    # Split
    train_df, val_df = train_test_split(df, test_size=0.15, stratify=df['_bin'], random_state=42)

    print(f"\nTrain: {len(train_df)} samples")
    print(f"Val:   {len(val_df)} samples")

    # Check for leakage
    train_texts = set(train_df['text_clean'])
    val_texts = set(val_df['text_clean'])
    leaked_texts = train_texts & val_texts

    print(f"\n{'='*70}")
    print("LEAKAGE ANALYSIS:")
    print(f"{'='*70}")

    if leaked_texts:
        print(f"⚠️  LEAKAGE DETECTED: {len(leaked_texts)} texts appear in BOTH train and val")
        print(f"\nExamples of leaked texts:")
        for text in list(leaked_texts)[:5]:
            train_count = (train_df['text_clean'] == text).sum()
            val_count = (val_df['text_clean'] == text).sum()
            print(f"  '{text[:60]}...'")
            print(f"    Train: {train_count} copies, Val: {val_count} copies")

        print(f"\n⚠️  RECOMMENDATION: Remove duplicates to prevent leakage")
    else:
        print(f"✓ NO LEAKAGE: All duplicate texts stayed within same split")
        print(f"\n✓ Stratified splitting successfully kept duplicates together")
        print(f"✓ RECOMMENDATION: Keep duplicates (no leakage risk, more training data)")

if __name__ == "__main__":
    check_leakage()
