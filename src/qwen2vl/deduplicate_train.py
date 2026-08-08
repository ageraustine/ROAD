"""
Remove exact duplicate texts from Train.csv to prevent data leakage.
Keep the first occurrence of each unique text.
"""
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent

def deduplicate_train():
    train_csv = REPO_ROOT / "dataset/Train.csv"
    output_csv = REPO_ROOT / "dataset/Train_deduped.csv"

    df = pd.read_csv(train_csv)
    print(f"Original dataset: {len(df)} samples")

    # Clean text for comparison
    df['text_clean'] = df['Target'].astype(str).str.lower().str.strip()

    # Find duplicates
    duplicates = df[df.duplicated(subset=['text_clean'], keep='first')]
    print(f"Found {len(duplicates)} duplicate samples")

    if len(duplicates) > 0:
        print("\nDuplicate examples:")
        for text in duplicates['text_clean'].head(5):
            count = (df['text_clean'] == text).sum()
            print(f"  '{text[:60]}...' appears {count} times")

    # Remove duplicates (keep first occurrence)
    df_deduped = df.drop_duplicates(subset=['text_clean'], keep='first')

    # Drop helper column
    df_deduped = df_deduped.drop(columns=['text_clean'])

    print(f"\nDeduplicated dataset: {len(df_deduped)} samples")
    print(f"Removed: {len(df) - len(df_deduped)} duplicates")

    # Save
    df_deduped.to_csv(output_csv, index=False)
    print(f"\nSaved to: {output_csv}")

    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Update config to use 'Train_deduped.csv' instead of 'Train.csv'")
    print("2. Retrain model")
    print("3. This should give more honest eval_score (closer to leaderboard)")

if __name__ == "__main__":
    deduplicate_train()
