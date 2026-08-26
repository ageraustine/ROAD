"""
Analyze entry similarity to detect:
1. Duplicates/near-duplicates
2. Template-based documents
3. Train/val similarity distribution
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

REPO_ROOT = Path(__file__).parent.parent.parent

def analyze_entry_similarity():
    train_csv = REPO_ROOT / "dataset/Train.csv"
    df = pd.read_csv(train_csv)

    print(f"Loaded {len(df)} samples")
    print()

    # Clean text
    df['text_clean'] = df['Target'].astype(str).str.lower().str.strip()

    # 1. EXACT DUPLICATES
    duplicates = df[df.duplicated(subset=['text_clean'], keep=False)]
    if len(duplicates) > 0:
        print("=" * 70)
        print("⚠️  EXACT DUPLICATES FOUND:")
        print("=" * 70)
        for text, group in duplicates.groupby('text_clean'):
            if len(group) > 1:
                print(f"Text: '{text[:60]}...'")
                print(f"  Appears {len(group)} times: IDs = {list(group['ID'][:5])}")
        print()
    else:
        print("✓ No exact duplicates found")
        print()

    # 2. NEAR-DUPLICATES (TF-IDF similarity)
    print("=" * 70)
    print("COMPUTING TF-IDF SIMILARITY (sample 1000 for speed)...")
    print("=" * 70)

    sample_df = df.sample(min(1000, len(df)), random_state=42)
    texts = sample_df['text_clean'].tolist()

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(
        max_features=500,
        ngram_range=(1, 3),  # Capture phrases
        min_df=2
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(texts)

        # Compute pairwise similarities
        similarities = cosine_similarity(tfidf_matrix)

        # Find near-duplicates (similarity > 0.8, excluding self)
        np.fill_diagonal(similarities, 0)  # Ignore self-similarity

        near_dupes = []
        for i in range(len(similarities)):
            for j in range(i + 1, len(similarities)):
                if similarities[i, j] > 0.8:
                    near_dupes.append((i, j, similarities[i, j]))

        if near_dupes:
            print(f"\n⚠️  Found {len(near_dupes)} near-duplicate pairs (similarity > 0.8):")
            for i, j, sim in near_dupes[:5]:  # Show first 5
                print(f"\nSimilarity: {sim:.3f}")
                print(f"  Text 1: {texts[i][:80]}")
                print(f"  Text 2: {texts[j][:80]}")
        else:
            print("\n✓ No near-duplicates found (threshold=0.8)")

        # 3. SIMILARITY DISTRIBUTION
        print("\n" + "=" * 70)
        print("SIMILARITY DISTRIBUTION:")
        print("=" * 70)

        # Get all pairwise similarities (upper triangle only)
        upper_tri_indices = np.triu_indices_from(similarities, k=1)
        all_sims = similarities[upper_tri_indices]

        percentiles = [10, 25, 50, 75, 90, 95, 99]
        print("\nPairwise similarity percentiles:")
        for p in percentiles:
            val = np.percentile(all_sims, p)
            print(f"  {p}th percentile: {val:.3f}")

        # Count by similarity ranges
        print("\nSimilarity distribution:")
        ranges = [
            (0.0, 0.2, "Very dissimilar"),
            (0.2, 0.4, "Dissimilar"),
            (0.4, 0.6, "Somewhat similar"),
            (0.6, 0.8, "Similar"),
            (0.8, 1.0, "Very similar"),
        ]
        for low, high, label in ranges:
            count = ((all_sims >= low) & (all_sims < high)).sum()
            pct = 100 * count / len(all_sims)
            print(f"  {label:20} [{low:.1f}-{high:.1f}): {count:6} pairs ({pct:5.1f}%)")

        # 4. TEMPLATE DETECTION (high similarity to many samples)
        print("\n" + "=" * 70)
        print("TEMPLATE DETECTION (samples similar to many others):")
        print("=" * 70)

        # Count how many samples each sample is similar to (>0.6)
        similar_counts = (similarities > 0.6).sum(axis=1)

        # Find samples similar to many others (potential templates)
        template_threshold = 5  # Similar to 5+ other samples
        potential_templates = np.where(similar_counts >= template_threshold)[0]

        if len(potential_templates) > 0:
            print(f"\n⚠️  Found {len(potential_templates)} potential template-based samples:")
            for idx in potential_templates[:10]:  # Show first 10
                count = similar_counts[idx]
                print(f"\n  Sample {idx}: similar to {count} other samples")
                print(f"    Text: {texts[idx][:100]}")
        else:
            print(f"\n✓ No obvious templates found (threshold={template_threshold})")

        # 5. WHAT THIS MEANS FOR VALIDATION
        print("\n" + "=" * 70)
        print("IMPLICATIONS FOR TRAIN/VAL SPLIT:")
        print("=" * 70)

        median_sim = np.median(all_sims)
        mean_sim = np.mean(all_sims)

        print(f"\nAverage pairwise similarity: {mean_sim:.3f}")
        print(f"Median pairwise similarity:  {median_sim:.3f}")

        if mean_sim > 0.5:
            print("\n⚠️  HIGH SIMILARITY detected!")
            print("   → Many documents share similar structure/vocabulary")
            print("   → Random split may put similar docs in train AND val")
            print("   → This could explain good eval_score but worse leaderboard")
            print("\n   RECOMMENDATION:")
            print("   - Use clustering-based split (dissimilar clusters in val)")
            print("   - Or use adversarial validation to select val set")
        elif mean_sim > 0.3:
            print("\n✓ MODERATE SIMILARITY")
            print("   → Some structure sharing (normal for legal/historical docs)")
            print("   → Current stratification should handle this")
        else:
            print("\n✓ LOW SIMILARITY")
            print("   → Documents are diverse")
            print("   → Current random stratified split is appropriate")

    except Exception as e:
        print(f"Error computing similarities: {e}")

if __name__ == "__main__":
    analyze_entry_similarity()
