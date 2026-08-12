"""
Text Difficulty Analysis for Historical Documents

Analyzes transcription complexity to identify "hard" text samples:
- Vocabulary diversity (rare words vs common boilerplate)
- Named entities (names, places)
- Numbers and dates
- Special characters (^, abbreviations)
- Historical spelling complexity
- Text length

Usage:
    python analyze_text_difficulty.py --config config_qwen3_8b_full.yaml

Output:
    dataset/text_difficulty.csv (ID, text metrics, difficulty_score)
"""

import argparse
import re
from pathlib import Path
from collections import Counter

import yaml
import pandas as pd
import numpy as np

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent


# Common boilerplate words/phrases in historical legal documents
BOILERPLATE_WORDS = {
    'the', 'and', 'of', 'in', 'to', 'a', 'all', 'said', 'one', 'that', 'for',
    'with', 'by', 'be', 'was', 'this', 'his', 'her', 'their', 'from', 'or',
    'as', 'an', 'at', 'are', 'have', 'had', 'upon', 'unto', 'aforesaid',
    'being', 'same', 'given', 'granted', 'made', 'sealed', 'delivered',
    'presence', 'witness', 'premises', 'bargain', 'sell', 'convey',
    'deed', 'will', 'testament', 'executor', 'land', 'acres', 'plantation',
    'island', 'barbados', 'thomas', 'parish', 'day', 'year', 'lord',
    'kingdom', 'england', 'scotland', 'ireland', 'grace', 'majesty',
    'thousand', 'hundred', 'first', 'second', 'third', 'fourth', 'fifth'
}

# Common historical abbreviations
ABBREVIATIONS = {
    '^', 'y^e', 's^d', 'w^th', 'Dec^d', 'S:', 'Esq:', 'Gent:', 'afores^d',
    'Rec^d', 'deliver^d'
}


def compute_vocabulary_diversity(text):
    """
    Measure lexical diversity (type-token ratio).

    Higher diversity = more unique words = harder (likely contains names/rare words)
    Lower diversity = repetitive boilerplate = easier

    Returns:
        score: 0-100 (higher = more diverse/harder)
    """
    words = text.lower().split()
    if len(words) == 0:
        return 0.0

    unique_words = len(set(words))
    total_words = len(words)

    # Type-token ratio
    ttr = unique_words / total_words

    # Normalize to 0-100 (typical TTR: 0.5-0.9 for short texts)
    score = min(100, ttr * 100)

    return score


def compute_rare_word_ratio(text, word_frequencies):
    """
    Measure ratio of rare words (not in common boilerplate).

    Returns:
        score: 0-100 (higher = more rare words = harder)
    """
    words = text.lower().split()
    if len(words) == 0:
        return 0.0

    # Count words not in boilerplate or not in top 20% of corpus
    rare_count = 0
    for word in words:
        # Remove punctuation for matching
        clean_word = re.sub(r'[^\w\s]', '', word)
        if clean_word not in BOILERPLATE_WORDS:
            # Check if rare in corpus
            if word_frequencies.get(word, 0) < word_frequencies.get('median_freq', 1):
                rare_count += 1

    rare_ratio = rare_count / len(words)

    # Normalize to 0-100
    score = min(100, rare_ratio * 200)  # Scale up (typical: 0.1-0.5)

    return score


def compute_named_entity_score(text):
    """
    Estimate presence of named entities (names, places).

    Heuristic: Capitalized words that are not sentence-start or boilerplate proper nouns.

    Returns:
        score: 0-100 (higher = more names = harder)
    """
    # Find all capitalized words
    capitalized = re.findall(r'\b[A-Z][a-z]+', text)

    if len(text.split()) == 0:
        return 0.0

    # Filter out known proper nouns in boilerplate
    known_proper = {'Barbados', 'England', 'Scotland', 'Ireland', 'Thomas', 'Lord',
                   'Kingdom', 'Island', 'Parish', 'August', 'January', 'February',
                   'March', 'April', 'May', 'June', 'July', 'September', 'October',
                   'November', 'December'}

    potential_names = [w for w in capitalized if w not in known_proper]

    # Ratio of potential names to total words
    name_ratio = len(potential_names) / len(text.split())

    # Normalize to 0-100
    score = min(100, name_ratio * 500)  # Scale up (typical: 0.05-0.2)

    return score


def compute_number_complexity(text):
    """
    Measure complexity of numbers (dates, amounts).

    Numbers are harder to transcribe (digit recognition errors).

    Returns:
        score: 0-100 (higher = more numbers = harder)
    """
    # Find all numbers
    numbers = re.findall(r'\b\d+\b', text)

    if len(text.split()) == 0:
        return 0.0

    number_ratio = len(numbers) / len(text.split())

    # Long numbers are harder
    avg_number_length = np.mean([len(n) for n in numbers]) if numbers else 0

    # Combine ratio and length
    score = min(100, number_ratio * 300 + avg_number_length * 5)

    return score


def compute_special_char_complexity(text):
    """
    Measure presence of special characters (^, abbreviations, unusual chars).

    Returns:
        score: 0-100 (higher = more special chars = harder)
    """
    # Count special characters
    special_chars = re.findall(r'[\^~\*\-]{2,}|y\^e|s\^d|w\^th|Dec\^d|afores\^d|Rec\^d|deliver\^d', text)

    if len(text) == 0:
        return 0.0

    special_ratio = len(special_chars) / len(text.split())

    # Normalize to 0-100
    score = min(100, special_ratio * 500)

    return score


def compute_historical_spelling_complexity(text):
    """
    Detect historical spelling variations that are hard to transcribe.

    Examples: "ffrance" (double-f), "y^e" (the), archaic spellings

    Returns:
        score: 0-100 (higher = more archaic = harder)
    """
    # Historical spelling patterns
    patterns = [
        r'\bff[a-z]',  # Double-f (ffrance, ffather)
        r'\bvp?on\b',  # vpon (upon)
        r'\bvnt[o|i]',  # vnto (unto)
        r'\bw[t]h',  # wth (with)
        r'\bwhich',  # olde spellings
        r'[aeiou]{3,}',  # Unusual vowel clusters
    ]

    archaic_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in patterns)

    if len(text.split()) == 0:
        return 0.0

    archaic_ratio = archaic_count / len(text.split())

    # Normalize to 0-100
    score = min(100, archaic_ratio * 500)

    return score


def compute_text_length_complexity(text):
    """
    Longer texts are generally harder (more room for errors).

    Returns:
        score: 0-100 (higher = longer = harder)
    """
    length = len(text)

    # Normalize based on dataset (typical: 20-150 chars)
    # 150+ chars = very long = high difficulty
    score = min(100, length / 2.0)

    return score


def compute_punctuation_complexity(text):
    """
    Heavy punctuation can indicate complex sentence structure.

    Returns:
        score: 0-100 (higher = more punctuation = harder)
    """
    punctuation = re.findall(r'[,;:\.!?\-\(\)]', text)

    if len(text) == 0:
        return 0.0

    punct_ratio = len(punctuation) / len(text)

    # Normalize to 0-100
    score = min(100, punct_ratio * 500)

    return score


def analyze_text(text, word_frequencies):
    """
    Comprehensive text difficulty analysis.

    Returns:
        dict of text metrics
    """
    metrics = {
        "text_length": len(text),
        "word_count": len(text.split()),
        "vocabulary_diversity": compute_vocabulary_diversity(text),
        "rare_word_ratio": compute_rare_word_ratio(text, word_frequencies),
        "named_entity_score": compute_named_entity_score(text),
        "number_complexity": compute_number_complexity(text),
        "special_char_complexity": compute_special_char_complexity(text),
        "historical_spelling": compute_historical_spelling_complexity(text),
        "text_length_complexity": compute_text_length_complexity(text),
        "punctuation_complexity": compute_punctuation_complexity(text),
    }

    # Composite difficulty score (weighted average)
    difficulty = (
        metrics["vocabulary_diversity"] * 0.15 +
        metrics["rare_word_ratio"] * 0.25 +
        metrics["named_entity_score"] * 0.20 +
        metrics["number_complexity"] * 0.15 +
        metrics["special_char_complexity"] * 0.10 +
        metrics["historical_spelling"] * 0.05 +
        metrics["text_length_complexity"] * 0.05 +
        metrics["punctuation_complexity"] * 0.05
    )

    metrics["difficulty_score"] = difficulty

    return metrics


def build_word_frequencies(df):
    """Build corpus-level word frequency dictionary."""
    all_words = []
    for text in df["Target"]:
        all_words.extend(text.lower().split())

    word_counts = Counter(all_words)
    total_words = len(all_words)

    # Convert to frequencies
    word_frequencies = {word: count / total_words for word, count in word_counts.items()}

    # Compute median frequency threshold
    median_freq = np.median(list(word_frequencies.values()))
    word_frequencies['median_freq'] = median_freq

    return word_frequencies


def main():
    parser = argparse.ArgumentParser(description="Analyze text difficulty")
    parser.add_argument(
        "--config",
        type=str,
        default="config_qwen3_8b_full.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: dataset/text_difficulty.csv)"
    )
    args = parser.parse_args()

    # Load config
    config_path = SCRIPT_DIR / args.config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]

    # Load training data
    train_csv = REPO_ROOT / data_cfg["train_csv"]
    df = pd.read_csv(train_csv)
    print(f"Loaded {len(df)} training samples")

    # Build word frequencies
    print("Building word frequency distribution...")
    word_frequencies = build_word_frequencies(df)
    print(f"  Vocabulary size: {len(word_frequencies)}")
    print(f"  Median word frequency: {word_frequencies['median_freq']:.6f}")

    # Analyze each text
    print("\nAnalyzing text difficulty...")
    results = []
    for _, row in df.iterrows():
        metrics = analyze_text(row["Target"], word_frequencies)
        metrics["ID"] = row["ID"]
        results.append(metrics)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Sort by difficulty
    results_df = results_df.sort_values("difficulty_score", ascending=False)

    # Reorder columns
    cols = ["ID", "difficulty_score", "text_length", "word_count",
            "vocabulary_diversity", "rare_word_ratio", "named_entity_score",
            "number_complexity", "special_char_complexity", "historical_spelling",
            "text_length_complexity", "punctuation_complexity"]
    results_df = results_df[cols]

    # Save
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = REPO_ROOT / "dataset" / "text_difficulty.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    print(f"\n✓ Saved text difficulty analysis to: {output_path}")

    # Print statistics
    print("\n" + "="*80)
    print("Text Difficulty Statistics")
    print("="*80)

    print(f"\nDifficulty Score Distribution (0=easy, 100=hard):")
    print(f"  Mean: {results_df['difficulty_score'].mean():.1f}")
    print(f"  Median: {results_df['difficulty_score'].median():.1f}")
    print(f"  Std: {results_df['difficulty_score'].std():.1f}")
    print(f"  Min: {results_df['difficulty_score'].min():.1f}")
    print(f"  Max: {results_df['difficulty_score'].max():.1f}")

    print(f"\nComplexity Metrics (0-100):")
    for metric in ["vocabulary_diversity", "rare_word_ratio", "named_entity_score",
                   "number_complexity", "special_char_complexity", "historical_spelling"]:
        mean_val = results_df[metric].mean()
        print(f"  {metric:25s}: {mean_val:5.1f} (±{results_df[metric].std():.1f})")

    # Identify hardest texts
    print(f"\n🔴 Top 10 Hardest Texts:")
    print("="*80)
    hardest = results_df.head(10)

    # Merge with original text
    hardest_with_text = hardest.merge(df[["ID", "Target"]], on="ID")

    for idx, row in hardest_with_text.iterrows():
        text_preview = row["Target"][:60] + "..." if len(row["Target"]) > 60 else row["Target"]
        print(f"\n{row['ID']}")
        print(f"  Difficulty: {row['difficulty_score']:.1f}")
        print(f"  Text: \"{text_preview}\"")
        print(f"  Rare words: {row['rare_word_ratio']:.1f} | Names: {row['named_entity_score']:.1f} | "
              f"Numbers: {row['number_complexity']:.1f} | Special: {row['special_char_complexity']:.1f}")

    # Identify easiest texts
    print(f"\n🟢 Top 10 Easiest Texts:")
    print("="*80)
    easiest = results_df.tail(10)

    # Merge with original text
    easiest_with_text = easiest.merge(df[["ID", "Target"]], on="ID")

    for idx, row in easiest_with_text.iterrows():
        text_preview = row["Target"][:60] + "..." if len(row["Target"]) > 60 else row["Target"]
        print(f"\n{row['ID']}")
        print(f"  Difficulty: {row['difficulty_score']:.1f}")
        print(f"  Text: \"{text_preview}\"")
        print(f"  Rare words: {row['rare_word_ratio']:.1f} | Names: {row['named_entity_score']:.1f} | "
              f"Numbers: {row['number_complexity']:.1f} | Special: {row['special_char_complexity']:.1f}")


if __name__ == "__main__":
    main()
