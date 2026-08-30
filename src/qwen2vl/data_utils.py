"""
Data loading and stratification logic for the Qwen-VL HTR fine-tuning pipeline.

Deliberately has NO torch/transformers/peft/cv2/PIL imports - only
pandas/numpy/sklearn/yaml. This lets `test_stratification.py` (and anything
else that only cares about the train/val split) run standalone, fast, and
without a GPU or any ML framework installed. train.py imports the functions
it needs from here rather than duplicating them, so there is a single source
of truth for the stratification logic.
"""

from pathlib import Path
import re
import random
from collections import Counter

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold
    GROUP_KFOLD_AVAILABLE = True
except ImportError:
    GROUP_KFOLD_AVAILABLE = False

# Resolved relative to this file's location, same convention as train.py's
# SCRIPT_DIR/REPO_ROOT - keep data_utils.py alongside train.py so both agree
# on where "dataset/" and the repo root actually are.
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent


def load_and_prepare_dataframe(data_cfg: dict) -> pd.DataFrame:
    """
    Load train_csv and merge in cluster_csv / document_condition.csv if present.

    Pulled out of train() (2026-08-30) so the same loading logic - including
    the median-fill for missing condition scores - is shared between actual
    training and standalone stratification testing, instead of the test script
    re-implementing (and risking drifting from) this merge logic.
    """
    train_csv = REPO_ROOT / data_cfg["train_csv"]
    df = pd.read_csv(train_csv)
    nan_count = df['Target'].isna().sum()
    print(f"Loaded {len(df)} samples from {train_csv.name}" + (f" ({nan_count} NaN targets)" if nan_count > 0 else ""))

    # Load document clusters if provided
    cluster_csv = data_cfg.get("cluster_csv")
    if cluster_csv:
        cluster_path = REPO_ROOT / cluster_csv
        if cluster_path.exists():
            cluster_df = pd.read_csv(cluster_path)
            df = df.merge(cluster_df, on="ID", how="left")
            group_col = data_cfg.get("group_col")
            if group_col and group_col in df.columns:
                print(f"Loaded document clusters from {cluster_path.name}")
                print(f"  {df[group_col].nunique()} unique clusters, avg {len(df)/df[group_col].nunique():.1f} samples/cluster")
            else:
                print(f"⚠️  Warning: cluster_csv provided but group_col '{group_col}' not found in merged data")
        else:
            print(f"⚠️  Warning: cluster_csv '{cluster_csv}' not found, proceeding without clustering")

    # Load document condition scores for adaptive augmentation (if available)
    condition_csv = REPO_ROOT / "dataset" / "document_condition.csv"
    if condition_csv.exists():
        condition_df = pd.read_csv(condition_csv)
        # Filter to successful analyses only
        condition_df = condition_df[condition_df["success"] == True]
        # Merge condition scores
        df = df.merge(condition_df[["ID", "condition_score"]], on="ID", how="left")
        # Fill missing with median (for any images that failed analysis)
        median_cond = df["condition_score"].median()
        n_missing = df["condition_score"].isna().sum()
        if n_missing > 0:
            df.loc[df["condition_score"].isna(), "condition_score"] = median_cond
        print(f"Loaded document condition scores from {condition_csv.name}")
        print(f"  Mean: {df['condition_score'].mean():.1f}, Median: {median_cond:.1f}, Std: {df['condition_score'].std():.1f}")
        if n_missing > 0:
            print(f"  Filled {n_missing} missing values with median")
    else:
        print(f"ℹ️  Document condition scores not found ({condition_csv.name}), proceeding without adaptive augmentation")

    return df


def compute_special_char_density(text: str) -> float:
    """
    Compute special character density as stratification proxy.

    Special characters indicate:
    - Document type (legal docs with £, dates, formal punctuation)
    - Scribe style (abbreviations, dashes)
    - Transcription complexity

    Returns: ratio of special chars to total chars (0.0 to 1.0)
    """
    if not text or pd.isna(text):
        return 0.0

    text = str(text)
    # Count non-alphanumeric, non-space characters
    special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
    return special_chars / max(len(text), 1)


def compute_digit_density(text: str) -> float:
    """
    Compute digit/number density as stratification proxy.

    Digits indicate:
    - Dates (1842, 15th)
    - Monetary amounts (£25-10-6)
    - Measurements (3 acres)
    - Different OCR challenge (digits often harder than letters)

    Returns: ratio of digits to total chars (0.0 to 1.0)
    """
    if not text or pd.isna(text):
        return 0.0

    text = str(text)
    digits = sum(1 for c in text if c.isdigit())
    return digits / max(len(text), 1)


def compute_uppercase_ratio(text: str) -> float:
    """
    Compute uppercase letter ratio as formality/emphasis indicator.

    Uppercase indicates:
    - Proper nouns (John Smith, London)
    - Formal language (WITNESSED, SEALED)
    - Emphasis and titles (Mr., Esq.)
    - Different capitalization patterns across document types

    Returns: ratio of uppercase letters to total letters (0.0 to 1.0)
    """
    if not text or pd.isna(text):
        return 0.0

    text = str(text)
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    uppercase = sum(1 for c in letters if c.isupper())
    return uppercase / len(letters)


def compute_lexical_diversity(text: str) -> float:
    """
    Compute lexical diversity (unique word ratio) as vocabulary complexity indicator.

    Lexical diversity indicates:
    - Repetitive/formulaic language (low diversity): "the said party... the said party"
    - Rich vocabulary (high diversity): "signed, sealed, witnessed, delivered, dated"
    - Document type (legal templates vs descriptive narratives)

    Returns: ratio of unique words to total words (0.0 to 1.0)
    """
    if not text or pd.isna(text):
        return 0.0

    words = str(text).lower().split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def compute_avg_word_length(text: str) -> float:
    """
    Compute average word length as vocabulary complexity indicator.

    Word length indicates:
    - Simple vocabulary (short words): "I see the man go"
    - Complex vocabulary (long words): "aforementioned beneficiary witnessed"
    - Different OCR challenge (longer words = more opportunities for errors)

    Returns: average characters per word
    """
    if not text or pd.isna(text):
        return 0.0

    words = [w for w in str(text).split() if w]
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)


def distribute_duplicate_groups(df: pd.DataFrame, val_split: float, seed: int) -> pd.Series:
    """
    Deliberately spread each duplicate/near-duplicate TEXT group (_dup_group
    >= 0, 2+ members) across train AND val, instead of leaving group
    placement to chance.

    This is the OPPOSITE mechanism from group_duplicate_text_in_split (which
    keeps a whole group on one side, for tasks where text-level leakage
    matters). Here, since duplicate TEXT doesn't imply duplicate visual
    content for OCR, the goal is coverage: make sure both splits see a
    calligraphic exemplar of every recurring phrase, rather than risk val
    being blind to it entirely.

    Why this needs to be explicit rather than left to the ordinary stratified
    split: on the real 4098-row training set (2026-08-30), 22 of 27 duplicate
    groups have exactly 2 members. Under a plain 90/10 stratified split with
    no group awareness, a 2-member group lands entirely in train with
    probability 0.9² = 81% by pure chance - val would see that phrase 0% of
    the time in the large majority of cases, purely from how the random
    split happened to fall, not from any deliberate decision.

    Method: for each group, shuffle its members with a seeded local RNG (so
    results are deterministic and reproducible run-to-run for a given seed,
    independent of anything else that consumes randomness elsewhere) and put
    max(1, round(n * val_split)) of them in val, capped at n-1 so at least
    one member always stays in train too. For every observed group size here
    (2-5), this puts exactly one member in val and the rest in train.

    Args:
        df: annotated dataframe with a _dup_group column (from
            compute_stratification_bins)
        val_split: target val fraction, used per-group to decide how many
            of that group's members go to val
        seed: for the local deterministic shuffle

    Returns:
        Series aligned to df's index: 'train' or 'val' for every member of a
        duplicate group (size >= 2), NaN for everything else (singletons -
        left to the ordinary stratified split to place).
    """
    assignment = pd.Series(pd.NA, index=df.index, dtype=object)
    rng = random.Random(seed)

    duplicated = df[df["_dup_group"] >= 0]
    if duplicated.empty:
        return assignment

    for _, group_df in duplicated.groupby("_dup_group"):
        indices = list(group_df.index)
        n = len(indices)
        if n < 2:
            continue  # shouldn't happen (groups are built from >=2 matches), but safe or the ordinary split may as well handle it

        rng.shuffle(indices)
        n_val = max(1, round(n * val_split))
        n_val = min(n_val, n - 1)  # always leave at least one member in train

        for idx in indices[:n_val]:
            assignment.at[idx] = "val"
        for idx in indices[n_val:]:
            assignment.at[idx] = "train"

    return assignment


def compute_ngram_jaccard(text1: str, text2: str, n: int = 3) -> float:
    """
    Compute character n-gram Jaccard similarity between two texts.

    Used for fuzzy boilerplate detection - historical legal documents often share
    90%+ boilerplate with only names/dates differing (e.g., "This Indenture made
    the [DATE] between [NAMES]...").

    Args:
        text1, text2: Texts to compare
        n: N-gram size (default 3-char for historical text)

    Returns:
        Jaccard similarity (0.0 to 1.0)
    """
    if not text1 or not text2:
        return 0.0

    # Normalize: lowercase, strip whitespace
    t1 = text1.lower().strip()
    t2 = text2.lower().strip()

    if t1 == t2:
        return 1.0

    # Generate character n-grams
    def get_ngrams(text, n):
        return set(text[i:i+n] for i in range(len(text) - n + 1))

    ngrams1 = get_ngrams(t1, n)
    ngrams2 = get_ngrams(t2, n)

    if not ngrams1 or not ngrams2:
        return 0.0

    # Jaccard similarity: |A ∩ B| / |A ∪ B|
    intersection = len(ngrams1 & ngrams2)
    union = len(ngrams1 | ngrams2)

    return intersection / union if union > 0 else 0.0


def compute_rare_vocabulary_flags(target_series: pd.Series, freq_threshold: int = 1) -> pd.Series:
    """
    Flag each document as containing at least one 'rare' word - a word whose
    frequency ACROSS THIS CORPUS (not general English) is <= freq_threshold.
    Default threshold=1 means hapax legomena: words that appear exactly once
    anywhere in the training set.

    Corpus-relative rarity is the deliberate definition, not absolute English
    rarity: a common English word that only happens to appear once in this
    particular dataset (e.g. a place name mentioned in only one deed) is
    exactly the kind of word the language-model prior can't help predict -
    which is the thing this flag is meant to catch for stratification.
    Tested against the real 4098-row training set (2026-08-30): threshold=1
    flags 48.2% of documents, giving a roughly balanced binary split, and is
    close to independent of condition_score (47.5%-49.4% across all three
    condition tiers) and only mildly correlated with text length (r=0.11) -
    it is not a repackaging of an existing stratification factor the way
    difficulty_score turned out to be.
    """
    def tokenize(text):
        return re.findall(r"[a-zA-Z']+", str(text).lower())

    tokens_per_doc = target_series.apply(tokenize)
    word_counts = Counter()
    for toks in tokens_per_doc:
        word_counts.update(toks)
    rare_words = {w for w, c in word_counts.items() if c <= freq_threshold}
    return tokens_per_doc.apply(lambda toks: any(w in rare_words for w in toks))


def classify_caret_types(target_series: pd.Series) -> tuple:
    """
    Split caret (^) occurrences into two visually distinct scribal phenomena.

    - 'interlineation': a caret preceded by whitespace/start-of-string and
      followed by a letter - marks a WHOLE WORD squeezed in above the line
      (e.g. "omitt Suffered or done ^by the said John Bawdon"). A genuinely
      distinct visual event: an inserted word, often cramped or smaller,
      wedged between existing lines.
    - 'superscript': a caret embedded directly inside a token (attached to a
      letter, digit, or punctuation on the left) - marks a raised/superscript
      letter SEQUENCE WITHIN a word, typically an abbreviation or ordinal
      contraction (e.g. "s^.d", "w^tsoever", "Exec:^rs", "m^r", "25:^th"). A
      different visual pattern: small raised letters within an otherwise
      normal word, not a separate inserted word.

    Verified on the real 4098-row training set (2026-08-30): 688 documents
    have at least one caret; 663 are superscript-only, 20 interlineation-only,
    5 have both.

    IMPORTANT: interlineation is too rare (~23-25 documents total, combining
    interlineation-only + both-types) to safely use as its own crossed
    stratification axis - tested directly: crossing condition(3) x
    text_type(5, with interlineation split out) x rare_vocab(2) produces
    bins with a SINGLE sample regardless of which factor is given priority in
    the text_type tier ordering, which breaks stratified splitting outright.
    Even on its own, uncrossed, val_split=0.1 would put only ~2-3
    interlineation samples in val - too few to say much either way. This
    function is for DIAGNOSTIC reporting (train/val breakdown by caret type,
    printed in make_splits) rather than for the stratification key itself -
    see compute_text_type, which folds both sub-types into one 'has_caret'
    tier for the actual bin construction, since that combined population
    (581) is large enough to stratify on safely.

    Returns:
        (has_interlineation, has_superscript) - two boolean Series aligned to
        target_series's index. A document can be True in both.
    """
    def classify(text):
        text = str(text)
        has_interlin = False
        has_superscr = False
        for m in re.finditer(r"\^", text):
            i = m.start()
            before = text[i - 1] if i > 0 else " "
            after = text[i + 1] if i + 1 < len(text) else ""
            if before.isspace() and after.isalpha():
                has_interlin = True
            else:
                has_superscr = True
        return has_interlin, has_superscr

    results = target_series.apply(classify)
    has_interlineation = results.apply(lambda t: t[0])
    has_superscript = results.apply(lambda t: t[1])
    return has_interlineation, has_superscript


def compute_text_type(df: pd.DataFrame) -> pd.Series:
    """
    Collapse has_digit/has_upper/has_caret into ONE categorical instead of
    crossing them as independent binary flags.

    Crossing has_digit x has_upper independently creates a near-empty cell:
    on the real 4098-row training set (2026-08-30), only 6 documents have
    digits WITHOUT any uppercase letter, so any scheme that crosses these two
    flags guarantees a ~6-sample cell before condition or anything else is
    even applied - this was the actual cause of the fallback-ladder collapse
    seen in production, not difficulty_score as initially suspected.

    UPDATED 2026-08-30: added a 'has_caret' tier, above names_only, folding in
    BOTH caret sub-types (interlineation and superscript - see
    classify_caret_types) as a single category. The two sub-types are real
    and visually distinct, but interlineation alone (~23-25/4098) is too rare
    to cross safely with condition/rare_vocab as its own tier - tested
    directly, produces singleton bins. They're tracked separately as
    diagnostic-only columns (_has_interlineation/_has_superscript, reported
    in make_splits' train/val summary) rather than folded into the
    stratification key.

    Priority order: has_digits > has_caret > names_only > plain. Digits stay
    top priority (rarest genuinely-independent category, 217/4098, hardest to
    predict without a language-model prior). Caret sits above names_only:
    interlineated/superscript script is a real, distinct visual phenomenon
    (cramped, raised, or inserted text) at least as significant as
    capitalization alone. Verified population (2026-08-30): plain=853,
    names_only=2447, has_caret=581, has_digits=217 (581 < the raw 688
    caret-containing documents because ~107 of those also contain a digit,
    which takes priority in this ordering).
    """
    has_digit = df["Target"].str.contains(r"\d", regex=True, na=False)
    has_upper = df["Target"].str.contains(r"[A-Z]", regex=True, na=False)
    has_interlin, has_superscr = classify_caret_types(df["Target"])
    has_caret = has_interlin | has_superscr

    text_type = pd.Series("plain", index=df.index)
    text_type = text_type.mask(has_upper, "names_only")
    text_type = text_type.mask(has_caret, "has_caret")   # caret takes priority over names_only
    text_type = text_type.mask(has_digit, "has_digits")  # digits take priority over everything else
    return text_type


def compute_stratification_bins(df: pd.DataFrame, data_cfg: dict) -> pd.DataFrame:
    """
    Annotate df with duplicate-group ids, text-difficulty scores, and the
    combined `_bin` stratification key - everything make_splits needs to
    decide train/val membership, computed but not yet split.

    Pulled out of make_splits (2026-08-30) so a lightweight diagnostic script
    can inspect bin populations (which bins are too small, what fallback
    stratification will fire) without duplicating this logic and without
    needing any of train.py's heavy ML imports (torch/transformers/peft) -
    this function only needs pandas/numpy/sklearn.
    """
    df = df.copy()

    # STEP 1: Identify duplicate and near-duplicate texts to prevent train/val leakage
    print("Checking for duplicate and near-duplicate texts (fuzzy boilerplate detection)...")
    df["_text_clean"] = df["Target"].astype(str).str.lower().str.strip()

    # REFINEMENT: Fuzzy duplicate detection using character n-gram Jaccard similarity
    # Historical legal docs share 90%+ boilerplate with only names/dates differing
    # Example: "This Indenture made [DATE] between [NAMES]..." → 95% similarity
    # Config: data.fuzzy_duplicate_threshold (default 0.90, set to 0 to disable)
    fuzzy_threshold = data_cfg.get("fuzzy_duplicate_threshold", 0.90)

    # First pass: exact duplicates (fast)
    text_counts = df["_text_clean"].value_counts()
    exact_duplicates = text_counts[text_counts > 1]

    # Initialize duplicate groups
    df["_dup_group"] = -1  # -1 = not a duplicate
    next_group_id = 0

    # Group exact duplicates first
    if len(exact_duplicates) > 0:
        for dup_text in exact_duplicates.index:
            df.loc[df["_text_clean"] == dup_text, "_dup_group"] = next_group_id
            next_group_id += 1

    exact_grouped = (df["_dup_group"] >= 0).sum()
    print(f"  Exact matches: {len(exact_duplicates)} unique texts ({exact_grouped} samples)")

    # Second pass: fuzzy duplicates (slower, O(n²) - only on ungrouped samples)
    ungrouped_indices = df[df["_dup_group"] == -1].index.tolist()
    fuzzy_grouped = 0

    if len(ungrouped_indices) > 1 and fuzzy_threshold > 0:
        print(f"  Scanning {len(ungrouped_indices)} ungrouped samples for fuzzy duplicates (threshold={fuzzy_threshold})...")

        # PERFORMANCE FIX (2026-08-30): the previous version recomputed each
        # text's n-gram set from scratch on EVERY pairwise comparison (via
        # compute_ngram_jaccard) and used pandas .loc scalar access inside the
        # O(n²) loop for the already-grouped check. Both are avoidable and
        # both cost real time at this scale: benchmarked at ~285s on the real
        # 4098-row training set (n=500/1000/1500 all extrapolate consistently
        # to that number via n² scaling) - long enough to make even a quick
        # stratification check impractical, and paid on every actual training
        # run too, since train() calls this same function via make_splits.
        #
        # Fix: precompute each text's n-gram set ONCE (O(n) work), track
        # dup-group membership in a plain Python list during the loop instead
        # of pandas .loc (pandas per-cell access has real per-call overhead
        # at millions of calls), and skip the full intersection/union
        # computation when a cheap upper bound already rules out reaching
        # threshold: Jaccard = |A∩B|/|A∪B| <= min(|A|,|B|)/max(|A|,|B|), so if
        # that ratio alone is below threshold, the true Jaccard can't reach it
        # either and the (more expensive) actual set operations are skipped.
        # Same algorithm, same grouping decisions, same threshold semantics -
        # only the implementation is faster.
        def _ngrams(text: str, n: int = 3) -> frozenset:
            return frozenset(text[i:i + n] for i in range(len(text) - n + 1))

        texts = [df.at[idx, "_text_clean"] for idx in ungrouped_indices]
        ngram_sets = [_ngrams(t, 3) for t in texts]
        sizes = [len(s) for s in ngram_sets]
        n_items = len(ungrouped_indices)

        # -1 = ungrouped, else a local group id (remapped to real _dup_group
        # ids only at the end, after we know how many fuzzy groups exist)
        local_group = [-1] * n_items
        next_local_id = 0

        for i in range(n_items):
            if local_group[i] >= 0:
                continue
            size_i = sizes[i]
            if size_i == 0:
                continue  # empty n-gram set (very short text) can't match anything

            similar_group = [i]
            for j in range(i + 1, n_items):
                if local_group[j] >= 0:
                    continue
                size_j = sizes[j]
                if size_j == 0:
                    continue

                # Cheap upper bound before paying for the actual set ops
                max_possible = min(size_i, size_j) / max(size_i, size_j)
                if max_possible < fuzzy_threshold:
                    continue

                inter = len(ngram_sets[i] & ngram_sets[j])
                union = size_i + size_j - inter
                similarity = inter / union if union > 0 else 0.0

                if similarity >= fuzzy_threshold:
                    similar_group.append(j)

            if len(similar_group) > 1:
                for pos in similar_group:
                    local_group[pos] = next_local_id
                next_local_id += 1
                fuzzy_grouped += len(similar_group)

        # Bulk-assign back to df in one vectorized pass (not per-cell .loc)
        if next_local_id > 0:
            id_map = {
                idx: next_group_id + local_id
                for idx, local_id in zip(ungrouped_indices, local_group)
                if local_id >= 0
            }
            df.loc[list(id_map.keys()), "_dup_group"] = list(id_map.values())
            next_group_id += next_local_id

        print(f"  Fuzzy matches: {fuzzy_grouped} samples grouped into {next_local_id} boilerplate clusters")
    else:
        print(f"  Fuzzy matching disabled (threshold={fuzzy_threshold})")

    total_grouped = (df["_dup_group"] >= 0).sum()
    if total_grouped > 0:
        print(f"  Total grouped: {total_grouped} samples ({exact_grouped} exact + {fuzzy_grouped} fuzzy)")

    # NOTE: text_difficulty.csv / difficulty_score used to be loaded here and
    # merged into df. Removed 2026-08-30: difficulty_score turned out to be
    # nearly redundant with the digit/upper flags (99.7% of "hard"-difficulty
    # documents also had has_upper=True on the real data - see
    # compute_text_type docstring), so it added stratification bins without
    # adding real independent information. Replaced by rare_vocab (see
    # compute_rare_vocabulary_flags), verified closer to independent of the
    # other factors. If per-document difficulty scoring is needed again for
    # some other purpose, recompute it explicitly rather than reviving this
    # merge - don't let it quietly re-enter the stratification key without
    # re-checking the same redundancy this removal was based on.

    # Compute text features for logging (not used in the stratification key,
    # but useful diagnostic info printed in the train/val summary below)
    df["_digit_density"] = df["Target"].apply(compute_digit_density)
    df["_uppercase_ratio"] = df["Target"].apply(compute_uppercase_ratio)
    df["_lexical_diversity"] = df["Target"].apply(compute_lexical_diversity)
    df["_special_char_density"] = df["Target"].apply(compute_special_char_density)
    df["_avg_word_length"] = df["Target"].apply(compute_avg_word_length)

    # STRATIFICATION STRATEGY (REVISED 2026-08-30, validated against the real
    # 4098-row training set):
    #
    # Previous scheme: condition (3) x difficulty (3) x digits (2) x names (2)
    # = 36 bins. Two problems, found by testing against real data rather than
    # assumed:
    #   1. difficulty_score (whether analysis-driven or the fallback formula
    #      digit_density*50 + uppercase_ratio*30 + lexical_diversity*20) is
    #      nearly redundant with the digit/upper flags it was crossed against:
    #      99.7% of "hard" documents also have has_upper=True, and 92% of
    #      digit-containing documents land in "hard" too. Crossing it added
    #      bin count without adding real independent information.
    #   2. Crossing has_digit x has_upper as independent binary flags creates
    #      a near-empty cell on its own: only 6/4098 documents have digits
    #      WITHOUT uppercase. This - not difficulty_score - was the actual
    #      cause of production runs collapsing to the weakest (4-bin)
    #      fallback: any scheme crossing these two flags guarantees a ~6-
    #      sample cell before condition or anything else is even applied.
    #
    # Current scheme: condition (3) x text_type (3) x rare_vocab (2) = 18 bins.
    #   - text_type collapses has_digit/has_upper into one 3-level categorical
    #     (see compute_text_type) instead of crossing them, eliminating the
    #     near-empty cell.
    #   - rare_vocab (see compute_rare_vocabulary_flags) replaces
    #     difficulty_score with a factor verified to be closer to independent
    #     of condition/text_type/length, rather than a repackaging of them.
    #   - text length was tested as a 4th factor and dropped: at 3 factors
    #     (18 bins) the smallest cell was already only 10-14 samples: adding
    #     a genuine 4th independent axis on top pushed bin count past what
    #     this dataset size supports (36 bins, smallest cell of 4, 4 bins
    #     under 10 samples) - the same fragmentation problem being fixed.
    #     Verified by direct bin-population testing, not assumed.
    #
    # Verified tier populations on the real training set (n=4098):
    #   condition: good=~23%, medium=~57%, poor_or_worse=~19.5%
    #   text_type: plain=928 (22.6%), names_only=2953 (72.1%), has_digits=217 (5.3%)
    #   rare_vocab: ~48% have_rare / ~52% common
    #   Resulting 18-bin scheme: smallest cell=10, no cell under 10 samples.
    #
    # UPDATE (2026-08-30, same day, following request to consider the caret
    # symbol '^'): folded a 'has_caret' tier into text_type, between
    # has_digits and names_only - see compute_text_type and
    # classify_caret_types docstrings. Scheme is now condition (3) x
    # text_type (4) x rare_vocab (2) = 24 bins, smallest cell=14 (verified).
    # The caret marks two visually distinct scribal phenomena - interlineated
    # word insertions and superscript abbreviation/ordinal contractions - but
    # only their UNION is folded into the stratification key; interlineation
    # alone (~23-25/4098) is too rare to cross safely (produces singleton
    # bins, tested directly). The two sub-types are still tracked and
    # reported separately as diagnostic-only columns
    # (_has_interlineation/_has_superscript) in make_splits' train/val
    # summary, just not used to constrain the split.

    # 1. Visual condition bins (Good/Medium/Poor-or-worse)
    has_condition = "condition_score" in df.columns and not df["condition_score"].isna().all()
    if has_condition:
        # Bin by visual degradation: good (<8.07), medium (8.07-23.38), poor_or_worse (>=23.38).
        # These thresholds match the adaptive augmentation tiers' first two
        # boundaries (the augmentation system itself further splits poor_or_worse
        # into poor/very_poor at 26.12 for augmentation-strength purposes only;
        # for stratification, poor+very_poor are combined into one tier since
        # each individually is too small - ~10%/~9% - to stratify on alone).
        #
        # FIXED bins, not qcut(q=3): qcut produces equal-COUNT tertiles (33/33/33
        # by construction), which on this data lands at (8.55, 13.94) - a "poor"
        # tier that's a full third of the data, NOT the same ~19.5% poor-condition
        # population the augmentation tiers and the rest of this design target.
        # Caught by direct comparison against the fixed-threshold version: qcut
        # gives 1824/1824/1824, fixed gives 1249/3156/1067 - materially different
        # splits, and only the fixed one is what was actually validated via
        # bin-population testing against the real 4098-row training set.
        #
        # These are the same precise percentiles (22.9th/80.5th) computed
        # directly from the uploaded document_condition.csv (n=5472,
        # success==True; mean=13.64, median=9.11, std=8.15 - text_contrast
        # still included in the composite, confirmed via exact-fit linear
        # regression against the five raw metrics).
        df["_condition_bin"] = pd.cut(
            df["condition_score"],
            bins=[0, 8.07, 23.38, 100],
            labels=["good_cond", "medium_cond", "poor_cond"],
            include_lowest=True
        )
        print(f"  ✓ Visual condition stratification enabled (3 fixed bins: <8.07 / 8.07-23.38 / >=23.38)")
    else:
        df["_condition_bin"] = "unknown_cond"
        print(f"  ⚠️  No condition_score found - using text-only stratification")

    # 2. Text type (plain / names_only / has_caret / has_digits) - replaces
    #    the old digit_bin x upper_bin cross; see compute_text_type docstring.
    df["_has_digit"] = df["Target"].str.contains(r"\d", regex=True, na=False)
    df["_has_upper"] = df["Target"].str.contains(r"[A-Z]", regex=True, na=False)
    df["_has_interlineation"], df["_has_superscript"] = classify_caret_types(df["Target"])
    df["_text_type"] = compute_text_type(df)

    # 3. Rare vocabulary flag (corpus-relative hapax legomena by default) -
    #    replaces difficulty_score in the stratification key.
    rare_freq_threshold = data_cfg.get("rare_vocab_freq_threshold", 1)
    df["_has_rare"] = compute_rare_vocabulary_flags(df["Target"], freq_threshold=rare_freq_threshold)
    df["_rare_bin"] = df["_has_rare"].map({True: "rare_vocab", False: "common_vocab"})

    # Combine: condition (3) × text_type (4) × rare_vocab (2) = 24 bins
    df["_bin"] = (df["_condition_bin"].astype(str) + "_" +
                  df["_text_type"].astype(str) + "_" +
                  df["_rare_bin"].astype(str))

    # Keep text_len for logging (no longer part of the stratification key,
    # but still useful diagnostic info)
    df["_text_len"] = df["Target"].str.len()

    # Store original index for duplicate-aware splitting
    df["_orig_idx"] = df.index

    return df


def make_splits(df: pd.DataFrame, data_cfg: dict):
    """Yield (train_df, val_df, fold_num). Stratified by semantic text properties from ground truth. Grouped by group_col when present."""
    df = compute_stratification_bins(df, data_cfg)
    has_condition = "condition_score" in df.columns and not df["condition_score"].isna().all()

    k_folds = data_cfg.get("k_folds", 1)
    seed = data_cfg.get("seed", 42)
    group_col = data_cfg.get("group_col")

    # Whether documents with duplicate/near-duplicate TEXT are forced into the
    # same split. Default OFF (2026-08-30, per explicit request): for an OCR/
    # HTR model the learning target is the visual stroke-to-character mapping,
    # not the text content. Two documents sharing the same transcribed phrase
    # (e.g. shared legal boilerplate: "By this publique Act and Instrument of
    # protest...") are commonly written by different scribes with completely
    # different calligraphy - different ink, slant, letterforms, stroke width.
    # There is no meaningful visual leakage between them just because the text
    # matches, so forcing them into the same split only costs stratification
    # granularity for a leakage risk that doesn't really apply to this task.
    #
    # group_col is a SEPARATE and still-fully-respected mechanism: if the
    # person provides an explicit physical-document/page cluster id (e.g.
    # pages from the same manuscript, or crops from the same scan), that IS a
    # genuine visual-leakage concern - same paper, same scribe, same scan
    # session - and grouping on it is unaffected by this flag.
    #
    # The duplicate/near-duplicate TEXT detection in compute_stratification_bins
    # still runs and is still reported (Exact matches / Fuzzy matches / Total
    # grouped) - that's kept as diagnostic info about how much boilerplate
    # overlap exists in the corpus, it's just no longer used to constrain
    # which split a sample lands in.
    #
    # Set data.group_duplicate_text_in_split: true in config to restore the
    # old (conservative) behavior if ever needed for a different task where
    # text-level leakage genuinely matters (e.g. if this pipeline were reused
    # for a text-generation task rather than OCR).
    group_by_text = data_cfg.get("group_duplicate_text_in_split", False)
    text_dup_exists = (df["_dup_group"] >= 0).any()
    has_duplicates = text_dup_exists and group_by_text

    helper = ["_text_clean", "_dup_group", "_orig_idx", "_digit_density", "_uppercase_ratio", "_lexical_diversity",
              "_special_char_density", "_avg_word_length", "_has_digit", "_has_upper", "_text_len",
              "_has_interlineation", "_has_superscript",
              "_condition_bin", "_text_type", "_has_rare", "_rare_bin", "_bin"]

    if group_col and group_col not in df.columns:
        raise ValueError(f"group_col '{group_col}' not in the CSV columns")

    if k_folds > 1:
        if has_duplicates or group_col:
            # Use StratifiedGroupKFold to keep duplicates/clusters together
            if not GROUP_KFOLD_AVAILABLE:
                raise RuntimeError("K-fold with duplicates needs scikit-learn >= 1.0 for StratifiedGroupKFold")

            # Create synthetic group column combining duplicates + user group_col
            if has_duplicates and group_col:
                # Combine both: duplicate group + user-specified group
                df["_fold_group"] = df["_dup_group"].astype(str) + "_" + df[group_col].astype(str)
                print(f"K-fold with duplicate-text awareness + grouped on '{group_col}'")
            elif has_duplicates:
                # Use duplicate group as fold group
                # Assign unique group ID to non-duplicates
                max_dup_group = df["_dup_group"].max()
                df["_fold_group"] = df.apply(
                    lambda row: row["_dup_group"] if row["_dup_group"] >= 0 else max_dup_group + 1 + row.name,
                    axis=1
                )
                print(f"K-fold with duplicate-text awareness (keeps {(df['_dup_group'] >= 0).sum()} duplicate-text samples together)")
            else:
                # Only user-specified group
                df["_fold_group"] = df[group_col]
                print(f"K-fold grouped on '{group_col}' ({df[group_col].nunique()} groups)")

            helper.append("_fold_group")

            splitter = StratifiedGroupKFold(n_splits=k_folds, shuffle=True, random_state=seed)
            split_iter = splitter.split(df, df["_bin"], groups=df["_fold_group"])

            for fold_num, (tr, va) in enumerate(split_iter, 1):
                train_df = df.iloc[tr].copy()
                val_df = df.iloc[va].copy()

                # This IS a real leakage check here (group_by_text=True means
                # the person explicitly wants duplicate text kept on one side).
                if text_dup_exists and group_by_text:
                    train_texts = set(train_df["_text_clean"])
                    val_texts = set(val_df["_text_clean"])
                    leaked = train_texts & val_texts
                    if leaked:
                        print(f"  ⚠️  Fold {fold_num}: {len(leaked)} duplicate texts leaked (should be 0)!")
                    else:
                        print(f"  ✓ Fold {fold_num}: No duplicate-text leakage")

                yield (train_df.drop(columns=helper).copy(),
                       val_df.drop(columns=helper).copy(),
                       fold_num)

        elif text_dup_exists:
            # No group_col, duplicate-text grouping is off (the default):
            # each duplicate-text group is explicitly DISTRIBUTED round-robin
            # across the k folds - same reasoning as the single-split case
            # (distribute_duplicate_groups docstring): letting StratifiedKFold
            # place small groups without group-awareness risks a fold seeing
            # zero exemplars of a recurring phrase purely by chance. sklearn's
            # splitters don't accept pre-assigned fold membership, so this
            # runs its own loop rather than producing an sklearn split_iter.
            strat_desc = "condition (3) × text_type (4) × rare_vocab (2) = 24 bins" if has_condition \
                else "text_type (4) × rare_vocab (2) = 8 bins"
            print(f"Stratified {k_folds}-fold: {strat_desc}")

            rng = random.Random(seed)
            dup_fold_assignment = {}  # index -> fold number (0-based)
            n_groups = 0
            for _, group_df in df[df["_dup_group"] >= 0].groupby("_dup_group"):
                n_groups += 1
                indices = list(group_df.index)
                rng.shuffle(indices)
                # BUG FIX: always starting the round-robin at fold 0 systematically
                # biased small (2-member, the majority here) groups toward folds
                # 0/1 only, since i%k_folds never exceeds the group size. Verified
                # on the real data before this fix: folds 1-5 got 27/27/5/3/1
                # duplicate-group val samples respectively - heavily skewed, not
                # a fair distribution. A random per-group starting offset spreads
                # which folds a small group can land on.
                start = rng.randrange(k_folds)
                for i, idx in enumerate(indices):
                    dup_fold_assignment[idx] = (start + i) % k_folds

            dup_index_list = list(dup_fold_assignment.keys())
            remainder_df = df.drop(index=dup_index_list)
            print(f"  ✓ Distributing {len(dup_index_list)} samples from {n_groups} duplicate-text "
                  f"groups round-robin across the {k_folds} folds")

            splitter = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
            remainder_split_iter = list(splitter.split(remainder_df, remainder_df["_bin"]))

            for fold_num0 in range(k_folds):
                tr_pos, va_pos = remainder_split_iter[fold_num0]
                remainder_train_idx = remainder_df.index[tr_pos]
                remainder_val_idx = remainder_df.index[va_pos]

                fold_dup_val_idx = [idx for idx, f in dup_fold_assignment.items() if f == fold_num0]
                fold_dup_train_idx = [idx for idx, f in dup_fold_assignment.items() if f != fold_num0]

                train_df = pd.concat([df.loc[remainder_train_idx], df.loc[fold_dup_train_idx]]) \
                    if fold_dup_train_idx else df.loc[remainder_train_idx].copy()
                val_df = pd.concat([df.loc[remainder_val_idx], df.loc[fold_dup_val_idx]]) \
                    if fold_dup_val_idx else df.loc[remainder_val_idx].copy()

                fold_num = fold_num0 + 1
                print(f"  Fold {fold_num}: {len(fold_dup_val_idx)} duplicate-group samples in val, "
                      f"{len(fold_dup_train_idx)} in train")

                yield (train_df.drop(columns=helper).copy(),
                       val_df.drop(columns=helper).copy(),
                       fold_num)

        else:
            # Standard k-fold (no group_col, no duplicate text at all)
            strat_desc = "condition (3) × text_type (4) × rare_vocab (2) = 24 bins" if has_condition else "text_type (4) × rare_vocab (2) = 8 bins"
            print(f"Stratified {k_folds}-fold: {strat_desc}")
            splitter = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
            split_iter = splitter.split(df, df["_bin"])

            for fold_num, (tr, va) in enumerate(split_iter, 1):
                train_df = df.iloc[tr].copy()
                val_df = df.iloc[va].copy()
                yield (train_df.drop(columns=helper).copy(),
                       val_df.drop(columns=helper).copy(),
                       fold_num)
    else:
        has_groups = group_col is not None

        if has_duplicates or has_groups:
            # Create unified group column combining duplicates + user group_col
            if has_duplicates and has_groups:
                # Combine both: duplicate group + document cluster
                df["_split_group"] = df["_dup_group"].astype(str) + "_" + df[group_col].astype(str)
                print(f"Splitting with duplicate-text awareness + document clustering ('{group_col}')...")
            elif has_duplicates:
                # Only duplicates - assign unique group ID to non-duplicates
                max_dup_group = df["_dup_group"].max()
                df["_split_group"] = df.apply(
                    lambda row: row["_dup_group"] if row["_dup_group"] >= 0 else max_dup_group + 1 + row.name,
                    axis=1
                )
                print("Splitting with duplicate-text-group awareness (keeps duplicate texts together)...")
            else:
                # Only user group_col (document clusters)
                df["_split_group"] = df[group_col]
                print(f"Splitting with document clustering ('{group_col}': {df[group_col].nunique()} clusters)...")

            helper.append("_split_group")

            # Strategy: For each group (duplicate/cluster), assign all members to train or val together
            # 1. Get one representative per group
            # 2. Split representatives with stratification
            # 3. Propagate split assignment to all group members

            # Get one representative per group (use first occurrence)
            group_representatives = df.groupby("_split_group", as_index=False).first()

            # Split representatives with stratification
            # Try full stratification, fall back if bins too small
            # `actual_strat_desc` records which level actually succeeded, so the
            # final summary print reflects reality instead of a static guess.
            #
            # Fallback order (revised 2026-08-30): full (condition x text_type x
            # rare_vocab) -> drop rare_vocab (condition x text_type) -> drop
            # condition too (text_type alone). text_type is the floor here
            # rather than a digit x upper cross, because text_type has no
            # near-empty cell (smallest real category is has_digits at 217/4098,
            # vs. the 6-sample digit-without-upper cell the old cross had) - so
            # falling back to it is a much safer worst case than the old ladder's
            # 4-bin digit x upper floor was.
            full_bins_desc = "condition (3) × text_type (4) × rare_vocab (2) = 24 bins" if has_condition \
                else "text_type (4) × rare_vocab (2) = 8 bins"
            try:
                split_train, split_val = train_test_split(
                    group_representatives,
                    test_size=data_cfg["val_split"],
                    stratify=group_representatives["_bin"],
                    random_state=seed
                )
                actual_strat_desc = full_bins_desc
                print(f"  ✓ Using full stratification ({full_bins_desc})")
            except ValueError:
                # Some bins have <2 samples after grouping - fall back to simpler stratification
                print(f"  ⚠️  Full stratification failed (some bins too small after grouping)")

                if has_condition:
                    try:
                        # Medium fallback: drop rare_vocab, keep condition x text_type (12 bins)
                        group_representatives["_simple_bin"] = (
                            group_representatives["_condition_bin"].astype(str) + "_" +
                            group_representatives["_text_type"].astype(str)
                        )
                        split_train, split_val = train_test_split(
                            group_representatives,
                            test_size=data_cfg["val_split"],
                            stratify=group_representatives["_simple_bin"],
                            random_state=seed
                        )
                        actual_strat_desc = "condition (3) × text_type (4) = 12 bins"
                        print(f"  ✓ Using medium stratification: condition (3) × text_type (4) = 12 bins")
                    except ValueError:
                        # Still too small, fall back to minimal: text_type alone (4 bins)
                        split_train, split_val = train_test_split(
                            group_representatives,
                            test_size=data_cfg["val_split"],
                            stratify=group_representatives["_text_type"],
                            random_state=seed
                        )
                        actual_strat_desc = "text_type (4) = 4 bins"
                        print(f"  ✓ Using minimal stratification: text_type (4) = 4 bins")
                else:
                    # No condition score, fall back straight to text_type alone (4 bins)
                    split_train, split_val = train_test_split(
                        group_representatives,
                        test_size=data_cfg["val_split"],
                        stratify=group_representatives["_text_type"],
                        random_state=seed
                    )
                    actual_strat_desc = "text_type (4) = 4 bins"
                    print(f"  ✓ Using simplified stratification: text_type (4) = 4 bins")

            # Now propagate: which groups went to train vs val?
            train_groups = set(split_train["_split_group"])
            val_groups = set(split_val["_split_group"])

            # Build train/val by group assignment
            train_mask = df["_split_group"].isin(train_groups)
            val_mask = df["_split_group"].isin(val_groups)

            train_df = df[train_mask].copy()
            val_df = df[val_mask].copy()

            # Verify no leakage (for duplicates, only enforced when group_by_text)
            if has_duplicates:
                train_texts = set(train_df["_text_clean"])
                val_texts = set(val_df["_text_clean"])
                leaked = train_texts & val_texts
                if leaked:
                    print(f"  ⚠️  WARNING: {len(leaked)} texts still leaked (should be 0)!")
                else:
                    print(f"  ✓ No duplicate-text leakage - all copies kept together")

            # Verify group separation (for document clusters)
            if has_groups:
                train_clusters = set(train_df[group_col])
                val_clusters = set(val_df[group_col])
                leaked_clusters = train_clusters & val_clusters
                if leaked_clusters:
                    print(f"  ⚠️  WARNING: {len(leaked_clusters)} clusters leaked across train/val!")
                else:
                    print(f"  ✓ No cluster leakage - {len(train_clusters)} clusters in train, {len(val_clusters)} in val")

        else:
            # No group_col, and duplicate-text grouping is off (the new
            # default): each duplicate/near-duplicate TEXT group is
            # explicitly DISTRIBUTED across train and val - not kept
            # together, and not left to chance either. See
            # distribute_duplicate_groups docstring for why this needs to be
            # explicit: verified on this dataset, 22/27 duplicate groups are
            # pairs, and a plain stratified split (ignoring group membership)
            # would put a pair entirely in train 81% of the time by pure
            # chance (0.9² at a 90/10 split) - val would see that recurring
            # phrase 0% of the time in the large majority of cases, purely
            # from how the random split happened to fall.
            if text_dup_exists:
                forced = distribute_duplicate_groups(df, data_cfg["val_split"], seed)
                forced_val_idx = forced[forced == "val"].index
                forced_train_idx = forced[forced == "train"].index
                remainder_df = df.drop(index=forced.dropna().index)
                n_groups = int(df.loc[df["_dup_group"] >= 0, "_dup_group"].nunique())

                # Adjust the remainder's val fraction so the OVERALL val
                # count (forced + remainder) still lands close to the
                # configured val_split, rather than the forced assignments
                # silently skewing the total ratio. Duplicates are a small
                # slice of this corpus (~2%) so the skew would be minor
                # either way, but correcting for it exactly is cheap.
                target_val_n = round(len(df) * data_cfg["val_split"])
                remaining_val_needed = max(0, target_val_n - len(forced_val_idx))
                remainder_val_split = (
                    min(1.0, remaining_val_needed / len(remainder_df)) if len(remainder_df) else 0.0
                )

                if remainder_val_split <= 0:
                    remainder_train_df, remainder_val_df = remainder_df, remainder_df.iloc[0:0]
                elif remainder_val_split >= 1:
                    remainder_train_df, remainder_val_df = remainder_df.iloc[0:0], remainder_df
                else:
                    try:
                        remainder_train_df, remainder_val_df = train_test_split(
                            remainder_df, test_size=remainder_val_split,
                            stratify=remainder_df["_bin"], random_state=seed
                        )
                    except ValueError:
                        # Removing duplicate-group members shrank some _bin cell
                        # below 2 - fall back to the coarsest stratification
                        # level (text_type alone) for just this remainder split.
                        print(f"  ⚠️  Full stratification on the remainder failed after "
                              f"removing duplicate-group members - falling back to text_type alone")
                        remainder_train_df, remainder_val_df = train_test_split(
                            remainder_df, test_size=remainder_val_split,
                            stratify=remainder_df["_text_type"], random_state=seed
                        )

                train_df = pd.concat([df.loc[forced_train_idx], remainder_train_df])
                val_df = pd.concat([df.loc[forced_val_idx], remainder_val_df])

                print(f"  ✓ Distributed {len(forced_val_idx) + len(forced_train_idx)} samples from "
                      f"{n_groups} duplicate-text groups across train/val "
                      f"({len(forced_val_idx)} val / {len(forced_train_idx)} train) - "
                      f"every group with 2+ members has at least one exemplar on each side")
            else:
                train_df, val_df = train_test_split(
                    df, test_size=data_cfg["val_split"], stratify=df["_bin"], random_state=seed
                )

            actual_strat_desc = "condition (3) × text_type (4) × rare_vocab (2) = 24 bins" if has_condition \
                else "text_type (4) × rare_vocab (2) = 8 bins"

        # Log distributions to verify stratification is working
        train_digits = train_df["_digit_density"]
        val_digits = val_df["_digit_density"]
        train_upper = train_df["_uppercase_ratio"]
        val_upper = val_df["_uppercase_ratio"]
        train_lex = train_df["_lexical_diversity"]
        val_lex = val_df["_lexical_diversity"]
        train_spec = train_df["_special_char_density"]
        val_spec = val_df["_special_char_density"]
        train_wlen = train_df["_avg_word_length"]
        val_wlen = val_df["_avg_word_length"]

        # Get condition scores for logging (if available)
        has_condition_logging = "condition_score" in train_df.columns and not train_df["condition_score"].isna().all()

        # print what actually ran (tracked per-branch above), not a static
        # recomputation that always claims the full bin scheme even when a
        # fallback (medium/minimal) fired due to small-bin ValueErrors.
        print(f"Stratified split: {actual_strat_desc}")
        print(f"  Train: {len(train_df)} samples")

        # Visual condition distribution
        if has_condition_logging:
            train_cond = train_df["condition_score"]
            print(f"    Visual condition: min={train_cond.min():.1f}, median={train_cond.median():.1f}, max={train_cond.max():.1f}")

        # text_type and rare_vocab distributions - the two factors actually
        # driving the stratification key now (difficulty_score removed 2026-08-30)
        train_text_type = train_df["_text_type"].value_counts(normalize=True).round(3).to_dict()
        val_text_type = val_df["_text_type"].value_counts(normalize=True).round(3).to_dict()
        train_rare_pct = train_df["_has_rare"].mean() * 100
        val_rare_pct = val_df["_has_rare"].mean() * 100
        # Caret sub-type counts - DIAGNOSTIC ONLY, not part of the stratification
        # key (see classify_caret_types docstring: interlineation is too rare,
        # ~23-25 total, to safely stratify on). Raw counts rather than
        # percentages here deliberately, since a percentage would overstate
        # precision on populations this small.
        train_interlin_n = int(train_df["_has_interlineation"].sum())
        val_interlin_n = int(val_df["_has_interlineation"].sum())
        train_superscr_n = int(train_df["_has_superscript"].sum())
        val_superscr_n = int(val_df["_has_superscript"].sum())
        print(f"    Text type: {train_text_type}")
        print(f"    Rare vocab: {train_rare_pct:.1f}%")
        print(f"    Caret sub-types: interlineation={train_interlin_n}, superscript={train_superscr_n}")
        print(f"    Digit density: min={train_digits.min():.3f}, median={train_digits.median():.3f}, max={train_digits.max():.3f}")
        print(f"    Uppercase ratio: min={train_upper.min():.3f}, median={train_upper.median():.3f}, max={train_upper.max():.3f}")
        print(f"    Lexical diversity: min={train_lex.min():.3f}, median={train_lex.median():.3f}, max={train_lex.max():.3f}")
        print(f"    Text length: min={train_df['_text_len'].min()}, median={train_df['_text_len'].median():.0f}, max={train_df['_text_len'].max()}")
        print(f"  Val:   {len(val_df)} samples")

        # Visual condition distribution
        if has_condition_logging:
            val_cond = val_df["condition_score"]
            print(f"    Visual condition: min={val_cond.min():.1f}, median={val_cond.median():.1f}, max={val_cond.max():.1f}")

        print(f"    Text type: {val_text_type}")
        print(f"    Rare vocab: {val_rare_pct:.1f}%")
        print(f"    Caret sub-types: interlineation={val_interlin_n}, superscript={val_superscr_n}")
        print(f"    Digit density: min={val_digits.min():.3f}, median={val_digits.median():.3f}, max={val_digits.max():.3f}")
        print(f"    Uppercase ratio: min={val_upper.min():.3f}, median={val_upper.median():.3f}, max={val_upper.max():.3f}")
        print(f"    Lexical diversity: min={val_lex.min():.3f}, median={val_lex.median():.3f}, max={val_lex.max():.3f}")
        print(f"    Text length: min={val_df['_text_len'].min()}, median={val_df['_text_len'].median():.0f}, max={val_df['_text_len'].max()}")

        yield train_df.drop(columns=helper), val_df.drop(columns=helper), None