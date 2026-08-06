# Ensemble Strategy Analysis

## Problem Discovered

**K-fold ensemble (char voting): 0.850 score**
**Single fold 1 alone: 0.855 score**

The ensemble is performing **worse** than a single fold! This indicates character-level voting is hurting rather than helping.

---

## Why Character-Level Voting Can Degrade Performance

### Example of Alignment Problem

**Fold 1**: `"Signed Sealed and delivered"`
**Fold 2**: `"Signed Sealed delivered"`
**Fold 3**: `"Signed and Sealed delivered"`

Character-level voting at each position:
```
Position 0-6: "Signed " ✓ (all agree)
Position 7:   'S' (fold 1,2) vs 'a' (fold 3) → 'S' wins
Position 8:   'e' (fold 1,2) vs 'n' (fold 3) → 'e' wins
...
Result: "Signed Sealed..." but positions are misaligned!
```

**Problem**: Once predictions differ in length or have insertions/deletions, character positions no longer correspond. Voting creates gibberish.

### Dilution Effect

If fold 1 has excellent predictions but folds 2-5 are mediocre, character voting dilutes the good predictions:

- **Without ensemble**: Use fold 1's great prediction → 0.855 score
- **With char voting**: Mix great + mediocre → 0.850 score (worse!)

---

## Available Ensemble Strategies

### 1. `char_voting` (Default, but problematic)

**How it works**: Vote for most common character at each position after padding.

**Pros**:
- Theoretically can fix single-character errors

**Cons**:
- Alignment issues when predictions differ in length
- Dilutes good predictions when quality varies
- Can create non-words

**Use when**: All folds are equally good and predictions differ only by minor typos

---

### 2. `majority` (Recommended alternative)

**How it works**: Pick the most common **full prediction** across folds.

**Example**:
```
Fold 1: "Signed Sealed and delivered"
Fold 2: "Signed Sealed and delivered"
Fold 3: "Signed and Sealed delivered"
Fold 4: "Signed Sealed and delivered"
Fold 5: "Signed Sealed and delivered"

Result: "Signed Sealed and delivered" (appears 4 times)
```

**Pros**:
- No alignment issues
- Preserves coherent predictions
- Simple and robust

**Cons**:
- If all predictions are unique, picks first one (arbitrary)

**Use when**: Folds mostly agree with minor variations

---

### 3. `longest` (Defensive strategy)

**How it works**: Pick the longest prediction.

**Rationale**: Historical documents often have complex text. Longer predictions likely captured more detail.

**Pros**:
- Simple
- Avoids truncated predictions

**Cons**:
- Can pick overly verbose or hallucinated text
- No quality weighting

**Use when**: Folds tend to under-predict (missing words)

---

### 4. `shortest` (Conservative strategy)

**How it works**: Pick the shortest prediction.

**Rationale**: Shorter predictions are more confident (model stopped generating).

**Pros**:
- Avoids hallucination
- More confident

**Cons**:
- May miss text
- Too conservative

**Use when**: Folds tend to over-predict (hallucinate text)

---

### 5. `first` (Baseline / No ensemble)

**How it works**: Use fold 1's predictions only.

**Rationale**: If one fold is clearly better, just use it!

**Pros**:
- No ensemble degradation
- Fast (no voting)

**Cons**:
- Throws away other folds' work
- If fold 1 got lucky, not reproducible

**Use when**: Fold 1 is clearly the best (as in your case: 0.855 vs 0.850)

---

## Usage

### Test Different Strategies

```bash
# Strategy 1: Character voting (current default, but problematic)
python inference.py --config config_qwen3_8b.yaml --kfold --ensemble-strategy char_voting

# Strategy 2: Majority voting (recommended)
python inference.py --config config_qwen3_8b.yaml --kfold --ensemble-strategy majority

# Strategy 3: Longest prediction
python inference.py --config config_qwen3_8b.yaml --kfold --ensemble-strategy longest

# Strategy 4: Shortest prediction
python inference.py --config config_qwen3_8b.yaml --kfold --ensemble-strategy shortest

# Strategy 5: First fold only (your current best)
python inference.py --config config_qwen3_8b.yaml --kfold --ensemble-strategy first
```

### Analyze Fold Agreement

Run analysis to see where folds disagree:

```bash
python analyze_folds.py --config config_qwen3_8b.yaml
```

**Output**:
- Agreement statistics (% of images where all folds agree)
- Examples of disagreement
- Comparison of all strategies
- Generates `submission_{strategy}.csv` files for testing

---

## Recommendations

### Step 1: Analyze Your Folds

```bash
python analyze_folds.py --config config_qwen3_8b.yaml
```

Check:
- **High agreement (>80%)**: Folds are consistent → `majority` strategy likely best
- **Low agreement (<50%)**: Folds disagree heavily → investigate fold quality

### Step 2: Test All Strategies

Submit all generated CSVs to the platform:
1. `submission_first.csv` - Single fold (your current best: 0.855)
2. `submission_majority.csv` - Most common full prediction
3. `submission_char_voting.csv` - Current ensemble (0.850)
4. `submission_longest.csv` - Longest prediction
5. `submission_shortest.csv` - Shortest prediction

### Step 3: Interpret Results

**If `first` is still best**:
- Fold 1 just got lucky with train/val split
- Other folds might be overfitted differently
- Consider: Train more folds, pick best 2-3 for ensemble

**If `majority` beats `first`**:
- Folds are mostly consistent
- Ensemble is working correctly
- Use `majority` going forward

**If `longest` or `shortest` wins**:
- Folds have systematic bias (under/over-predicting)
- Investigate training (might be stopping too early/late)

---

## Why Fold 1 Might Be Best

### Possible Reasons

1. **Random split luck**: Fold 1's validation data happened to be more representative of test set
2. **Overfitting variance**: Other folds overfitted to their specific validation data differently
3. **Training instability**: Early stopping triggered at different points for each fold
4. **Data stratification**: Stratified split by text length might have grouped "easier" samples in fold 1

### What to Do

**Short-term**: Just use fold 1 if it's clearly best (0.855 > 0.850)

**Long-term**:
1. Train more folds (10 folds instead of 5)
2. Select top 3-5 folds by validation performance
3. Ensemble only the best folds (not all of them)
4. Use `majority` voting on the selected folds

---

## Advanced: Quality-Weighted Ensemble

**Not yet implemented**, but could help:

```python
# Weight by validation CER/WER
fold_weights = [0.054, 0.058, 0.062, 0.056, 0.060]  # Lower = better
weights = [1/cer for cer in fold_weights]
weights = [w/sum(weights) for w in weights]  # Normalize

# Weighted voting
for fold, weight in zip(folds, weights):
    # Give higher weight to better folds
    ...
```

This would prevent bad folds from dragging down the ensemble.

---

## Summary

**Your finding is valuable**: It reveals that character-level voting doesn't work well for this task!

**Immediate actions**:
1. ✅ Use `--ensemble-strategy first` for now (matches your 0.855 score)
2. ✅ Run `analyze_folds.py` to understand fold disagreement
3. ✅ Test `--ensemble-strategy majority` (might beat 0.855 if folds mostly agree)

**If majority voting still loses to first fold**:
- Fold 1 got lucky
- Train more folds and select the best ones
- Ensemble only high-quality folds
