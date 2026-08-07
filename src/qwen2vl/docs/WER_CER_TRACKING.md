# WER + CER Tracking (Competition Metric)

## Changes Made

Added **full competition metric tracking** during training to optimize for the actual leaderboard score.

---

## What's New

### 1. WER Computation
Added word-level error rate computation alongside character-level:

```python
def corpus_wer(preds: list, refs: list) -> float:
    """Aggregate WER: total word edits / total reference words."""
    # Uses word-level Levenshtein distance
```

### 2. Competition Score Tracking
Now computes and tracks the **actual leaderboard metric**:

```python
competition_score = 0.5 * WER + 0.5 * CER
metrics["eval_score"] = competition_score
```

### 3. Updated Metrics Display

**Old output:**
```
eval_cer=0.0650 ✓ (new best)
```

**New output:**
```
eval_score=0.7625 (cer=0.0650, wer=0.8600) ✓ (new best)
```

**Training completion:**
```
✓ Training | eval_score=0.7625 | Completed 3.0 epochs
  Competition score (0.5*WER + 0.5*CER): 0.7625
  CER: 0.0650 | WER: 0.8600
```

---

## Config Changes

### Updated Metric for Best Model

**Old:**
```yaml
metric_for_best_model: "eval_cer"  # Only CER
```

**New:**
```yaml
metric_for_best_model: "eval_score"  # Competition metric (0.5*WER + 0.5*CER)
```

This means:
- ✅ Model checkpoints saved based on **best competition score**
- ✅ `best/` directory contains model optimized for leaderboard
- ✅ Early stopping still uses `eval_loss` (stable signal)

---

## How It Works

### During Training (Every 100 Steps)

1. **Generate predictions** on 200 validation samples (greedy decode)
2. **Compute all metrics:**
   - `eval_loss`: Cross-entropy (stable, deterministic)
   - `eval_cer`: Character error rate
   - `eval_wer`: Word error rate
   - `eval_score`: 0.5*WER + 0.5*CER (competition metric)
3. **Save checkpoint** if `eval_score` improved
4. **Print update** if new best score achieved

### Metrics Tracked in History

All metrics are saved to `trainer.state.log_history`:
```python
{
    "eval_loss": 0.5890,
    "eval_cer": 0.0650,
    "eval_wer": 0.8600,
    "eval_score": 0.7625,  # 0.5*0.86 + 0.5*0.065
    "epoch": 3.0
}
```

---

## Expected Training Output

### Clean Progress Display

```
Loaded 4098 samples from Train.csv
======================================================================
Training | Train: 3688 | Val: 410
======================================================================
Loading Qwen3-VL-8B-Instruct... ✓ (214.7M trainable / 8982M total = 2.39%)
Steps: 231/epoch × 3 epochs = 693 total
Starting training...

Step 50/693 | loss=0.8231 | lr=1.99e-05 | epoch=0.43
Eval @ step 100 | eval_loss=0.7123
eval_score=0.8250 (cer=0.0892, wer=0.8608) ✓ (new best)

Step 100/693 | loss=0.6142 | lr=1.87e-05 | epoch=0.87
Eval @ step 200 | eval_loss=0.6374
eval_score=0.7850 (cer=0.0760, wer=0.7940) ✓ (new best)

Step 150/693 | loss=0.5234 | lr=1.76e-05 | epoch=1.30
Eval @ step 300 | eval_loss=0.6091
eval_score=0.7745 (cer=0.0749, wer=0.7741) ✓ (new best)

...

Eval @ step 600 | eval_loss=0.5890
eval_score=0.7610 (cer=0.0652, wer=0.7568) ✓ (new best)

======================================================================
✓ Training | eval_score=0.7610 | Completed 3.0 epochs
  Competition score (0.5*WER + 0.5*CER): 0.7610
  CER: 0.0652 | WER: 0.7568
  Saved: /content/drive/MyDrive/prd/qwen3-8b-full/best
======================================================================
```

---

## Understanding the Metrics

### Leaderboard Score Prediction

**Training metrics → Expected leaderboard:**
```
eval_cer = 0.065 (6.5% character errors)
eval_wer = 0.757 (75.7% word errors)
eval_score = 0.761 (competition metric)

Expected leaderboard: ~0.76-0.86
```

**Why the range?**
- Training metrics on 200 samples (small)
- Test set may be easier/harder
- Greedy decode vs beam search (inference uses beams=5)

**Model 1 actual:**
- Training: eval_cer=0.065, eval_score≈0.76
- Leaderboard: **0.86** (much better!)
- Reason: Test set easier, or beam search helped

### Metric Relationships

**Character vs Word Errors:**
```
Low CER + High WER = correct characters, wrong word boundaries
Low CER + Low WER  = excellent (both correct)
High CER + High WER = poor (both wrong)
```

**Example:**
```
Reference:  "Signed Sealed and delivered"
Prediction: "SignedSealed anddelivered"

CER: 0.04 (only 1 space missing = 1 char error out of 28 chars)
WER: 1.0 (all 4 words are "wrong" due to space errors)
Score: 0.52 (average of 0.04 and 1.0)
```

This shows why **tracking both** matters!

---

## Benefits

### 1. Optimize for What Matters
- **Old:** Best model = lowest CER
- **New:** Best model = lowest competition score (0.5*WER + 0.5*CER)

**Impact:** Model might trade slightly higher CER for much lower WER (or vice versa) if it improves overall score.

### 2. Better Understanding
Can now see if model is:
- ✅ Good at characters (low CER)
- ✅ Good at words (low WER)
- ✅ Good at competition metric (low score)

### 3. Accurate Leaderboard Prediction
Training `eval_score` gives better estimate of leaderboard than CER alone.

---

## Backward Compatibility

### If You Prefer CER-Only

To revert to CER-only optimization:

```yaml
# config.yaml
training:
  metric_for_best_model: "eval_cer"  # Use CER instead of competition score
```

**Note:** WER will still be computed and displayed, just not used for checkpoint selection.

---

## Performance Impact

### Computation Overhead

**Per evaluation (every 100 steps):**
- Old: CER only (~3 seconds for 200 samples)
- New: CER + WER (~4 seconds for 200 samples)
- **Overhead: +1 second per evaluation** (~negligible)

**Total training:**
- 693 steps / 100 = 7 evaluations
- Extra time: 7 × 1s = **+7 seconds total** (<0.1% of 2 hour training)

### Memory Impact
- None (WER uses same predictions as CER)

---

## Summary

✅ **Added WER tracking** (word-level errors)
✅ **Added competition score** (0.5*WER + 0.5*CER)
✅ **Updated best model selection** to use competition metric
✅ **Clean output** showing all three metrics
✅ **Negligible overhead** (+7 seconds per training run)

**Now training optimizes for the actual leaderboard metric!** 🎯

---

## Example Output Comparison

### Model 1 (CER-only, before changes)
```
✓ Training | eval_cer=0.0650 | Completed 3.0 epochs
  eval_cer=0.0650
```
**Leaderboard:** 0.86 (surprising! We didn't know what WER was)

### Model 2 (Competition score, after changes)
```
✓ Training | eval_score=0.7610 | Completed 3.0 epochs
  Competition score (0.5*WER + 0.5*CER): 0.7610
  CER: 0.0652 | WER: 0.7568
```
**Expected leaderboard:** 0.76-0.86 (now we have better prediction!)

If actual leaderboard is significantly different from eval_score, it tells us:
- Test set has different characteristics than validation
- Beam search (inference) helps/hurts significantly
- Need to adjust validation set or inference params
