# K-Fold Logging Improvements

Cleaned up logging output to make K-Fold training easier to follow.

## Changes Made

### 1. Reduced Library Logging Noise

**Before:** Verbose transformers/accelerate logs cluttered output

**After:**
```python
# Set to WARNING level for K-Fold
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("accelerate").setLevel(logging.WARNING)
```

### 2. Less Frequent Training Logs

**Before:** `logging_steps=10` (logs every 10 steps)

**After:**
```python
logging_steps = 50 if is_kfold else 10  # Less noise for K-Fold
```

### 3. Disabled Progress Bars for K-Fold

**Before:** Multiple progress bars per fold cluttered output

**After:**
```python
disable_tqdm=is_kfold  # Clean K-Fold output
```

### 4. Cleaner Fold Headers

**Before:**
```
Fold 1 - Train: 3278, Val: 820
```

**After:**
```
======================================================================
🔄 Fold 1/5 - Train: 3278, Val: 820
======================================================================

🚀 Training Fold 1/5... (progress bars disabled for cleaner K-Fold output)
```

### 5. Better Fold Completion Messages

**Before:**
```
Fold 1 - Best eval_loss: 0.5842
```

**After:**
```
======================================================================
✅ Fold 1/5 Complete - Best eval_loss: 0.5842
   Saved to: outputs/qwen3-2b-kfold/fold_1/best
======================================================================
```

### 6. Enhanced Summary Display

**Before:**
```
K-FOLD RESULTS SUMMARY
Fold 1: eval_loss=0.5842 (train=3278, val=820)
Fold 2: eval_loss=0.5791 (train=3278, val=820)
...
Average eval_loss: 0.5820 ± 0.0025
```

**After:**
```
======================================================================
🎯 K-FOLD CROSS-VALIDATION RESULTS
======================================================================

  Fold 1/5: eval_loss = 0.5842
  Fold 2/5: eval_loss = 0.5791
  Fold 3/5: eval_loss = 0.5834
  Fold 4/5: eval_loss = 0.5809
  Fold 5/5: eval_loss = 0.5823

  ──────────────────────────────────────────────────────────────────
  📊 Average:  0.5820 ± 0.0025
  📈 Best:     0.5791 (Fold 2)
  📉 Worst:    0.5842 (Fold 1)

======================================================================
✅ All folds complete! Use --kfold flag for ensemble inference.
======================================================================
```

### 7. Initial K-Fold Banner

**Before:**
```
K-FOLD CROSS-VALIDATION: 5 folds
Total samples: 4098
```

**After:**
```
======================================================================
🔄 K-FOLD CROSS-VALIDATION
======================================================================
  Folds: 5
  Total samples: 4098
  Per fold: ~3278 train, ~820 val
  Progress bars: Disabled for cleaner output
======================================================================
```

---

## What You'll See Now

### During Training (Each Fold)

```
======================================================================
🔄 Fold 1/5 - Train: 3278, Val: 820
======================================================================

🚀 Training Fold 1/5... (progress bars disabled for cleaner K-Fold output)

Loading model: Qwen/Qwen3-VL-2B-Instruct
✓ Using Flash Attention 2
trainable params: 104,595,456 || all params: 2,232,127,488

[Minimal training logs - only every 50 steps]

======================================================================
✅ Fold 1/5 Complete - Best eval_loss: 0.5842
   Saved to: outputs/qwen3-2b-kfold/fold_1/best
======================================================================
```

### After All Folds Complete

```
======================================================================
🎯 K-FOLD CROSS-VALIDATION RESULTS
======================================================================

  Fold 1/5: eval_loss = 0.5842
  Fold 2/5: eval_loss = 0.5791
  Fold 3/5: eval_loss = 0.5834
  Fold 4/5: eval_loss = 0.5809
  Fold 5/5: eval_loss = 0.5823

  ──────────────────────────────────────────────────────────────────
  📊 Average:  0.5820 ± 0.0025
  📈 Best:     0.5791 (Fold 2)
  📉 Worst:    0.5842 (Fold 1)

======================================================================
✅ All folds complete! Use --kfold flag for ensemble inference.
======================================================================

📄 Summary saved to: outputs/qwen3-2b-kfold/kfold_summary.txt
```

---

## Benefits

✅ **Cleaner output** - No progress bar clutter
✅ **Easier to track** - Clear fold numbers (1/5, 2/5, etc.)
✅ **Key info prominent** - eval_loss, best/worst folds highlighted
✅ **Less noise** - Library warnings suppressed
✅ **Better summary** - Shows average, best, worst with emojis

---

## Single Model Training (Unchanged)

When `k_folds=1`, you still get:
- Detailed training schedule
- Normal logging frequency (every 10 steps)
- Progress bars enabled
- Full verbosity

The improvements only apply to K-Fold mode for cleaner multi-fold training.
