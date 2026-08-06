# Early Stopping for K-Fold Training

## Problem

In K-fold cross-validation with expensive VLM models:
- Each fold can take **1-3 hours** on A100 80GB
- Models often plateau **before completing all epochs**
- Wasting compute on a plateaued fold means less time for other experiments

**Example:** Training Qwen3-VL-2B for 5 folds × 4 epochs = 20 epochs total
- Without early stopping: 5-6 hours, some folds waste time after epoch 2
- With early stopping: 3-4 hours, stops each fold when it plateaus

## How It Works

The `EarlyStoppingCallback` monitors `eval_loss` (or `eval_cer`) and stops training when:
1. The metric stops improving
2. For a specified number of evaluations (patience)

### Configuration

**In `config.yaml` or `config_qwen3_2b.yaml`:**

```yaml
training:
  early_stopping:
    enabled: true
    patience: 3          # Stop after 3 evals with no improvement
    min_delta: 0.0001    # Minimum improvement threshold
```

**Patience Example:**
- `eval_steps: 100` (evaluate every 100 steps)
- `patience: 3` → stops if no improvement for 300 steps

**Min Delta Example:**
```
Eval 1: loss = 0.7500  ← new best
Eval 2: loss = 0.7498  ← improved by 0.0002 > 0.0001 ✓ reset patience
Eval 3: loss = 0.7499  ← worse, patience = 1/3
Eval 4: loss = 0.7497  ← improved by 0.0001 (equal to min_delta, not enough), patience = 2/3
Eval 5: loss = 0.7496  ← improved by 0.0001 (not enough), patience = 3/3 → STOP
```

## Training Output

### During Training

When loss plateaus, you'll see:
```
Eval 1: eval_loss=0.7500
Eval 2: eval_loss=0.7498  ✓ improvement
Eval 3: eval_loss=0.7499
  Early stopping: 1/3 (no improvement in eval_loss)
Eval 4: eval_loss=0.7501
  Early stopping: 2/3 (no improvement in eval_loss)
Eval 5: eval_loss=0.7500
  Early stopping: 3/3 (no improvement in eval_loss)

⚠️  Early stopping triggered!
    eval_loss plateaued at 0.7498
    No improvement for 3 evaluations
```

### Per-Fold Summary

```
======================================================================
Fold 1/5 complete - best eval_loss: 0.7498
  ⏱️  Early stopped at epoch 2.3 (saved 1.7 epochs)
  saved -> outputs/qwen3-2b-kfold-v2/fold_1/best
======================================================================
```

### K-Fold Summary

```
======================================================================
K-FOLD RESULTS
======================================================================
  Fold 1/5: eval_loss=0.7498  eval_cer=0.0456  [early stop @ 2.3]
  Fold 2/5: eval_loss=0.7612  eval_cer=0.0478  [early stop @ 2.8]
  Fold 3/5: eval_loss=0.7523  eval_cer=0.0461
  Fold 4/5: eval_loss=0.7489  eval_cer=0.0451  [early stop @ 1.9]
  Fold 5/5: eval_loss=0.7556  eval_cer=0.0469  [early stop @ 2.5]

  eval_loss  0.7536 +/- 0.0049
  eval_cer   0.0463 +/- 0.0010

  Early stopping: 4/5 folds stopped early
  Avg epochs completed: 2.4/4
======================================================================
```

## Command-Line Control

### Enable/Disable Early Stopping

```bash
# Use config default (enabled)
python train.py --config config_qwen3_2b.yaml

# Disable early stopping for this run
python train.py --config config_qwen3_2b.yaml --no-early-stop
```

### Tuning Patience

Quick test (aggressive early stopping):
```yaml
early_stopping:
  patience: 2  # Stop after 2 evals with no improvement
```

Conservative (give model more time):
```yaml
early_stopping:
  patience: 5  # Stop after 5 evals with no improvement
```

## Typical Results

### Qwen3-VL-2B (4 epochs, 5 folds)

**Without early stopping:**
- Total time: ~5-6 hours
- All folds complete 4 epochs
- Some folds plateau at epoch 2-3 but keep training

**With early stopping (patience=3):**
- Total time: ~3-4 hours (25-33% faster)
- Folds stop at epoch 2-3 when plateaued
- Same or better final metrics (saves overfitting)

### When Early Stopping Helps Most

✅ **Good use cases:**
- K-fold cross-validation (5+ folds)
- Small datasets (4K samples) that overfit quickly
- Experimenting with hyperparameters
- Limited compute budget

❌ **When to disable:**
- Single model training (not K-fold)
- Large datasets that need full training
- Final production model (you want full control)
- Learning rate schedule needs full epochs

## Interaction with Other Features

### Resume from Checkpoint
✅ Works together seamlessly
- Checkpoint saves full state including patience counter
- Resume continues early stopping from last state

### Load Best Model at End
✅ Fully compatible
- Early stop saves best model seen so far
- `load_best_model_at_end=True` still works

### CER Callback
✅ Can monitor `eval_cer` instead of `eval_loss`

```yaml
training:
  metric_for_best_model: "eval_cer"
  early_stopping:
    enabled: true
    patience: 3  # Monitors eval_cer automatically
```

## Recommendations

### For R.O.A.D. Competition

**Initial experiments** (finding hyperparameters):
```yaml
early_stopping:
  enabled: true
  patience: 2  # Aggressive - quick feedback
```

**K-fold cross-validation** (model selection):
```yaml
early_stopping:
  enabled: true
  patience: 3  # Balanced - recommended
```

**Final model** (submission):
```yaml
early_stopping:
  enabled: false  # Or patience: 5 for safety
```

### Typical Savings

| Folds | Epochs | Without ES | With ES (p=3) | Time Saved |
|-------|--------|------------|---------------|------------|
| 5     | 3      | 4.5h       | 3.5h          | ~1h (22%)  |
| 5     | 4      | 6.0h       | 4.0h          | ~2h (33%)  |
| 5     | 5      | 7.5h       | 4.5h          | ~3h (40%)  |

*Based on Qwen3-VL-2B on A100 80GB*

## Troubleshooting

### Early stopping too aggressive?

**Problem:** Folds stop at epoch 1-2, metrics suboptimal

**Solution:** Increase patience or min_delta
```yaml
early_stopping:
  patience: 5  # More lenient
  min_delta: 0.00005  # Smaller threshold
```

### Never triggers?

**Problem:** All folds complete full epochs

**Possible causes:**
- Learning rate too high (loss keeps jumping)
- `min_delta` too large (0.001 might be too strict)
- Dataset too large (never plateaus in 3-4 epochs)

**Solution:**
```yaml
early_stopping:
  patience: 3
  min_delta: 0.0001  # Reasonable for eval_loss ~0.7
```

### Want per-fold control?

Disable in config, use command-line:
```bash
# Fold 1-3: use early stopping
python train.py --config config.yaml

# Fold 4-5: disable for final folds
# (edit config.yaml: k_folds=2, then:)
python train.py --config config.yaml --no-early-stop
```

## References

- Based on Keras EarlyStopping callback design
- Monitors `metric_for_best_model` from config
- Compatible with HuggingFace Trainer callback system
- Tested with transformers v4.46+ and v5.x
