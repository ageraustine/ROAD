# Early Stopping Metric Fix

## Problem Identified

**Original Bug:**
Early stopping was monitoring the same metric as `metric_for_best_model`, which could be `eval_cer`.

**Why This Was Wrong:**
- `eval_cer`: Computed on 200 samples, generative (noisy), high variance
- Noisy signals cause false plateaus → premature stopping
- Or mask real plateaus → training continues unnecessarily

## Solution

**Separate the two decisions:**

### 1. Early Stopping → Always monitor `eval_loss`
```yaml
early_stopping:
  metric: "eval_loss"  # Stable, deterministic, low variance
  patience: 3
```

**Why eval_loss:**
- Computed on entire validation set (800+ samples)
- Teacher-forced (deterministic)
- Low variance → reliable plateau detection

### 2. Best Model Selection → Can use `eval_cer`
```yaml
metric_for_best_model: "eval_cer"  # What matters for leaderboard
```

**Why eval_cer for best model:**
- Matches competition metric (WER/CER)
- Generative (reflects real inference)
- Variance is okay for picking minimum

## Changes Made

### 1. train.py (lines 758-777)
```python
# OLD (WRONG):
early_stop_metric = metric  # Inherited from metric_for_best_model

# NEW (CORRECT):
early_stop_metric = early_stop_cfg.get("metric", "eval_loss")  # Defaults to eval_loss
```

### 2. All config files
Added explicit `metric: "eval_loss"` to early_stopping sections:
- `config.yaml`
- `config_qwen3_2b.yaml`
- `config_qwen3_8b.yaml`

## Example Trajectories

### With Noisy eval_cer Monitoring (WRONG)
```
Step 50:  eval_cer=0.054, eval_loss=0.560
Step 100: eval_cer=0.056, eval_loss=0.555  ← CER worse (noise), patience +1
Step 150: eval_cer=0.055, eval_loss=0.552  ← CER better, patience reset
Step 200: eval_cer=0.057, eval_loss=0.551  ← CER worse (noise), patience +1
Step 250: eval_cer=0.056, eval_loss=0.550  ← CER worse, patience +2
Step 300: eval_cer=0.058, eval_loss=0.550  ← CER worse, patience +3 → STOP

STOPPED TOO EARLY - eval_loss was still improving until step 250!
```

### With Stable eval_loss Monitoring (CORRECT)
```
Step 50:  eval_loss=0.560, eval_cer=0.054
Step 100: eval_loss=0.555, eval_cer=0.056  ← Improving
Step 150: eval_loss=0.552, eval_cer=0.055  ← Improving
Step 200: eval_loss=0.551, eval_cer=0.057  ← Improving
Step 250: eval_loss=0.550, eval_cer=0.056  ← Improving
Step 300: eval_loss=0.550, eval_cer=0.053  ← Plateau, patience +1
Step 350: eval_loss=0.550, eval_cer=0.054  ← Plateau, patience +2
Step 400: eval_loss=0.550, eval_cer=0.052  ← Plateau, patience +3 → STOP

Best model saved: eval_cer=0.052 (from step 400)
STOPPED AT TRUE PLATEAU - eval_loss converged, picked best CER checkpoint
```

## Verification

Check your training logs:
```bash
grep "early_stopping:" outputs/qwen3-8b-v1/fold_1/logs.txt
```

Should see:
```
early_stopping: metric=eval_loss, patience=3, min_delta=0.0001
```

NOT:
```
early_stopping: metric=eval_cer, patience=3, min_delta=0.0001
```

## Impact

**For your 0.56 results:**
- If early stopping was monitoring eval_cer, it might have stopped prematurely
- Or continued training past plateau (noise masked convergence)
- With eval_loss monitoring: More reliable training termination

**Expected improvement:**
- More consistent fold results (lower variance)
- Better use of compute (stop at true plateau, not noise)
- No change to final performance (same best checkpoint selection)
