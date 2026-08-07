# Resumable K-Fold Inference

## Problem Solved

**Before:** K-fold ensemble inference takes ~1.7 hours for 5 folds. If it crashes or disconnects (common on Colab), you lose all progress and must restart from fold 1.

**Now:** Each fold's predictions are saved to disk after completion. If interrupted, resume from the last completed fold automatically.

---

## How It Works

### Caching Mechanism

```
outputs/qwen3-8b-v1/
├── fold_1/
│   └── best/           # Model checkpoint
├── fold_2/
│   └── best/
├── ...
└── inference_cache/    # ← NEW: Prediction cache
    ├── fold_1_predictions.json
    ├── fold_2_predictions.json
    ├── fold_3_predictions.json
    └── ...
```

**Each fold's predictions saved as JSON:**
```json
{
  "ABC123": "Signed Sealed and delivered",
  "XYZ789": "This done and protested the day",
  ...
}
```

### Resume Logic

```python
for fold_num in [1, 2, 3, 4, 5]:
    json_cache = f"inference_cache/fold_{fold_num}_predictions.json"
    csv_cache = f"inference_cache/fold_{fold_num}_predictions.csv"

    if json_cache exists:
        # Load JSON cache (instant, preferred format)
        predictions = load_json(json_cache)
    elif csv_cache exists:
        # Load CSV cache and convert to JSON
        predictions = load_csv(csv_cache)
        save_json(json_cache, predictions)  # Convert for future speed
    else:
        # Run inference (~20 minutes)
        model = load_model(fold_checkpoint)
        predictions = run_inference(model, test_images)
        # Save to cache
        save_json(json_cache, predictions)
```

### CSV Import Support

**NEW**: You can now import existing predictions from CSV files!

If you already have predictions for some folds (e.g., from a previous run or different script), place them in the inference cache directory as CSV:

```
outputs/qwen3-8b-v1/inference_cache/
└── fold_1_predictions.csv  # Your existing results
```

**CSV Format** (same as submission.csv):
```csv
ID,Target
ABC123,Signed Sealed and delivered
XYZ789,This done and protested the day
```

**Behavior**:
1. Script detects CSV file
2. Loads predictions from CSV
3. Automatically converts to JSON for faster future loads
4. Continues with remaining folds

**Example**:
```bash
# You have fold 1 results from another source
cp my_fold1_results.csv outputs/qwen3-8b-v1/inference_cache/fold_1_predictions.csv

# Run K-fold inference
python inference.py --config config_qwen3_8b.yaml --kfold

# Output:
# ⏩ Fold 1: Loading cached predictions from fold_1_predictions.csv (CSV)
#    Loaded 1373 predictions
#    Converting to JSON format for future speed...
#    Saved as fold_1_predictions.json
# 🔄 Fold 2: Running predictions...
```

---

## Usage

### Basic K-Fold Inference (Auto-Resume)

```bash
python inference.py --config config_qwen3_8b.yaml --kfold
```

**First run:**
```
K-FOLD ENSEMBLE INFERENCE: 5 folds
Inference cache: outputs/qwen3-8b-v1/inference_cache

🔄 Fold 1: Loading model from outputs/qwen3-8b-v1/fold_1/best
   Running predictions (batch_size=2)...
Fold 1: 100%|████████████| 1373/1373 [18:32<00:00]
   Saving predictions to cache: fold_1_predictions.json

🔄 Fold 2: Loading model from outputs/qwen3-8b-v1/fold_2/best
   Running predictions (batch_size=2)...
[CRASH/DISCONNECT at fold 2, 50% done]
```

**Resume (automatic):**
```bash
# Same command - automatically detects cached predictions
python inference.py --config config_qwen3_8b.yaml --kfold
```

**Output:**
```
K-FOLD ENSEMBLE INFERENCE: 5 folds
Inference cache: outputs/qwen3-8b-v1/inference_cache

⏩ Fold 1: Loading cached predictions from fold_1_predictions.json
   Loaded 1373 predictions

🔄 Fold 2: Loading model from outputs/qwen3-8b-v1/fold_2/best
   Running predictions (batch_size=2)...
[Continues from where it left off]
```

---

### Force Re-run All Folds

If you need to regenerate predictions (e.g., changed inference params):

```bash
python inference.py --config config_qwen3_8b.yaml --kfold --clear-cache
```

**Output:**
```
🗑️  Clearing inference cache...
   Cache cleared.

🔄 Fold 1: Loading model...
[Re-runs all folds from scratch]
```

---

## Time Savings

### Scenario 1: Complete Run (No Interruption)
```
Before: 1.7 hours (no overhead)
Now:    1.7 hours + ~2 seconds (JSON save per fold)
Overhead: Negligible (~0.1%)
```

### Scenario 2: Interrupted at Fold 3
```
Without resumability:
- Fold 1: 20 min ✓
- Fold 2: 20 min ✓
- Fold 3: 10 min (interrupted) ❌
- Restart: 60 min (folds 1-3 again) ← WASTED
- Total: 110 min

With resumability:
- Fold 1: 20 min ✓ (saved)
- Fold 2: 20 min ✓ (saved)
- Fold 3: 10 min (interrupted) ❌
- Resume: Instant (load folds 1-2), 10 min (finish fold 3)
- Total: 60 min (40% faster)
```

### Scenario 3: Multiple Interruptions
```
Colab disconnects 3 times during ensemble:
- Without resumability: Start over 3 times = 5.1 hours wasted
- With resumability: Resume each time = 1.7 hours total ✓
```

---

## Cache Management

### View Cache Status

```bash
ls -lh outputs/qwen3-8b-v1/inference_cache/
```

**Output:**
```
fold_1_predictions.json   1.2M   (completed)
fold_2_predictions.json   1.2M   (completed)
fold_3_predictions.json   1.2M   (completed)
# fold 4, 5 missing = need to run
```

### Cache Size

Each fold's predictions:
```
1373 test images × ~50 chars average = ~70KB text
JSON formatting: ~1.2MB per fold
Total for 5 folds: ~6MB (negligible)
```

### Manual Cache Cleanup

```bash
# Remove all cached predictions
rm -rf outputs/qwen3-8b-v1/inference_cache/

# Remove specific fold
rm outputs/qwen3-8b-v1/inference_cache/fold_3_predictions.json
```

---

## Edge Cases Handled

### 1. Partial Fold Completion

**Problem:** Fold 3 crashes halfway through (686/1373 images)

**Solution:**
- No cache file saved until fold completes
- On resume, fold 3 restarts from beginning
- Only completed folds are cached

**Why:** Partial predictions are unreliable (might be from wrong checkpoint, bad state)

### 2. Changed Test Set

**Problem:** You modify Test.csv or add/remove images

**Solution:**
- Use `--clear-cache` to regenerate all predictions
- Cache doesn't track test set changes automatically

**Best practice:**
```bash
# If test set changed
python inference.py --config config.yaml --kfold --clear-cache
```

### 3. Changed Model Checkpoints

**Problem:** You retrain a fold (e.g., fold_3/best updated)

**Solution:**
```bash
# Clear cache for changed fold
rm outputs/qwen3-8b-v1/inference_cache/fold_3_predictions.json

# Or clear all
python inference.py --config config.yaml --kfold --clear-cache
```

### 4. Changed Inference Parameters

**Problem:** You change `num_beams` from 5 → 3

**Solution:**
- Cache doesn't track inference params
- Must use `--clear-cache` to regenerate

**Future improvement:** Could hash inference params and include in cache filename

---

## Implementation Details

### Cache File Format

```json
{
  "img_id_1": "predicted text 1",
  "img_id_2": "predicted text 2",
  ...
}
```

**Simple key-value:** Image ID → Prediction text
**Human readable:** Can inspect/debug manually
**Compact:** ~1.2MB per fold (compressed)

### Memory Efficiency

**Loading cached predictions:**
```python
# Instant load, no model in memory
with open(cache_file) as f:
    predictions = json.load(f)  # ~1.2MB RAM
```

**Running inference:**
```python
# Model in VRAM, predictions in RAM
model = load_model(checkpoint)  # ~16GB VRAM
predictions = {}  # ~1.2MB RAM (builds up)
# ... inference ...
torch.cuda.empty_cache()  # Free VRAM before next fold
```

**Sequential fold processing:**
- Only 1 model loaded at a time
- Previous folds' predictions already on disk
- Minimal RAM footprint

---

## Comparison with Training Checkpoints

| Feature | Training | Inference |
|---------|----------|-----------|
| **What's saved** | Model weights + optimizer state | Predictions (text) |
| **Size per checkpoint** | ~500MB (LoRA) | ~1.2MB (JSON) |
| **Resume granularity** | Per-step (every 50 steps) | Per-fold (after completion) |
| **Resume speed** | Minutes (reload model + state) | Instant (load JSON) |
| **Storage overhead** | High (~2.5GB for 5 checkpoints) | Low (~6MB for 5 folds) |

**Why different:**
- Training: Need fine-grained resume (expensive to restart)
- Inference: Coarse resume okay (folds are atomic, ~20 min each)

---

## Troubleshooting

### Cache Not Loading

**Symptoms:**
```
🔄 Fold 1: Loading model...
(Expected: ⏩ Fold 1: Loading cached predictions)
```

**Causes:**
1. Cache file doesn't exist (check `inference_cache/` directory)
2. Cache file corrupted (invalid JSON)
3. Using `--clear-cache` flag

**Fix:**
```bash
# Check cache exists
ls outputs/qwen3-8b-v1/inference_cache/

# Validate JSON
python -m json.tool outputs/.../fold_1_predictions.json
```

### Wrong Predictions After Resume

**Symptoms:** Ensemble results differ from expected

**Causes:**
1. Test set changed after some folds cached
2. Model checkpoints updated after cache created

**Fix:**
```bash
# Regenerate all predictions
python inference.py --config config.yaml --kfold --clear-cache
```

### Out of Disk Space

**Symptoms:** Error saving cache file

**Solution:**
```bash
# Cache is only 6MB total, check disk:
df -h

# If really tight, can skip caching by modifying code
# (not recommended - defeats the purpose)
```

---

## Best Practices

### 1. Always Use K-Fold with Caching

```bash
# Good (default behavior)
python inference.py --config config_qwen3_8b.yaml --kfold

# Bad (no caching, wasted on interruption)
python inference.py --config config_qwen3_8b.yaml --kfold --clear-cache
# (only use --clear-cache when needed)
```

### 2. Clear Cache When Necessary

**When to use `--clear-cache`:**
- ✅ Changed test set (Test.csv)
- ✅ Retrained models (updated checkpoints)
- ✅ Changed inference params (num_beams, max_tokens)
- ✅ Changed image preprocessing (max_pixels)
- ❌ Just re-running for same setup (waste time)

### 3. Archive Cache for Reproducibility

```bash
# After generating final submission
tar -czf qwen3-8b-v1-inference-cache.tar.gz \
  outputs/qwen3-8b-v1/inference_cache/

# Can recreate submission later without re-inference
```

### 4. Monitor Progress

```bash
# In another terminal, watch cache directory
watch -n 5 'ls -lh outputs/qwen3-8b-v1/inference_cache/'

# See which folds completed
```

### 5. Import Existing Predictions (CSV)

**Use case**: You already have predictions from some folds (e.g., different experiment, previous run)

```bash
# Copy your existing CSV results to cache directory
cp fold_1_results.csv outputs/qwen3-8b-v1/inference_cache/fold_1_predictions.csv
cp fold_2_results.csv outputs/qwen3-8b-v1/inference_cache/fold_2_predictions.csv

# Run ensemble - will use CSV cache for folds 1-2, run inference for 3-5
python inference.py --config config_qwen3_8b.yaml --kfold
```

**CSV format requirements**:
```csv
ID,Target
img_id_1,transcription text 1
img_id_2,transcription text 2
```

**Important**:
- CSV must have exactly 1373 rows (test set size)
- All IDs must match Test.csv
- Script will auto-convert CSV → JSON for future speed

---

## Performance Impact

### Overhead

**Per fold:**
- Save JSON: ~0.5 seconds (1.2MB write)
- Load JSON: ~0.1 seconds (1.2MB read)

**Total for 5 folds:**
- Save: ~2.5 seconds
- Load (all cached): ~0.5 seconds

**Compared to inference time:**
- Inference per fold: ~20 minutes
- Cache overhead: <0.1%

**Verdict:** Essentially free

### Benefits

**Robustness:**
- No progress lost on disconnection
- Can stop/resume anytime

**Development:**
- Test ensemble combinations quickly
- Iterate on ensemble logic without re-inference

**Production:**
- Reliable submission generation
- Can regenerate submission if needed (instant from cache)

---

## Summary

**Simple addition, huge impact:**
- Added cache directory: 1 line
- Save after fold: 3 lines
- Load before fold: 5 lines
- **Result: Resumable inference, 40-100% time savings on interruption**

**Perfect for:**
- Colab (frequent disconnections)
- Iterating on ensemble strategy
- Generating multiple submissions quickly

**No downsides:**
- Negligible overhead (<0.1%)
- Minimal disk usage (6MB)
- Automatic, no user action needed

🎯 **Enable by default, use `--clear-cache` only when needed.**
