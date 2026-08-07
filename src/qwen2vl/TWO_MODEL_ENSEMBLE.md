# Two-Model Ensemble Strategy

## Concept

Instead of 5-fold CV where each model sees only 80% of data (3,278 samples), train **two models that each see 90% of data (3,688 samples)** with different train/val splits.

**Best of both worlds:**
- ✅ More data per model (+410 samples = +12.5%)
- ✅ Ensemble diversity from different splits
- ✅ Faster training (2 models × 2h = 4h vs 5 models × 2h = 10h)

---

## Strategy

### Model 1 (Already Trained)
```yaml
seed: 42
output_dir: "/content/drive/MyDrive/prd/qwen3-8b-full"
Result: 0.86 leaderboard score
```

### Model 2 (To Train)
```yaml
seed: 123  # Different split
output_dir: "/content/drive/MyDrive/prd/qwen3-8b-full-v2"
Expected: ~0.85-0.87 (similar to model 1)
```

### Ensemble
Combine predictions from both models using:
- **Shortest** (worked best for k-fold: 0.859)
- Or **Majority** voting
- Or **Character voting**

**Expected ensemble result: 0.87-0.88** (better than single model 0.86)

---

## Step-by-Step Workflow

### Step 1: Train Model 2 (Different Seed)

```bash
cd /content/ROAD/src/qwen2vl

# Train second model with seed=123
python train.py --config config_qwen3_8b_full_v2.yaml
```

**Expected output:**
```
Training | Train: 3688 | Val: 410  # Different 410 samples than model 1
...
✓ Training | eval_cer=0.0640-0.0660 | Completed 3.0 epochs
Saved: /content/drive/MyDrive/prd/qwen3-8b-full-v2/best
```

**Training time:** ~2 hours on A100

---

### Step 2: Generate Predictions from Both Models

```bash
# Model 1 predictions (already done if you have submission_full.csv)
python inference.py --config config_qwen3_8b_full.yaml
# Generates: submission_full.csv

# Model 2 predictions
python inference.py --config config_qwen3_8b_full_v2.yaml
# Generates: submission_full_v2.csv
```

**Each inference:** ~10-15 minutes for 1373 test images

---

### Step 3: Ensemble the Two Models

```bash
# Try different ensemble strategies
python ensemble_two_models.py --config1 config_qwen3_8b_full.yaml \
                              --config2 config_qwen3_8b_full_v2.yaml \
                              --strategy shortest \
                              --output submission_ensemble_shortest.csv

python ensemble_two_models.py --strategy majority \
                              --output submission_ensemble_majority.csv

python ensemble_two_models.py --strategy char_voting \
                              --output submission_ensemble_char.csv
```

**Output:**
```
Loading predictions from:
  Model 1: submission_full.csv
  Model 2: submission_full_v2.csv

Ensembling 1373 predictions (strategy=shortest)...

Model agreement: 1189/1373 (86.6%)

Examples of disagreement:
Image: ABC123
  Model 1: "Signed Sealed and delivered"
  Model 2: "Signed Sealed delivered"
  Ensemble: "Signed Sealed delivered"  (shortest)

✅ Saved ensemble to submission_ensemble_shortest.csv
```

---

### Step 4: Test All Submissions

Submit to leaderboard:
1. `submission_full.csv` (model 1 only: 0.86)
2. `submission_full_v2.csv` (model 2 only: expected ~0.86)
3. `submission_ensemble_shortest.csv` (expected: **0.87-0.88**)
4. `submission_ensemble_majority.csv`
5. `submission_ensemble_char.csv`

---

## Why This Works Better Than K-Fold

### Data Efficiency

**5-Fold K-Fold:**
```
Fold 1: 3,278 samples (80%) → model 1
Fold 2: 3,278 samples (80%) → model 2
Fold 3: 3,278 samples (80%) → model 3
Fold 4: 3,278 samples (80%) → model 4
Fold 5: 3,278 samples (80%) → model 5

Each model sees: 3,278 samples
Ensemble: 5 models with less data each
```

**2-Model Ensemble:**
```
Model 1: 3,688 samples (90%, seed=42)  → better model
Model 2: 3,688 samples (90%, seed=123) → better model

Each model sees: 3,688 samples (+410 more!)
Ensemble: 2 strong models
```

### Training Time

**K-Fold:** 5 models × 2h = **10 hours**
**Two-Model:** 2 models × 2h = **4 hours** (2.5× faster!)

### Diversity

**K-Fold diversity sources:**
- Different 80% train splits
- Different validation sets
- Models may overfit differently

**Two-Model diversity sources:**
- Different 90% train splits (still 3,688 samples each)
- Different validation sets
- Same hyperparameters (controlled comparison)

**Key insight:** With 90% data per model, random splits still provide enough diversity while keeping individual models strong.

---

## Expected Performance Analysis

### Individual Model Performance

**Model 1 (seed=42):**
- Train: 3,688 samples
- eval_cer: 0.0650
- Leaderboard: **0.86**

**Model 2 (seed=123):**
- Train: 3,688 samples (different split)
- Expected eval_cer: 0.0640-0.0670
- Expected leaderboard: **0.85-0.87**

### Ensemble Performance

**Agreement rate:** ~85-90% (models will agree on most images)

**Disagreement handling:**
- **Shortest strategy:** When models disagree, pick shorter prediction
  - Works if both models tend to over-generate slightly
  - Worked best for k-fold (0.859)

- **Majority strategy:** With 2 models, falls back to first model
  - Not useful with only 2 models (no majority)

- **Character voting:** Align and vote per character
  - Can help fix small differences
  - May create artifacts on large differences

**Expected ensemble result:**
- Conservative estimate: **0.87** (+0.01 from 0.86)
- Optimistic estimate: **0.88** (+0.02 from 0.86)

---

## Comparison Table

| Metric | K-Fold (5 models) | Two-Model Ensemble | Improvement |
|--------|-------------------|-------------------|-------------|
| Data per model | 3,278 samples | 3,688 samples | +12.5% |
| Training time | 10 hours | 4 hours | 2.5× faster |
| Best single model | 0.855 (fold 1) | 0.86 (model 1) | +0.5% |
| Best ensemble | 0.859 (shortest) | 0.87-0.88 (expected) | +1-2% |
| Models to manage | 5 | 2 | Simpler |

---

## Troubleshooting

### Model 2 Gets Much Worse Performance

**Problem:** Model 2 eval_cer = 0.10 (much worse than 0.065)

**Possible causes:**
1. Bad random seed got unlucky split
2. Training diverged (check training logs)

**Solution:**
Try another seed (e.g., seed=456) until you get similar performance.

### Models Agree 99%

**Problem:** Model agreement >95% (too similar)

**Cause:** Seed 123 might produce very similar split to seed 42

**Solution:**
Use more different seed (e.g., seed=999) or manually shuffle data before splitting.

### Ensemble Worse Than Best Single Model

**Problem:** Ensemble 0.85 < Model 1 0.86

**Cause:** Model 2 is significantly weaker, dragging down ensemble

**Solution:**
1. Check model 2's individual score first
2. If model 2 < 0.85, retrain with different seed
3. Only ensemble if both models are strong (>0.85)

---

## Advanced: Three-Model Ensemble

If you have time and want to push further:

```bash
# Train third model with seed=456
# Copy config_qwen3_8b_full_v2.yaml → config_qwen3_8b_full_v3.yaml
# Change seed to 456 and output_dir to qwen3-8b-full-v3

python train.py --config config_qwen3_8b_full_v3.yaml
```

Then ensemble all three:
- Majority voting now makes sense (2-out-of-3 wins)
- More diversity
- Diminishing returns (3rd model adds less than 2nd)

**Expected:** 0.88-0.89 (marginal gain over two-model)

---

## Summary

**What to do:**
1. ✅ Train model 2 with `config_qwen3_8b_full_v2.yaml` (2 hours)
2. ✅ Run inference on both models (30 minutes)
3. ✅ Ensemble with `ensemble_two_models.py` (instant)
4. ✅ Submit and compare

**Expected outcome:**
- Model 1: 0.86 (already achieved)
- Model 2: ~0.86
- **Ensemble: 0.87-0.88** (new best!)

**Time investment:** ~2.5 hours total
**Expected gain:** +1-2% leaderboard improvement

Good luck! 🚀
