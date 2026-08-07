# Model V2 Training Strategy

## Goal: Ensemble Diversity + Equal Quality

Model 2 should be **different but equally strong** as Model 1 to maximize ensemble benefit.

---

## Key Differences from Model 1

### 1. Data Split (Primary Diversity Source)
| | Model 1 | Model 2 |
|---|---------|---------|
| **Seed** | 42 | 123 |
| **Train samples** | 3,688 (90%) | 3,688 (90%) - DIFFERENT samples |
| **Val samples** | 410 (10%) | 410 (10%) - DIFFERENT samples |

**Impact**: Models see mostly different validation samples, learn from slightly different training distributions.

### 2. Training Duration
| | Model 1 | Model 2 |
|---|---------|---------|
| **Epochs** | 3 | **4** (+33%) |
| **Total steps** | 693 | **924** (+231 steps) |

**Why 4 epochs:**
- Model 1 plateaued but was still improving slightly at epoch 3
- Extra epoch gives model 2 more time to find different local optimum
- With lower LR, needs more steps to converge

### 3. Learning Rate Schedule
| | Model 1 | Model 2 |
|---|---------|---------|
| **Initial LR** | 2e-5 | **1.8e-5** (-10%) |
| **Warmup ratio** | 0.1 | **0.15** (+50% warmup) |
| **Schedule** | Cosine | Cosine |

**Why slower:**
- Lower LR → more careful optimization → finds different solution
- Longer warmup → more stable early training
- With 4 epochs, LR decays slower → learns more gradually

### 4. Regularization
| | Model 1 | Model 2 |
|---|---------|---------|
| **Weight decay** | 0.1 | **0.08** (-20%) |
| **LoRA dropout** | 0.15 | **0.1** (-33%) |

**Why less aggressive:**
- Model 1 was conservative (high regularization)
- Model 2 explores more freely (learns different patterns)
- Balance: not so loose it overfits, but different from v1

### 5. LoRA Capacity
| | Model 1 | Model 2 |
|---|---------|---------|
| **LoRA rank** | 64 | **96** (+50%) |
| **Trainable params** | 214.7M | ~322M (+50%) |
| **LoRA alpha** | 128 | 128 (same) |

**Why higher capacity:**
- More parameters → can learn more complex patterns
- Different rank → different low-rank decomposition → different representation
- Still efficient (322M << 8.9B full model)

### 6. Augmentation
| | Model 1 | Model 2 |
|---|---------|---------|
| **Brightness** | 0.3 | **0.35** |
| **Contrast** | 0.3 | **0.25** |
| **Rotation** | 0.1, ±1° | **0.15, ±1.5°** |

**Why different:**
- Different augmentation → model sees slightly different data
- Higher brightness, more rotation → handles variations better
- Lower contrast → focuses less on contrast-based features

---

## Expected Training Behavior

### Model 1 Trajectory (Actual)
```
Step 200: eval_cer=0.0760, eval_loss=0.6374
Step 300: eval_cer=0.0749, eval_loss=0.6091
Step 400: eval_cer=0.0685, eval_loss=0.5914  ← Biggest improvement
Step 500: eval_cer=0.0661, eval_loss=0.5952
Step 600: eval_cer=0.0652, eval_loss=0.5890
Step 693: eval_cer=0.0650, eval_loss=0.5883  ← Plateaued

Final: eval_cer=0.0650, train_loss=0.4351
Gap: 0.17 (some overfitting)
Leaderboard: 0.86
```

### Model 2 Predictions (Expected)
```
Step 200: eval_cer~0.078, eval_loss~0.64  (slower start due to lower LR)
Step 300: eval_cer~0.072, eval_loss~0.61
Step 400: eval_cer~0.068, eval_loss~0.60
Step 500: eval_cer~0.065, eval_loss~0.59
Step 600: eval_cer~0.062, eval_loss~0.585 (still improving)
Step 700: eval_cer~0.061, eval_loss~0.583
Step 800: eval_cer~0.060, eval_loss~0.582
Step 924: eval_cer~0.059-0.061, eval_loss~0.580-0.585

Final: eval_cer=0.059-0.062 (similar or slightly better)
Gap: 0.16-0.18 (similar overfitting)
Leaderboard: 0.85-0.87 (expected similar to model 1)
```

**Key differences:**
- Slower convergence (lower LR, more warmup)
- Continues improving beyond 693 steps (extra epoch)
- May find slightly different optimum (different regularization, capacity)

---

## Why This Creates Good Ensemble

### 1. Independent Errors
**Model 1:**
- Trained on samples [1, 2, 3, ..., 3688] with seed 42
- Learns patterns specific to this split
- Makes errors on specific types of images

**Model 2:**
- Trained on samples [different 3688] with seed 123
- ~20% different training samples (820 samples)
- Learns different patterns, makes errors on different images

### 2. Different Solutions
**Model 1:**
- Higher regularization → conservative predictions
- Rank 64 → specific feature representation
- 3 epochs → earlier stopping point

**Model 2:**
- Lower regularization → explores more
- Rank 96 → richer feature representation
- 4 epochs → more converged solution

**Result**: When they disagree (~15% of time), they offer different perspectives.

### 3. Complementary Strengths
**Ensemble scenarios:**

**Scenario A: Both agree (85% of cases)**
```
Model 1: "Signed Sealed and delivered"
Model 2: "Signed Sealed and delivered"
Ensemble: "Signed Sealed and delivered" ✓ (high confidence)
```

**Scenario B: Model 1 hallucinated (7% of cases)**
```
Model 1: "Signed Sealed and delivered on this day" (over-generated)
Model 2: "Signed Sealed and delivered"
Ensemble (shortest): "Signed Sealed and delivered" ✓ (corrected)
```

**Scenario C: Model 2 hallucinated (7% of cases)**
```
Model 1: "Signed Sealed delivered"
Model 2: "Signed Sealed and delivered the"
Ensemble (shortest): "Signed Sealed delivered" ✓ (corrected)
```

**Scenario D: Both wrong differently (1% of cases)**
```
Model 1: "Signed Sealed and delivered"
Model 2: "Signed Sealed delivered"
Ensemble (shortest): "Signed Sealed delivered" (picks one)
Ground truth: "Signed and Sealed" (both missed)
```

---

## Expected Ensemble Results

### Agreement Analysis
Based on k-fold experience where models trained on 80% data:
- **Agreement rate**: 86-90% (models produce identical predictions)
- **Disagreement rate**: 10-14% (137-192 images)

With 90% data per model (vs 80% in k-fold):
- **Agreement rate**: 88-92% (models are stronger, more likely to agree on correct answer)
- **Disagreement rate**: 8-12% (110-165 images)

### Ensemble Strategies Performance

**Shortest (Best for k-fold, expected best here):**
```
Model 1: 0.86
Model 2: 0.86 (expected)
Ensemble: 0.87-0.88

Improvement: +1-2%
Why: Picks shorter on disagreements, both models tend to over-generate slightly
```

**Majority (Not useful with 2 models):**
```
With 2 models, majority = first model
Same as model 1: 0.86
```

**Character voting (Risky):**
```
May help: 0.87
May hurt: 0.85
Why: Can create artifacts when models disagree significantly
```

---

## Training Time Estimate

### Model 1 (Actual)
```
3 epochs × 231 steps/epoch = 693 steps
Time: 7189 seconds = ~2.0 hours
```

### Model 2 (Expected)
```
4 epochs × 231 steps/epoch = 924 steps
Time: ~924/693 × 7189s = ~9585s = ~2.7 hours
```

**Overhead from extra complexity:**
- Higher LoRA rank (96 vs 64): +~15% compute
- More augmentation probability: Negligible

**Total time**: **~2.8-3.0 hours**

---

## Success Criteria

### Minimum Success (Model 2 is usable)
- eval_cer < 0.070
- Leaderboard score > 0.84
- Agreement with model 1: 85-95%

### Expected Success (Model 2 matches model 1)
- eval_cer: 0.059-0.065
- Leaderboard score: 0.85-0.87
- Agreement with model 1: 88-92%
- **Ensemble beats both: 0.87-0.88**

### Outstanding Success (Model 2 beats model 1)
- eval_cer < 0.058
- Leaderboard score: 0.87+
- **Ensemble: 0.88-0.89**

---

## Monitoring During Training

Watch for these signs:

### Good Signs ✓
- eval_cer steadily decreasing (like model 1)
- eval_loss plateaus around 0.58-0.59
- Train/eval gap < 0.20 (not overfitting too much)
- Stops around epoch 3.5-4.0 (early stopping working)

### Warning Signs ⚠️
- eval_cer > 0.08 at epoch 1 (training too slow)
- eval_loss not decreasing after epoch 2 (stuck)
- Train/eval gap > 0.25 (overfitting badly)

### Bad Signs ❌
- eval_cer > 0.10 at epoch 2 (something wrong)
- eval_loss increasing (diverging)
- Early stop at epoch 1 (bad initialization)

If you see warning/bad signs, consider:
1. Reducing LoRA rank back to 64
2. Increasing weight decay to 0.09
3. Using seed 456 instead of 123

---

## Post-Training Checklist

After training completes:

- [ ] Check final eval_cer (should be 0.059-0.065)
- [ ] Check train/eval gap (should be 0.15-0.20)
- [ ] Run inference: `python inference.py --config config_qwen3_8b_full_v2.yaml`
- [ ] Check individual leaderboard score (should be 0.85-0.87)
- [ ] Ensemble with model 1: `python ensemble_two_models.py --strategy shortest`
- [ ] Submit ensemble (expected: 0.87-0.88)
- [ ] If ensemble < 0.86, investigate disagreement patterns

---

## Summary

**Model 2 Configuration = Controlled Diversity:**

✅ Different data split (seed 123)
✅ More epochs (4 vs 3)
✅ Lower LR (1.8e-5 vs 2e-5)
✅ Higher capacity (rank 96 vs 64)
✅ Less regularization (dropout 0.1, weight_decay 0.08)
✅ Different augmentation mix

**Expected outcome:**
- Model 2: 0.85-0.87 (similar to model 1)
- Ensemble: **0.87-0.88** (better than both)
- Training time: ~3 hours
- Total time to ensemble: ~4 hours

**Risk mitigation:**
- Not too different (same architecture, similar hyperparams)
- Early stopping prevents overfitting
- Can always fall back to model 1 (0.86) if model 2 fails

Good luck! 🚀
