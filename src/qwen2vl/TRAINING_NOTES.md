# Training Notes - Qwen2-VL OCR

## Run 1 Results (Original Config)

**Training Details:**
- Model: Qwen2-VL-7B-Instruct
- LoRA: r=64, alpha=128, dropout=0.05
- Epochs: 5
- Weight decay: 0.01
- Augmentation: moderate (p=0.2-0.3)

**Observations:**
- Best model: **Epoch 1.7** (eval_loss=0.7017)
- Train loss: 1.463 → 0.23 (continued dropping)
- Eval loss: 0.92 → 0.70 (plateaued after epoch 2)
- **Overfitting detected** - train loss keeps dropping but eval loss stops improving

**Key Finding:** Model converges early around epochs 1-2, further training doesn't help validation performance.

---

## Data Split Strategy (Small Dataset Optimization)

### Stratified Splitting by Text Length

**Problem:** With only 4,098 samples, random split can create unrepresentative validation sets.

**Solution:** Stratify by text length (quintiles)
```python
# Create 5 length bins
df['length_bin'] = pd.qcut(df['text_length'], q=5, labels=False)

# Stratified split ensures each bin proportionally represented
train_df, val_df = train_test_split(df, test_size=0.1, stratify=df['length_bin'])
```

**Why text length?**
- WER/CER metrics weight longer texts more heavily
- Validation metrics more stable when length distribution matches training
- Reduces variance in eval scores across runs

**Alternative strategies:**
- **Stratify by difficulty:** Character diversity, special characters
- **K-fold CV:** 5-fold for more reliable metrics (5x training time)
- **Larger val split:** 15% for more stable metrics (less training data)

**Current choice:** 10% stratified by length (balanced trade-off)

**To analyze your split:** `python analyze_split.py`

---

## Run 2 Config (Anti-Overfitting Tuning)

### Changes Made

| Parameter | Run 1 | Run 2 | Reason |
|-----------|-------|-------|--------|
| **epochs** | 5 | **3** | Best model found at epoch 1-2 |
| **weight_decay** | 0.01 | **0.05** | Stronger L2 regularization |
| **lora_dropout** | 0.05 | **0.1** | More aggressive dropout |
| **p_blur** | 0.2 | **0.0** | Disabled - documents already degraded |
| **p_noise** | 0.2 | **0.0** | Disabled - documents already have grain |
| **p_brightness** | 0.3 | **0.3** | Realistic scan exposure variations |
| **p_contrast** | 0.3 | **0.3** | Realistic ink fade variations |
| **p_rotate** | 0.1 | **0.1** | Conservative alignment correction |
| **max_rotation** | 2° | **1°** | More conservative rotation range |
| **output_dir** | run1 | **run2** | New experiment |

### Augmentation Philosophy

**Conservative approach for historical documents:**
- ✓ **Keep:** Brightness, contrast, small rotation (simulate scanning variations)
- ✗ **Remove:** Blur, noise (documents already have natural degradation)
- **Rationale:** Adding artificial degradation on top of already-faded/damaged text risks making characters illegible

### Expected Impact

**Training time:** ~2 hours (down from 3.5 hours) on A100 without Flash Attention

**Expected results:**
- Lower train loss ceiling (more regularization)
- Better generalization (higher eval performance)
- Less overfitting (train/eval gap reduced)
- **Cleaner training signal** - model learns character shapes without artificial noise

### When to Use Each Config

**Run 1 Config (Original):**
- ✓ Exploring maximum model capacity
- ✓ Have more training data (>10K samples)
- ✓ Want to see full training curve

**Run 2 Config (Current):**
- ✓ **Optimizing for competition** (better generalization)
- ✓ Limited training data (~4K samples)
- ✓ Faster iteration cycles

---

## Inference

**Always use the `best` checkpoint:**
```bash
python inference.py  # uses checkpoint from config.yaml
```

The `best` checkpoint is automatically saved at the step with lowest eval_loss during training.

---

## Further Tuning Ideas

If Run 2 still overfits:
1. **Even more augmentation:** Add elastic distortion, cutout
2. **Reduce LoRA rank:** Try r=32 (less capacity)
3. **Reduce learning rate:** Try 1e-5 (slower convergence)

If Run 2 underfits (train/eval both high):
4. **Increase LoRA rank:** Try r=128
5. **Reduce regularization:** weight_decay=0.02
6. **More epochs:** Back to 5 epochs

---

## Run 3 Config (Increased Capacity + K-Fold CV)

### Changes from Run 2

| Parameter | Run 2 | Run 3 | Reason |
|-----------|-------|-------|--------|
| **lora_r** | 64 | **128** | 2x capacity for better learning |
| **lora_alpha** | 128 | **256** | Maintain 2:1 ratio with rank |
| **weight_decay** | 0.05 | **0.02** | Reduce regularization (was too strong) |
| **lora_dropout** | 0.1 | **0.05** | Reduce dropout |
| **epochs** | 3 | **5** | More training time with early stopping |
| **learning_rate** | 2e-5 | **1.5e-5** | More stable for larger rank |
| **warmup_ratio** | 0.1 | **0.15** | Longer warmup |
| **max_pixels** | 2.016M | **2.5M** | Higher resolution (~1581x1581) |
| **k_folds** | N/A | **5** | K-Fold CV for limited data |

### K-Fold Cross-Validation

**Setup (config.yaml):**
```yaml
data:
  k_folds: 5  # 5-fold CV
  # val_split only used if k_folds=1
```

**How it works:**
- Trains 5 models using stratified k-fold split by text length
- Each fold: 80% train (~3,278 samples), 20% val (~820 samples)
- Every sample used for training in 4 folds, validation in 1 fold
- More efficient use of limited data (4,098 samples)

**Training:**
```bash
python train.py  # trains all 5 folds automatically
```

**Outputs:**
```
outputs/qwen2vl-7b-run3/
├── fold_1/best/
├── fold_2/best/
├── fold_3/best/
├── fold_4/best/
├── fold_5/best/
└── kfold_summary.txt
```

**Inference (Ensemble):**
```bash
python inference.py --kfold
```

- Loads all 5 fold models
- Each predicts on test set
- Character-level majority voting for ensemble
- Expected improvement: 1-3% over single model

**Expected Run 3 Performance:**
- Target: eval_loss **0.55-0.60** (vs Run 2's 0.67)
- With ensemble: **WER ~8-10%**, **CER ~3-4%**
- Training time: ~15-17 hours (5 folds × 3 hours)

---

## Competition Strategy

**Updated strategy with K-Fold CV:**

**Phase 1 (Current):**
1. ✓ Run 1: Baseline (eval_loss=0.7017)
2. ✓ Run 2: Anti-overfitting (eval_loss=0.67)
3. **Run 3: Increased capacity + K-Fold CV (target: 0.55-0.60)**

**Phase 2 (For final submission):**
1. Train Run 3 with K-Fold CV (5 models)
2. Generate ensemble predictions with `--kfold` flag
3. Optional: Post-processing (spell check, language model)

**Why K-Fold > Multi-Seed:**
- Uses all data more efficiently (every sample in train & val)
- More diverse models (different train/val splits)
- Better ensemble (5 models vs 3 seeds)
- More reliable metrics (averaged across 5 folds)

Expected combined improvement: **2-4% over single model**
