# Qwen3-VL-2B Setup Guide

## Why Qwen3-VL-2B?

**Qwen3-VL-2B has specific improvements for your historical OCR task:**

1. **Enhanced OCR capabilities:**
   - 32 languages (vs 19 in Qwen2-VL)
   - Robust to blur, low light, and tilt
   - Better with rare/ancient characters (1600s spelling!)
   - Improved long-document structure parsing

2. **DeepStack architecture:**
   - Captures fine-grained details in degraded text
   - Better for faded ink and aged paper

3. **Practical advantages:**
   - 3x faster training (~1h/fold vs 3h for 7B)
   - Lower VRAM (can use larger batches/higher resolution)
   - More K-Fold iterations in same time

---

## Installation (Colab)

### Step 1: Update Transformers

Qwen3-VL requires bleeding-edge transformers:

```bash
pip install -q --upgrade git+https://github.com/huggingface/transformers
```

**Time:** ~2 minutes

### Step 2: Verify Compatibility

```bash
cd /content/ROAD/src/qwen2vl/
python test_qwen3_compatibility.py
```

**Expected output:**
```
✓ Model loaded successfully
✓ VRAM used: 5.2GB
✓ VRAM available for training: 74.8GB
✓ Inference successful
COMPATIBILITY TEST PASSED ✓
```

**If test fails:** Check transformers version or GPU availability

---

## Training

### Quick Start (Single Model)

Test Qwen3-VL-2B with simple split first:

```bash
cd /content/ROAD/src/qwen2vl/
python train.py --config config_qwen3_2b.yaml
```

**Training time:** ~1.5 hours (7 epochs, single model)

### Full K-Fold CV (Recommended)

For best competition performance:

```bash
cd /content/ROAD/src/qwen2vl/
python train.py --config config_qwen3_2b.yaml
```

**Training time:** ~5-6 hours (5 folds × 1 hour each)

**Outputs:**
```
outputs/qwen3-2b-kfold/
├── fold_1/best/
├── fold_2/best/
├── fold_3/best/
├── fold_4/best/
├── fold_5/best/
└── kfold_summary.txt
```

---

## Inference

### K-Fold Ensemble (Best Performance)

```bash
cd /content/ROAD/src/qwen2vl/
python inference.py --config config_qwen3_2b.yaml --kfold
```

**Loads all 5 models and ensembles predictions**

### Single Model

```bash
python inference.py --config config_qwen3_2b.yaml
```

**Output:** `submission.csv` in repo root

---

## Configuration Details

**Key differences from Qwen2-VL-7B config:**

| Parameter | Qwen2-VL-7B | Qwen3-VL-2B | Reason |
|-----------|-------------|-------------|--------|
| **batch_size** | 4 | **8** | 2B uses less VRAM |
| **gradient_accum** | 4 | **2** | Maintain effective batch=16 |
| **epochs** | 5 | **7** | Smaller model trains faster |
| **learning_rate** | 1.5e-5 | **3e-5** | Smaller model needs stronger signal |
| **lora_r** | 128 | **96** | Balanced for 2B |
| **weight_decay** | 0.02 | **0.01** | Lighter regularization |
| **max_pixels** | 2.5M | **3M** | More VRAM available |

---

## Expected Performance

**Single Model:**
- eval_loss: ~0.58-0.63
- WER: ~10-13%
- CER: ~4-5%

**K-Fold Ensemble (5 models):**
- WER: ~9-11% (1-2% improvement)
- CER: ~3-4%
- Combined: ~6-7.5%

**Why it might match/beat 7B:**
- Architecture improvements for OCR
- Task-specific optimizations (ancient text, degradation)
- Better fine-grained detail capture (DeepStack)

---

## Troubleshooting

**Error: `AutoModelForVision2Seq` not found**
```bash
# Update transformers to latest
pip install --upgrade git+https://github.com/huggingface/transformers
```

**Error: CUDA out of memory**
```yaml
# In config_qwen3_2b.yaml, reduce:
batch_size: 6  # down from 8
max_pixels: 2500000  # down from 3M
```

**Model downloads slowly**
```bash
# Pre-download model
huggingface-cli download Qwen/Qwen3-VL-2B-Instruct
```

---

## Comparison Strategy

**Train both and compare:**

1. **Qwen3-VL-2B K-Fold** (6 hours) → Submit to leaderboard
2. **Qwen2-VL-7B K-Fold** (15 hours) → Submit to leaderboard
3. **Compare scores** → Use best for final submission

**Or ensemble both:**
- 5 folds of 2B + 3-5 folds of 7B = 8-10 models
- Maximum diversity
- Expected: **best possible score** (WER ~7-9%)

---

## Quick Command Reference

```bash
# Test compatibility
python test_qwen3_compatibility.py

# Train with K-Fold
python train.py --config config_qwen3_2b.yaml

# Inference with ensemble
python inference.py --config config_qwen3_2b.yaml --kfold

# Check training progress
tail -f outputs/qwen3-2b-kfold/fold_1/trainer_state.json
```
