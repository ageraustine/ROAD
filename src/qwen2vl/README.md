# Qwen2-VL Fine-tuning Pipeline

Competition-optimized pipeline for historical handwriting recognition using Qwen2-VL-7B on A100 80GB.

## Architecture

- **Base Model**: Qwen/Qwen2-VL-7B-Instruct
- **Fine-tuning**: LoRA (rank 64, alpha 128)
- **Precision**: BF16 (no quantization)
- **Attention**: Flash Attention 2
- **Augmentation**: Multi-stage document degradation simulation

## Files

```
qwen2vl/
├── config.yaml      # Main configuration
├── train.py         # Training script with augmentation
├── inference.py     # Submission generation
├── requirements.txt # Dependencies
└── README.md        # This file
```

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Verify config paths
# Edit config.yaml to ensure:
#   - general.repo_root points to project root
#   - All paths are correct
```

## Training

```bash
python train.py
```

**Expected outputs:**
- `outputs/qwen2vl-7b-run1/best/` - Best checkpoint by eval loss
- `outputs/qwen2vl-7b-run1/final/` - Final checkpoint

**Training time:** ~6-8 hours on A100 80GB (5 epochs, 4098 samples)

**Memory usage:** ~45-55GB VRAM

## Inference

```bash
# Use best checkpoint
python inference.py

# Or specify custom checkpoint
python inference.py --checkpoint outputs/qwen2vl-7b-run1/final
```

**Output:** `submission.csv` in repo root

## Configuration Details

### Model Config
```yaml
model:
  name: "Qwen/Qwen2-VL-7B-Instruct"
  use_flash_attention: true
  use_quantization: false
  torch_dtype: "bfloat16"
```

### Training Config
```yaml
training:
  batch_size: 4                     # Per device
  gradient_accumulation_steps: 4    # Effective batch = 16
  epochs: 5
  learning_rate: 2.0e-5
  warmup_ratio: 0.1
  lora_r: 64                        # Higher than baseline (16)
  lora_alpha: 128
```

### Augmentation
```yaml
augmentation:
  enabled: true
  p_blur: 0.2          # Gaussian blur
  p_noise: 0.2         # Gaussian noise
  p_brightness: 0.3    # Brightness adjustment
  p_contrast: 0.3      # Contrast adjustment
  p_rotate: 0.1        # Small rotation (±2°)
```

## Hyperparameter Tuning Notes

**What we optimized:**
- LoRA rank: 64 (vs baseline 16) - more capacity for historical styles
- Learning rate: 2e-5 (vs 3e-5) - more stable with larger rank
- Effective batch: 16 (vs 16) - good balance for convergence
- Max pixels: 2M (vs 1.5M) - preserve more detail

**What to tune if needed:**
- `epochs`: Increase to 7-10 if underfitting
- `learning_rate`: Lower to 1e-5 if training unstable
- `lora_r`: Increase to 128 for more capacity (watch VRAM)
- `augmentation probabilities`: Adjust based on error analysis

## Expected Performance

| Metric | Expected Range | Notes |
|--------|----------------|-------|
| WER | 12-15% | Word Error Rate |
| CER | 4-6% | Character Error Rate |
| Combined Score | 8-10.5% | Competition metric (0.5*WER + 0.5*CER) |

**Improvement opportunities:**
1. Ensemble with TrOCR (different architecture)
2. Test-time augmentation (3-5 passes, vote)
3. Post-processing (spell correction, historical language model)
4. Longer training (10-15 epochs with early stopping)

## Troubleshooting

**OOM (Out of Memory):**
- Reduce `batch_size` to 2
- Reduce `max_pixels` to 1.5M
- Reduce `lora_r` to 32

**Underfitting (high train/val loss):**
- Increase `epochs` to 10
- Increase `lora_r` to 128
- Decrease `lora_dropout` to 0.03

**Overfitting (low train, high val loss):**
- Increase `weight_decay` to 0.05
- Increase augmentation probabilities by 0.1
- Add more augmentation types

**Slow training:**
- Increase `batch_size` if VRAM allows
- Reduce `dataloader_num_workers` if CPU bound
- Verify Flash Attention is enabled (check logs)

## Competition Strategy

**Phase 1 (Current):**
- [x] Qwen2-VL-7B baseline with augmentation
- [ ] Error analysis on validation set
- [ ] Hyperparameter tuning based on errors

**Phase 2:**
- [ ] TrOCR-large fine-tuning (architectural diversity)
- [ ] Ensemble Qwen + TrOCR
- [ ] Test-time augmentation

**Phase 3:**
- [ ] Post-processing pipeline
- [ ] Historical language model for error correction
- [ ] Final ensemble weights optimization
