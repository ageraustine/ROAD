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

### Local Setup
```bash
# Install dependencies
pip install -r requirements.txt
```

### Google Colab Setup
Use `train_colab.ipynb` in the repo root - it handles everything automatically:
- GPU verification (A100 recommended)
- Repository cloning and updates
- Dataset download
- Fast dependency installation (pre-built Flash Attention wheels)
- Training and inference

**Workflow:**
1. First run: Clones repo and sets up environment (~3 min setup)
2. Subsequent runs: Optional cell to pull latest code updates
3. All paths auto-detected - no manual configuration needed

**Performance notes:**
- Flash Attention installs in ~30 seconds using pre-built wheels (vs 10-15 min from source)
- Flash Attention is **optional** - training works fine without it (20-30% slower, same accuracy)
- Auto-fallback to standard attention if Flash Attention unavailable

## Training

```bash
python train.py
```

**Expected outputs:**
- `outputs/qwen2vl-7b-run2/best/` - Best checkpoint by eval loss (use this!)
- `outputs/qwen2vl-7b-run2/final/` - Final checkpoint

**Training time:** ~2 hours on A100 80GB (3 epochs, 4098 samples, no Flash Attention)

**Memory usage:** ~45-55GB VRAM

**Config Optimization:**
Current config is tuned to prevent overfitting observed in initial training runs:
- Reduced epochs: 3 (best model typically found at epoch 1-2)
- Increased regularization: weight_decay=0.05, lora_dropout=0.1
- More augmentation: higher probabilities for blur, noise, brightness, contrast

See `TRAINING_NOTES.md` for detailed analysis of training runs and tuning decisions.

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
  # Conservative approach for historical documents
  p_blur: 0.0          # Disabled (documents already degraded)
  p_noise: 0.0         # Disabled (documents already have grain)
  p_brightness: 0.3    # Scan exposure variations
  p_contrast: 0.3      # Ink fade variations
  p_rotate: 0.1        # Alignment variations (±1°)
```

**Philosophy:** Historical documents are already faded and degraded. Adding artificial blur/noise on top risks making text illegible and confuses the model. Focus on realistic scanning variations only.

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

**Flash Attention installation issues:**
- Set `use_flash_attention: false` in config.yaml
- Training will work fine with standard attention (20-30% slower)
- Accuracy is identical, only speed/memory affected

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
- Check if Flash Attention is enabled (see logs at model load)

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
