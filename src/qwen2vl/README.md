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

### K-Fold Cross-Validation (Recommended for Competition)

**For maximum performance on limited data (4,098 samples), use 5-Fold CV:**

```bash
# config.yaml: set k_folds=5
python train.py
```

**How it works:**
- Trains 5 models, each on 80% data (3,278 samples), validates on 20% (820 samples)
- Every sample used for training in 4 folds, validation in 1 fold
- More reliable metrics (averaged across 5 folds)
- Better final predictions through ensemble

**Expected outputs:**
```
outputs/qwen2vl-7b-run3/
├── fold_1/
│   ├── best/      # Best checkpoint for fold 1
│   └── final/
├── fold_2/best/   # Best checkpoint for fold 2
├── fold_3/best/
├── fold_4/best/
├── fold_5/best/
└── kfold_summary.txt  # Summary of all folds
```

**Training time:** ~15-17 hours on A100 80GB (5 folds × 3 hours each)

**Memory usage:** ~45-55GB VRAM per fold

### Simple Train/Val Split (Faster Iteration)

**For quick testing, use simple 90/10 split:**

```bash
# config.yaml: set k_folds=1
python train.py
```

**Expected outputs:**
- `outputs/qwen2vl-7b-run3/best/` - Best checkpoint by eval loss
- `outputs/qwen2vl-7b-run3/final/` - Final checkpoint

**Training time:** ~3 hours on A100 80GB (5 epochs, run3 config)

**Config Optimization (Run 3):**
Increased capacity to improve from Run 2's eval_loss=0.67:
- LoRA rank: 128 (doubled from 64)
- Reduced regularization: weight_decay=0.02, lora_dropout=0.05
- More training: 5 epochs (with early stopping)
- Higher resolution: 2.5M pixels

See `TRAINING_NOTES.md` for detailed analysis of training runs and tuning decisions.

## Inference

### K-Fold Ensemble (Best Performance)

```bash
# Ensemble predictions from all 5 folds
python inference.py --kfold

# Uses all fold checkpoints from config.yaml output_dir
# e.g., outputs/qwen2vl-7b-run3/fold_*/best/
```

**How it works:**
- Loads all 5 fold models
- Each model predicts on test set
- Combines predictions (majority vote for each character position)
- **Expected improvement:** 1-3% better than single model

### Single Model Inference

```bash
# Use best checkpoint from config
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

### Data Split Strategy

**Stratified sampling by text length** (critical for small datasets):
```yaml
data:
  val_split: 0.1  # 10% = ~410 samples
  seed: 42
```

**Why stratify by text length?**
- With only 4,098 samples, random split can create unrepresentative validation sets
- WER/CER metrics weight longer texts more heavily
- Ensures validation metrics are stable and reliable

**Implementation:** `train.py` automatically creates 5 text length bins and ensures each is proportionally represented in train/val.

**To analyze your split:** `python analyze_split.py`

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
