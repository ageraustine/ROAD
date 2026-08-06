# CLAUDE.md - R.O.A.D. Project Guide

## Project Overview

**Reclaiming Our Atlantic Destiny (R.O.A.D.)** - A Zindi ML competition for handwritten text recognition (HTR) on historical Barbados archival documents.

**Goal**: Build an OCR model to transcribe colonial-era handwritten text (deeds, wills, census records) from scanned images.

## Dataset Structure

```
dataset/
├── Train.csv          # 4098 samples (ID, Target)
├── Test.csv           # 1373 samples (ID only)
├── SampleSubmission.csv
└── images/            # 5472 JPG images
```

- **ID**: Image filename without extension (e.g., `uGI8F9Er0c5XwdnX`)
- **Target**: Ground truth transcription text

## Evaluation Metrics

Final score = 0.5 * WER + 0.5 * CER (lower is better)

- **WER**: Word Error Rate
- **CER**: Character Error Rate
- Longer transcriptions weighted more heavily

## Submission Format

```csv
ID,Target
MzQuRiUbPFsq6Azy,transcribed text here
```

## Starter Approaches

Located in `src/`:

| Approach | Directory | Best For | Training |
|----------|-----------|----------|----------|
| VLM | `VLM/` | Highest accuracy (Qwen2-VL) | Yes |
| Kraken-OCR | `Kraken-OCR/` | Historical documents | Yes |
| Paddle-OCR | `Paddle-OCR/` | Fast inference | Limited |

### Each starter contains:
- `setup.sh` - Environment setup (creates conda env)
- `config.yaml` - Configuration paths
- `inference.py` - Generate predictions
- `eval_metrics.py` - Calculate WER/CER
- `train.py` or `trainer.py` - Training script (VLM/Kraken)

## Image Analysis Tool

Before training, analyze your dataset to determine optimal image resolution settings:

```bash
# Analyze dataset/images to find optimal max_pixels
bash run_image_analysis.sh
```

This provides:
- Image dimension statistics (width, height, pixels)
- Resize impact at different thresholds
- Recommendations for config.yaml max_pixels setting
- Critical for balancing quality vs training speed

See `IMAGE_ANALYSIS_README.md` for details.

## Quick Start

```bash
# Example: VLM approach
cd src/VLM
bash setup.sh
conda activate vlm_env
# Edit config.yaml with correct paths
python trainer.py      # Train
python inference.py    # Generate submission.csv
python eval_metrics.py # Evaluate
```

## Key Challenges

- Faded ink and degraded pages
- Unfamiliar historical handwriting styles
- Variable text lengths

## System Requirements

- Python 3.11
- 16GB+ RAM (32GB recommended)
- CUDA GPU recommended (11.8+ for Kraken, 12.x for Paddle/VLM)

## Important Paths

- Images: `dataset/images/{ID}.jpg`
- Update `repo_root` and `base_image_dir` in config.yaml files

---

## Competition Pipeline (src/qwen2vl/)

**Optimized for winning with A100 80GB VRAM**

### Architecture
- **Model**: Qwen2-VL-7B-Instruct (full precision, no quantization)
- **Fine-tuning**: LoRA (r=64, alpha=128) on full model
- **Flash Attention 2**: Optional (improves speed 20-30%, auto-fallback if unavailable)
- **Precision**: BF16 for optimal A100 utilization

### Key Features
- **Conservative augmentation**: Brightness, contrast, rotation only (no blur/noise on already-degraded documents)
- **Curriculum learning ready**: Easy/hard sample separation
- **High-res processing**: 2M pixels (~1344x1500) - analyze with `run_image_analysis.sh`
- **Beam search**: 5 beams for better decoding
- **Auto-resume**: Training resumes from checkpoint if interrupted (critical for Colab)
- **Early stopping**: Monitors eval_loss (stable signal) to detect plateaus, while best model selection uses eval_cer (competition metric). Saves 25-40% time in K-fold.

### Training Setup

**Step 0: Analyze Dataset (Recommended)**
```bash
# Analyze images to determine optimal max_pixels setting
bash run_image_analysis.sh
# Update max_pixels in config files based on recommendations
```

**Option 1: Google Colab (Recommended)**
```
1. Open train_colab.ipynb in Google Colab
2. Set Runtime to GPU (A100)
3. Run all cells
```

**Option 2: Local/Remote GPU**
```bash
cd src/qwen2vl
pip install -r requirements.txt
python train.py

# Expected training time: ~6-8 hours on A100 80GB
# Auto-resumes from checkpoint if interrupted
```

### Inference

```bash
python inference.py
# Generates submission.csv in repo root
```

### Configuration (config.yaml)

**Training hyperparameters:**
- Batch size: 4 (effective 16 with grad accumulation)
- Learning rate: 2e-5 with cosine schedule
- Epochs: 5
- LoRA rank: 64 (higher capacity than baseline)

**Augmentation (conservative for historical docs):**
- Blur: Disabled (documents already degraded)
- Noise: Disabled (documents already have grain)
- Brightness: 30% (scan exposure variations)
- Contrast: 30% (ink fade variations)
- Rotation: 10% (±1 degree alignment)

### Expected Performance
- Baseline 7B without augmentation: WER ~15-20%, CER ~5-8%
- With augmentation: WER ~12-15%, CER ~4-6%
- Ensemble potential: Additional 2-3% improvement

### Next Steps for Improvement
1. Error analysis on validation set
2. Add TrOCR ensemble
3. Post-processing with historical language model
4. Test-time augmentation (TTA)
