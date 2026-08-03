# Barbados Road Challenge – OCR Starter Guide

Welcome to the Barbados Road Challenge OCR Starter Pack. This repository contains **three independent OCR approaches**, each with its own environment, dependencies, and workflow.

The goal of this guide is to help beginners quickly set up an environment, run inference, and evaluate results.

---

# Table of Contents

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Before You Start](#before-you-start)
- [Choose an OCR Approach](#choose-an-ocr-approach)
  - VLM (Vision Language Model)
  - Kraken-OCR
  - Paddle-OCR
- [Troubleshooting](#troubleshooting)
- [Additional Resources](#additional-resources)

---

# Overview

This starter pack includes three OCR approaches:

| Approach | Best For | Training Support | Recommended CUDA |
|-----------|-----------|------------------|------------------|
| VLM | Highest OCR accuracy and multimodal understanding | Yes | CUDA 11.8 or 12.x |
| Kraken-OCR | Historical and challenging documents | Yes | CUDA 11.8+ (Optional) |
| Paddle-OCR | Fast production-ready OCR | Limited/Fine-tuning Optional | CUDA 12.x |

Each approach folder contains:

```text
setup.sh            Environment setup
requirements.txt    Python dependencies
config.yaml         Configuration
inference.py        Run predictions
train.py/trainer.py Training script (if available)
eval_metrics.py     Evaluation metrics
```

---

# System Requirements

## Minimum Requirements

- Linux, macOS, or Windows (WSL2 recommended)
- Python 3.11
- 16 GB RAM
- 20 GB available storage

## Recommended Requirements

- 32 GB RAM
- NVIDIA CUDA-compatible system

## CUDA Requirements

The most important requirement is your **CUDA version**, not a specific GPU model.

| Approach | Recommended CUDA Version |
|-----------|--------------------------|
| VLM | CUDA 11.8 or CUDA 12.x |
| Kraken-OCR | CUDA 11.8+ |
| Paddle-OCR | CUDA 12.x |

Check your CUDA installation:

```bash
nvidia-smi
```

Example output:

```text
CUDA Version: 12.1
```

If CUDA is not installed, download it from NVIDIA's CUDA Toolkit website.

---

# Before You Start

From the project root, go to the starters folder:

```bash
cd code/scripts/Starters
```

Choose one OCR approach below. Each one uses its **own conda environment**.

---

# Choose an OCR Approach

Each OCR approach is completely independent and should be installed separately.

---

<details>
<summary><strong>📖 VLM (Vision Language Model)</strong></summary>

## Description

Uses Vision Language Models (for example Qwen2-VL) for OCR.

Best choice when accuracy is the primary goal.

### Environment Name

```bash
vlm_env
```

### Setup

Navigate to the folder:

```bash
cd code/scripts/Starters/VLM
```

Run the setup script:

```bash
bash setup.sh
```

Activate the environment:

```bash
conda activate vlm_env
```

### Configure

Edit:

```bash
config.yaml
```

Important settings in `config.yaml`:

```yaml
general:
  repo_root:

training:
  model_id:
  train_csv:
  base_image_dir:

inference:
  model_path:
  test_csv:
  base_image_dir:
  output_csv:

evaluation:
  pred_csv:
  pub_gt:
  priv_gt:
  weight_factor:
```

### Train

```bash
python trainer.py
```

### Run Inference

```bash
python inference.py
```

This will:

1. Load the model
2. Process test images
3. Generate OCR predictions
4. Create:

```text
submission.csv
```

Expected prediction columns:

```text
ID,Target
```

### Evaluate

```bash
python eval_metrics.py
```

Metrics include:

- Character Error Rate (CER)
- Word Error Rate (WER)

### Recommended CUDA

```text
CUDA 11.8 or CUDA 12.x
```

</details>

---

<details>
<summary><strong>📖 Kraken-OCR</strong></summary>

## Description

Kraken is specialized for historical and difficult OCR tasks.

It can run on CPU or CUDA-enabled systems.

### Environment Name

```bash
kraken_ocr_env
```

### Setup

```bash
cd code/scripts/Starters/Kraken-OCR
bash setup.sh
```

Activate:

```bash
conda activate kraken_ocr_env
```

### Configure

Edit:

```bash
config.yaml
```

Important settings in `config.yaml`:

```yaml
general:
  repo_root:

training:
  train_csv:
  base_image_dir:

inference:
  model_path:
  test_csv:
  base_image_dir:
  output_csv:

evaluation:
  pred_csv:
  pub_gt:
  priv_gt:
  weight_factor:
```

### Train

```bash
python train.py
```

### Run Inference

```bash
python inference.py
```

Expected prediction columns:

```text
ID,Target
```

### Evaluate

```bash
python eval_metrics.py
```

### Recommended CUDA

```text
CUDA 11.8+
```

(Optional)

</details>

---

<details>
<summary><strong>📖 Paddle-OCR</strong></summary>

## Description

PaddleOCR is optimized for fast OCR pipelines and production deployments.

Includes:

- Text detection
- Orientation classification
- Text recognition

### Environment Name

```bash
paddle_ocr_env
```

### Setup

```bash
cd code/scripts/Starters/Paddle-OCR
bash setup.sh
```

Activate:

```bash
conda activate paddle_ocr_env
```

### Configure

Edit:

```bash
config.yaml
```

Important settings in `config.yaml`:

```yaml
general:
  repo_root:

inference:
  test_csv:
  base_image_dir:
  output_csv:
  use_gpu:
  use_angle_cls:
  lang:

evaluation:
  pred_csv:
  pub_gt:
  priv_gt:
  weight_factor:
```

### Run Inference

```bash
python inference.py
```

Pipeline:

1. Detect text regions
2. Classify orientation
3. Recognize text
4. Export results

Expected prediction columns:

```text
ID,Target
```

### Evaluate

```bash
python eval_metrics.py
```

### Recommended CUDA

```text
CUDA 12.x
```

</details>

---

