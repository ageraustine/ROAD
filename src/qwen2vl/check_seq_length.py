#!/usr/bin/env python3
"""
check_seq_lengths.py - Measures REAL tokenized sequence lengths for a sample
of the training data, using the actual processor and config. Run this on the
pod before guessing another max_seq_length value.

Usage (from src/qwen2vl/):
    python3 check_seq_lengths.py --config config_qwen3_8b_full.yaml --n 50
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml
from transformers import AutoProcessor
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config_qwen3_8b_full.yaml")
parser.add_argument("--n", type=int, default=50, help="Number of samples to check")
args = parser.parse_args()

config_path = SCRIPT_DIR / "configs" / args.config
with open(config_path) as f:
    cfg = yaml.safe_load(f)

model_name = cfg["model"]["name"]
max_pixels = cfg["training"]["max_pixels"]
train_csv = cfg["data"]["train_csv"]
image_dir = cfg["data"]["image_dir"]

repo_root = SCRIPT_DIR.parent.parent
print(f">> Loading processor for {model_name} ...")
processor = AutoProcessor.from_pretrained(model_name)

df = pd.read_csv(repo_root / train_csv)
if len(df) > args.n:
    df = df.sample(n=args.n, random_state=42)

OCR_PROMPT = "Transcribe the handwritten text in this image."  # adjust if train.py's actual prompt differs
ASSISTANT_HEADER = "<|im_start|>assistant\n"

lengths = []
image_col = "ID" if "ID" in df.columns else df.columns[0]
target_col = "Target" if "Target" in df.columns else df.columns[-1]

for _, row in df.iterrows():
    img_path = repo_root / image_dir / f"{row[image_col]}.jpg"
    if not img_path.exists():
        continue
    img = Image.open(img_path).convert("RGB")

    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": OCR_PROMPT},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": str(row[target_col])}]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    batch = processor(text=[text], images=[img], padding=False, truncation=False, return_tensors="pt")
    seq_len = batch["input_ids"].shape[1]
    lengths.append(seq_len)

lengths.sort()
n = len(lengths)
print(f"\n>> Measured {n} samples at max_pixels={max_pixels}")
print(f"   min:    {lengths[0]}")
print(f"   median: {lengths[n // 2]}")
print(f"   p95:    {lengths[int(n * 0.95)]}")
print(f"   max:    {lengths[-1]}")
print(f"\n>> Recommended max_seq_length: {lengths[-1] + 100} (max + safety margin)")
print("   (if this is much higher than expected, max_pixels is producing more")
print("   vision tokens than a naive pixel/1024 estimate - DeepStack's")
print("   multi-level feature fusion can add overhead beyond simple compression math)")