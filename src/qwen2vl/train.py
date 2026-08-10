"""
Qwen-VL Fine-tuning for Historical HTR
Supports Qwen2-VL / Qwen2.5-VL / Qwen3-VL. Optimized for a single A100 80GB.

Refactor notes (vs. original):
  - Validation no longer augmented (separate eval collator).
  - Label masking anchored on the assistant header tokens, fails loudly.
  - Padding masked via attention_mask, not by token id (pad/eos collision safe).
  - NaN targets dropped instead of becoming the string "nan".
  - Optional CER metric via greedy decode on a fixed val subset.
  - Gradient checkpointing made PEFT-safe.
  - Per-fold model teardown; deterministic device placement.
"""

import gc
import os
import math
import json
import random
import inspect
import argparse
import logging
import warnings
from pathlib import Path

import yaml
import torch
import pandas as pd
import numpy as np
import cv2  # Used in augmentation code (even if currently disabled)
from io import BytesIO
from PIL import Image, ImageFilter, ImageEnhance
from sklearn.model_selection import train_test_split, StratifiedKFold

# Suppress verbose warnings and progress bars
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
from datasets import Dataset
from tqdm import tqdm

import transformers
from transformers import (
    AutoProcessor,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model

try:
    from sklearn.model_selection import StratifiedGroupKFold
    GROUP_KFOLD_AVAILABLE = True
except ImportError:
    GROUP_KFOLD_AVAILABLE = False

import transformers

# v5 dropped a pile of TrainingArguments fields (warmup_ratio, overwrite_output_dir,
# evaluation_strategy, per_gpu_*) and renamed from_pretrained's torch_dtype -> dtype.
TF_MAJOR = int(transformers.__version__.split(".")[0])


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent

OCR_PROMPT = (
    "Transcribe the handwritten text in this image exactly as written. "
    "Preserve spelling, punctuation, and line breaks. Output only the transcription."
)

ASSISTANT_HEADER = "<|im_start|>assistant\n"


# ─────────────────────────────────────────────────────────────
# VERSION COMPATIBILITY
# ─────────────────────────────────────────────────────────────

def compute_schedule(n_samples: int, batch_size: int, grad_accum: int,
                     epochs: int, warmup_ratio: float):
    """
    Optimizer-step counts, used to express warmup as an integer.

    The original script used floor division on the sample count, which
    under-counts whenever the last batch is partial. The Trainer itself uses
    ceil at both levels, so this matches what actually runs and the cosine
    schedule lands on the true final step.
    """
    steps_per_epoch = math.ceil(math.ceil(n_samples / batch_size) / grad_accum)
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    return steps_per_epoch, total_steps, warmup_steps


def supported_kwargs(cls, kwargs: dict) -> dict:
    """
    Drop kwargs the installed version's __init__ doesn't accept, and say which.

    Avoids whack-a-mole across transformers v4/v5, where fields get removed
    one at a time and each one costs a full crash to discover.
    """
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return kwargs

    params = sig.parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs  # accepts **kwargs, nothing to filter

    valid = set(params)
    dropped = sorted(k for k in kwargs if k not in valid)
    if dropped:
        print(f"  [compat] {cls.__name__} on transformers {transformers.__version__} "
              f"does not accept: {', '.join(dropped)} - dropped")
    return {k: v for k, v in kwargs.items() if k in valid}


# ─────────────────────────────────────────────────────────────
# MODEL CLASS RESOLUTION
# ─────────────────────────────────────────────────────────────

def resolve_model_class(model_name: str):
    """
    Pick the right model class. Order matters: 'Qwen2.5-VL' contains 'Qwen2',
    so the 2.5 check must come first or it silently loads the wrong class.
    """
    name = model_name.lower().replace("_", ".").replace("-", ".")

    if "qwen3.vl" in name:
        from transformers import Qwen3VLForConditionalGeneration
        return Qwen3VLForConditionalGeneration, "qwen3"
    if "qwen2.5.vl" in name:
        from transformers import Qwen2_5_VLForConditionalGeneration
        return Qwen2_5_VLForConditionalGeneration, "qwen2.5"
    if "qwen2.vl" in name:
        from transformers import Qwen2VLForConditionalGeneration
        return Qwen2VLForConditionalGeneration, "qwen2"

    from transformers import AutoModelForVision2Seq
    return AutoModelForVision2Seq, "auto"


# ─────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────

def levenshtein(a: str, b: str) -> int:
    """Edit distance, O(min(len)) memory. Avoids a jiwer/evaluate dependency."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(
                prev[j] + 1,        # deletion
                curr[j - 1] + 1,    # insertion
                prev[j - 1] + (ca != cb),  # substitution
            ))
        prev = curr
    return prev[-1]


def corpus_cer(preds: list, refs: list) -> float:
    """Aggregate CER: total edits / total reference chars (not mean of per-sample CER)."""
    total_edits = sum(levenshtein(p, r) for p, r in zip(preds, refs))
    total_chars = sum(len(r) for r in refs)
    return total_edits / max(total_chars, 1)


def corpus_wer(preds: list, refs: list) -> float:
    """
    Aggregate WER: total word edits / total reference words.

    Word tokenization: split on whitespace (simple but effective for historical docs).
    """
    total_edits = 0
    total_words = 0

    for pred, ref in zip(preds, refs):
        # Simple word tokenization: split on whitespace
        pred_words = pred.split()
        ref_words = ref.split()

        # Compute word-level edit distance
        word_edits = word_levenshtein(pred_words, ref_words)
        total_edits += word_edits
        total_words += len(ref_words)

    return total_edits / max(total_words, 1)


def word_levenshtein(a: list, b: list) -> int:
    """Levenshtein distance at word level (instead of character level)."""
    if len(a) < len(b):
        a, b = b, a

    if not b:
        return len(a)

    prev = list(range(len(b) + 1))

    for i, a_word in enumerate(a, 1):
        curr = [i]
        for j, b_word in enumerate(b, 1):
            # Cost: 0 if words match, 1 if different
            cost = 0 if a_word == b_word else 1
            curr.append(min(
                prev[j] + 1,      # deletion
                curr[j - 1] + 1,  # insertion
                prev[j - 1] + cost  # substitution
            ))
        prev = curr

    return prev[-1]


# ─────────────────────────────────────────────────────────────
# AUGMENTATION
# ─────────────────────────────────────────────────────────────

class ImageAugmenter:
    """
    Conservative augmentation for already-degraded historical documents.
    Training only - never applied to validation.
    """

    def __init__(self, cfg: dict):
        self.enabled = cfg.get("enabled", False)
        self.p_blur = cfg.get("p_blur", 0.0)
        self.p_noise = cfg.get("p_noise", 0.0)
        self.p_brightness = cfg.get("p_brightness", 0.3)
        self.p_contrast = cfg.get("p_contrast", 0.3)
        self.p_rotate = cfg.get("p_rotate", 0.1)
        self.max_rotation = cfg.get("max_rotation", 1)

        # Advanced HTR augmentations
        self.p_morphology = cfg.get("p_morphology", 0.0)  # dilate/erode (ink thickness)
        self.p_shear = cfg.get("p_shear", 0.0)  # slant jitter (scribe variation)
        self.max_shear = cfg.get("max_shear", 8)  # degrees
        self.p_resolution = cfg.get("p_resolution", 0.0)  # resolution jitter
        self.p_jpeg = cfg.get("p_jpeg", 0.0)  # JPEG artifacts

    def __call__(self, img: Image.Image) -> Image.Image:
        if not self.enabled:
            return img

        if random.random() < self.p_blur:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

        if random.random() < self.p_brightness:
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))

        if random.random() < self.p_contrast:
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))

        if random.random() < self.p_rotate:
            angle = random.uniform(-self.max_rotation, self.max_rotation)
            img = img.rotate(angle, fillcolor=(255, 255, 255), expand=False)

        if random.random() < self.p_noise:
            arr = np.array(img).astype(np.float32)
            arr += np.random.normal(0, random.uniform(5, 15), arr.shape)
            img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

        # Morphological ops (models ink thickness, pen pressure, ink loading)
        if random.random() < self.p_morphology:
            arr = np.array(img)
            kernel_size = random.choice([2, 3])
            kernel = np.ones((kernel_size, kernel_size), np.uint8)

            # Dilate (thicken) or erode (thin) with equal probability
            if random.random() < 0.5:
                arr = cv2.dilate(arr, kernel, iterations=1)
            else:
                arr = cv2.erode(arr, kernel, iterations=1)

            # Slight blur to simulate ink diffusion into paper fibers
            arr = cv2.GaussianBlur(arr, (3, 3), 0.5)
            img = Image.fromarray(arr)

        # Shear/slant jitter (models different scribe handwriting angles)
        if random.random() < self.p_shear:
            angle_deg = random.uniform(-self.max_shear, self.max_shear)
            angle_rad = np.deg2rad(angle_deg)
            w, h = img.size

            # Affine transform for horizontal shear
            shear_factor = np.tan(angle_rad)
            img = img.transform(
                (w, h),
                Image.AFFINE,
                (1, shear_factor, -shear_factor * h / 2, 0, 1, 0),
                fillcolor=(255, 255, 255),
                resample=Image.BILINEAR
            )

        # Resolution jitter (prevents overfitting to exact pixel budget)
        if random.random() < self.p_resolution:
            w, h = img.size
            scale = random.uniform(0.6, 1.0)
            new_w, new_h = int(w * scale), int(h * scale)

            # Downscale then upscale back (simulates lower resolution scans)
            img = img.resize((new_w, new_h), Image.BILINEAR)
            img = img.resize((w, h), Image.BILINEAR)

        # JPEG artifacts (models scan compression)
        if random.random() < self.p_jpeg:
            buffer = BytesIO()
            quality = random.randint(60, 90)
            img.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            img = Image.open(buffer).convert('RGB')

        return img


# ─────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────

def load_image(path: str, max_pixels: int = 2016000) -> Image.Image:
    """Load and downscale (LANCZOS) while preserving aspect ratio."""
    img = Image.open(path).convert("RGB")
    w, h = img.size

    if w * h > max_pixels:
        scale = (max_pixels / (w * h)) ** 0.5
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    return img


def build_dataset(df: pd.DataFrame, image_dir: Path, image_ext: str = ".jpg",
                  split_name: str = "") -> Dataset:
    """
    Build a HF dataset, reporting every dropped row rather than silently skipping.
    """
    samples = []
    dropped = {"missing_image": 0, "empty_target": 0, "bad_id": 0}

    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc=f"Building {split_name or 'dataset'}", leave=False):
        img_id = str(row["ID"]).strip()
        if not img_id or img_id.lower() == "nan":
            dropped["bad_id"] += 1
            continue

        target = row.get("Target")
        # pd.isna catches NaN/None. str(np.nan) would otherwise yield "nan".
        if target is None or (isinstance(target, float) and pd.isna(target)):
            dropped["empty_target"] += 1
            continue

        target = str(target).strip()
        if not target:
            dropped["empty_target"] += 1
            continue

        img_path = image_dir / f"{img_id}{image_ext}"
        if not img_path.exists():
            dropped["missing_image"] += 1
            continue

        samples.append({"id": img_id, "image_path": str(img_path), "text": target})

    n_dropped = sum(dropped.values())
    label = split_name or "dataset"
    # Only print if samples were dropped
    if n_dropped:
        detail = ", ".join(f"{k}={v}" for k, v in dropped.items() if v)
        print(f"  {label}: {len(samples)} kept ({n_dropped} dropped: {detail})")

    if not samples:
        raise RuntimeError(
            f"{label} is empty. Check image_dir, image_ext, and the ID column."
        )

    return Dataset.from_list(samples)


# ─────────────────────────────────────────────────────────────
# COLLATOR
# ─────────────────────────────────────────────────────────────

class OCRCollator:
    """
    Collator for Qwen-VL OCR training.

    Loss is computed only on the assistant turn. The span is located by finding
    the assistant header token sequence, not by string-matching the target text
    (BPE merges whitespace differently in context, so the old approach could
    silently match nothing and train on the prompt).
    """

    def __init__(self, processor, max_pixels: int, augmenter: ImageAugmenter = None,
                 max_length: int = 2048, strict: bool = True):
        self.processor = processor
        self.max_pixels = max_pixels
        self.augmenter = augmenter
        self.max_length = max_length
        self.strict = strict
        self.mask_failures = 0

        tokenizer = processor.tokenizer
        self.header_ids = tokenizer.encode(ASSISTANT_HEADER, add_special_tokens=False)
        if not self.header_ids:
            raise RuntimeError(
                f"Assistant header {ASSISTANT_HEADER!r} tokenized to nothing. "
                "This chat template is not Qwen-style; adjust ASSISTANT_HEADER."
            )

    def __call__(self, examples: list) -> dict:
        images, texts = [], []

        for ex in examples:
            img = load_image(ex["image_path"], self.max_pixels)
            if self.augmenter is not None:
                img = self.augmenter(img)
            images.append(img)
            texts.append(ex["text"])

        messages_batch = [
            [
                {"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": OCR_PROMPT},
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": label}]},
            ]
            for img, label in zip(images, texts)
        ]

        texts_formatted = [
            self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            for m in messages_batch
        ]

        batch = self.processor(
            text=texts_formatted,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()

        for i in range(labels.size(0)):
            seq = batch["input_ids"][i].tolist()
            start = self._rfind_subsequence(seq, self.header_ids)

            if start == -1:
                self.mask_failures += 1
                if self.strict:
                    raise RuntimeError(
                        "Could not locate the assistant header in the tokenized "
                        f"sequence for sample {i}. Target was:\n  {texts[i][:120]!r}\n"
                        "Set training.strict_masking=false to skip these instead."
                    )
                # Mask the whole sample rather than train on the prompt.
                labels[i, :] = -100
                continue

            # Mask everything up to and including the header; the target and the
            # trailing <|im_end|> stay supervised so the model learns to stop.
            labels[i, : start + len(self.header_ids)] = -100

        # Mask padding by position. Masking by pad_token_id would also wipe any
        # legitimate occurrence of that id, and breaks if pad_token == eos_token.
        if "attention_mask" in batch:
            labels[batch["attention_mask"] == 0] = -100

        batch["labels"] = labels
        return batch

    @staticmethod
    def _rfind_subsequence(sequence: list, subsequence: list) -> int:
        """Last occurrence - the assistant header is the final one in the turn."""
        n = len(subsequence)
        for i in range(len(sequence) - n, -1, -1):
            if sequence[i:i + n] == subsequence:
                return i
        return -1


# ─────────────────────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────────────────────

class CERCallback(TrainerCallback):
    """
    Greedy-decode a fixed val subset and inject `eval_cer` into the metrics dict.

    eval_loss is teacher-forced and only loosely tracks the leaderboard metric.
    Mutating the metrics dict here works because Trainer.evaluate() fires
    on_evaluate before returning, so `metric_for_best_model: eval_cer` is valid.
    """

    def __init__(self, processor, dataset, max_pixels: int, n_samples: int = 200,
                 max_new_tokens: int = 256, batch_size: int = 4, seed: int = 42,
                 is_kfold: bool = False, verbose: bool = False):
        self.processor = processor
        self.max_pixels = max_pixels
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.is_kfold = is_kfold
        self.verbose = verbose

        n = min(n_samples, len(dataset))
        idx = np.random.RandomState(seed).choice(len(dataset), n, replace=False)
        self.samples = [dataset[int(i)] for i in sorted(idx)]

    @torch.no_grad()
    def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
        if metrics is None or model is None:
            return

        was_training = model.training
        model.eval()

        cache_owner = getattr(model, "config", None)
        prev_cache = getattr(cache_owner, "use_cache", None) if cache_owner else None
        if cache_owner is not None:
            cache_owner.use_cache = True

        tokenizer = self.processor.tokenizer
        prev_side = tokenizer.padding_side
        tokenizer.padding_side = "left"  # required for batched generation

        preds, refs = [], []
        try:
            for i in range(0, len(self.samples), self.batch_size):
                chunk = self.samples[i:i + self.batch_size]
                images = [load_image(s["image_path"], self.max_pixels) for s in chunk]

                prompts = [
                    self.processor.apply_chat_template(
                        [{"role": "user", "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": OCR_PROMPT},
                        ]}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for img in images
                ]

                inputs = self.processor(
                    text=prompts, images=images, padding=True, return_tensors="pt"
                ).to(model.device)

                out = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    num_beams=1,  # greedy: this is a monitoring signal, not the submission
                    repetition_penalty=1.1,  # Lighter penalty for training eval (less aggressive than inference)
                    eos_token_id=tokenizer.eos_token_id,  # Explicitly enforce stop token
                    pad_token_id=tokenizer.pad_token_id,
                )
                trimmed = out[:, inputs["input_ids"].shape[1]:]
                decoded = self.processor.batch_decode(trimmed, skip_special_tokens=True)

                preds.extend(d.strip() for d in decoded)
                refs.extend(s["text"] for s in chunk)
        finally:
            tokenizer.padding_side = prev_side
            if cache_owner is not None and prev_cache is not None:
                cache_owner.use_cache = prev_cache
            if was_training:
                model.train()

        # Compute both CER and WER (competition uses 0.5*WER + 0.5*CER)
        cer = corpus_cer(preds, refs)
        wer = corpus_wer(preds, refs)
        competition_score = 0.5 * wer + 0.5 * cer

        metrics["eval_cer"] = cer
        metrics["eval_wer"] = wer
        metrics["eval_score"] = competition_score  # Competition metric

        # Track best competition score (what matters for leaderboard)
        if not hasattr(self, 'best_score'):
            self.best_score = float('inf')
            self.best_cer = float('inf')
            self.best_wer = float('inf')

        if competition_score < self.best_score:
            self.best_score = competition_score
            self.best_cer = cer
            self.best_wer = wer
            if not getattr(self, 'is_kfold', False):
                print(f"eval_score={competition_score:.4f} (cer={cer:.4f}, wer={wer:.4f}) ✓ (new best)")
        elif getattr(self, 'verbose', False):
            print(f"eval_score={competition_score:.4f} (cer={cer:.4f}, wer={wer:.4f})")


class ProgressCallback(TrainerCallback):
    """Print clean progress updates instead of verbose tqdm bars."""

    def __init__(self, log_every_n_steps: int = 50):
        self.log_every_n_steps = log_every_n_steps
        self.last_log_step = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Print clean training progress."""
        if logs is None:
            return

        # Only log at intervals
        if state.global_step - self.last_log_step < self.log_every_n_steps:
            return

        self.last_log_step = state.global_step

        # Build compact log line
        parts = []
        if "loss" in logs:
            parts.append(f"loss={logs['loss']:.4f}")
        if "learning_rate" in logs:
            parts.append(f"lr={logs['learning_rate']:.2e}")
        if "epoch" in logs:
            parts.append(f"epoch={logs['epoch']:.2f}")

        if parts:
            progress = f"Step {state.global_step}/{state.max_steps}"
            print(f"{progress} | {' | '.join(parts)}")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Print evaluation results (eval_loss only - CER callback handles eval_cer)."""
        if metrics is None:
            return

        # Only print eval_loss (CER callback handles eval_cer printing)
        if "eval_loss" in metrics:
            print(f"Eval @ step {state.global_step} | eval_loss={metrics['eval_loss']:.4f}")


class EarlyStoppingCallback(TrainerCallback):
    """
    Stop training when eval_loss plateaus (no improvement for N evaluations).

    Critical for K-fold CV where each fold can take hours - don't waste compute
    on a fold that stopped improving.
    """

    def __init__(self, patience: int = 3, min_delta: float = 0.0001,
                 metric: str = "eval_loss", greater_is_better: bool = False,
                 verbose: bool = False):
        """
        Args:
            patience: Number of evaluations with no improvement before stopping
            min_delta: Minimum change to qualify as improvement
            metric: Metric to monitor (eval_loss or eval_cer)
            greater_is_better: True if higher is better (False for loss/cer)
            verbose: Print patience counter on every evaluation (False = only on stop)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric
        self.greater_is_better = greater_is_better
        self.verbose = verbose

        self.best_value = float('inf') if not greater_is_better else float('-inf')
        self.patience_counter = 0
        self.stopped_epoch = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None or self.metric not in metrics:
            return

        current_value = metrics[self.metric]

        # Check if improvement
        if self.greater_is_better:
            improved = current_value > (self.best_value + self.min_delta)
        else:
            improved = current_value < (self.best_value - self.min_delta)

        if improved:
            self.best_value = current_value
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        # Log progress (only if verbose)
        if self.verbose and self.patience_counter > 0:
            print(f"  Early stopping: {self.patience_counter}/{self.patience} "
                  f"(no improvement in {self.metric})")

        # Stop if patience exceeded
        if self.patience_counter >= self.patience:
            print(f"\n⚠️  Early stopping triggered!")
            print(f"    {self.metric} plateaued at {self.best_value:.4f}")
            print(f"    No improvement for {self.patience} evaluations")
            control.should_training_stop = True
            self.stopped_epoch = state.epoch


# ─────────────────────────────────────────────────────────────
# CHECKPOINT UTILITIES
# ─────────────────────────────────────────────────────────────

def get_last_checkpoint(output_dir: Path) -> str | None:
    """
    Find the last checkpoint in output directory for resuming training.

    Returns:
        Path to last checkpoint, or None if no checkpoints found.
    """
    if not output_dir.exists():
        return None

    # Look for checkpoint-* directories (created by Trainer during training)
    checkpoints = [d for d in output_dir.iterdir()
                   if d.is_dir() and d.name.startswith("checkpoint-")]

    if not checkpoints:
        return None

    # Sort by step number (checkpoint-100, checkpoint-200, etc.)
    def get_step_number(ckpt_path):
        try:
            return int(ckpt_path.name.split("-")[-1])
        except (ValueError, IndexError):
            return 0

    checkpoints.sort(key=get_step_number)
    last_checkpoint = checkpoints[-1]

    return str(last_checkpoint)


# ─────────────────────────────────────────────────────────────
# MODEL SETUP
# ─────────────────────────────────────────────────────────────

def setup_model(cfg: dict):
    """Load model + processor and attach LoRA."""
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    model_name = model_cfg["name"]

    model_class, family = resolve_model_class(model_name)
    model_short_name = model_name.split("/")[-1]  # Just "Qwen3-VL-8B-Instruct"
    print(f"Loading {model_short_name}...", end=" ", flush=True)

    if model_class.__name__ == "AutoModelForVision2Seq":
        raise RuntimeError(
            f"No dedicated class resolved for {model_name}. This usually means "
            "transformers is too old (Qwen3-VL needs >=4.57). Upgrade rather than "
            "falling back silently."
        )

    dtype = torch.bfloat16 if model_cfg.get("torch_dtype", "bfloat16") == "bfloat16" \
        else torch.float16

    # Explicit single-GPU placement. device_map="auto" installs accelerate
    # dispatch hooks and turns into naive pipeline parallelism on multi-GPU,
    # which conflicts with Trainer.
    model_kwargs = {
        "device_map": {"": 0} if torch.cuda.is_available() else None,
        "trust_remote_code": True,
    }

    # v5 renamed from_pretrained's torch_dtype -> dtype for every model. On v4
    # only Qwen3-VL takes `dtype`. Passing the wrong one is silently ignored
    # (it lands in **kwargs) and you get fp32 weights and an OOM.
    if TF_MAJOR >= 5 or family == "qwen3":
        model_kwargs["dtype"] = dtype
    else:
        model_kwargs["torch_dtype"] = dtype

    if model_cfg.get("use_flash_attention", True):
        try:
            import flash_attn  # noqa: F401
            model_kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            model_kwargs["attn_implementation"] = "sdpa"
    else:
        model_kwargs["attn_implementation"] = "sdpa"

    model = model_class.from_pretrained(model_name, **model_kwargs)
    model.config.use_cache = False

    # Processor. Passing max_pixels makes the visual token count deterministic
    # instead of relying on the processor's default cap.
    processor_kwargs = {"trust_remote_code": True}
    if train_cfg.get("max_pixels"):
        processor_kwargs["max_pixels"] = train_cfg["max_pixels"]
    if train_cfg.get("min_pixels"):
        processor_kwargs["min_pixels"] = train_cfg["min_pixels"]
    processor = AutoProcessor.from_pretrained(model_name, **processor_kwargs)

    # Gradient checkpointing must be enabled BEFORE the PEFT wrap, and needs
    # input grads enabled or nothing flows back through the frozen base.
    if train_cfg.get("gradient_checkpointing", True):
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    lora_config = LoraConfig(
        r=train_cfg["lora_r"],
        lora_alpha=train_cfg["lora_alpha"],
        target_modules=train_cfg["lora_target_modules"],
        modules_to_save=train_cfg.get("lora_modules_to_save"),
        lora_dropout=train_cfg["lora_dropout"],
        use_rslora=train_cfg.get("use_rslora", False),  # Rank-Stabilized LoRA
        # use_dora=train_cfg.get("use_dora", False),
        rank_pattern=train_cfg.get("rank_pattern"),  # Asymmetric ranks (e.g., vision=64, llm=16)
        alpha_pattern=train_cfg.get("alpha_pattern"),  # Matching alpha for asymmetric ranks
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # Verify LoRA targeting - check structure, not just total
    from peft.tuners.lora import LoraLayer
    lora_modules = [n for n, m in model.named_modules() if isinstance(m, LoraLayer)]
    vision_modules = sum(1 for n in lora_modules if ".visual." in n)
    llm_modules = sum(1 for n in lora_modules if ".language_model." in n)

    # Count trainable params
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    trainable_pct = 100 * n_trainable / n_total

    print(f"✓ ({n_trainable/1e6:.1f}M trainable / {n_total/1e6:.0f}M total = {trainable_pct:.2f}%)")
    print(f"  LoRA modules: {len(lora_modules)} total ({vision_modules} vision / {llm_modules} llm)")

    # Sanity checks
    if n_trainable == 0:
        raise RuntimeError("LoRA matched zero modules. Check lora_target_modules regex.")
    if vision_modules == 0:
        raise RuntimeError(f"LoRA matched 0 vision modules (expected ~116). Check target_modules regex.")

    return model, processor


# ─────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────

def train_single_fold(train_df, val_df, image_dir, fold_output_dir, cfg,
                      fold_num=None, n_folds=None):
    """Train one fold (or a single model when fold_num is None). Returns metrics dict."""
    # Suppress verbose library logs
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("transformers.trainer").setLevel(logging.WARNING)
    logging.getLogger("accelerate").setLevel(logging.ERROR)
    logging.getLogger("torch").setLevel(logging.ERROR)
    logging.getLogger("peft").setLevel(logging.WARNING)

    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    aug_cfg = cfg.get("augmentation", {})

    fold_output_dir.mkdir(parents=True, exist_ok=True)
    is_kfold = fold_num is not None
    fold_str = f"Fold {fold_num}/{n_folds}" if is_kfold else "Training"

    print(f"\n{'=' * 70}")
    print(f"{fold_str} | Train: {len(train_df)} | Val: {len(val_df)}")
    print(f"{'=' * 70}")

    image_ext = data_cfg.get("image_ext", ".jpg")
    train_dataset = build_dataset(train_df, image_dir, image_ext, "train")
    val_dataset = build_dataset(val_df, image_dir, image_ext, "val")

    model, processor = setup_model(cfg)

    max_pixels = train_cfg["max_pixels"]
    max_length = train_cfg.get("max_seq_length", 2048)
    strict = train_cfg.get("strict_masking", True)

    # Two collators. The eval one has no augmenter, so eval_loss is a fixed
    # measurement rather than a fresh random draw each time.
    train_collator = OCRCollator(processor, max_pixels, ImageAugmenter(aug_cfg),
                                 max_length, strict)
    eval_collator = OCRCollator(processor, max_pixels, None, max_length, strict)

    eval_bs = train_cfg.get("eval_batch_size", train_cfg["batch_size"])
    metric = train_cfg.get("metric_for_best_model", "eval_loss")

    # Express warmup as an explicit integer step count. transformers v5 removed
    # warmup_ratio; its warmup_steps accepts a float there but not on v4, so an
    # int is the only form that works on both. (The original script did this too.)
    steps_per_epoch, total_steps, warmup_steps = compute_schedule(
        n_samples=len(train_dataset),
        batch_size=train_cfg["batch_size"],
        grad_accum=train_cfg["gradient_accumulation_steps"],
        epochs=train_cfg["epochs"],
        warmup_ratio=train_cfg.get("warmup_ratio", 0.1),
    )
    print(f"Steps: {steps_per_epoch}/epoch × {train_cfg['epochs']} epochs = {total_steps} total")

    # Reduce logging frequency for non-kfold (less spam)
    log_steps = train_cfg.get("logging_steps", 50 if not is_kfold else 25)

    ta_kwargs = dict(
        output_dir=str(fold_output_dir),
        per_device_train_batch_size=train_cfg["batch_size"],
        per_device_eval_batch_size=eval_bs,
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        num_train_epochs=train_cfg["epochs"],
        learning_rate=train_cfg["learning_rate"],
        warmup_steps=warmup_steps,
        lr_scheduler_type=train_cfg["lr_scheduler"],
        weight_decay=train_cfg["weight_decay"],
        max_grad_norm=train_cfg["max_grad_norm"],
        bf16=True,
        logging_steps=log_steps,
        logging_strategy="steps",
        logging_first_step=False,
        logging_nan_inf_filter=True,  # Suppress NaN/inf warnings
        eval_strategy="steps",
        eval_steps=train_cfg["eval_steps"],
        save_strategy="steps",
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model=metric,
        greater_is_better=False,  # both eval_loss and eval_cer: lower is better
        dataloader_num_workers=train_cfg["num_workers"],
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        label_names=["labels"],  # PEFT wrappers hide this from Trainer's autodetect
        report_to="none",
        seed=data_cfg.get("seed", 42),
        data_seed=data_cfg.get("seed", 42),
        disable_tqdm=True,  # Disable all progress bars (too verbose)
        log_level="error",  # Suppress info logs from trainer
        log_level_replica="error",
    )
    args = TrainingArguments(**supported_kwargs(TrainingArguments, ta_kwargs))

    callbacks = []

    # Progress callback (replaces verbose tqdm with clean updates)
    if not is_kfold:
        callbacks.append(ProgressCallback(log_every_n_steps=log_steps))

    # Optional CER computation callback
    if train_cfg.get("compute_cer", True):
        callbacks.append(CERCallback(
            processor=processor,
            dataset=val_dataset,
            max_pixels=max_pixels,
            n_samples=train_cfg.get("cer_samples", 200),
            max_new_tokens=cfg.get("inference", {}).get("max_new_tokens", 256),
            batch_size=eval_bs,
            seed=data_cfg.get("seed", 42),
            is_kfold=is_kfold,
            verbose=False,  # Only print improvements
        ))
    elif metric == "eval_cer":
        raise ValueError("metric_for_best_model='eval_cer' requires compute_cer: true")

    # Early stopping callback (prevent wasting compute on plateaued folds)
    # IMPORTANT: Early stopping should monitor eval_loss (stable signal), while
    # metric_for_best_model can be eval_cer (what matters for leaderboard).
    early_stop_cfg = train_cfg.get("early_stopping", {})
    if early_stop_cfg.get("enabled", True):
        patience = early_stop_cfg.get("patience", 3)
        min_delta = early_stop_cfg.get("min_delta", 0.0001)

        # Use dedicated early stopping metric (defaults to eval_loss for stability)
        # eval_loss: computed on full val set, deterministic, low variance
        # eval_cer: computed on 200 samples, generative, high variance (bad for early stop)
        early_stop_metric = early_stop_cfg.get("metric", "eval_loss")

        callbacks.append(EarlyStoppingCallback(
            patience=patience,
            min_delta=min_delta,
            metric=early_stop_metric,
            greater_is_better=False,  # eval_loss and eval_cer: lower is better
            verbose=False,  # Don't print patience counter every eval
        ))

    # Create trainer without default callbacks (too verbose)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=train_collator,
        callbacks=callbacks,
    )

    # Remove default progress callback (replaced by our ProgressCallback)
    trainer.remove_callback(transformers.trainer_callback.ProgressCallback)
    # Trainer has no eval_data_collator arg; swap it in for the eval dataloader only.
    _base_get_eval = trainer.get_eval_dataloader

    def get_eval_dataloader(eval_dataset=None):
        saved, trainer.data_collator = trainer.data_collator, eval_collator
        try:
            return _base_get_eval(eval_dataset)
        finally:
            trainer.data_collator = saved

    trainer.get_eval_dataloader = get_eval_dataloader

    # Check for existing checkpoint to resume from
    resume_checkpoint = None
    if train_cfg.get("resume_from_checkpoint", True):
        resume_checkpoint = get_last_checkpoint(fold_output_dir)
        if resume_checkpoint:
            print(f"⏩ Resuming from {Path(resume_checkpoint).name}\n")
        else:
            print(f"Starting training...\n")
    else:
        print(f"Starting training...\n")

    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # Check if early stopping was triggered
    early_stop_callback = None
    for cb in callbacks:
        if isinstance(cb, EarlyStoppingCallback):
            early_stop_callback = cb
            break

    early_stopped = early_stop_callback.stopped_epoch if early_stop_callback else None

    # load_best_model_at_end=True means the in-memory model IS the best
    # checkpoint here. The original script also wrote a "final/" dir with these
    # same weights, which was misleading - dropped.
    best_dir = fold_output_dir / "best"
    trainer.save_model(str(best_dir))
    processor.save_pretrained(str(best_dir))

    history = [h for h in trainer.state.log_history if "eval_loss" in h]
    result = {
        "fold": fold_num,
        "best_metric": trainer.state.best_metric,
        "metric_name": metric,
        "best_eval_loss": min((h["eval_loss"] for h in history), default=None),
        "best_eval_cer": min((h["eval_cer"] for h in history if "eval_cer" in h), default=None),
        "best_eval_wer": min((h["eval_wer"] for h in history if "eval_wer" in h), default=None),
        "best_eval_score": min((h["eval_score"] for h in history if "eval_score" in h), default=None),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "mask_failures": train_collator.mask_failures + eval_collator.mask_failures,
        "early_stopped": early_stopped is not None,
        "stopped_epoch": early_stopped,
        "epochs_completed": trainer.state.epoch,
    }

    print(f"\n{'=' * 70}")
    status = f"Early stopped at epoch {early_stopped:.1f}" if early_stopped else f"Completed {result['epochs_completed']:.1f} epochs"
    print(f"✓ {fold_str} | {metric}={result['best_metric']:.4f} | {status}")

    # Print competition metrics
    if result["best_eval_score"] is not None:
        print(f"  Competition score (0.5*WER + 0.5*CER): {result['best_eval_score']:.4f}")
    if result["best_eval_cer"] is not None and result["best_eval_wer"] is not None:
        print(f"  CER: {result['best_eval_cer']:.4f} | WER: {result['best_eval_wer']:.4f}")

    if result["mask_failures"]:
        print(f"  ⚠️  {result['mask_failures']} samples failed label masking")
    print(f"  Saved: {best_dir}")
    print(f"{'=' * 70}\n")

    # Explicit teardown. Sequential folds otherwise fragment VRAM and OOM
    # around fold 4.
    del trainer, model, train_collator, eval_collator
    gc.collect()
    torch.cuda.empty_cache()

    return result


def compute_special_char_density(text: str) -> float:
    """
    Compute special character density as stratification proxy.

    Special characters indicate:
    - Document type (legal docs with £, dates, formal punctuation)
    - Scribe style (abbreviations, dashes)
    - Transcription complexity

    Returns: ratio of special chars to total chars (0.0 to 1.0)
    """
    if not text or pd.isna(text):
        return 0.0

    text = str(text)
    # Count non-alphanumeric, non-space characters
    special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
    return special_chars / max(len(text), 1)


def compute_digit_density(text: str) -> float:
    """
    Compute digit/number density as stratification proxy.

    Digits indicate:
    - Dates (1842, 15th)
    - Monetary amounts (£25-10-6)
    - Measurements (3 acres)
    - Different OCR challenge (digits often harder than letters)

    Returns: ratio of digits to total chars (0.0 to 1.0)
    """
    if not text or pd.isna(text):
        return 0.0

    text = str(text)
    digits = sum(1 for c in text if c.isdigit())
    return digits / max(len(text), 1)


def compute_uppercase_ratio(text: str) -> float:
    """
    Compute uppercase letter ratio as formality/emphasis indicator.

    Uppercase indicates:
    - Proper nouns (John Smith, London)
    - Formal language (WITNESSED, SEALED)
    - Emphasis and titles (Mr., Esq.)
    - Different capitalization patterns across document types

    Returns: ratio of uppercase letters to total letters (0.0 to 1.0)
    """
    if not text or pd.isna(text):
        return 0.0

    text = str(text)
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    uppercase = sum(1 for c in letters if c.isupper())
    return uppercase / len(letters)


def compute_lexical_diversity(text: str) -> float:
    """
    Compute lexical diversity (unique word ratio) as vocabulary complexity indicator.

    Lexical diversity indicates:
    - Repetitive/formulaic language (low diversity): "the said party... the said party"
    - Rich vocabulary (high diversity): "signed, sealed, witnessed, delivered, dated"
    - Document type (legal templates vs descriptive narratives)

    Returns: ratio of unique words to total words (0.0 to 1.0)
    """
    if not text or pd.isna(text):
        return 0.0

    words = str(text).lower().split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def compute_avg_word_length(text: str) -> float:
    """
    Compute average word length as vocabulary complexity indicator.

    Word length indicates:
    - Simple vocabulary (short words): "I see the man go"
    - Complex vocabulary (long words): "aforementioned beneficiary witnessed"
    - Different OCR challenge (longer words = more opportunities for errors)

    Returns: average characters per word
    """
    if not text or pd.isna(text):
        return 0.0

    words = [w for w in str(text).split() if w]
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)


def make_splits(df: pd.DataFrame, data_cfg: dict):
    """Yield (train_df, val_df, fold_num). Stratified by semantic text properties from ground truth. Grouped by group_col when present."""
    df = df.copy()

    # STEP 1: Identify duplicate texts to prevent train/val leakage
    print("Checking for duplicate texts...")
    df["_text_clean"] = df["Target"].astype(str).str.lower().str.strip()

    # Find duplicate texts and assign group IDs
    text_counts = df["_text_clean"].value_counts()
    duplicate_texts = text_counts[text_counts > 1]

    if len(duplicate_texts) > 0:
        print(f"  Found {len(duplicate_texts)} unique texts with duplicates ({duplicate_texts.sum()} total samples)")
        print(f"  Assigning duplicate group IDs to keep copies together in same split...")

        # Assign a unique group ID to each duplicate text
        df["_dup_group"] = -1  # -1 = not a duplicate
        for group_id, dup_text in enumerate(duplicate_texts.index):
            df.loc[df["_text_clean"] == dup_text, "_dup_group"] = group_id

        n_grouped = (df["_dup_group"] >= 0).sum()
        print(f"  Grouped {n_grouped} duplicate samples into {len(duplicate_texts)} groups")
    else:
        print("  No duplicate texts found")
        df["_dup_group"] = -1

    # STEP 2: Compute semantic text-based features (fast, interpretable, ground-truth based)
    print("Computing semantic text features (digits, uppercase, lexical diversity, special chars, word length)...")
    df["_digit_density"] = df["Target"].apply(compute_digit_density)
    df["_uppercase_ratio"] = df["Target"].apply(compute_uppercase_ratio)
    df["_lexical_diversity"] = df["Target"].apply(compute_lexical_diversity)
    df["_special_char_density"] = df["Target"].apply(compute_special_char_density)
    df["_avg_word_length"] = df["Target"].apply(compute_avg_word_length)

    # Stratify by digit density (2 bins: has_numbers vs minimal_numbers)
    try:
        df["_digit_bin"] = pd.qcut(df["_digit_density"], q=2, labels=["no_nums", "has_nums"], duplicates="drop")
    except ValueError:
        df["_digit_bin"] = "has_nums"

    # Stratify by uppercase ratio (2 bins: informal vs formal)
    try:
        df["_upper_bin"] = pd.qcut(df["_uppercase_ratio"], q=2, labels=["informal", "formal"], duplicates="drop")
    except ValueError:
        df["_upper_bin"] = "informal"

    # Stratify by lexical diversity (3 bins: repetitive/moderate/diverse vocabulary)
    try:
        df["_lex_bin"] = pd.qcut(df["_lexical_diversity"], q=3, labels=["repetitive", "moderate", "diverse"], duplicates="drop")
    except ValueError:
        df["_lex_bin"] = "moderate"

    # Combine: digits (2) × uppercase (2) × lexical_diversity (3) = 12 bins
    # Captures: numeric content, formality, vocabulary complexity
    df["_bin"] = (df["_digit_bin"].astype(str) + "_" +
                  df["_upper_bin"].astype(str) + "_" +
                  df["_lex_bin"].astype(str))

    k_folds = data_cfg.get("k_folds", 1)
    seed = data_cfg.get("seed", 42)
    group_col = data_cfg.get("group_col")

    # Store original index for duplicate-aware splitting
    df["_orig_idx"] = df.index

    helper = ["_text_clean", "_dup_group", "_orig_idx", "_digit_density", "_uppercase_ratio", "_lexical_diversity",
              "_special_char_density", "_avg_word_length", "_digit_bin", "_upper_bin", "_lex_bin", "_bin"]

    if group_col and group_col not in df.columns:
        raise ValueError(f"group_col '{group_col}' not in the CSV columns")

    if k_folds > 1:
        # K-fold with duplicate awareness
        has_duplicates = (df["_dup_group"] >= 0).any()

        if has_duplicates or group_col:
            # Use StratifiedGroupKFold to keep duplicates together
            if not GROUP_KFOLD_AVAILABLE:
                raise RuntimeError("K-fold with duplicates needs scikit-learn >= 1.0 for StratifiedGroupKFold")

            # Create synthetic group column combining duplicates + user group_col
            if has_duplicates and group_col:
                # Combine both: duplicate group + user-specified group
                df["_fold_group"] = df["_dup_group"].astype(str) + "_" + df[group_col].astype(str)
                print(f"K-fold with duplicate awareness + grouped on '{group_col}'")
            elif has_duplicates:
                # Use duplicate group as fold group
                # Assign unique group ID to non-duplicates
                max_dup_group = df["_dup_group"].max()
                df["_fold_group"] = df.apply(
                    lambda row: row["_dup_group"] if row["_dup_group"] >= 0 else max_dup_group + 1 + row.name,
                    axis=1
                )
                print(f"K-fold with duplicate awareness (keeps {(df['_dup_group'] >= 0).sum()} duplicate samples together)")
            else:
                # Only user-specified group
                df["_fold_group"] = df[group_col]
                print(f"K-fold grouped on '{group_col}' ({df[group_col].nunique()} groups)")

            helper.append("_fold_group")

            splitter = StratifiedGroupKFold(n_splits=k_folds, shuffle=True, random_state=seed)
            split_iter = splitter.split(df, df["_bin"], groups=df["_fold_group"])
        else:
            # Standard k-fold (no duplicates, no grouping)
            print(f"Stratified {k_folds}-fold by semantic features: digits (2) × uppercase (2) × lexical_diversity (3) = 12 bins.")
            splitter = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
            split_iter = splitter.split(df, df["_bin"])

        for fold_num, (tr, va) in enumerate(split_iter, 1):
            train_df = df.iloc[tr].copy()
            val_df = df.iloc[va].copy()

            # Verify no duplicate leakage across folds
            if has_duplicates:
                train_texts = set(train_df["_text_clean"])
                val_texts = set(val_df["_text_clean"])
                leaked = train_texts & val_texts
                if leaked:
                    print(f"  ⚠️  Fold {fold_num}: {len(leaked)} duplicate texts leaked (should be 0)!")
                else:
                    print(f"  ✓ Fold {fold_num}: No duplicate leakage")

            yield (train_df.drop(columns=helper).copy(),
                   val_df.drop(columns=helper).copy(),
                   fold_num)
    else:
        # STEP 3: Split with duplicate awareness
        has_duplicates = (df["_dup_group"] >= 0).any()

        if has_duplicates:
            print("Splitting with duplicate-group awareness (keeps duplicate texts together)...")

            # Strategy: For each duplicate group, assign all members to train or val together
            # 1. Get one representative per duplicate group
            # 2. Combine with unique samples
            # 3. Split representatives + unique samples with stratification
            # 4. Propagate split assignment to all group members

            # Separate duplicates from unique samples
            dup_mask = df["_dup_group"] >= 0
            dup_df = df[dup_mask].copy()
            unique_df = df[~dup_mask].copy()

            # Get one representative per duplicate group (use first occurrence)
            # Keep _orig_idx so we can map back
            dup_representatives = dup_df.groupby("_dup_group", as_index=False).first()

            # Combine representatives with unique samples for stratified splitting
            split_df = pd.concat([unique_df, dup_representatives], ignore_index=True)

            # Split representatives + unique samples with stratification
            split_train, split_val = train_test_split(
                split_df, test_size=data_cfg["val_split"], stratify=split_df["_bin"], random_state=seed
            )

            # Now propagate: which duplicate groups went to train vs val?
            train_dup_groups = set(split_train[split_train["_dup_group"] >= 0]["_dup_group"])
            val_dup_groups = set(split_val[split_val["_dup_group"] >= 0]["_dup_group"])

            # Build train/val by:
            # - Unique samples from split_train/split_val directly
            # - All members of duplicate groups assigned to train/val
            train_mask = (df["_dup_group"] == -1) & (df["_orig_idx"].isin(split_train["_orig_idx"]))
            train_mask = train_mask | (df["_dup_group"].isin(train_dup_groups))

            val_mask = (df["_dup_group"] == -1) & (df["_orig_idx"].isin(split_val["_orig_idx"]))
            val_mask = val_mask | (df["_dup_group"].isin(val_dup_groups))

            train_df = df[train_mask].copy()
            val_df = df[val_mask].copy()

            # Verify no leakage
            train_texts = set(train_df["_text_clean"])
            val_texts = set(val_df["_text_clean"])
            leaked = train_texts & val_texts
            if leaked:
                print(f"  ⚠️  WARNING: {len(leaked)} texts still leaked (should be 0)!")
            else:
                print(f"  ✓ No duplicate text leakage - all copies kept together")

        else:
            # No duplicates: standard stratified split
            train_df, val_df = train_test_split(
                df, test_size=data_cfg["val_split"], stratify=df["_bin"], random_state=seed
            )

        # Log distributions to verify stratification is working
        train_digits = train_df["_digit_density"]
        val_digits = val_df["_digit_density"]
        train_upper = train_df["_uppercase_ratio"]
        val_upper = val_df["_uppercase_ratio"]
        train_lex = train_df["_lexical_diversity"]
        val_lex = val_df["_lexical_diversity"]
        train_spec = train_df["_special_char_density"]
        val_spec = val_df["_special_char_density"]
        train_wlen = train_df["_avg_word_length"]
        val_wlen = val_df["_avg_word_length"]

        print(f"Stratified split by semantic features: digits (2) × uppercase (2) × lexical_diversity (3) = 12 bins:")
        print(f"  Train: {len(train_df)} samples")
        print(f"    Digit density: min={train_digits.min():.3f}, median={train_digits.median():.3f}, max={train_digits.max():.3f}")
        print(f"    Uppercase ratio: min={train_upper.min():.3f}, median={train_upper.median():.3f}, max={train_upper.max():.3f}")
        print(f"    Lexical diversity: min={train_lex.min():.3f}, median={train_lex.median():.3f}, max={train_lex.max():.3f}")
        print(f"    Special chars: min={train_spec.min():.3f}, median={train_spec.median():.3f}, max={train_spec.max():.3f}")
        print(f"    Avg word length: min={train_wlen.min():.1f}, median={train_wlen.median():.1f}, max={train_wlen.max():.1f}")
        print(f"  Val:   {len(val_df)} samples")
        print(f"    Digit density: min={val_digits.min():.3f}, median={val_digits.median():.3f}, max={val_digits.max():.3f}")
        print(f"    Uppercase ratio: min={val_upper.min():.3f}, median={val_upper.median():.3f}, max={val_upper.max():.3f}")
        print(f"    Lexical diversity: min={val_lex.min():.3f}, median={val_lex.median():.3f}, max={val_lex.max():.3f}")
        print(f"    Special chars: min={val_spec.min():.3f}, median={val_spec.median():.3f}, max={val_spec.max():.3f}")
        print(f"    Avg word length: min={val_wlen.min():.1f}, median={val_wlen.median():.1f}, max={val_wlen.max():.1f}")

        yield train_df.drop(columns=helper), val_df.drop(columns=helper), None


def train(cfg: dict):
    data_cfg, train_cfg = cfg["data"], cfg["training"]

    train_csv = REPO_ROOT / data_cfg["train_csv"]
    image_dir = REPO_ROOT / data_cfg["image_dir"]
    base_output_dir = REPO_ROOT / train_cfg["output_dir"]

    df = pd.read_csv(train_csv)
    nan_count = df['Target'].isna().sum()
    print(f"Loaded {len(df)} samples from {train_csv.name}" + (f" ({nan_count} NaN targets)" if nan_count > 0 else ""))

    k_folds = data_cfg.get("k_folds", 1)
    n_folds = k_folds if k_folds > 1 else None
    results = []

    for train_df, val_df, fold_num in make_splits(df, data_cfg):
        out_dir = base_output_dir / f"fold_{fold_num}" if fold_num else base_output_dir
        results.append(train_single_fold(
            train_df, val_df, image_dir, out_dir, cfg, fold_num, n_folds
        ))

    if len(results) > 1:
        print(f"\n{'=' * 70}")
        print(f"K-FOLD SUMMARY ({n_folds} folds)")
        print(f"{'=' * 70}")
        for r in results:
            stop_tag = f" [stopped @ {r['stopped_epoch']:.1f}]" if r.get("early_stopped") else ""
            score_str = f" | score={r['best_eval_score']:.4f}" if r.get("best_eval_score") is not None else ""
            cer_wer = ""
            if r.get("best_eval_cer") is not None and r.get("best_eval_wer") is not None:
                cer_wer = f" (cer={r['best_eval_cer']:.4f}, wer={r['best_eval_wer']:.4f})"
            print(f"  Fold {r['fold']}: loss={r['best_eval_loss']:.4f}{score_str}{cer_wer}{stop_tag}")

        losses = [r["best_eval_loss"] for r in results]
        scores = [r["best_eval_score"] for r in results if r.get("best_eval_score") is not None]
        cers = [r["best_eval_cer"] for r in results if r.get("best_eval_cer") is not None]
        wers = [r["best_eval_wer"] for r in results if r.get("best_eval_wer") is not None]

        print(f"\nAverage: loss={np.mean(losses):.4f}±{np.std(losses):.4f}", end="")
        if scores:
            print(f" | score={np.mean(scores):.4f}±{np.std(scores):.4f}", end="")
        if cers and wers:
            print(f" (cer={np.mean(cers):.4f}, wer={np.mean(wers):.4f})")
        else:
            print()

        print(f"{'=' * 70}\n")

    base_output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = base_output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({"config": cfg, "results": results}, f, indent=2, default=str)
    print(f"Summary -> {summary_path}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--folds", type=int, default=None,
                        help="Override data.k_folds (use 1 for a quick single-split run)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training.epochs")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start training from scratch, ignoring existing checkpoints")
    parser.add_argument("--no-early-stop", action="store_true",
                        help="Disable early stopping (train full epochs regardless of plateau)")
    args = parser.parse_args()

    config_path = SCRIPT_DIR / args.config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if args.folds is not None:
        cfg["data"]["k_folds"] = args.folds
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.no_resume:
        cfg["training"]["resume_from_checkpoint"] = False
    if args.no_early_stop:
        if "early_stopping" not in cfg["training"]:
            cfg["training"]["early_stopping"] = {}
        cfg["training"]["early_stopping"]["enabled"] = False

    seed = cfg["data"].get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    train(cfg)


if __name__ == "__main__":
    main()