"""
Qwen2-VL Fine-tuning for Historical HTR
Optimized for A100 80GB - Competition Setup
"""

import os
import random
import argparse
from pathlib import Path

import yaml
import torch
import pandas as pd
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from sklearn.model_selection import train_test_split, StratifiedKFold
from datasets import Dataset
from tqdm import tqdm

from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent

OCR_PROMPT = (
    "Transcribe the handwritten text in this image exactly as written. "
    "Preserve spelling, punctuation, and line breaks. Output only the transcription."
)


# ─────────────────────────────────────────────────────────────
# AUGMENTATION
# ─────────────────────────────────────────────────────────────

class ImageAugmenter:
    """
    Augmentations for historical document images.

    Philosophy: Conservative augmentation for already-degraded documents.
    - Focus on realistic scanning variations (brightness, contrast, rotation)
    - Avoid artificial degradation (blur, noise) on already-faded text
    - All augmentations are configurable and can be disabled (p=0.0)
    """

    def __init__(self, cfg: dict):
        self.enabled = cfg.get("enabled", False)
        self.p_blur = cfg.get("p_blur", 0.2)
        self.p_noise = cfg.get("p_noise", 0.2)
        self.p_brightness = cfg.get("p_brightness", 0.3)
        self.p_contrast = cfg.get("p_contrast", 0.3)
        self.p_rotate = cfg.get("p_rotate", 0.1)
        self.max_rotation = cfg.get("max_rotation", 2)

    def __call__(self, img: Image.Image) -> Image.Image:
        if not self.enabled:
            return img

        # Gaussian blur
        if random.random() < self.p_blur:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

        # Brightness
        if random.random() < self.p_brightness:
            factor = random.uniform(0.8, 1.2)
            img = ImageEnhance.Brightness(img).enhance(factor)

        # Contrast
        if random.random() < self.p_contrast:
            factor = random.uniform(0.8, 1.2)
            img = ImageEnhance.Contrast(img).enhance(factor)

        # Small rotation
        if random.random() < self.p_rotate:
            angle = random.uniform(-self.max_rotation, self.max_rotation)
            img = img.rotate(angle, fillcolor=(255, 255, 255), expand=False)

        # Gaussian noise (via numpy)
        if random.random() < self.p_noise:
            arr = np.array(img).astype(np.float32)
            noise = np.random.normal(0, random.uniform(5, 15), arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)

        return img


# ─────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────

def load_image(path: str, max_pixels: int = 2016000) -> Image.Image:
    """Load and resize image while preserving aspect ratio."""
    img = Image.open(path).convert("RGB")
    w, h = img.size

    # Resize if too large
    pixels = w * h
    if pixels > max_pixels:
        scale = (max_pixels / pixels) ** 0.5
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

    return img


def build_dataset(df: pd.DataFrame, image_dir: Path) -> Dataset:
    """Build HuggingFace dataset from dataframe."""
    samples = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building dataset"):
        img_id = str(row["ID"]).strip()
        target = str(row.get("Target", "")).strip()

        if not img_id or not target:
            continue

        img_path = image_dir / f"{img_id}.jpg"
        if not img_path.exists():
            continue

        samples.append({
            "id": img_id,
            "image_path": str(img_path),
            "text": target,
        })

    print(f"Dataset: {len(samples)} samples")
    return Dataset.from_list(samples)


# ─────────────────────────────────────────────────────────────
# COLLATOR
# ─────────────────────────────────────────────────────────────

class OCRCollator:
    """Data collator for Qwen2-VL OCR training."""

    def __init__(self, processor, max_pixels: int, augmenter: ImageAugmenter = None):
        self.processor = processor
        self.max_pixels = max_pixels
        self.augmenter = augmenter

    def __call__(self, examples: list) -> dict:
        images = []
        texts = []

        for ex in examples:
            img = load_image(ex["image_path"], self.max_pixels)

            # Apply augmentation during training
            if self.augmenter:
                img = self.augmenter(img)

            images.append(img)
            texts.append(ex["text"])

        # Build messages for chat template
        messages_batch = []
        for img, label in zip(images, texts):
            messages_batch.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": OCR_PROMPT},
                        {"type": "image", "image": img},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": label}],
                },
            ])

        # Apply chat template
        texts_formatted = [
            self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            for m in messages_batch
        ]

        # Process batch
        batch = self.processor(
            text=texts_formatted,
            images=images,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        # Create labels (mask everything except assistant response)
        input_ids = batch["input_ids"].clone()
        labels = input_ids.clone()

        tokenizer = self.processor.tokenizer
        pad_id = tokenizer.pad_token_id
        img_token_id = tokenizer.convert_tokens_to_ids(self.processor.image_token)

        for i, label_text in enumerate(texts):
            # Encode the label
            label_ids = tokenizer.encode(label_text, add_special_tokens=False)
            seq = input_ids[i].tolist()

            # Find where label starts in sequence
            start_idx = self._find_subsequence(seq, label_ids)
            if start_idx != -1:
                labels[i, :start_idx] = -100

        # Mask padding and image tokens
        labels[labels == pad_id] = -100
        labels[labels == img_token_id] = -100

        batch["labels"] = labels
        return batch

    @staticmethod
    def _find_subsequence(sequence: list, subsequence: list) -> int:
        """Find start index of subsequence in sequence."""
        n = len(subsequence)
        for i in range(len(sequence) - n + 1):
            if sequence[i:i + n] == subsequence:
                return i
        return -1


# ─────────────────────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────────────────────

class SaveBestCallback(TrainerCallback):
    """Save best model based on eval loss."""

    def __init__(self, output_dir: Path):
        self.best_loss = float("inf")
        self.output_dir = output_dir / "best"

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        eval_loss = metrics.get("eval_loss", float("inf"))

        if eval_loss < self.best_loss:
            self.best_loss = eval_loss
            print(f"\n>>> New best model! Loss: {eval_loss:.4f}")
            control.should_save = True


# ─────────────────────────────────────────────────────────────
# TRAINER SETUP
# ─────────────────────────────────────────────────────────────

def setup_model(cfg: dict):
    """Load and configure model with LoRA."""

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    # Determine dtype
    dtype = torch.bfloat16 if model_cfg.get("torch_dtype") == "bfloat16" else torch.float16

    # Load model
    model_kwargs = {
        "torch_dtype": dtype,
        "device_map": "auto",
        "trust_remote_code": True,
    }

    # Flash attention (optional)
    use_flash = model_cfg.get("use_flash_attention", True)

    print(f"Loading model: {model_cfg['name']}")

    if use_flash:
        try:
            # Try with Flash Attention 2
            model_kwargs["attn_implementation"] = "flash_attention_2"
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_cfg["name"],
                **model_kwargs,
            )
            print("✓ Using Flash Attention 2")
        except Exception as e:
            print(f"⚠️  Flash Attention 2 not available: {e}")
            print("  Falling back to standard attention...")
            # Fallback to standard attention
            model_kwargs.pop("attn_implementation", None)
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_cfg["name"],
                **model_kwargs,
            )
            print("✓ Using standard attention")
    else:
        print("Flash Attention 2 disabled in config")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_cfg["name"],
            **model_kwargs,
        )
        print("✓ Using standard attention")

    # Disable cache for training
    model.config.use_cache = False

    # Setup LoRA
    lora_config = LoraConfig(
        r=train_cfg["lora_r"],
        lora_alpha=train_cfg["lora_alpha"],
        target_modules=train_cfg["lora_target_modules"],
        lora_dropout=train_cfg["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()

    # Load processor
    processor = AutoProcessor.from_pretrained(
        model_cfg["name"],
        trust_remote_code=True,
    )

    return model, processor


def train_single_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    image_dir: Path,
    fold_output_dir: Path,
    cfg: dict,
    fold_num: int = None,
):
    """Train a single fold or single model."""
    train_cfg = cfg["training"]
    aug_cfg = cfg.get("augmentation", {})

    fold_output_dir.mkdir(parents=True, exist_ok=True)

    fold_str = f"Fold {fold_num}" if fold_num is not None else "Model"
    print(f"\n{'='*70}")
    print(f"{fold_str} - Train: {len(train_df)}, Val: {len(val_df)}")
    print(f"{'='*70}\n")

    # Build datasets
    train_dataset = build_dataset(train_df, image_dir)
    val_dataset = build_dataset(val_df, image_dir)

    # Model & processor
    model, processor = setup_model(cfg)

    # Augmenter (only for training)
    augmenter = ImageAugmenter(aug_cfg)

    # Collators
    train_collator = OCRCollator(processor, train_cfg["max_pixels"], augmenter)

    # Training arguments
    args = TrainingArguments(
        output_dir=str(fold_output_dir),
        per_device_train_batch_size=train_cfg["batch_size"],
        per_device_eval_batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        num_train_epochs=train_cfg["epochs"],
        learning_rate=train_cfg["learning_rate"],
        warmup_ratio=train_cfg["warmup_ratio"],
        lr_scheduler_type=train_cfg["lr_scheduler"],
        weight_decay=train_cfg["weight_decay"],
        max_grad_norm=train_cfg["max_grad_norm"],
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=train_cfg["eval_steps"],
        save_strategy="steps",
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model=train_cfg["metric_for_best_model"],
        greater_is_better=False,
        dataloader_num_workers=train_cfg["num_workers"],
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to="none",
    )

    # Callbacks
    callbacks = [SaveBestCallback(fold_output_dir)]

    # Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=train_collator,
        callbacks=callbacks,
    )

    # Train
    print(f"Starting training for {fold_str}...")
    trainer.train()

    # Save final
    final_dir = fold_output_dir / "final"
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))
    print(f"Saved final model to {final_dir}")

    # Save best
    best_dir = fold_output_dir / "best"
    best_dir.mkdir(exist_ok=True)
    trainer.save_model(str(best_dir))
    processor.save_pretrained(str(best_dir))
    print(f"Saved best model to {best_dir}")

    # Get best eval loss
    best_metrics = trainer.state.best_metric
    print(f"{fold_str} - Best eval_loss: {best_metrics:.4f}\n")

    return best_metrics


def train(cfg: dict):
    """Main training function with K-Fold CV support."""

    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    # Paths
    train_csv = REPO_ROOT / data_cfg["train_csv"]
    image_dir = REPO_ROOT / data_cfg["image_dir"]
    base_output_dir = REPO_ROOT / train_cfg["output_dir"]

    # Load data
    print(f"Loading data from {train_csv}")
    df = pd.read_csv(train_csv)

    # Prepare stratification by text length
    print("Preparing stratified splits by text length...")
    df['text_length'] = df['Target'].str.len()
    df['length_bin'] = pd.qcut(df['text_length'], q=5, labels=False, duplicates='drop')

    # K-Fold or simple split
    k_folds = data_cfg.get("k_folds", 1)

    if k_folds > 1:
        # K-Fold Cross-Validation
        print(f"\n{'='*70}")
        print(f"K-FOLD CROSS-VALIDATION: {k_folds} folds")
        print(f"Total samples: {len(df)}")
        print(f"{'='*70}\n")

        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=data_cfg["seed"])
        fold_results = []

        for fold_num, (train_idx, val_idx) in enumerate(skf.split(df, df['length_bin']), 1):
            train_df = df.iloc[train_idx].drop(columns=['text_length', 'length_bin']).copy()
            val_df = df.iloc[val_idx].drop(columns=['text_length', 'length_bin']).copy()

            fold_output_dir = base_output_dir / f"fold_{fold_num}"

            best_loss = train_single_fold(
                train_df=train_df,
                val_df=val_df,
                image_dir=image_dir,
                fold_output_dir=fold_output_dir,
                cfg=cfg,
                fold_num=fold_num,
            )

            fold_results.append({
                'fold': fold_num,
                'eval_loss': best_loss,
                'train_samples': len(train_df),
                'val_samples': len(val_df),
            })

        # Print summary
        print(f"\n{'='*70}")
        print("K-FOLD RESULTS SUMMARY")
        print(f"{'='*70}")
        for r in fold_results:
            print(f"Fold {r['fold']}: eval_loss={r['eval_loss']:.4f} "
                  f"(train={r['train_samples']}, val={r['val_samples']})")

        avg_loss = np.mean([r['eval_loss'] for r in fold_results])
        std_loss = np.std([r['eval_loss'] for r in fold_results])
        print(f"\nAverage eval_loss: {avg_loss:.4f} ± {std_loss:.4f}")
        print(f"{'='*70}\n")

        # Save summary
        summary_file = base_output_dir / "kfold_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"K-Fold Cross-Validation Results ({k_folds} folds)\n")
            f.write(f"{'='*70}\n\n")
            for r in fold_results:
                f.write(f"Fold {r['fold']}: eval_loss={r['eval_loss']:.4f}\n")
            f.write(f"\nAverage: {avg_loss:.4f} ± {std_loss:.4f}\n")
        print(f"Saved summary to {summary_file}")

    else:
        # Simple train/val split (single model)
        print(f"\n{'='*70}")
        print("SIMPLE TRAIN/VAL SPLIT (Single Model)")
        print(f"{'='*70}\n")

        train_df, val_df = train_test_split(
            df,
            test_size=data_cfg["val_split"],
            stratify=df['length_bin'],
            random_state=data_cfg["seed"],
        )

        train_df = train_df.drop(columns=['text_length', 'length_bin'])
        val_df = val_df.drop(columns=['text_length', 'length_bin'])

        train_single_fold(
            train_df=train_df,
            val_df=val_df,
            image_dir=image_dir,
            fold_output_dir=base_output_dir,
            cfg=cfg,
            fold_num=None,
        )


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    config_path = SCRIPT_DIR / args.config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Set seed
    seed = cfg["data"].get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train(cfg)


if __name__ == "__main__":
    main()
