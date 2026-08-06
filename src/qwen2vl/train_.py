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
import json
import random
import argparse
import logging
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
    print(f"  {label}: {len(samples)} kept / {len(df)} rows", end="")
    if n_dropped:
        detail = ", ".join(f"{k}={v}" for k, v in dropped.items() if v)
        print(f"  [DROPPED {n_dropped}: {detail}]")
    else:
        print()

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
                 max_new_tokens: int = 256, batch_size: int = 4, seed: int = 42):
        self.processor = processor
        self.max_pixels = max_pixels
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size

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

        cer = corpus_cer(preds, refs)
        metrics["eval_cer"] = cer
        print(f"  eval_cer = {cer:.4f}  (greedy, n={len(preds)})")


# ─────────────────────────────────────────────────────────────
# MODEL SETUP
# ─────────────────────────────────────────────────────────────

def setup_model(cfg: dict):
    """Load model + processor and attach LoRA."""
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    model_name = model_cfg["name"]

    model_class, family = resolve_model_class(model_name)
    print(f"Loading {model_name}  ->  {model_class.__name__} (family={family})")

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

    # Qwen3-VL's from_pretrained uses `dtype`; older ones use `torch_dtype`.
    if family == "qwen3":
        model_kwargs["dtype"] = dtype
    else:
        model_kwargs["torch_dtype"] = dtype

    if model_cfg.get("use_flash_attention", True):
        try:
            import flash_attn  # noqa: F401
            model_kwargs["attn_implementation"] = "flash_attention_2"
            print("  attn: flash_attention_2")
        except ImportError:
            print("  attn: sdpa (flash-attn not installed)")
            model_kwargs["attn_implementation"] = "sdpa"
    else:
        model_kwargs["attn_implementation"] = "sdpa"
        print("  attn: sdpa (disabled in config)")

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
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Fail loudly if the target_modules matched nothing (a typo yields a model
    # with zero trainable params that trains happily and learns nothing).
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if n_trainable == 0:
        raise RuntimeError("LoRA matched zero modules. Check lora_target_modules.")

    return model, processor


# ─────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────

def train_single_fold(train_df, val_df, image_dir, fold_output_dir, cfg,
                      fold_num=None, n_folds=None):
    """Train one fold (or a single model when fold_num is None). Returns metrics dict."""
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("accelerate").setLevel(logging.WARNING)

    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    aug_cfg = cfg.get("augmentation", {})

    fold_output_dir.mkdir(parents=True, exist_ok=True)
    is_kfold = fold_num is not None
    fold_str = f"Fold {fold_num}/{n_folds}" if is_kfold else "Model"

    print(f"\n{'=' * 70}")
    print(f"{fold_str} - Train: {len(train_df)}, Val: {len(val_df)}")
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

    args = TrainingArguments(
        output_dir=str(fold_output_dir),
        per_device_train_batch_size=train_cfg["batch_size"],
        per_device_eval_batch_size=eval_bs,
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        num_train_epochs=train_cfg["epochs"],
        learning_rate=train_cfg["learning_rate"],
        warmup_ratio=train_cfg.get("warmup_ratio", 0.1),
        lr_scheduler_type=train_cfg["lr_scheduler"],
        weight_decay=train_cfg["weight_decay"],
        max_grad_norm=train_cfg["max_grad_norm"],
        bf16=True,
        logging_steps=train_cfg.get("logging_steps", 25),
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
        disable_tqdm=is_kfold,
    )

    callbacks = []
    if train_cfg.get("compute_cer", True):
        callbacks.append(CERCallback(
            processor=processor,
            dataset=val_dataset,
            max_pixels=max_pixels,
            n_samples=train_cfg.get("cer_samples", 200),
            max_new_tokens=cfg.get("inference", {}).get("max_new_tokens", 256),
            batch_size=eval_bs,
            seed=data_cfg.get("seed", 42),
        ))
    elif metric == "eval_cer":
        raise ValueError("metric_for_best_model='eval_cer' requires compute_cer: true")

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=train_collator,
        callbacks=callbacks,
    )
    # Trainer has no eval_data_collator arg; swap it in for the eval dataloader only.
    _base_get_eval = trainer.get_eval_dataloader

    def get_eval_dataloader(eval_dataset=None):
        saved, trainer.data_collator = trainer.data_collator, eval_collator
        try:
            return _base_get_eval(eval_dataset)
        finally:
            trainer.data_collator = saved

    trainer.get_eval_dataloader = get_eval_dataloader

    print(f"\nTraining {fold_str}...")
    trainer.train()

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
        "best_eval_cer": min((h["eval_cer"] for h in history if "eval_cer" in h),
                             default=None),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "mask_failures": train_collator.mask_failures + eval_collator.mask_failures,
    }

    print(f"\n{'=' * 70}")
    print(f"{fold_str} complete - best {metric}: {result['best_metric']:.4f}")
    if result["best_eval_cer"] is not None:
        print(f"  best eval_cer: {result['best_eval_cer']:.4f}")
    if result["mask_failures"]:
        print(f"  WARNING: {result['mask_failures']} samples failed label masking")
    print(f"  saved -> {best_dir}")
    print(f"{'=' * 70}\n")

    # Explicit teardown. Sequential folds otherwise fragment VRAM and OOM
    # around fold 4.
    del trainer, model, train_collator, eval_collator
    gc.collect()
    torch.cuda.empty_cache()

    return result


def make_splits(df: pd.DataFrame, data_cfg: dict):
    """Yield (train_df, val_df, fold_num). Grouped by group_col when present."""
    df = df.copy()
    df["_len"] = df["Target"].astype(str).str.len()
    df["_bin"] = pd.qcut(df["_len"], q=5, labels=False, duplicates="drop")

    k_folds = data_cfg.get("k_folds", 1)
    seed = data_cfg.get("seed", 42)
    group_col = data_cfg.get("group_col")
    helper = ["_len", "_bin"]

    if group_col and group_col not in df.columns:
        raise ValueError(f"group_col '{group_col}' not in the CSV columns")

    if k_folds > 1:
        if group_col:
            if not GROUP_KFOLD_AVAILABLE:
                raise RuntimeError("group_col needs scikit-learn >= 1.0 for StratifiedGroupKFold")
            print(f"Grouped 5-fold on '{group_col}' "
                  f"({df[group_col].nunique()} groups) - prevents writer/page leakage")
            splitter = StratifiedGroupKFold(n_splits=k_folds, shuffle=True, random_state=seed)
            split_iter = splitter.split(df, df["_bin"], groups=df[group_col])
        else:
            print("Stratified 5-fold by text length. NOTE: if rows share a source "
                  "page or scribe, set data.group_col to avoid leakage.")
            splitter = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
            split_iter = splitter.split(df, df["_bin"])

        for fold_num, (tr, va) in enumerate(split_iter, 1):
            yield (df.iloc[tr].drop(columns=helper).copy(),
                   df.iloc[va].drop(columns=helper).copy(),
                   fold_num)
    else:
        train_df, val_df = train_test_split(
            df, test_size=data_cfg["val_split"], stratify=df["_bin"], random_state=seed
        )
        yield train_df.drop(columns=helper), val_df.drop(columns=helper), None


def train(cfg: dict):
    data_cfg, train_cfg = cfg["data"], cfg["training"]

    train_csv = REPO_ROOT / data_cfg["train_csv"]
    image_dir = REPO_ROOT / data_cfg["image_dir"]
    base_output_dir = REPO_ROOT / train_cfg["output_dir"]

    print(f"Loading {train_csv}")
    df = pd.read_csv(train_csv)
    print(f"  {len(df)} rows, {df['Target'].isna().sum()} NaN targets")

    k_folds = data_cfg.get("k_folds", 1)
    n_folds = k_folds if k_folds > 1 else None
    results = []

    for train_df, val_df, fold_num in make_splits(df, data_cfg):
        out_dir = base_output_dir / f"fold_{fold_num}" if fold_num else base_output_dir
        results.append(train_single_fold(
            train_df, val_df, image_dir, out_dir, cfg, fold_num, n_folds
        ))

    if len(results) > 1:
        print(f"\n{'=' * 70}\nK-FOLD RESULTS\n{'=' * 70}")
        for r in results:
            line = f"  Fold {r['fold']}/{n_folds}: eval_loss={r['best_eval_loss']:.4f}"
            if r["best_eval_cer"] is not None:
                line += f"  eval_cer={r['best_eval_cer']:.4f}"
            print(line)

        losses = [r["best_eval_loss"] for r in results]
        print(f"\n  eval_loss  {np.mean(losses):.4f} +/- {np.std(losses):.4f}")

        cers = [r["best_eval_cer"] for r in results if r["best_eval_cer"] is not None]
        if cers:
            print(f"  eval_cer   {np.mean(cers):.4f} +/- {np.std(cers):.4f}")
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
    args = parser.parse_args()

    config_path = SCRIPT_DIR / args.config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if args.folds is not None:
        cfg["data"]["k_folds"] = args.folds
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs

    seed = cfg["data"].get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    train(cfg)


if __name__ == "__main__":
    main()