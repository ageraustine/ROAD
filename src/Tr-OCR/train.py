"""
TrOCR Fine-tuning for Historical HTR

Line-level handwriting recognition. This is the second track to run alongside
the Qwen-VL script - the two models fail differently (TrOCR produces garbage on
illegible text, VLMs produce fluent wrong text), so their errors decorrelate and
they ensemble well.

Requires SINGLE-LINE image crops. TrOCR resizes every input to 384x384 with no
aspect-ratio preservation; feed it a full page and the lines collapse into an
unreadable smear.

Same CSV contract as train.py: columns ID, Target; images at <image_dir>/<ID><ext>.
"""

import gc
import os
import math
import json
import random
import inspect
import argparse
import logging
from pathlib import Path

import yaml
import torch
import pandas as pd
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
from sklearn.model_selection import train_test_split, StratifiedKFold
from datasets import Dataset
from tqdm import tqdm

import transformers
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

try:
    from sklearn.model_selection import StratifiedGroupKFold
    GROUP_KFOLD_AVAILABLE = True
except ImportError:
    GROUP_KFOLD_AVAILABLE = False

try:
    from torchvision.transforms import ElasticTransform
    import torchvision.transforms.functional as TF
    ELASTIC_AVAILABLE = True
except ImportError:
    ELASTIC_AVAILABLE = False

TF_MAJOR = int(transformers.__version__.split(".")[0])

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent


# ─────────────────────────────────────────────────────────────
# COMPAT (shared behaviour with train.py)
# ─────────────────────────────────────────────────────────────

def compute_schedule(n_samples, batch_size, grad_accum, epochs, warmup_ratio):
    steps_per_epoch = math.ceil(math.ceil(n_samples / batch_size) / grad_accum)
    total_steps = steps_per_epoch * epochs
    return steps_per_epoch, total_steps, max(1, int(total_steps * warmup_ratio))


def supported_kwargs(cls, kwargs: dict) -> dict:
    """Drop kwargs the installed version doesn't accept (v5 removed warmup_ratio etc)."""
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return kwargs
    params = sig.parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    dropped = sorted(k for k in kwargs if k not in params)
    if dropped:
        print(f"  [compat] {cls.__name__} on transformers {transformers.__version__} "
              f"does not accept: {', '.join(dropped)} - dropped")
    return {k: v for k, v in kwargs.items() if k in params}


# ─────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────

def levenshtein(a, b) -> int:
    """Works on strings (CER) or token lists (WER)."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]


def error_rate(preds, refs, word_level=False) -> float:
    """Aggregate edits / total reference units. Not the mean of per-sample rates."""
    if word_level:
        preds = [p.split() for p in preds]
        refs = [r.split() for r in refs]
    edits = sum(levenshtein(p, r) for p, r in zip(preds, refs))
    total = sum(len(r) for r in refs)
    return edits / max(total, 1)


# ─────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────

class LinePreprocessor:
    """
    Match TrOCR's pretraining domain (IAM: dark text, light background).

    The medieval-manuscript ablation work found contrast normalization to be one
    of three knobs that actually moved CER, so this is worth A/B-ing rather than
    assuming.
    """

    def __init__(self, cfg: dict):
        self.autocontrast = cfg.get("autocontrast", True)
        self.cutoff = cfg.get("autocontrast_cutoff", 2)  # % clipped per tail
        self.grayscale = cfg.get("grayscale", False)
        self.invert_if_dark = cfg.get("invert_if_dark", True)

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.grayscale:
            img = ImageOps.grayscale(img).convert("RGB")

        if self.autocontrast:
            img = ImageOps.autocontrast(img, cutoff=self.cutoff)

        # Light-on-dark scans (some archival negatives) confuse a model trained
        # on dark-on-light. Flip them based on mean intensity.
        if self.invert_if_dark and np.array(ImageOps.grayscale(img)).mean() < 110:
            img = ImageOps.invert(img.convert("RGB"))

        return img.convert("RGB")


class LineAugmenter:
    """
    Training-only. Rotation and elastic are the two that measurably beat the
    baseline in the published HTR ablations; the rest are mild scan-variation.
    """

    def __init__(self, cfg: dict):
        self.enabled = cfg.get("enabled", False)
        self.p_rotate = cfg.get("p_rotate", 0.3)
        self.max_rotation = cfg.get("max_rotation", 2)
        self.p_elastic = cfg.get("p_elastic", 0.3)
        self.elastic_alpha = cfg.get("elastic_alpha", 30.0)
        self.elastic_sigma = cfg.get("elastic_sigma", 5.0)
        self.p_brightness = cfg.get("p_brightness", 0.3)
        self.p_contrast = cfg.get("p_contrast", 0.3)
        self.p_dilate = cfg.get("p_dilate", 0.0)

        self._elastic = None
        if self.enabled and self.p_elastic > 0:
            if ELASTIC_AVAILABLE:
                self._elastic = ElasticTransform(
                    alpha=self.elastic_alpha, sigma=self.elastic_sigma, fill=255
                )
            else:
                print("  [warn] torchvision unavailable - elastic augmentation disabled")

    def __call__(self, img: Image.Image) -> Image.Image:
        if not self.enabled:
            return img

        if random.random() < self.p_rotate:
            angle = random.uniform(-self.max_rotation, self.max_rotation)
            img = img.rotate(angle, fillcolor=(255, 255, 255),
                             expand=False, resample=Image.BILINEAR)

        if self._elastic is not None and random.random() < self.p_elastic:
            img = TF.to_pil_image(self._elastic(TF.to_tensor(img)))

        if random.random() < self.p_brightness:
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))

        if random.random() < self.p_contrast:
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))

        return img


# ─────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────

def build_dataset(df, image_dir: Path, image_ext=".jpg", split_name="") -> Dataset:
    samples = []
    dropped = {"missing_image": 0, "empty_target": 0, "bad_id": 0}

    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc=f"Building {split_name or 'dataset'}", leave=False):
        img_id = str(row["ID"]).strip()
        if not img_id or img_id.lower() == "nan":
            dropped["bad_id"] += 1
            continue

        target = row.get("Target")
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
        print(f"  [DROPPED {n_dropped}: "
              f"{', '.join(f'{k}={v}' for k, v in dropped.items() if v)}]")
    else:
        print()

    if not samples:
        raise RuntimeError(f"{label} is empty. Check image_dir / image_ext / ID column.")
    return Dataset.from_list(samples)


class TrOCRCollator:
    """
    Images -> pixel_values (384x384), text -> labels with pad masked to -100.

    Much simpler than the VLM case: no chat template, no prompt to mask, no
    assistant-span search. The whole target sequence is supervised.
    """

    def __init__(self, processor, preprocessor, augmenter=None, max_target_length=128):
        self.processor = processor
        self.preprocessor = preprocessor
        self.augmenter = augmenter
        self.max_target_length = max_target_length
        self.truncated = 0

    def __call__(self, examples: list) -> dict:
        images, texts = [], []
        for ex in examples:
            img = Image.open(ex["image_path"]).convert("RGB")
            img = self.preprocessor(img)
            if self.augmenter is not None:
                img = self.augmenter(img)
            images.append(img)
            texts.append(ex["text"])

        pixel_values = self.processor(images=images, return_tensors="pt").pixel_values

        tokenized = self.processor.tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt",
        )
        labels = tokenized.input_ids.clone()

        # Count silent truncation - a truncated target teaches the model to stop early.
        self.truncated += int((tokenized.attention_mask.sum(1)
                               >= self.max_target_length).sum())

        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels}


# ─────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────

def setup_model(cfg: dict):
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    name = model_cfg["name"]

    print(f"Loading {name}")
    processor = TrOCRProcessor.from_pretrained(name)
    model = VisionEncoderDecoderModel.from_pretrained(name)

    tok = processor.tokenizer

    # TrOCR ships without these wired up. Skipping this step is the single most
    # common cause of a run that trains fine and then generates nothing.
    model.config.decoder_start_token_id = tok.cls_token_id
    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = tok.sep_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    gen = model.generation_config
    gen.decoder_start_token_id = tok.cls_token_id
    gen.pad_token_id = tok.pad_token_id
    gen.eos_token_id = tok.sep_token_id
    gen.max_length = train_cfg.get("max_target_length", 128)
    gen.num_beams = train_cfg.get("eval_num_beams", 1)
    gen.early_stopping = True
    gen.no_repeat_ngram_size = train_cfg.get("no_repeat_ngram_size", 0)
    gen.length_penalty = train_cfg.get("length_penalty", 1.0)

    # Optional encoder freezing. The medieval-HTR ablation found layer freezing
    # to be one of the three settings that actually mattered on small datasets.
    n_freeze = train_cfg.get("freeze_encoder_layers", 0)
    if n_freeze == -1:
        for p in model.encoder.parameters():
            p.requires_grad = False
        print("  froze the entire encoder")
    elif n_freeze > 0:
        try:
            layers = model.encoder.encoder.layer
        except AttributeError:
            layers = None
        if layers is None:
            print(f"  [warn] could not locate encoder layers; freezing skipped")
        else:
            for p in model.encoder.embeddings.parameters():
                p.requires_grad = False
            for layer in layers[:n_freeze]:
                for p in layer.parameters():
                    p.requires_grad = False
            print(f"  froze embeddings + first {n_freeze}/{len(layers)} encoder layers")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  trainable: {trainable / 1e6:.1f}M / {total / 1e6:.1f}M "
          f"({100 * trainable / total:.1f}%)")

    if train_cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    return model, processor


def make_compute_metrics(processor):
    tok = processor.tokenizer

    def compute_metrics(eval_pred):
        pred_ids, label_ids = eval_pred.predictions, eval_pred.label_ids
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]

        label_ids = np.where(label_ids != -100, label_ids, tok.pad_token_id)

        preds = [s.strip() for s in tok.batch_decode(pred_ids, skip_special_tokens=True)]
        refs = [s.strip() for s in tok.batch_decode(label_ids, skip_special_tokens=True)]

        return {
            "cer": error_rate(preds, refs, word_level=False),
            "wer": error_rate(preds, refs, word_level=True),
        }

    return compute_metrics


# ─────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────

def train_single_fold(train_df, val_df, image_dir, fold_output_dir, cfg,
                      fold_num=None, n_folds=None):
    logging.getLogger("transformers").setLevel(logging.WARNING)

    train_cfg, data_cfg = cfg["training"], cfg["data"]
    fold_output_dir.mkdir(parents=True, exist_ok=True)

    is_kfold = fold_num is not None
    fold_str = f"Fold {fold_num}/{n_folds}" if is_kfold else "Model"
    print(f"\n{'=' * 70}\n{fold_str} - Train: {len(train_df)}, Val: {len(val_df)}\n{'=' * 70}")

    image_ext = data_cfg.get("image_ext", ".jpg")
    train_dataset = build_dataset(train_df, image_dir, image_ext, "train")
    val_dataset = build_dataset(val_df, image_dir, image_ext, "val")

    model, processor = setup_model(cfg)

    preprocessor = LinePreprocessor(cfg.get("preprocessing", {}))
    max_target_length = train_cfg.get("max_target_length", 128)

    train_collator = TrOCRCollator(processor, preprocessor,
                                   LineAugmenter(cfg.get("augmentation", {})),
                                   max_target_length)
    eval_collator = TrOCRCollator(processor, preprocessor, None, max_target_length)

    steps_per_epoch, total_steps, warmup_steps = compute_schedule(
        len(train_dataset), train_cfg["batch_size"],
        train_cfg["gradient_accumulation_steps"], train_cfg["epochs"],
        train_cfg.get("warmup_ratio", 0.1),
    )
    print(f"  schedule: {steps_per_epoch} steps/epoch, {total_steps} total, "
          f"{warmup_steps} warmup")

    metric = train_cfg.get("metric_for_best_model", "eval_cer")

    ta_kwargs = dict(
        output_dir=str(fold_output_dir),
        per_device_train_batch_size=train_cfg["batch_size"],
        per_device_eval_batch_size=train_cfg.get("eval_batch_size",
                                                 train_cfg["batch_size"]),
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        num_train_epochs=train_cfg["epochs"],
        learning_rate=train_cfg["learning_rate"],
        warmup_steps=warmup_steps,  # integer: works on transformers v4 and v5
        lr_scheduler_type=train_cfg.get("lr_scheduler", "cosine"),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        bf16=torch.cuda.is_available(),
        predict_with_generate=True,  # gives compute_metrics real decoded text
        generation_max_length=max_target_length,
        generation_num_beams=train_cfg.get("eval_num_beams", 1),
        logging_steps=train_cfg.get("logging_steps", 25),
        eval_strategy="steps",
        eval_steps=train_cfg["eval_steps"],
        save_strategy="steps",
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg.get("save_total_limit", 2),
        load_best_model_at_end=True,
        metric_for_best_model=metric,
        greater_is_better=False,
        dataloader_num_workers=train_cfg.get("num_workers", 8),
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to="none",
        seed=data_cfg.get("seed", 42),
        data_seed=data_cfg.get("seed", 42),
        disable_tqdm=is_kfold,
    )
    args = Seq2SeqTrainingArguments(**supported_kwargs(Seq2SeqTrainingArguments, ta_kwargs))

    # v5 renamed Trainer's `tokenizer` argument to `processing_class`.
    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=train_collator,
        compute_metrics=make_compute_metrics(processor),
    )
    if "processing_class" in inspect.signature(Seq2SeqTrainer.__init__).parameters:
        trainer_kwargs["processing_class"] = processor
    else:
        trainer_kwargs["tokenizer"] = processor

    trainer = Seq2SeqTrainer(**trainer_kwargs)

    # Swap in the un-augmented collator for eval only. Augmenting validation
    # makes every metric a fresh random draw.
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

    best_dir = fold_output_dir / "best"
    trainer.save_model(str(best_dir))
    processor.save_pretrained(str(best_dir))

    history = [h for h in trainer.state.log_history if "eval_cer" in h]
    result = {
        "fold": fold_num,
        "best_metric": trainer.state.best_metric,
        "metric_name": metric,
        "best_cer": min((h["eval_cer"] for h in history), default=None),
        "best_wer": min((h["eval_wer"] for h in history), default=None),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "truncated_targets": train_collator.truncated,
    }

    print(f"\n{'=' * 70}")
    print(f"{fold_str} complete - CER {result['best_cer']:.4f}  WER {result['best_wer']:.4f}")
    if result["truncated_targets"]:
        print(f"  WARNING: {result['truncated_targets']} targets hit "
              f"max_target_length={max_target_length} and were truncated")
    print(f"  saved -> {best_dir}\n{'=' * 70}\n")

    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def make_splits(df, data_cfg):
    df = df.copy()
    df["_len"] = df["Target"].astype(str).str.len()
    df["_bin"] = pd.qcut(df["_len"], q=5, labels=False, duplicates="drop")

    k_folds = data_cfg.get("k_folds", 1)
    seed = data_cfg.get("seed", 42)
    group_col = data_cfg.get("group_col")
    helper = ["_len", "_bin"]

    if group_col and group_col not in df.columns:
        raise ValueError(f"group_col '{group_col}' not in CSV columns")

    if k_folds > 1:
        if group_col:
            if not GROUP_KFOLD_AVAILABLE:
                raise RuntimeError("group_col needs sklearn >= 1.0")
            print(f"Grouped {k_folds}-fold on '{group_col}' "
                  f"({df[group_col].nunique()} groups)")
            splitter = StratifiedGroupKFold(n_splits=k_folds, shuffle=True,
                                            random_state=seed)
            split_iter = splitter.split(df, df["_bin"], groups=df[group_col])
        else:
            print(f"Stratified {k_folds}-fold by text length "
                  "(set data.group_col if rows share a page or scribe)")
            splitter = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
            split_iter = splitter.split(df, df["_bin"])

        for fold_num, (tr, va) in enumerate(split_iter, 1):
            yield (df.iloc[tr].drop(columns=helper).copy(),
                   df.iloc[va].drop(columns=helper).copy(), fold_num)
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
    lengths = df["Target"].astype(str).str.len()
    print(f"  {len(df)} rows | target chars: median {lengths.median():.0f}, "
          f"p95 {lengths.quantile(0.95):.0f}, max {lengths.max()}")
    print(f"  {df['Target'].isna().sum()} NaN targets")

    k_folds = data_cfg.get("k_folds", 1)
    n_folds = k_folds if k_folds > 1 else None
    results = []

    for train_df, val_df, fold_num in make_splits(df, data_cfg):
        out_dir = base_output_dir / f"fold_{fold_num}" if fold_num else base_output_dir
        results.append(train_single_fold(train_df, val_df, image_dir, out_dir,
                                         cfg, fold_num, n_folds))

    if len(results) > 1:
        print(f"\n{'=' * 70}\nK-FOLD RESULTS\n{'=' * 70}")
        for r in results:
            print(f"  Fold {r['fold']}/{n_folds}: CER={r['best_cer']:.4f} "
                  f"WER={r['best_wer']:.4f}")
        cers = [r["best_cer"] for r in results]
        print(f"\n  CER {np.mean(cers):.4f} +/- {np.std(cers):.4f}\n{'=' * 70}\n")

    base_output_dir.mkdir(parents=True, exist_ok=True)
    summary = base_output_dir / "summary.json"
    with open(summary, "w") as f:
        json.dump({"config": cfg, "results": results}, f, indent=2, default=str)
    print(f"Summary -> {summary}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_trocr.yaml")
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    with open(SCRIPT_DIR / args.config) as f:
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