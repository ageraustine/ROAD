"""
Qwen2-VL Inference for Historical HTR
Generates submission.csv for competition
"""

import os
import argparse
from pathlib import Path
from collections import Counter

import yaml
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm

from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel

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
# UTILS
# ─────────────────────────────────────────────────────────────

def load_image(path: str, max_pixels: int = 2016000) -> Image.Image:
    """Load and resize image while preserving aspect ratio."""
    img = Image.open(path).convert("RGB")
    w, h = img.size

    pixels = w * h
    if pixels > max_pixels:
        scale = (max_pixels / pixels) ** 0.5
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

    return img


def clean_output(text: str) -> str:
    """Clean model output, removing any chat artifacts."""
    text = str(text)

    # Remove common chat template artifacts
    artifacts = ["<|assistant|>", "<|user|>", "assistant", "Assistant:"]
    for artifact in artifacts:
        if artifact in text:
            text = text.split(artifact)[-1]

    return " ".join(text.split()).strip()


# ─────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, model_name: str, use_flash: bool = True):
    """Load fine-tuned model with LoRA weights."""

    dtype = torch.bfloat16

    model_kwargs = {
        "torch_dtype": dtype,
        "device_map": "auto",
        "trust_remote_code": True,
    }

    print(f"Loading base model: {model_name}")

    if use_flash:
        try:
            # Try with Flash Attention 2
            model_kwargs["attn_implementation"] = "flash_attention_2"
            base_model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                **model_kwargs,
            )
            print("✓ Using Flash Attention 2")
        except Exception as e:
            print(f"⚠️  Flash Attention 2 not available: {e}")
            print("  Falling back to standard attention...")
            # Fallback to standard attention
            model_kwargs.pop("attn_implementation", None)
            base_model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                **model_kwargs,
            )
            print("✓ Using standard attention")
    else:
        print("Flash Attention 2 disabled")
        base_model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            **model_kwargs,
        )
        print("✓ Using standard attention")

    print(f"Loading LoRA weights from: {checkpoint_path}")
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    return model, processor


# ─────────────────────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────────────────────

def predict_single(
    model,
    processor,
    image: Image.Image,
    max_new_tokens: int = 256,
    num_beams: int = 5,
) -> str:
    """Generate transcription for a single image."""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": OCR_PROMPT},
                {"type": "image", "image": image},
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[prompt],
        images=[image],
        return_tensors="pt",
        padding=True,
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return clean_output(decoded)


# ─────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────

def ensemble_predictions(predictions: list) -> str:
    """
    Ensemble multiple predictions using character-level voting.

    Args:
        predictions: List of prediction strings from different models

    Returns:
        Ensembled prediction string
    """
    if len(predictions) == 1:
        return predictions[0]

    # Pad all predictions to same length
    max_len = max(len(p) for p in predictions)
    padded = [p + ' ' * (max_len - len(p)) for p in predictions]

    # Vote for each character position
    result = []
    for i in range(max_len):
        chars = [p[i] for p in padded]
        # Most common character wins (majority vote)
        most_common = Counter(chars).most_common(1)[0][0]
        result.append(most_common)

    # Join and strip trailing spaces
    return ''.join(result).rstrip()


def run_kfold_inference(cfg: dict):
    """Run K-Fold ensemble inference on test set."""

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    inf_cfg = cfg["inference"]

    # Paths
    test_csv = REPO_ROOT / data_cfg["test_csv"]
    image_dir = REPO_ROOT / data_cfg["image_dir"]
    base_output_dir = REPO_ROOT / train_cfg["output_dir"]
    output_csv = REPO_ROOT / inf_cfg["output_csv"]

    # Find all fold checkpoints
    k_folds = data_cfg.get("k_folds", 5)
    fold_checkpoints = []

    for fold_num in range(1, k_folds + 1):
        fold_checkpoint = base_output_dir / f"fold_{fold_num}" / "best"
        if fold_checkpoint.exists():
            fold_checkpoints.append(fold_checkpoint)
        else:
            print(f"Warning: Fold {fold_num} checkpoint not found: {fold_checkpoint}")

    if not fold_checkpoints:
        print(f"Error: No fold checkpoints found in {base_output_dir}")
        print(f"Expected: fold_1/best/, fold_2/best/, ..., fold_{k_folds}/best/")
        return None

    print(f"\n{'='*70}")
    print(f"K-FOLD ENSEMBLE INFERENCE: {len(fold_checkpoints)} folds")
    print(f"{'='*70}\n")

    # Load test data
    print(f"Loading test data from {test_csv}")
    df = pd.read_csv(test_csv)
    print(f"Test samples: {len(df)}\n")

    # Collect predictions from all folds
    all_fold_predictions = []

    for fold_idx, checkpoint in enumerate(fold_checkpoints, 1):
        print(f"Loading Fold {fold_idx} model from {checkpoint}")
        model, processor = load_model(
            str(checkpoint),
            model_cfg["name"],
            use_flash=model_cfg.get("use_flash_attention", True),
        )

        print(f"Running predictions for Fold {fold_idx}...")
        fold_predictions = {}

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Fold {fold_idx}"):
            img_id = str(row["ID"]).strip()
            img_path = image_dir / f"{img_id}.jpg"

            if not img_path.exists():
                fold_predictions[img_id] = ""
                continue

            image = load_image(str(img_path))
            pred = predict_single(
                model,
                processor,
                image,
                max_new_tokens=inf_cfg["max_new_tokens"],
                num_beams=inf_cfg["num_beams"],
            )
            fold_predictions[img_id] = pred

        all_fold_predictions.append(fold_predictions)

        # Free memory
        del model
        torch.cuda.empty_cache()
        print()

    # Ensemble predictions
    print("Ensembling predictions across folds...")
    results = []

    for _, row in df.iterrows():
        img_id = str(row["ID"]).strip()

        # Collect predictions from all folds for this image
        img_predictions = [fold_preds.get(img_id, "") for fold_preds in all_fold_predictions]

        # Ensemble
        ensembled = ensemble_predictions(img_predictions)
        results.append({"ID": img_id, "Target": ensembled})

    # Save submission
    out_df = pd.DataFrame(results)
    out_df.to_csv(output_csv, index=False)
    print(f"\n{'='*70}")
    print(f"Saved ensemble submission to {output_csv}")
    print(f"{'='*70}\n")

    return out_df


def run_inference(cfg: dict, checkpoint_override: str = None):
    """Run single-model inference on test set and generate submission."""

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    inf_cfg = cfg["inference"]

    # Paths
    test_csv = REPO_ROOT / data_cfg["test_csv"]
    image_dir = REPO_ROOT / data_cfg["image_dir"]

    if checkpoint_override:
        checkpoint = Path(checkpoint_override)
    else:
        checkpoint = REPO_ROOT / inf_cfg["checkpoint"]

    output_csv = REPO_ROOT / inf_cfg["output_csv"]

    # Load model
    model, processor = load_model(
        str(checkpoint),
        model_cfg["name"],
        use_flash=model_cfg.get("use_flash_attention", True),
    )

    # Load test data
    print(f"Loading test data from {test_csv}")
    df = pd.read_csv(test_csv)
    print(f"Test samples: {len(df)}")

    # Run predictions
    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
        img_id = str(row["ID"]).strip()
        img_path = image_dir / f"{img_id}.jpg"

        if not img_path.exists():
            print(f"Warning: Image not found: {img_path}")
            results.append({"ID": img_id, "Target": ""})
            continue

        image = load_image(str(img_path))

        pred = predict_single(
            model,
            processor,
            image,
            max_new_tokens=inf_cfg["max_new_tokens"],
            num_beams=inf_cfg["num_beams"],
        )

        results.append({"ID": img_id, "Target": pred})

    # Save submission
    out_df = pd.DataFrame(results)
    out_df.to_csv(output_csv, index=False)
    print(f"Saved submission to {output_csv}")

    return out_df


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Qwen2-VL HTR Inference - Generate competition submission"
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path (single model only)")
    parser.add_argument("--output", default=None, help="Override output CSV path")
    parser.add_argument(
        "--kfold",
        action="store_true",
        help="Use K-Fold ensemble inference (loads all fold_*/best/ checkpoints)",
    )
    args = parser.parse_args()

    config_path = SCRIPT_DIR / args.config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # CLI overrides
    if args.output:
        cfg["inference"]["output_csv"] = args.output

    if args.kfold:
        # K-Fold ensemble inference
        print("Running K-Fold ensemble inference...")
        run_kfold_inference(cfg)
    else:
        # Single model inference
        if args.checkpoint:
            print(f"Using custom checkpoint: {args.checkpoint}")
        run_inference(cfg, checkpoint_override=args.checkpoint)


if __name__ == "__main__":
    main()
