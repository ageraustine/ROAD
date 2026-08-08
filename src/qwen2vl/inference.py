"""
Qwen2-VL Inference for Historical HTR
Generates submission.csv for competition
"""

import os
import json
import shutil
import argparse
import warnings
from pathlib import Path
from collections import Counter

import yaml
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm

from transformers import AutoProcessor
from peft import PeftModel

# Suppress repetitive padding side warnings during batched inference
warnings.filterwarnings(
    "ignore",
    message=".*padding_side.*",
    category=UserWarning,
    module="transformers"
)

# Support both Qwen2-VL and Qwen3-VL
try:
    from transformers import Qwen3VLForConditionalGeneration
    QWEN3_AVAILABLE = True
except ImportError:
    QWEN3_AVAILABLE = False

try:
    from transformers import Qwen2VLForConditionalGeneration
    QWEN2_AVAILABLE = True
except ImportError:
    QWEN2_AVAILABLE = False

# Fallback to generic class
if not QWEN3_AVAILABLE and not QWEN2_AVAILABLE:
    from transformers import AutoModelForVision2Seq

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


def remove_repetitions(text: str, max_repetitions: int = 5) -> str:
    """
    Detect and truncate excessive repetitive patterns in generated text.

    Catches patterns like:
    - "* * * * * * * * * * * *..." (word repetition)
    - "- - - - - - - - - - - -..." (token repetition)
    - "the the the the the..." (phrase repetition)

    Args:
        text: Generated text that may contain repetitions
        max_repetitions: Maximum allowed consecutive repetitions before truncating

    Returns:
        Text truncated at first excessive repetition point
    """
    if not text or len(text) < 10:
        return text

    tokens = text.split()
    if len(tokens) < max_repetitions:
        return text

    # Check for consecutive word repetitions
    for i in range(len(tokens) - max_repetitions):
        word = tokens[i]
        # Count consecutive occurrences
        consecutive = 1
        for j in range(i + 1, len(tokens)):
            if tokens[j] == word:
                consecutive += 1
            else:
                break

        # If we find excessive repetitions, truncate
        if consecutive > max_repetitions:
            # Keep text up to (but not including) the repetition
            truncated = " ".join(tokens[:i])
            return truncated.strip()

    # Check for phrase repetitions (2-3 word sequences)
    for phrase_len in [2, 3]:
        for i in range(len(tokens) - (phrase_len * max_repetitions)):
            phrase = tuple(tokens[i:i + phrase_len])
            consecutive = 1

            # Count consecutive phrase occurrences
            pos = i + phrase_len
            while pos + phrase_len <= len(tokens):
                if tuple(tokens[pos:pos + phrase_len]) == phrase:
                    consecutive += 1
                    pos += phrase_len
                else:
                    break

            # Truncate if excessive
            if consecutive > max_repetitions:
                truncated = " ".join(tokens[:i])
                return truncated.strip()

    return text


def clean_output(text: str) -> str:
    """Clean model output, removing any chat artifacts and excessive repetitions."""
    text = str(text)

    # Remove common chat template artifacts
    artifacts = ["<|assistant|>", "<|user|>", "assistant", "Assistant:"]
    for artifact in artifacts:
        if artifact in text:
            text = text.split(artifact)[-1]

    # Remove excessive repetitive patterns (defensive layer)
    text = remove_repetitions(text, max_repetitions=5)

    return " ".join(text.split()).strip()


# ─────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, model_name: str, use_flash: bool = True, verbose: bool = False):
    """Load fine-tuned model with LoRA weights."""

    # Detect model type
    is_qwen3 = "Qwen3" in model_name or "qwen3" in model_name.lower()
    is_qwen2 = "Qwen2" in model_name or "qwen2" in model_name.lower()

    # Setup kwargs based on model type
    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
    }

    # Qwen3 uses dtype="auto", Qwen2 uses torch_dtype=torch.bfloat16
    if is_qwen3:
        model_kwargs["dtype"] = "auto"
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    # Select appropriate model class
    if is_qwen3 and QWEN3_AVAILABLE:
        model_class = Qwen3VLForConditionalGeneration
    elif is_qwen2 and QWEN2_AVAILABLE:
        model_class = Qwen2VLForConditionalGeneration
    else:
        model_class = AutoModelForVision2Seq

    if use_flash:
        try:
            # Try with Flash Attention 2
            model_kwargs["attn_implementation"] = "flash_attention_2"
            base_model = model_class.from_pretrained(
                model_name,
                **model_kwargs,
            )
        except Exception:
            # Fallback to standard attention
            model_kwargs.pop("attn_implementation", None)
            base_model = model_class.from_pretrained(
                model_name,
                **model_kwargs,
            )
    else:
        base_model = model_class.from_pretrained(
            model_name,
            **model_kwargs,
        )

    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    # Set left padding for decoder-only model (required for correct batched generation)
    processor.tokenizer.padding_side = 'left'

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
    repetition_penalty: float = 1.0,
) -> str:
    """Generate transcription for a single image."""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": OCR_PROMPT},
            ],
        }
    ]

    # Qwen3 style: tokenize=True in apply_chat_template
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
            repetition_penalty=repetition_penalty,
        )

    # Trim input prompt from generated output (Qwen3 style)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    # Decode only the generated part
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    return clean_output(output_text[0] if output_text else "")


def predict_batch(
    model,
    processor,
    images: list[Image.Image],
    max_new_tokens: int = 256,
    num_beams: int = 5,
    repetition_penalty: float = 1.0,
) -> list[str]:
    """
    Generate transcriptions for a batch of images.

    Args:
        model: The model to use for generation
        processor: The processor for tokenization
        images: List of PIL images to process
        max_new_tokens: Maximum tokens to generate
        num_beams: Number of beams for beam search
        repetition_penalty: Penalty for token repetition (>1.0 = less repetition)

    Returns:
        List of transcription strings (same order as input images)
    """
    if not images:
        return []

    # Build messages for each image
    messages_batch = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": OCR_PROMPT},
                ],
            }
        ]
        for img in images
    ]

    # Apply chat template to each conversation
    text_inputs = [
        processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        for messages in messages_batch
    ]

    # Process batch (processor handles padding automatically)
    inputs = processor(
        text=text_inputs,
        images=images,
        padding=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
            repetition_penalty=repetition_penalty,
        )

    # Trim input prompt from generated output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    # Decode all outputs
    output_texts = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    # Clean each output
    return [clean_output(text) for text in output_texts]


# ─────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────

def ensemble_predictions(predictions: list, strategy: str = "char_voting") -> str:
    """
    Ensemble multiple predictions using various strategies.

    Args:
        predictions: List of prediction strings from different models
        strategy: Ensemble strategy - one of:
            - "char_voting": Character-level majority voting (default, may degrade quality)
            - "majority": Pick most common full prediction
            - "longest": Pick longest prediction
            - "shortest": Pick shortest prediction
            - "first": Use first fold only (no ensemble)

    Returns:
        Ensembled prediction string
    """
    if len(predictions) == 1:
        return predictions[0]

    if strategy == "char_voting":
        # Character-level voting (original method)
        max_len = max(len(p) for p in predictions)
        padded = [p + ' ' * (max_len - len(p)) for p in predictions]

        result = []
        for i in range(max_len):
            chars = [p[i] for p in padded]
            most_common = Counter(chars).most_common(1)[0][0]
            result.append(most_common)

        return ''.join(result).rstrip()

    elif strategy == "majority":
        # Pick most common full prediction
        counter = Counter(predictions)
        return counter.most_common(1)[0][0]

    elif strategy == "longest":
        # Pick longest prediction
        return max(predictions, key=len)

    elif strategy == "shortest":
        # Pick shortest prediction
        return min(predictions, key=len)

    elif strategy == "first":
        # Use first fold only
        return predictions[0]

    else:
        raise ValueError(f"Unknown ensemble strategy: {strategy}")


def run_kfold_inference(cfg: dict, clear_cache: bool = False, ensemble_strategy: str = "char_voting"):
    """
    Run K-Fold ensemble inference on test set with batching and resumability.

    Saves predictions for each fold to disk. If interrupted, can resume from
    last completed fold without re-running inference.

    Args:
        cfg: Configuration dictionary
        clear_cache: If True, delete cached predictions and re-run all folds
        ensemble_strategy: Strategy for combining fold predictions (char_voting, majority, longest, shortest, first)
    """

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    inf_cfg = cfg["inference"]

    # Paths
    test_csv = REPO_ROOT / data_cfg["test_csv"]
    image_dir = REPO_ROOT / data_cfg["image_dir"]
    base_output_dir = REPO_ROOT / train_cfg["output_dir"]
    output_csv = REPO_ROOT / inf_cfg["output_csv"]

    # Inference cache directory for resumability
    inference_cache_dir = base_output_dir / "inference_cache"
    inference_cache_dir.mkdir(parents=True, exist_ok=True)

    # Clear cache if requested
    if clear_cache:
        if inference_cache_dir.exists():
            shutil.rmtree(inference_cache_dir)
            inference_cache_dir.mkdir(parents=True, exist_ok=True)
        print("🗑️  Cache cleared\n")

    # Find all fold checkpoints
    k_folds = data_cfg.get("k_folds", 5)
    fold_checkpoints = []

    for fold_num in range(1, k_folds + 1):
        fold_checkpoint = base_output_dir / f"fold_{fold_num}" / "best"
        if fold_checkpoint.exists():
            fold_checkpoints.append((fold_num, fold_checkpoint))
        else:
            print(f"Warning: Fold {fold_num} checkpoint not found: {fold_checkpoint}")

    if not fold_checkpoints:
        print(f"Error: No fold checkpoints found in {base_output_dir}")
        print(f"Expected: fold_1/best/, fold_2/best/, ..., fold_{k_folds}/best/")
        return None

    print(f"\n{'='*70}")
    print(f"K-FOLD ENSEMBLE: {len(fold_checkpoints)} folds | Strategy: {ensemble_strategy}")
    print(f"Test samples: {len(pd.read_csv(test_csv))}")
    print(f"{'='*70}\n")

    # Load test data
    df = pd.read_csv(test_csv)

    # Get batch size from config
    batch_size = inf_cfg.get("batch_size", 2)

    # Collect predictions from all folds (with caching)
    all_fold_predictions = []

    for fold_num, checkpoint in fold_checkpoints:
        # Check for cached predictions (JSON first, then CSV)
        json_cache_file = inference_cache_dir / f"fold_{fold_num}_predictions.json"
        csv_cache_file = inference_cache_dir / f"fold_{fold_num}_predictions.csv"

        # Priority 1: JSON cache (fastest)
        if json_cache_file.exists():
            print(f"⏩ Fold {fold_num}: Cached ({len(json.load(open(json_cache_file)))} predictions)")
            with open(json_cache_file, 'r') as f:
                fold_predictions = json.load(f)
            all_fold_predictions.append(fold_predictions)
            continue

        # Priority 2: CSV cache (if JSON doesn't exist)
        if csv_cache_file.exists():
            print(f"⏩ Fold {fold_num}: Cached CSV → converting to JSON...")
            cache_df = pd.read_csv(csv_cache_file)

            # Convert CSV to dict format
            fold_predictions = dict(zip(
                cache_df['ID'].astype(str).str.strip(),
                cache_df['Target'].astype(str)
            ))

            all_fold_predictions.append(fold_predictions)

            # Convert CSV to JSON for faster future loads
            with open(json_cache_file, 'w') as f:
                json.dump(fold_predictions, f, indent=2)
            continue

        # Run inference for this fold
        print(f"🔄 Fold {fold_num}: Running inference (batch_size={batch_size})...", flush=True)
        model, processor = load_model(
            str(checkpoint),
            model_cfg["name"],
            use_flash=model_cfg.get("use_flash_attention", True),
        )

        fold_predictions = {}

        # Process in batches
        batch_ids = []
        batch_images = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Fold {fold_num}",
                          mininterval=2.0, ncols=80):
            img_id = str(row["ID"]).strip()
            img_path = image_dir / f"{img_id}.jpg"

            if not img_path.exists():
                fold_predictions[img_id] = ""
                continue

            image = load_image(str(img_path))
            batch_ids.append(img_id)
            batch_images.append(image)

            # Process batch when full
            if len(batch_images) >= batch_size:
                preds = predict_batch(
                    model,
                    processor,
                    batch_images,
                    max_new_tokens=inf_cfg["max_new_tokens"],
                    num_beams=inf_cfg["num_beams"],
                    repetition_penalty=inf_cfg.get("repetition_penalty", 1.0),
                )
                for bid, pred in zip(batch_ids, preds):
                    fold_predictions[bid] = pred

                batch_ids = []
                batch_images = []

        # Process remaining images
        if batch_images:
            preds = predict_batch(
                model,
                processor,
                batch_images,
                max_new_tokens=inf_cfg["max_new_tokens"],
                num_beams=inf_cfg["num_beams"],
                repetition_penalty=inf_cfg.get("repetition_penalty", 1.0),
            )
            for bid, pred in zip(batch_ids, preds):
                fold_predictions[bid] = pred

        # Save predictions to cache (for resumability)
        with open(json_cache_file, 'w') as f:
            json.dump(fold_predictions, f, indent=2)
        print(f"   ✓ Cached to {json_cache_file.name}\n")

        all_fold_predictions.append(fold_predictions)

        # Free memory
        del model
        torch.cuda.empty_cache()

    # Ensemble predictions
    print(f"\nEnsembling {len(df)} predictions (strategy={ensemble_strategy})...", end=" ", flush=True)
    results = []

    for _, row in df.iterrows():
        img_id = str(row["ID"]).strip()

        # Collect predictions from all folds for this image
        img_predictions = [fold_preds.get(img_id, "") for fold_preds in all_fold_predictions]

        # Ensemble
        ensembled = ensemble_predictions(img_predictions, strategy=ensemble_strategy)
        results.append({"ID": img_id, "Target": ensembled})

    # Save submission
    out_df = pd.DataFrame(results)
    out_df.to_csv(output_csv, index=False)
    print(f"✓\n✅ Saved: {output_csv}\n")

    return out_df


def run_inference(cfg: dict, checkpoint_override: str = None):
    """Run single-model inference on test set and generate submission with batching."""

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
    print(f"Loading model...", end=" ", flush=True)
    model, processor = load_model(
        str(checkpoint),
        model_cfg["name"],
        use_flash=model_cfg.get("use_flash_attention", True),
    )
    print("✓")

    # Load test data
    df = pd.read_csv(test_csv)
    batch_size = inf_cfg.get("batch_size", 2)
    print(f"Running inference: {len(df)} samples (batch_size={batch_size})")

    # Run predictions in batches
    results = []
    batch_ids = []
    batch_images = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Predicting",
                      mininterval=2.0, ncols=80):
        img_id = str(row["ID"]).strip()
        img_path = image_dir / f"{img_id}.jpg"

        if not img_path.exists():
            print(f"Warning: Image not found: {img_path}")
            results.append({"ID": img_id, "Target": ""})
            continue

        image = load_image(str(img_path))
        batch_ids.append(img_id)
        batch_images.append(image)

        # Process batch when full
        if len(batch_images) >= batch_size:
            preds = predict_batch(
                model,
                processor,
                batch_images,
                max_new_tokens=inf_cfg["max_new_tokens"],
                num_beams=inf_cfg["num_beams"],
                repetition_penalty=inf_cfg.get("repetition_penalty", 1.0),
            )
            for bid, pred in zip(batch_ids, preds):
                results.append({"ID": bid, "Target": pred})

            batch_ids = []
            batch_images = []

    # Process remaining images
    if batch_images:
        preds = predict_batch(
            model,
            processor,
            batch_images,
            max_new_tokens=inf_cfg["max_new_tokens"],
            num_beams=inf_cfg["num_beams"],
            repetition_penalty=inf_cfg.get("repetition_penalty", 1.0),
        )
        for bid, pred in zip(batch_ids, preds):
            results.append({"ID": bid, "Target": pred})

    # Save submission
    out_df = pd.DataFrame(results)
    out_df.to_csv(output_csv, index=False)
    print(f"✅ Saved: {output_csv}")

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
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear cached fold predictions and re-run all inference (K-Fold only)",
    )
    parser.add_argument(
        "--ensemble-strategy",
        default="char_voting",
        choices=["char_voting", "majority", "longest", "shortest", "first"],
        help="Ensemble strategy (K-Fold only): char_voting (default), majority (most common full prediction), longest, shortest, first (fold 1 only)",
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
        run_kfold_inference(cfg, clear_cache=args.clear_cache, ensemble_strategy=args.ensemble_strategy)
    else:
        # Single model inference
        if args.checkpoint:
            print(f"Using custom checkpoint: {args.checkpoint}")
        run_inference(cfg, checkpoint_override=args.checkpoint)


if __name__ == "__main__":
    main()
