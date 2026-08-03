"""
Qwen2-VL Inference for Historical HTR
Generates submission.csv for competition
"""

import os
import argparse
from pathlib import Path

import yaml
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
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

    if use_flash:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    print(f"Loading base model: {model_name}")
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        **model_kwargs,
    )

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

def run_inference(cfg: dict):
    """Run inference on test set and generate submission."""

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    inf_cfg = cfg["inference"]

    # Paths
    test_csv = REPO_ROOT / data_cfg["test_csv"]
    image_dir = REPO_ROOT / data_cfg["image_dir"]
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path")
    parser.add_argument("--output", default=None, help="Override output CSV path")
    args = parser.parse_args()

    config_path = SCRIPT_DIR / args.config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # CLI overrides
    if args.checkpoint:
        cfg["inference"]["checkpoint"] = args.checkpoint
    if args.output:
        cfg["inference"]["output_csv"] = args.output

    run_inference(cfg)


if __name__ == "__main__":
    main()
