"""
Inference script for Qwen2-VL OCR (path-safe + config-driven + PEFT-safe)
"""

import os
import argparse
import yaml
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm

from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig
)

from peft import PeftModel


# -----------------------------
# Utils
# -----------------------------

def clean_output(text: str) -> str:
    text = str(text)

    for tag in ["assistant", "user", "<|assistant|>", "<|user|>"]:
        if tag in text:
            text = text.split(tag)[-1]

    return " ".join(text.split()).strip()


def load_image(path):
    img = Image.open(path).convert("RGB")

    w, h = img.size
    aspect = w / h

    if aspect > 10:
        img.thumbnail((2048, 384))
    elif aspect > 5:
        img.thumbnail((2048, 512))
    else:
        img.thumbnail((1536, 768))

    return img


def to_bool(x, default=False):
    if x is None:
        return default
    if isinstance(x, bool):
        return x
    return str(x).lower() in ("1", "true", "yes")


# -----------------------------
# Prompt (must match training)
# -----------------------------

OCR_PROMPT = (
    "Transcribe EXACTLY as seen. Preserve line breaks, punctuation, and spacing. "
    "Do not add explanations. Return verbatim transcription. No paraphrasing. No normalization."
)


# -----------------------------
# Path builder
# -----------------------------

def get_record_id(row):
    value = row.get(
        "ID",
        row.get(
            "new_id",
            row.get("new id", row.get("id", row.get("trapp_id", "")))
        ),
    )
    if value is None:
        return ""
    value = str(value).strip()
    return "" if value.lower() == "nan" else value


def build_image_path(base_dir, record_id):
    return os.path.join(base_dir, f"{record_id}.jpg")


# -----------------------------
# Prediction
# -----------------------------

def predict(model, processor, image, max_new_tokens=128, num_beams=1):

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
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=num_beams,
        )

    decoded = processor.batch_decode(output, skip_special_tokens=True)[0]
    return clean_output(decoded)


# -----------------------------
# Main inference
# -----------------------------

def run_inference(
    model_path,
    config_path,
    test_csv=None,
    base_image_dir=None,
    output_csv="submission.csv",
    device=None,
    use_bf16=None,
    max_new_tokens=None,
    num_beams=None,
):

    # -----------------------------
    # Load config
    # -----------------------------
    cfg = {}
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}

    cfg_inf = cfg.get("inference", {})
    cfg_general = cfg.get("general", {})

    # -----------------------------
    # ROOT (SINGLE SOURCE OF TRUTH)
    # -----------------------------
    repo_root = cfg_general.get("repo_root")
    if repo_root is None:
        raise ValueError("❌ Missing general.repo_root in config.yaml")

    # -----------------------------
    # Resolve model path
    # -----------------------------
    model_path = model_path or cfg_inf.get("model_path")
    if model_path is None:
        raise ValueError("❌ Missing model_path (CLI or config.yaml)")

    # -----------------------------
    # Resolve paths safely
    # -----------------------------
    test_csv = test_csv or cfg_inf.get("test_csv", "test_split.csv")
    base_image_dir = base_image_dir or cfg_inf.get("base_image_dir", "data")
    output_csv = output_csv or cfg_inf.get("output_csv", "submission.csv")

    if not os.path.isabs(test_csv):
        test_csv = os.path.join(repo_root, test_csv)

    if not os.path.isabs(base_image_dir):
        base_image_dir = os.path.join(repo_root, base_image_dir)

    # -----------------------------
    # Runtime params
    # -----------------------------
    device = device or cfg_inf.get("device", "auto")
    max_new_tokens = max_new_tokens or cfg_inf.get("max_new_tokens", 128)
    num_beams = num_beams or cfg_inf.get("num_beams", 1)
    use_bf16 = to_bool(use_bf16, cfg_inf.get("use_bf16", False))

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] device={device}")
    print(f"[INFO] test_csv={test_csv}")
    print(f"[INFO] image_dir={base_image_dir}")
    print(f"[INFO] model_path={model_path}")

    # -----------------------------
    # Load model
    # -----------------------------
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    dtype = torch.bfloat16 if (use_bf16 and torch.cuda.is_bf16_supported()) else torch.float16

    base_model_id = "Qwen/Qwen2-VL-2B-Instruct"

    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model_id,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()

    processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)

    # -----------------------------
    # Load dataset
    # -----------------------------
    df = pd.read_csv(test_csv)

    results = []

    for row in tqdm(df.to_dict("records")):

        record_id = get_record_id(row)
        image_path = build_image_path(base_image_dir, record_id)

        if not os.path.exists(image_path):
            results.append({"ID": record_id, "Target": ""})
            continue

        image = load_image(image_path)

        pred = predict(
            model,
            processor,
            image,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )

        results.append({"ID": record_id, "Target": pred})

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_csv, index=False)

    print("[DONE] saved:", output_csv)
    return out_df


# -----------------------------
# CLI
# -----------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", default=None)
    parser.add_argument("--config", default="config.yaml")

    parser.add_argument("--test_csv", default=None)
    parser.add_argument("--base_image_dir", default=None)
    parser.add_argument("--output_csv", default=None)

    parser.add_argument("--device", default=None)
    parser.add_argument("--use_bf16", default=None)

    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--num_beams", type=int, default=None)

    args = parser.parse_args()

    use_bf16 = to_bool(args.use_bf16) if args.use_bf16 is not None else None

    run_inference(
        model_path=args.model_path,
        config_path=args.config,
        test_csv=args.test_csv,
        base_image_dir=args.base_image_dir,
        output_csv=args.output_csv,
        device=args.device,
        use_bf16=use_bf16,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
    )