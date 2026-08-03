"""Kraken OCR inference for Barbados Road Challenge test set."""

import argparse
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm

from kraken.containers import BBoxLine, Segmentation
from kraken.lib.models import load_any
from kraken.pageseg import segment
from kraken.rpred import rpred as kraken_rpred
import torch


def resolve(path, base):
    if os.path.isabs(path):
        return path
    return os.path.join(base, path)


def load_config(config_path):
    cfg = {}
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
    return cfg


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


def clean_output(text: str) -> str:
    text = str(text)
    for tag in ["assistant", "user", "<|assistant|>", "<|user|>"]:
        if tag in text:
            text = text.split(tag)[-1]
    return " ".join(text.split()).strip()


def otsu_threshold(image: Image.Image) -> int:
    array = np.asarray(image.convert("L"))
    hist, _ = np.histogram(array, bins=256, range=(0, 255))
    total = array.size
    sum_all = np.dot(np.arange(256), hist)
    sumB = 0
    wB = 0
    maximum = 0.0
    threshold = 0
    for i in range(256):
        wB += hist[i]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
        sumB += i * hist[i]
        mB = sumB / wB
        mF = (sum_all - sumB) / wF
        between = wB * wF * (mB - mF) ** 2
        if between >= maximum:
            threshold = i
            maximum = between
    return threshold


def binarize_image(image: Image.Image, threshold: Optional[int] = None) -> Tuple[Image.Image, int]:
    gray = image.convert("L")
    if threshold is None:
        threshold = otsu_threshold(gray)
    binary = gray.point(lambda p: 255 if p > threshold else 0, mode="1")
    return binary, threshold


def create_fallback_segmentation(image: Image.Image, text_direction: str) -> Segmentation:
    bbox = (0, 0, image.width, image.height)
    line = BBoxLine(id="full_page", bbox=bbox, text_direction=text_direction)
    return Segmentation(
        type="bbox",
        imagename=None,
        text_direction=text_direction,
        script_detection=False,
        lines=[line],
    )


def segment_image(image: Image.Image, text_direction: str, pad: int, threshold: Optional[int]):
    binary, threshold_value = binarize_image(image, threshold)
    try:
        segmented = segment(binary, text_direction=text_direction, pad=pad)
        if isinstance(segmented, dict) or not getattr(segmented, 'lines', None):
            return create_fallback_segmentation(binary, text_direction), binary, threshold_value
        return segmented, binary, threshold_value
    except Exception:
        fallback = create_fallback_segmentation(binary, text_direction)
        return fallback, binary, threshold_value


def infer_image(model, image_path: str, text_direction: str, pad: int, threshold: Optional[int], bidi_reordering=True) -> str:
    with Image.open(image_path) as img:
        seg, binary, threshold_value = segment_image(img, text_direction=text_direction, pad=pad, threshold=threshold)

    text_segments = []
    for rec in kraken_rpred(model, binary, seg, pad=pad, bidi_reordering=bidi_reordering):
        if getattr(rec, "prediction", None) is not None:
            text_segments.append(str(rec.prediction))

    if not text_segments:
        return ""
    return clean_output(" ".join(text_segments))


def run_inference(
    config_path=None,
    test_csv=None,
    base_image_dir=None,
    model_path=None,
    output_csv=None,
    device=None,
    use_gpu=None,
    pad=None,
    threshold=None,
    text_direction=None,
    bidi_reordering=None,
):
    cfg = load_config(config_path)
    cfg_inf = cfg.get("inference", {})
    cfg_general = cfg.get("general", {})

    repo_root = cfg_general.get("repo_root")
    if repo_root is None:
        raise ValueError("Missing general.repo_root in config.yaml")

    test_csv = test_csv or cfg_inf.get("test_csv", "code/scripts/data/Test.csv")
    base_image_dir = base_image_dir or cfg_inf.get("base_image_dir", "code/scripts/data/images")
    model_path = model_path or cfg_inf.get("model_path", "code/scripts/Kraken-OCR/kraken_model.mlmodel")
    output_csv = output_csv or cfg_inf.get("output_csv", "submission.csv")
    device = device or cfg_inf.get("device", "auto")
    use_gpu = use_gpu if use_gpu is not None else cfg_inf.get("use_gpu", True)
    pad = pad if pad is not None else cfg_inf.get("pad", 16)
    threshold = threshold if threshold is not None else cfg_inf.get("threshold", None)
    text_direction = text_direction or cfg_inf.get("text_direction", "horizontal-lr")
    bidi_reordering = bidi_reordering if bidi_reordering is not None else cfg_inf.get("bidi_reordering", True)

    test_csv = resolve(test_csv, repo_root)
    base_image_dir = resolve(base_image_dir, repo_root)
    model_path = resolve(model_path, repo_root)
    if not os.path.exists(model_path) and config_path:
        config_dir = os.path.dirname(os.path.abspath(config_path))
        alt_model_path = resolve(cfg_inf.get("model_path", model_path), config_dir)
        if os.path.exists(alt_model_path):
            model_path = alt_model_path
    output_csv = resolve(output_csv, repo_root)

    if device == "auto":
        device = "cuda" if bool(use_gpu) and torch.cuda.is_available() else "cpu"

    print(f"[INFO] device={device}")
    print(f"[INFO] test_csv={test_csv}")
    print(f"[INFO] image_dir={base_image_dir}")
    print(f"[INFO] model_path={model_path}")
    print(f"[INFO] output_csv={output_csv}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Kraken model not found: {model_path}")

    model = load_any(model_path, device=device)

    df = pd.read_csv(test_csv)
    results = []

    for row in tqdm(df.to_dict("records"), desc="OCR", unit="row"):
        record_id = get_record_id(row)
        image_path = build_image_path(base_image_dir, record_id)

        if not os.path.exists(image_path):
            results.append({"ID": record_id, "Target": ""})
            continue

        try:
            pred_text = infer_image(model, image_path, text_direction=text_direction, pad=pad, threshold=threshold, bidi_reordering=bidi_reordering)
        except Exception as exc:
            print(f"[WARN] failed to infer {image_path}: {exc}")
            pred_text = ""
        results.append({"ID": record_id, "Target": pred_text})

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_csv, index=False)

    print("[DONE] saved:", output_csv)
    return out_df


def parse_args():
    parser = argparse.ArgumentParser(description="Run Kraken OCR inference on the Barbados Road Challenge test set.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--test_csv", default=None, help="Override test CSV file")
    parser.add_argument("--base_image_dir", default=None, help="Override base image directory")
    parser.add_argument("--model_path", default=None, help="Override Kraken model file or directory")
    parser.add_argument("--output_csv", default=None, help="Override output CSV path")
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"], help="Device to use")
    parser.add_argument("--use_gpu", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=None, help="Enable GPU for Kraken inference")
    parser.add_argument("--pad", type=int, default=None, help="Padding for line segmentation")
    parser.add_argument("--threshold", type=int, default=None, help="Explicit binarization threshold (0-255)")
    parser.add_argument("--text_direction", default=None, help="Text direction for Kraken page segmentation")
    parser.add_argument("--bidi_reordering", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=None, help="Enable bidirectional reordering")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(
        config_path=args.config,
        test_csv=args.test_csv,
        base_image_dir=args.base_image_dir,
        model_path=args.model_path,
        output_csv=args.output_csv,
        device=args.device,
        use_gpu=args.use_gpu,
        pad=args.pad,
        threshold=args.threshold,
        text_direction=args.text_direction,
        bidi_reordering=args.bidi_reordering,
    )
