"""PaddleOCR inference for Barbados Road Challenge test set."""

import os
import argparse
import yaml
import pandas as pd
from tqdm import tqdm

import paddle
from paddle.base import libpaddle

if not hasattr(libpaddle.AnalysisConfig, "set_optimization_level"):
    def set_optimization_level(self, level):
        return None
    libpaddle.AnalysisConfig.set_optimization_level = set_optimization_level

if not hasattr(paddle.nn, "Conv2d") and hasattr(paddle.nn, "Conv2D"):
    class Conv2d(paddle.nn.Conv2D):
        def __init__(self, *args, bias=None, **kwargs):
            if bias is not None:
                if bias is False:
                    kwargs["bias_attr"] = False
                elif bias is True:
                    kwargs["bias_attr"] = None
                else:
                    kwargs["bias_attr"] = bias
            super().__init__(*args, **kwargs)

    paddle.nn.Conv2d = Conv2d

if not hasattr(paddle.nn, "Conv1d") and hasattr(paddle.nn, "Conv1D"):
    class Conv1d(paddle.nn.Conv1D):
        def __init__(self, *args, bias=None, **kwargs):
            if bias is not None:
                if bias is False:
                    kwargs["bias_attr"] = False
                elif bias is True:
                    kwargs["bias_attr"] = None
                else:
                    kwargs["bias_attr"] = bias
            super().__init__(*args, **kwargs)

    paddle.nn.Conv1d = Conv1d

from paddleocr import PaddleOCR


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


def run_inference(
    config_path=None,
    test_csv=None,
    base_image_dir=None,
    output_csv=None,
    device=None,
    use_gpu=None,
    use_angle_cls=None,
    lang=None,
    det_db_thresh=None,
    det_db_box_thresh=None,
):
    cfg = load_config(config_path)
    cfg_inf = cfg.get("inference", {})
    cfg_general = cfg.get("general", {})

    repo_root = cfg_general.get("repo_root")
    if repo_root is None:
        raise ValueError("Missing general.repo_root in config.yaml")

    test_csv = test_csv or cfg_inf.get("test_csv", "code/scripts/data/Test.csv")
    base_image_dir = base_image_dir or cfg_inf.get("base_image_dir", "code/scripts/data/images")
    output_csv = output_csv or cfg_inf.get("output_csv", "submission.csv")
    device = device or cfg_inf.get("device", "auto")
    use_gpu = use_gpu if use_gpu is not None else cfg_inf.get("use_gpu", True)
    lang = lang or cfg_inf.get("lang", "en")
    use_angle_cls = use_angle_cls if use_angle_cls is not None else cfg_inf.get("use_angle_cls", True)
    det_db_thresh = det_db_thresh or cfg_inf.get("det_db_thresh", 0.3)
    det_db_box_thresh = det_db_box_thresh or cfg_inf.get("det_db_box_thresh", 0.5)

    test_csv = resolve(test_csv, repo_root)
    base_image_dir = resolve(base_image_dir, repo_root)
    output_csv = resolve(output_csv, repo_root)

    device = device or cfg_inf.get("device", "auto")
    if device == "auto":
        device = "gpu" if bool(use_gpu) else "cpu"

    if device.lower() not in ("gpu", "cpu"):
        raise ValueError("Unsupported device: {device}. Use 'auto', 'gpu', or 'cpu'.")

    use_gpu_flag = bool(use_gpu) and device.lower() in ("gpu", "cuda")

    print(f"[INFO] device={device}")
    print(f"[INFO] test_csv={test_csv}")
    print(f"[INFO] image_dir={base_image_dir}")
    print(f"[INFO] output_csv={output_csv}")

    ocr = PaddleOCR(
        use_angle_cls=use_angle_cls,
        lang=lang,
        use_gpu=use_gpu_flag,
        det_db_thresh=det_db_thresh,
        det_db_box_thresh=det_db_box_thresh,
    )

    df = pd.read_csv(test_csv)
    results = []

    def extract_text_from_ocr_item(item):
        if item is None:
            return None
        if isinstance(item, (list, tuple)):
            if len(item) == 0:
                return None
            if len(item) == 1:
                return extract_text_from_ocr_item(item[0])
            if isinstance(item[1], (list, tuple)) and len(item[1]) >= 1:
                if isinstance(item[1][0], str):
                    return item[1][0]
            if isinstance(item[-1], (list, tuple)) and len(item[-1]) >= 1:
                if isinstance(item[-1][0], str):
                    return item[-1][0]
            if isinstance(item[0], (list, tuple)):
                return extract_text_from_ocr_item(item[0])
            return None
        return None

    for row in tqdm(df.to_dict("records")):
        record_id = get_record_id(row)
        image_path = build_image_path(base_image_dir, record_id)

        if not os.path.exists(image_path):
            results.append({"ID": record_id, "Target": ""})
            continue

        raw_results = ocr.ocr(image_path, cls=use_angle_cls)
        if not raw_results:
            results.append({"ID": record_id, "Target": ""})
            continue

        texts = []
        for item in raw_results:
            text = extract_text_from_ocr_item(item)
            if text is not None:
                texts.append(str(text))

        pred_text = clean_output(" ".join(texts))
        results.append({"ID": record_id, "Target": pred_text})

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_csv, index=False)

    print("[DONE] saved:", output_csv)
    return out_df


def parse_args():
    parser = argparse.ArgumentParser(description="Run PaddleOCR inference on the Barbados Road Challenge test set.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--test_csv", default=None, help="Override test CSV file")
    parser.add_argument("--base_image_dir", default=None, help="Override base image directory")
    parser.add_argument("--output_csv", default=None, help="Override output CSV path")
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "gpu"], help="Device to use")
    parser.add_argument("--use_gpu", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=None, help="Enable GPU for PaddleOCR")
    parser.add_argument(
        "--use_angle_cls",
        type=lambda x: str(x).lower() in ("1", "true", "yes"),
        default=None,
        help="Enable angle classification",
    )
    parser.add_argument("--lang", default=None, help="Language for PaddleOCR")
    parser.add_argument("--det_db_thresh", type=float, default=None, help="Detection threshold")
    parser.add_argument("--det_db_box_thresh", type=float, default=None, help="Detection box threshold")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(
        config_path=args.config,
        test_csv=args.test_csv,
        base_image_dir=args.base_image_dir,
        output_csv=args.output_csv,
        device=args.device,
        use_gpu=args.use_gpu,
        use_angle_cls=args.use_angle_cls,
        lang=args.lang,
        det_db_thresh=args.det_db_thresh,
        det_db_box_thresh=args.det_db_box_thresh,
    )
