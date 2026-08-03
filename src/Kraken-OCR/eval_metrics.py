import os
import sys
import yaml
import pandas as pd

SCRIPT_DIR = os.path.dirname(__file__)
EVAL_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "evaluations"))
if EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)

from wer import compute_weighted_wer
from cer import compute_weighted_cer


def resolve(path, base):
    if os.path.isabs(path):
        return path
    return os.path.join(base, path)


def evaluate_set(gt_csv_path, pred_csv_path, set_name="SET", weight_factor=0.5, cer_weight=0.7, wer_weight=0.3):
    """
    Evaluate predictions against ground truth using WER and CER metrics.
    """
    wer_result = compute_weighted_wer(
        gt_csv_path,
        pred_csv_path,
        weight_factor=weight_factor,
    )
    cer_result = compute_weighted_cer(
        gt_csv_path,
        pred_csv_path,
        weight_factor=weight_factor,
    )

    if wer_result["matched"] == 0:
        print(f"===== {set_name} =====")
        print("No matching samples found!")
        return {"set": set_name, "cer": 1.0, "wer": 1.0, "final": 1.0}

    wer_score = wer_result["score"]
    cer_score = cer_result["score"]
    final_score = cer_weight * cer_score + wer_weight * wer_score

    print(f"\n===== {set_name} =====")
    print(f"Weight factor: {weight_factor}")
    print(f"Samples: {wer_result['matched']}/{wer_result['total_reference']}")
    print(f"CER:   {cer_score:.4f} (weight: {cer_weight})")
    print(f"WER:   {wer_score:.4f} (weight: {wer_weight})")
    print(f"FINAL: {final_score:.4f} ({cer_weight}*CER + {wer_weight}*WER)\n")

    return {
        "set": set_name,
        "cer": cer_score,
        "wer": wer_score,
        "final": final_score,
    }


if __name__ == "__main__":
    default_config = os.path.join(os.path.dirname(__file__), "config.yaml")

    cfg = {}
    if os.path.exists(default_config):
        with open(default_config) as f:
            cfg = yaml.safe_load(f) or {}

    cfg_eval = cfg.get("evaluation", {})

    SCRIPT_DIR = os.path.dirname(__file__)
    REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

    pred_csv = resolve(cfg_eval.get("pred_csv", "submission.csv"), SCRIPT_DIR)
    pub_gt = resolve(cfg_eval.get("pub_gt", "code/scripts/data/pub_ref.csv"), REPO_ROOT)
    priv_gt = resolve(cfg_eval.get("priv_gt", "code/scripts/data/priv_ref.csv"), REPO_ROOT)
    weight_factor = float(cfg_eval.get("weight_factor", 0.5))

    if not os.path.exists(pred_csv):
        raise FileNotFoundError(f"Prediction CSV not found: {pred_csv}")
    if not os.path.exists(pub_gt):
        raise FileNotFoundError(f"Public GT not found: {pub_gt}")
    if not os.path.exists(priv_gt):
        raise FileNotFoundError(f"Private GT not found: {priv_gt}")

    # Optional read to keep behavior consistent with current script expectations
    pd.read_csv(pred_csv)

    evaluate_set(pub_gt, pred_csv, "PUBLIC", weight_factor=weight_factor)
    evaluate_set(priv_gt, pred_csv, "PRIVATE", weight_factor=weight_factor)

