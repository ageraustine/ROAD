"""Kraken OCR training helper for Barbados Road Challenge with Auto-Model Downloader."""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml


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


def clean_output(text: str) -> str:
    text = str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.splitlines()]
    return " ".join([line for line in lines if line])


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


def get_record_text(row):
    return clean_output(row.get("Target", row.get("text", row.get("label", ""))))


def build_image_path(base_dir, record_id):
    return os.path.join(base_dir, f"{record_id}.jpg")


def prepare_training_data(csv_path, base_image_dir, train_data_dir, max_samples=None):
    df = pd.read_csv(csv_path)
    os.makedirs(train_data_dir, exist_ok=True)

    manifest_path = os.path.join(train_data_dir, "training_manifest.txt")
    with open(manifest_path, "w", encoding="utf-8") as manifest:
        added = 0
        missing = 0
        for _, row in df.iterrows():
            record_id = get_record_id(row)
            label = get_record_text(row)
            if not record_id or not label:
                continue

            image_path = build_image_path(base_image_dir, record_id)
            if not os.path.exists(image_path):
                missing += 1
                continue

            source_name = os.path.basename(image_path)
            target_image = os.path.join(train_data_dir, source_name)
            
            # Using Path dynamically extracts the true stem file name without any extension
            label_base = Path(image_path).stem
            while '.' in label_base:
                label_base = Path(label_base).stem
                
            target_label = os.path.join(train_data_dir, label_base + ".gt.txt")

            if not os.path.exists(target_image):
                try:
                    os.symlink(os.path.abspath(image_path), target_image)
                except OSError:
                    shutil.copy2(image_path, target_image)

            with open(target_label, "w", encoding="utf-8") as fp:
                fp.write(label)

            manifest.write(f"{target_image}\n")
            added += 1
            if max_samples and added >= max_samples:
                break

    if missing:
        print(f"[WARN] skipped {missing} missing images")
    print(f"[DATA] prepared {added} training samples in {train_data_dir}")
    return manifest_path


def download_pretrained_model(model_identifier):
    """Downloads a public Kraken model if it is not a locally reachable path."""
    # If the identifier points directly to a valid local file, use it as-is
    if os.path.exists(model_identifier):
        print(f"[MODEL] Using existing local model: {model_identifier}")
        return model_identifier

    print(f"[MODEL] Target '{model_identifier}' not found locally.")
    print(f"[MODEL] Attempting to fetch pretrained model via 'kraken get'...")
    
    try:
        # Use Kraken CLI to query and fetch the model definition/weights
        # This places downloaded .mlmodel configurations into your default active path
        cmd = ["kraken", "get", model_identifier]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Parse output to capture where Kraken stored the extracted .mlmodel file
        for line in result.stdout.splitlines():
            if "model files:" in line.lower() or ".mlmodel" in line.lower():
                print(f"[MODEL] Successfully fetched: {line.strip()}")
                
        # Return the target identifier; Kraken's fine-tuning system accepts downloaded IDs/Zenodo DOIs natively
        return model_identifier
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed downloading pretrained model via Kraken: {e.stderr}", file=sys.stderr)
        print("[ERROR] Please provide a valid file path or valid Zenodo DOI ID.", file=sys.stderr)
        raise e


def run_training(
    config_path=None,
    batch_size=None,
    epochs=None,
    output=None,
    train_data_dir=None,
    partition=None,
    spec=None,
    log_dir=None,
    format_type=None,
    force_binarization=None,
    train_csv=None,
    base_image_dir=None,
    load_model=None,
    append=None,
    load_hyper_parameters=False,
    resize=None,
):
    cfg = load_config(config_path)
    cfg_train = cfg.get("training", {})
    cfg_general = cfg.get("general", {})

    repo_root = cfg_general.get("repo_root")
    if repo_root is None:
        raise ValueError("Missing general.repo_root in config.yaml")

    train_csv = train_csv or cfg_train.get("train_csv", "code/scripts/data/Train.csv")
    base_image_dir = base_image_dir or cfg_train.get("base_image_dir", "code/scripts/data/images")
    output = output or cfg_train.get("output", "code/scripts/Kraken-OCR/kraken_model")
    
    # Set a robust community base model default if config or CLI arguments do not specify one
    load_model = load_model or cfg_train.get("load_model", "10.5281/zenodo.10592716")
    
    append = append if append is not None else cfg_train.get("append", None)
    load_hyper_parameters = load_hyper_parameters or cfg_train.get("load_hyper_parameters", False)
    resize = resize or cfg_train.get("resize", None)
    train_data_dir = train_data_dir or cfg_train.get("train_data_dir", "code/scripts/Kraken-OCR/kraken_train_data")
    batch_size = batch_size or cfg_train.get("batch_size", 16)
    epochs = epochs or cfg_train.get("epochs", 10)
    partition = partition or cfg_train.get("partition", 0.9)
    spec = spec or cfg_train.get("spec", None)
    log_dir = log_dir or cfg_train.get("log_dir", "code/scripts/Kraken-OCR/training_logs")
    format_type = format_type or cfg_train.get("format_type", "path")
    force_binarization = force_binarization if force_binarization is not None else cfg_train.get("force_binarization", False)

    train_csv = resolve(train_csv, repo_root)
    base_image_dir = resolve(base_image_dir, repo_root)
    output = resolve(output, repo_root)
    train_data_dir = resolve(train_data_dir, repo_root)
    log_dir = resolve(log_dir, repo_root)
    
    # FIX: Explicitly create the TensorBoard log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # If the output directory contains a folder structure, ensure it exists too
    if os.path.dirname(output):
        os.makedirs(os.path.dirname(output), exist_ok=True)
    
    # Resolve local pathing logic or fetch the base model parameters on-the-fly
    if load_model is not None:
        if os.path.exists(resolve(load_model, repo_root)):
            load_model = resolve(load_model, repo_root)
        else:
            load_model = download_pretrained_model(load_model)

    manifest_path = prepare_training_data(train_csv, base_image_dir, train_data_dir)

    args = [
        "-t",
        manifest_path,
        "-f",
        format_type,
        "-o",
        output,
        "-N",
        str(epochs),
        "-B",
        str(batch_size),
        "-p",
        str(partition),
        "--log-dir",
        log_dir,
    ]

    if spec is not None:
        args.extend(["-s", spec])

    if load_model is not None:
        args.extend(["-i", load_model])
        if load_hyper_parameters:
            args.append("--load-hyper-parameters")
        if append is not None:
            args.extend(["-a", str(append)])
        if resize is None:
            resize = "new"
        if resize is not None:
            args.extend(["--resize", resize])

    if force_binarization:
        args.append("--force-binarization")

    print("[TRAIN] running Kraken recognition training")
    ketos_cmd = shutil.which("ketos") or "ketos"
    cmd = [ketos_cmd, "--device", "auto", "train"] + args
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Kraken OCR model from train_split.csv")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--batch_size", type=int, default=None, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--output", default=None, help="Output model prefix")
    parser.add_argument("--train_data_dir", default=None, help="Directory to create symlinked training images and gt files")
    parser.add_argument("--partition", type=float, default=None, help="Train/validation split ratio")
    parser.add_argument("--spec", default=None, help="VGSL spec to use for model architecture")
    parser.add_argument("--log_dir", default=None, help="TensorBoard log directory")
    parser.add_argument("--format_type", default=None, help="Training format type")
    parser.add_argument("--force_binarization", action="store_true", help="Force binarization for training input")
    parser.add_argument("--resize", default=None, help="Resize output layer when loading a pretrained model: new, add, union, both, fail")
    parser.add_argument("--load_model", default=None, help="Path or Zenodo DOI ID of pretrained model to continue training")
    parser.add_argument("--append", type=int, default=None, help="Remove layers before argument and then append spec to loaded model")
    parser.add_argument("--load_hyper_parameters", action="store_true", help="Load hyperparameters from the pretrained model")
    parser.add_argument("--train_csv", default=None, help="Override train CSV path")
    parser.add_argument("--base_image_dir", default=None, help="Override base image directory")
    args = parser.parse_args()

    run_training(
        config_path=args.config,
        batch_size=args.batch_size,
        epochs=args.epochs,
        output=args.output,
        train_data_dir=args.train_data_dir,
        partition=args.partition,
        spec=args.spec,
        log_dir=args.log_dir,
        format_type=args.format_type,
        force_binarization=args.force_binarization,
        train_csv=args.train_csv,
        base_image_dir=args.base_image_dir,
        load_model=args.load_model,
        append=args.append,
        load_hyper_parameters=args.load_hyper_parameters,
        resize=args.resize,
    )