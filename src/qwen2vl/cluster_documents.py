"""
Document Clustering via Vision Embeddings

Clusters training images by visual similarity (scribe, page, scan condition)
to enable document-aware k-fold splitting. Solves the fold variance problem
where random splits leak same-document samples across train/val.

Usage:
    python cluster_documents.py --config config_qwen3_8b_full.yaml

Output:
    dataset/document_clusters.csv (ID, cluster_id)

Then use in train.py:
    group_col="cluster_id" in data config
"""

import argparse
import warnings
import logging
import os
from pathlib import Path

import yaml
import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import KMeans

# Suppress all warnings and verbose logs
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent


def resolve_model_class(model_name: str):
    """
    Pick the right model class for Qwen2/2.5/3-VL models.
    Copied from train.py for consistency.
    """
    name = model_name.lower()

    if "qwen3.vl" in name or "qwen3-vl" in name:
        from transformers import Qwen3VLForConditionalGeneration
        return Qwen3VLForConditionalGeneration, "qwen3"
    if "qwen2.5.vl" in name or "qwen2.5-vl" in name:
        from transformers import Qwen2_5_VLForConditionalGeneration
        return Qwen2_5_VLForConditionalGeneration, "qwen2.5"
    if "qwen2.vl" in name or "qwen2-vl" in name:
        from transformers import Qwen2VLForConditionalGeneration
        return Qwen2VLForConditionalGeneration, "qwen2"

    raise RuntimeError(f"Unknown Qwen-VL model: {model_name}")


def load_vision_tower(model_name: str, device: str = "cuda"):
    """Load only the vision tower from Qwen model (frozen, for embeddings)."""
    print(f"Loading vision tower from {model_name}...", end=" ", flush=True)

    # Resolve model class
    model_class, _ = resolve_model_class(model_name)

    # Load full model (we only need vision tower, but easier to load complete model)
    model = model_class.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": device} if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    # Extract and freeze vision tower
    # Qwen3-VL uses model.model.visual instead of model.visual
    if hasattr(model, 'visual'):
        vision_tower = model.visual
    elif hasattr(model, 'model') and hasattr(model.model, 'visual'):
        vision_tower = model.model.visual
    else:
        raise AttributeError(f"Cannot find vision tower in {model_class.__name__}")

    vision_tower.eval()
    for param in vision_tower.parameters():
        param.requires_grad = False

    print("✓")
    return vision_tower, model.config


def extract_vision_embeddings(
    vision_tower,
    image_paths: list,
    batch_size: int = 8,
    device: str = "cuda"
):
    """
    Extract vision embeddings for a list of images.

    Returns:
        embeddings: np.ndarray of shape (n_images, embedding_dim)
    """
    from transformers import AutoProcessor

    # Load processor for image preprocessing
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen3-VL-8B-Instruct",
        trust_remote_code=True,
    )

    all_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting embeddings"):
            batch_paths = image_paths[i:i + batch_size]

            # Load images
            images = []
            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    images.append(img)
                except Exception:
                    # Silent fallback: blank image
                    images.append(Image.new("RGB", (224, 224), (255, 255, 255)))

            # Preprocess images (this handles resizing, normalization, etc)
            # We just need pixel_values, not text processing
            try:
                # For Qwen3-VL, we need to provide a dummy text to get image preprocessing
                inputs = processor(
                    text=[""] * len(images),  # Dummy text
                    images=images,
                    return_tensors="pt",
                    padding=True,
                )

                pixel_values = inputs["pixel_values"].to(device, dtype=torch.bfloat16)

                # Pass through vision tower
                # Qwen3-VL needs grid_thw parameter (temporal, height, width grid)
                if "image_grid_thw" in inputs:
                    grid_thw = inputs["image_grid_thw"].to(device)
                    vision_outputs = vision_tower(pixel_values, grid_thw=grid_thw)
                else:
                    # Fallback for older models
                    vision_outputs = vision_tower(pixel_values)

                # Extract hidden states from output object
                # vision_outputs is BaseModelOutputWithDeepstackFeatures
                if hasattr(vision_outputs, 'last_hidden_state'):
                    hidden_states = vision_outputs.last_hidden_state
                elif hasattr(vision_outputs, 'hidden_states'):
                    hidden_states = vision_outputs.hidden_states
                elif isinstance(vision_outputs, tuple):
                    hidden_states = vision_outputs[0]
                else:
                    # Direct tensor
                    hidden_states = vision_outputs

                # hidden_states shape: (batch, seq_len, hidden_dim)
                # Mean pool over sequence dimension to get one vector per image
                pooled = hidden_states.mean(dim=1)  # (batch, hidden_dim)

                all_embeddings.append(pooled.cpu().numpy())

            except Exception as e:
                # Fallback: zero embeddings
                dummy_dim = 1024  # Typical hidden dim
                all_embeddings.append(np.zeros((len(images), dummy_dim)))
                print(f"⚠️  Error processing batch {i//batch_size}: {e}")

    # Concatenate all batches
    embeddings = np.vstack(all_embeddings)

    # Diagnostics: Check if embeddings are valid
    print(f"\nEmbedding diagnostics:")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Mean: {embeddings.mean():.6f}")
    print(f"  Std: {embeddings.std():.6f}")
    print(f"  Min: {embeddings.min():.6f}")
    print(f"  Max: {embeddings.max():.6f}")

    # Check for all-zero embeddings
    zero_count = (embeddings.sum(axis=1) == 0).sum()
    if zero_count > 0:
        print(f"  ⚠️  WARNING: {zero_count}/{len(embeddings)} embeddings are all zeros!")
        print(f"  This suggests batch processing errors. Check error messages above.")

    return embeddings


def cluster_embeddings(embeddings: np.ndarray, n_clusters: int, seed: int = 42):
    """
    Cluster embeddings using KMeans.

    Args:
        embeddings: (n_samples, embedding_dim)
        n_clusters: Number of clusters (estimate: 30-80 for typical dataset)
        seed: Random seed for reproducibility

    Returns:
        cluster_labels: np.ndarray of shape (n_samples,)
    """
    print(f"Clustering {len(embeddings)} images into {n_clusters} clusters...", end=" ", flush=True)

    # Normalize embeddings (cosine similarity-based clustering)
    embeddings_normalized = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings_normalized)

    print("✓")

    # Print cluster size statistics
    unique, counts = np.unique(cluster_labels, return_counts=True)
    print(f"Cluster sizes: min={counts.min()}, max={counts.max()}, "
          f"median={int(np.median(counts))}, mean={counts.mean():.1f}")
    print(f"Unique clusters: {len(unique)} (expected: {n_clusters})")

    # Check if clustering failed (all in one cluster)
    if len(unique) == 1:
        print(f"\n⚠️  CLUSTERING FAILED: All samples assigned to cluster {unique[0]}")
        print(f"  This usually means:")
        print(f"  1. All embeddings are identical (check diagnostics above)")
        print(f"  2. All embeddings are zeros (batch processing failed)")
        print(f"  3. Normalization issue (divide by zero)")
        print(f"\n  Normalized embedding stats:")
        print(f"    Mean: {embeddings_normalized.mean():.6f}")
        print(f"    Std: {embeddings_normalized.std():.6f}")
        print(f"    Contains NaN: {np.isnan(embeddings_normalized).any()}")
        print(f"    All zeros: {(embeddings_normalized == 0).all()}")

    return cluster_labels


def main():
    parser = argparse.ArgumentParser(description="Cluster documents by visual similarity")
    parser.add_argument(
        "--config",
        type=str,
        default="config_qwen3_8b_full.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=None,
        help="Number of clusters (default: auto-estimate from dataset size)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: dataset/document_clusters.csv)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for embedding extraction"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)"
    )
    args = parser.parse_args()

    # Load config
    config_path = SCRIPT_DIR / args.config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    # Resolve paths (handle both relative and absolute paths)
    train_csv_path = data_cfg["train_csv"]
    image_dir_path = data_cfg["image_dir"]

    # Convert to Path and resolve
    train_csv = Path(train_csv_path)
    if not train_csv.is_absolute():
        train_csv = REPO_ROOT / train_csv_path

    image_dir = Path(image_dir_path)
    if not image_dir.is_absolute():
        image_dir = REPO_ROOT / image_dir_path

    image_ext = data_cfg.get("image_ext", ".jpg")

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = REPO_ROOT / "dataset" / "document_clusters.csv"

    # Load training data
    df = pd.read_csv(train_csv)
    print(f"Loaded {len(df)} training samples from {train_csv.name}")

    # Build image paths
    image_paths = [image_dir / f"{row['ID']}{image_ext}" for _, row in df.iterrows()]

    # Check missing images
    missing = [p for p in image_paths if not p.exists()]
    if missing:
        print(f"⚠️  Warning: {len(missing)}/{len(image_paths)} images not found (will use blank fallback)")

    # Auto-estimate number of clusters if not provided
    # Rule of thumb: sqrt(n_samples) to n_samples/50
    if args.n_clusters is None:
        # Conservative: aim for ~50-100 samples per cluster
        n_clusters = max(30, min(80, len(df) // 60))
        print(f"Auto-estimated n_clusters: {n_clusters} (for {len(df)} samples)")
    else:
        n_clusters = args.n_clusters

    # Load vision tower
    vision_tower, _ = load_vision_tower(
        model_cfg["name"],
        device=args.device
    )

    # Extract embeddings
    print(f"\nExtracting vision embeddings (batch_size={args.batch_size})...")
    embeddings = extract_vision_embeddings(
        vision_tower,
        image_paths,
        batch_size=args.batch_size,
        device=args.device
    )
    print(f"Embeddings shape: {embeddings.shape}")

    # Cluster
    cluster_labels = cluster_embeddings(embeddings, n_clusters, seed=42)

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame({
        "ID": df["ID"],
        "cluster_id": cluster_labels,
    })
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved document clusters to: {output_path}")

    # Print usage instructions
    print("\n" + "="*70)
    print("To use in training, update your config:")
    print("="*70)
    print("""
data:
  train_csv: "dataset/Train.csv"
  cluster_csv: "dataset/document_clusters.csv"  # ADD THIS
  group_col: "cluster_id"  # ADD THIS
    """)
    print("="*70)
    print("\nThis will ensure StratifiedGroupKFold keeps document clusters together,")
    print("fixing fold variance caused by same-document leakage across train/val.")


if __name__ == "__main__":
    main()
