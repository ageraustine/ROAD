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
from pathlib import Path

import yaml
import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent


def load_vision_tower(model_name: str, device: str = "cuda"):
    """Load only the vision tower from Qwen model (frozen, for embeddings)."""
    print(f"Loading vision tower from {model_name}...", end=" ", flush=True)

    # Use AutoModel to handle Qwen2/Qwen3 automatically
    from transformers import AutoModelForVision2Seq

    # Load full model (we only need vision tower, but easier to load complete model)
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": device} if torch.cuda.is_available() else None,
        trust_remote_code=True,
        ignore_mismatched_sizes=True,  # Handle Qwen2 vs Qwen3 differences
    )

    # Extract and freeze vision tower
    vision_tower = model.visual
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
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    # Use blank image as fallback
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
                # Output shape: (batch, num_patches, hidden_dim)
                vision_outputs = vision_tower(pixel_values)

                # Pool to get single vector per image
                # Mean pooling over patches
                if isinstance(vision_outputs, tuple):
                    vision_outputs = vision_outputs[0]

                # vision_outputs shape: (batch, seq_len, hidden_dim)
                # Mean pool over sequence dimension
                pooled = vision_outputs.mean(dim=1)  # (batch, hidden_dim)

                all_embeddings.append(pooled.cpu().numpy())

            except Exception as e:
                print(f"Error processing batch {i}: {e}")
                # Fallback: zero embeddings
                dummy_dim = 1024  # Typical hidden dim
                all_embeddings.append(np.zeros((len(images), dummy_dim)))

    # Concatenate all batches
    embeddings = np.vstack(all_embeddings)

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
    print(f"Loading training data from {train_csv}...")
    df = pd.read_csv(train_csv)
    print(f"Found {len(df)} training samples")

    # Debug: print paths
    print(f"Image directory: {image_dir}")
    print(f"Image extension: {image_ext}")
    print(f"Image dir exists: {image_dir.exists()}")
    if image_dir.exists():
        sample_files = list(image_dir.glob("*"))[:3]
        print(f"Sample files in dir: {[f.name for f in sample_files]}")

    # Build image paths
    image_paths = [image_dir / f"{row['ID']}{image_ext}" for _, row in df.iterrows()]

    # Check missing images
    missing = [p for p in image_paths if not p.exists()]
    if missing:
        print(f"⚠️  Warning: {len(missing)} images not found")
        print(f"First missing path: {missing[0]}")
        print(f"Will use fallback (blank images)")

    # Auto-estimate number of clusters if not provided
    # Rule of thumb: sqrt(n_samples) to n_samples/50
    if args.n_clusters is None:
        # Conservative: aim for ~50-100 samples per cluster
        n_clusters = max(30, min(80, len(df) // 60))
        print(f"Auto-estimated n_clusters: {n_clusters} (for {len(df)} samples)")
    else:
        n_clusters = args.n_clusters

    # Load vision tower
    vision_tower, model_config = load_vision_tower(
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
