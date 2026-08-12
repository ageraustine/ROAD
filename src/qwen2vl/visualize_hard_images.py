"""
Visualize Hard Images

Shows the hardest/easiest images side-by-side to understand quality issues.

Usage:
    python visualize_hard_images.py --config config_qwen3_8b_full.yaml --n 10

Output:
    dataset/hard_images_visualization.png
"""

import argparse
from pathlib import Path

import yaml
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent


def create_visualization(quality_df, image_dir, image_ext, n=10, output_path=None):
    """Create side-by-side visualization of hardest and easiest images."""

    # Get top N hardest and easiest
    hardest = quality_df.nlargest(n, "difficulty_score")
    easiest = quality_df.nsmallest(n, "difficulty_score")

    # Create figure
    fig, axes = plt.subplots(n, 2, figsize=(12, 4*n))

    for i in range(n):
        # Hard image (left column)
        hard_row = hardest.iloc[i]
        hard_path = image_dir / f"{hard_row['ID']}{image_ext}"

        if hard_path.exists():
            img = Image.open(hard_path).convert("RGB")
            axes[i, 0].imshow(img)
        else:
            axes[i, 0].text(0.5, 0.5, "Image not found", ha='center', va='center')

        axes[i, 0].set_title(
            f"HARD #{i+1}: {hard_row['ID']}\n"
            f"Difficulty={hard_row['difficulty_score']:.1f}, "
            f"Contrast={hard_row['contrast']:.1f}, "
            f"Fade={hard_row['ink_fade']:.1f}",
            fontsize=8
        )
        axes[i, 0].axis('off')

        # Easy image (right column)
        easy_row = easiest.iloc[i]
        easy_path = image_dir / f"{easy_row['ID']}{image_ext}"

        if easy_path.exists():
            img = Image.open(easy_path).convert("RGB")
            axes[i, 1].imshow(img)
        else:
            axes[i, 1].text(0.5, 0.5, "Image not found", ha='center', va='center')

        axes[i, 1].set_title(
            f"EASY #{i+1}: {easy_row['ID']}\n"
            f"Difficulty={easy_row['difficulty_score']:.1f}, "
            f"Contrast={easy_row['contrast']:.1f}, "
            f"Fade={easy_row['ink_fade']:.1f}",
            fontsize=8
        )
        axes[i, 1].axis('off')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to: {output_path}")
    else:
        plt.show()

    plt.close()


def print_detailed_stats(quality_df, train_df, cluster_df):
    """Print detailed statistics about hard images."""

    # Merge with training data
    df = quality_df.merge(train_df, on="ID", how="left")

    if cluster_df is not None:
        df = df.merge(cluster_df, on="ID", how="left")

    # Define difficulty thresholds
    hard_threshold = df["difficulty_score"].quantile(0.9)  # Top 10%
    easy_threshold = df["difficulty_score"].quantile(0.1)  # Bottom 10%

    hard_df = df[df["difficulty_score"] >= hard_threshold]
    easy_df = df[df["difficulty_score"] <= easy_threshold]

    print("\n" + "="*80)
    print("Detailed Analysis: Hard vs Easy Images")
    print("="*80)

    print(f"\nHard images (top 10%, n={len(hard_df)}):")
    print(f"  Avg difficulty: {hard_df['difficulty_score'].mean():.2f}")
    print(f"  Avg text length: {hard_df['Target'].str.len().mean():.1f} chars")

    if "cluster_id" in hard_df.columns:
        print(f"  Unique clusters: {hard_df['cluster_id'].nunique()}/{hard_df['cluster_id'].max()+1}")
        print(f"  Most common clusters: {hard_df['cluster_id'].value_counts().head(3).to_dict()}")

    print(f"\nEasy images (bottom 10%, n={len(easy_df)}):")
    print(f"  Avg difficulty: {easy_df['difficulty_score'].mean():.2f}")
    print(f"  Avg text length: {easy_df['Target'].str.len().mean():.1f} chars")

    if "cluster_id" in easy_df.columns:
        print(f"  Unique clusters: {easy_df['cluster_id'].nunique()}/{easy_df['cluster_id'].max()+1}")
        print(f"  Most common clusters: {easy_df['cluster_id'].value_counts().head(3).to_dict()}")

    # Check if hard images are evenly distributed across clusters
    if "cluster_id" in df.columns:
        print("\n" + "="*80)
        print("Cluster Difficulty Distribution")
        print("="*80)

        cluster_difficulty = df.groupby("cluster_id")["difficulty_score"].agg(["mean", "count"])
        cluster_difficulty = cluster_difficulty.sort_values("mean", ascending=False)

        print(f"\n🔴 Top 5 hardest clusters:")
        for cluster_id, row in cluster_difficulty.head(5).iterrows():
            print(f"  Cluster {cluster_id:2d}: Avg difficulty = {row['mean']:.2f} ({int(row['count'])} images)")

        print(f"\n🟢 Top 5 easiest clusters:")
        for cluster_id, row in cluster_difficulty.tail(5).iterrows():
            print(f"  Cluster {cluster_id:2d}: Avg difficulty = {row['mean']:.2f} ({int(row['count'])} images)")

        cluster_std = cluster_difficulty["mean"].std()
        print(f"\nCluster difficulty variance: {cluster_std:.2f}")

        if cluster_std > 2.0:
            print(f"  🔴 HIGH variance - some clusters are significantly harder!")
            print(f"  This could contribute to fold variance if hard clusters")
            print(f"  are unevenly distributed across folds.")
        elif cluster_std > 1.0:
            print(f"  🟡 Moderate variance - slight difficulty imbalance")
        else:
            print(f"  🟢 Low variance - clusters have similar difficulty")


def main():
    parser = argparse.ArgumentParser(description="Visualize hard/easy images")
    parser.add_argument(
        "--config",
        type=str,
        default="config_qwen3_8b_full.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--quality_csv",
        type=str,
        default=None,
        help="Path to quality CSV (default: dataset/image_quality.csv)"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of hard/easy images to visualize"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output visualization path (default: dataset/hard_images_viz.png)"
    )
    parser.add_argument(
        "--skip_viz",
        action="store_true",
        help="Skip visualization, only print stats"
    )
    args = parser.parse_args()

    # Load config
    config_path = SCRIPT_DIR / args.config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]

    # Load quality scores
    if args.quality_csv:
        quality_path = Path(args.quality_csv)
    else:
        quality_path = REPO_ROOT / "dataset" / "image_quality.csv"

    if not quality_path.exists():
        print(f"❌ Quality CSV not found: {quality_path}")
        print(f"\nRun this first:")
        print(f"  python analyze_image_quality.py --config {args.config}")
        return

    quality_df = pd.read_csv(quality_path)
    quality_df = quality_df[quality_df["success"] == True]  # Only successful analyses
    print(f"Loaded quality scores for {len(quality_df)} images")

    # Load training data
    train_csv = REPO_ROOT / data_cfg["train_csv"]
    train_df = pd.read_csv(train_csv)

    # Load clusters if available
    cluster_df = None
    cluster_csv = data_cfg.get("cluster_csv")
    if cluster_csv:
        cluster_path = REPO_ROOT / cluster_csv
        if cluster_path.exists():
            cluster_df = pd.read_csv(cluster_path)
            print(f"Loaded {cluster_df['cluster_id'].nunique()} document clusters")

    # Print detailed stats
    print_detailed_stats(quality_df, train_df, cluster_df)

    # Create visualization
    if not args.skip_viz:
        image_dir = REPO_ROOT / data_cfg["image_dir"]
        image_ext = data_cfg.get("image_ext", ".jpg")

        if args.output:
            output_path = Path(args.output)
        else:
            output_path = REPO_ROOT / "dataset" / "hard_images_viz.png"

        create_visualization(
            quality_df,
            image_dir,
            image_ext,
            n=args.n,
            output_path=output_path
        )


if __name__ == "__main__":
    main()
