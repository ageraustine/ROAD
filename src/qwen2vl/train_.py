"""
Qwen-VL Fine-tuning for Historical HTR
Supports Qwen2-VL / Qwen2.5-VL / Qwen3-VL. Optimized for a single A100 80GB.

Refactor notes (vs. original):
  - Validation no longer augmented (separate eval collator).
  - Label masking anchored on the assistant header tokens, fails loudly.
  - Padding masked via attention_mask, not by token id (pad/eos collision safe).
  - NaN targets dropped instead of becoming the string "nan".
  - Optional CER metric via greedy decode on a fixed val subset.
  - Gradient checkpointing made PEFT-safe.
  - Per-fold model teardown; deterministic device placement.
"""

import gc
import os
import math
import json
import random
import inspect
import argparse
import logging
import warnings
from pathlib import Path

import yaml
import torch
import pandas as pd
import numpy as np
import cv2  # Used in augmentation code (even if currently disabled)
from io import BytesIO
from PIL import Image, ImageFilter, ImageEnhance
from sklearn.model_selection import train_test_split, StratifiedKFold
from scipy.ndimage import gaussian_filter  # For elastic deformation

# Suppress verbose warnings and progress bars
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
from datasets import Dataset
from tqdm import tqdm

import transformers
from transformers import (
    AutoProcessor,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model

try:
    from sklearn.model_selection import StratifiedGroupKFold
    GROUP_KFOLD_AVAILABLE = True
except ImportError:
    GROUP_KFOLD_AVAILABLE = False

import transformers

# v5 dropped a pile of TrainingArguments fields (warmup_ratio, overwrite_output_dir,
# evaluation_strategy, per_gpu_*) and renamed from_pretrained's torch_dtype -> dtype.
TF_MAJOR = int(transformers.__version__.split(".")[0])


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent

OCR_PROMPT = (
    "Transcribe the handwritten text in this image exactly as written. "
    "Preserve spelling, punctuation, and line breaks. Output only the transcription."
)

ASSISTANT_HEADER = "<|im_start|>assistant\n"


# ─────────────────────────────────────────────────────────────
# VERSION COMPATIBILITY
# ─────────────────────────────────────────────────────────────

def compute_schedule(n_samples: int, batch_size: int, grad_accum: int,
                     epochs: int, warmup_ratio: float):
    """
    Optimizer-step counts, used to express warmup as an integer.

    The original script used floor division on the sample count, which
    under-counts whenever the last batch is partial. The Trainer itself uses
    ceil at both levels, so this matches what actually runs and the cosine
    schedule lands on the true final step.
    """
    steps_per_epoch = math.ceil(math.ceil(n_samples / batch_size) / grad_accum)
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    return steps_per_epoch, total_steps, warmup_steps


def supported_kwargs(cls, kwargs: dict) -> dict:
    """
    Drop kwargs the installed version's __init__ doesn't accept, and say which.

    Avoids whack-a-mole across transformers v4/v5, where fields get removed
    one at a time and each one costs a full crash to discover.
    """
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return kwargs

    params = sig.parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs  # accepts **kwargs, nothing to filter

    valid = set(params)
    dropped = sorted(k for k in kwargs if k not in valid)
    if dropped:
        print(f"  [compat] {cls.__name__} on transformers {transformers.__version__} "
              f"does not accept: {', '.join(dropped)} - dropped")
    return {k: v for k, v in kwargs.items() if k in valid}


# ─────────────────────────────────────────────────────────────
# MODEL CLASS RESOLUTION
# ─────────────────────────────────────────────────────────────

def resolve_model_class(model_name: str):
    """
    Pick the right model class. Order matters: 'Qwen2.5-VL' contains 'Qwen2',
    so the 2.5 check must come first or it silently loads the wrong class.
    """
    name = model_name.lower().replace("_", ".").replace("-", ".")

    if "qwen3.vl" in name:
        from transformers import Qwen3VLForConditionalGeneration
        return Qwen3VLForConditionalGeneration, "qwen3"
    if "qwen2.5.vl" in name:
        from transformers import Qwen2_5_VLForConditionalGeneration
        return Qwen2_5_VLForConditionalGeneration, "qwen2.5"
    if "qwen2.vl" in name:
        from transformers import Qwen2VLForConditionalGeneration
        return Qwen2VLForConditionalGeneration, "qwen2"

    from transformers import AutoModelForVision2Seq
    return AutoModelForVision2Seq, "auto"


# ─────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────

def levenshtein(a: str, b: str) -> int:
    """Edit distance, O(min(len)) memory. Avoids a jiwer/evaluate dependency."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(
                prev[j] + 1,        # deletion
                curr[j - 1] + 1,    # insertion
                prev[j - 1] + (ca != cb),  # substitution
            ))
        prev = curr
    return prev[-1]


def corpus_cer(preds: list, refs: list) -> float:
    """Aggregate CER: total edits / total reference chars (not mean of per-sample CER)."""
    total_edits = sum(levenshtein(p, r) for p, r in zip(preds, refs))
    total_chars = sum(len(r) for r in refs)
    return total_edits / max(total_chars, 1)


def corpus_wer(preds: list, refs: list) -> float:
    """
    Aggregate WER: total word edits / total reference words.

    Word tokenization: split on whitespace (simple but effective for historical docs).
    """
    total_edits = 0
    total_words = 0

    for pred, ref in zip(preds, refs):
        # Simple word tokenization: split on whitespace
        pred_words = pred.split()
        ref_words = ref.split()

        # Compute word-level edit distance
        word_edits = word_levenshtein(pred_words, ref_words)
        total_edits += word_edits
        total_words += len(ref_words)

    return total_edits / max(total_words, 1)


def word_levenshtein(a: list, b: list) -> int:
    """Levenshtein distance at word level (instead of character level)."""
    if len(a) < len(b):
        a, b = b, a

    if not b:
        return len(a)

    prev = list(range(len(b) + 1))

    for i, a_word in enumerate(a, 1):
        curr = [i]
        for j, b_word in enumerate(b, 1):
            # Cost: 0 if words match, 1 if different
            cost = 0 if a_word == b_word else 1
            curr.append(min(
                prev[j] + 1,      # deletion
                curr[j - 1] + 1,  # insertion
                prev[j - 1] + cost  # substitution
            ))
        prev = curr

    return prev[-1]


# ─────────────────────────────────────────────────────────────
# AUGMENTATION
# ─────────────────────────────────────────────────────────────

class ImageAugmenter:
    """
    Conservative augmentation for already-degraded historical documents.
    Training only - never applied to validation.

    REFINEMENTS APPLIED (2026-08-13):
    ✓ Edge-Refinement Padding: Geometric transforms (rotation, shear, elastic) now use
      sampled background color from image edges instead of pure white (255,255,255).
      Prevents high-contrast rectangular border artifacts on cream/yellowed manuscripts.

    ✓ Scale-Aware Morphology: Dilation/erosion kernel sizes now scale with image height
      (k = max(1, int(h × 0.005))) to maintain consistent relative ink thickness changes
      across different resolutions. Low-res images get smaller kernels to avoid erasing
      fine strokes; high-res images get larger kernels for visible effect.

    ✓ Text Truncation Prevention: Rotation and shear now use expand=True and dynamic
      canvas sizing to prevent clipping character terminals and descenders near edges.

    PERFORMANCE NOTE (CPU Bottleneck):
    Heavy CPU-bound operations (SciPy gaussian_filter, OpenCV transforms, PIL conversions)
    occur in the data collator during batch prep, potentially bottlenecking high-throughput
    GPUs like A100 80GB.

    FUTURE OPTIMIZATION (GPU Acceleration):
    For maximum throughput, consider migrating to GPU-native augmentation:
    - Kornia (kornia.augmentation): PyTorch-native geometric transforms on GPU tensors
    - Albumentations (albumentation.pytorch): Widely-used, supports GPU via ToTensorV2
    - Custom CUDA kernels for elastic deformation (most expensive op currently)

    Migration would eliminate PIL ↔ NumPy ↔ PyTorch roundtrips and move elastic warp,
    shear, blur, and morphological ops directly into the GPU pipeline. Expected speedup:
    20-40% reduction in epoch time on A100.
    """

    def __init__(self, cfg: dict):
        self.enabled = cfg.get("enabled", False)
        self.p_blur = cfg.get("p_blur", 0.0)
        self.p_noise = cfg.get("p_noise", 0.0)
        self.p_brightness = cfg.get("p_brightness", 0.3)
        self.p_contrast = cfg.get("p_contrast", 0.3)
        self.p_rotate = cfg.get("p_rotate", 0.1)
        self.max_rotation = cfg.get("max_rotation", 1)

        # Advanced HTR augmentations
        self.p_morphology = cfg.get("p_morphology", 0.0)  # dilate/erode (ink thickness)
        self.p_shear = cfg.get("p_shear", 0.0)  # slant jitter (scribe variation)
        self.max_shear = cfg.get("max_shear", 8)  # degrees
        self.p_resolution = cfg.get("p_resolution", 0.0)  # OLD resolution jitter (downscale blur)
        self.p_jpeg = cfg.get("p_jpeg", 0.0)  # JPEG artifacts

        # NEW: Document-condition augmentations (based on analysis)
        self.p_elastic = cfg.get("p_elastic", 0.0)  # Paper warping/curling
        self.elastic_alpha = cfg.get("elastic_alpha", 25)  # Displacement strength
        self.elastic_sigma = cfg.get("elastic_sigma", 6)  # Smoothness
        self.elastic_interpolation = cfg.get("elastic_interpolation", "bicubic")

        self.p_color_jitter = cfg.get("p_color_jitter", 0.0)  # Paper color variance
        self.hue_jitter = cfg.get("hue_jitter", 0.05)  # ±5%
        self.saturation_jitter = cfg.get("saturation_jitter", 0.1)  # ±10%

        self.p_resolution_jitter = cfg.get("p_resolution_jitter", 0.0)  # NEW proper resolution jitter
        self.min_pixels_ratio = cfg.get("min_pixels_ratio", 0.7)
        self.max_pixels_ratio = cfg.get("max_pixels_ratio", 1.0)

    def _sample_background_color(self, img: Image.Image) -> tuple:
        """
        Sample mean background color from image edges (border replication alternative).

        Historical manuscripts are cream, yellowed, or dark brown - not pure white.
        This prevents high-contrast rectangular border artifacts from geometric transforms.

        Args:
            img: PIL Image

        Returns:
            (R, G, B) tuple of mean edge color
        """
        arr = np.array(img)
        h, w = arr.shape[:2]

        # Sample 5% border from all four edges
        border_size = max(1, int(min(h, w) * 0.05))

        # Concatenate all edge pixels
        top = arr[:border_size, :].reshape(-1, 3)
        bottom = arr[-border_size:, :].reshape(-1, 3)
        left = arr[:, :border_size].reshape(-1, 3)
        right = arr[:, -border_size:].reshape(-1, 3)

        edge_pixels = np.vstack([top, bottom, left, right])

        # Mean color across all edge pixels
        mean_color = edge_pixels.mean(axis=0).astype(int)

        return tuple(mean_color)

    def __call__(self, img: Image.Image, condition_score: float = None) -> Image.Image:
        """
        Apply augmentation with optional adaptive strength based on document condition.

        STRATEGY (4-class system based on dataset analysis):
        Updated 2026-08-13 after adding text_contrast detector.

        Excellent condition (< 19, 50% of data):
            - COMMON samples → aggressive augmentation
            - Synthesize degradation (blur, noise, color jitter) to add variety
            - Standard geometric augmentation

        Medium condition (19-37, 35% of data):
            - COMMON samples → standard augmentation
            - Moderate degradation + standard geometric

        Poor condition (37-42, 12% of data):
            - RARE samples → INCREASE geometric diversity to prevent overfitting
            - DISABLE degradation simulation (already degraded/faded)
            - INCREASE geometric augmentation 1.5x

        Very Poor condition (>42, 3% of data):
            - RARE outliers → MAXIMUM geometric diversity
            - DISABLE all degradation (already destroyed/severely faded)
            - INCREASE geometric augmentation 2.5x (prevent memorization)

        Args:
            img: Input image
            condition_score: Optional document condition score (0-100, higher = worse)
        """
        if not self.enabled:
            return img

        # Adaptive augmentation based on document condition (4 classes)
        # Thresholds updated 2026-08-13 after adding text_contrast detector
        if condition_score is not None:
            if condition_score < 19:  # Excellent condition (50% of data)
                # Synthesize degradation + standard geometric
                p_degradation_mult = 1.2  # Add blur, noise, color variance
                p_elastic_mult = 1.0
                p_resolution_mult = 1.0
                p_rotation_mult = 1.0
                min_pixels_override = 0.75
                p_color_mult = 1.2
                p_brightness_mult = 1.0
                p_contrast_mult = 1.0
                elastic_alpha_override = self.elastic_alpha

            elif condition_score < 37:  # Medium condition (35% of data)
                # Standard augmentation
                p_degradation_mult = 1.0
                p_elastic_mult = 1.0
                p_resolution_mult = 1.0
                p_rotation_mult = 1.0
                min_pixels_override = self.min_pixels_ratio
                p_color_mult = 1.0
                p_brightness_mult = 1.0
                p_contrast_mult = 1.0
                elastic_alpha_override = self.elastic_alpha

            elif condition_score < 42:  # Poor condition (12% of data - RARE!)
                # RARE: No degradation, MORE geometric (1.5x)
                # Includes faded-text documents (high text_contrast score)
                p_degradation_mult = 0.0  # Already degraded/faded
                p_elastic_mult = 1.5      # More warping
                p_resolution_mult = 1.5   # More resolution variance
                p_rotation_mult = 1.5     # More rotation
                min_pixels_override = 0.65
                p_color_mult = 0.0        # Already discolored/faded
                p_brightness_mult = 0.5   # Careful with faded ink
                p_contrast_mult = 0.5     # Careful with faded ink
                elastic_alpha_override = self.elastic_alpha * 1.4

            else:  # Very Poor condition (>42, 3% of data - RARE!)
                # RARE: Extreme outliers need MAXIMUM geometric diversity
                # Severely faded text + physical damage
                p_degradation_mult = 0.0  # Already destroyed/severely faded
                p_elastic_mult = 2.5      # MAXIMUM warping (2.5x)
                p_resolution_mult = 2.5   # MAXIMUM resolution variance
                p_rotation_mult = 2.0     # MAXIMUM rotation
                min_pixels_override = 0.5  # Aggressive downsampling OK
                p_color_mult = 0.0        # Already discolored/faded
                p_brightness_mult = 0.3   # Minimal (very faded)
                p_contrast_mult = 0.3     # Minimal (very faded)
                elastic_alpha_override = self.elastic_alpha * 1.8  # Strongest warping
        else:
            # No condition score - use defaults
            p_degradation_mult = 1.0
            p_elastic_mult = 1.0
            p_resolution_mult = 1.0
            p_rotation_mult = 1.0
            min_pixels_override = self.min_pixels_ratio
            p_color_mult = 1.0
            p_brightness_mult = 1.0
            p_contrast_mult = 1.0
            elastic_alpha_override = self.elastic_alpha

        # DEGRADATION augmentations (disabled for poor-condition docs)
        if random.random() < (self.p_blur * p_degradation_mult):
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

        if random.random() < (self.p_noise * p_degradation_mult):
            arr = np.array(img).astype(np.float32)
            arr += np.random.normal(0, random.uniform(5, 15), arr.shape)
            img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

        # SCANNER-SETTING augmentations (reduced for poor-condition docs)
        if random.random() < (self.p_brightness * p_brightness_mult):
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))

        if random.random() < (self.p_contrast * p_contrast_mult):
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))

        # GEOMETRIC augmentations (INCREASED for poor-condition docs)
        if random.random() < (self.p_rotate * p_rotation_mult):
            angle = random.uniform(-self.max_rotation, self.max_rotation)
            # REFINEMENT: Use sampled background color instead of white + expand to prevent clipping
            bg_color = self._sample_background_color(img)
            img = img.rotate(angle, fillcolor=bg_color, expand=True, resample=Image.BICUBIC)

        # DEGRADATION: Morphological ops (models ink thickness, pen pressure)
        # Disabled for poor-condition docs (ink already varied)
        if random.random() < (self.p_morphology * p_degradation_mult):
            arr = np.array(img)
            h, w = arr.shape[:2]

            # REFINEMENT: Scale-aware kernel size based on image resolution
            # k = max(1, int(h × 0.005)) ensures consistent relative ink thickness
            # Low-res: smaller kernel (avoids erasing fine strokes)
            # High-res: larger kernel (visible effect at higher DPI)
            kernel_size = max(1, int(h * 0.005))
            kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1  # Must be odd
            kernel = np.ones((kernel_size, kernel_size), np.uint8)

            # Dilate (thicken) or erode (thin) with equal probability
            if random.random() < 0.5:
                arr = cv2.dilate(arr, kernel, iterations=1)
            else:
                arr = cv2.erode(arr, kernel, iterations=1)

            # Slight blur to simulate ink diffusion (also scale-aware)
            blur_kernel = max(3, kernel_size // 2)
            blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
            arr = cv2.GaussianBlur(arr, (blur_kernel, blur_kernel), 0.5)
            img = Image.fromarray(arr)

        # GEOMETRIC: Shear/slant jitter (models different scribe handwriting angles)
        # Keep enabled for all conditions (independent of document damage)
        if random.random() < self.p_shear:
            angle_deg = random.uniform(-self.max_shear, self.max_shear)
            angle_rad = np.deg2rad(angle_deg)
            w, h = img.size

            # REFINEMENT: Use sampled background color + expand to prevent text clipping
            bg_color = self._sample_background_color(img)
            shear_factor = np.tan(angle_rad)

            # Calculate expanded output size to prevent clipping
            # Horizontal shear displaces top edge by shear_factor * h
            offset = abs(shear_factor * h)
            new_w = int(w + offset)

            # Affine transform for horizontal shear with expanded canvas
            img = img.transform(
                (new_w, h),
                Image.AFFINE,
                (1, shear_factor, -shear_factor * h / 2 if shear_factor > 0 else 0, 0, 1, 0),
                fillcolor=bg_color,
                resample=Image.BICUBIC  # Better quality than BILINEAR for text
            )

        # DEGRADATION: OLD Resolution jitter (downscale→upscale blur)
        # Disabled for poor-condition docs (already blurry)
        if random.random() < (self.p_resolution * p_degradation_mult):
            w, h = img.size
            scale = random.uniform(0.6, 1.0)
            new_w, new_h = int(w * scale), int(h * scale)

            # Downscale then upscale back (simulates lower resolution scans)
            img = img.resize((new_w, new_h), Image.BILINEAR)
            img = img.resize((w, h), Image.BILINEAR)

        # DEGRADATION: JPEG artifacts (models scan compression)
        # Disabled for poor-condition docs (already have artifacts)
        if random.random() < (self.p_jpeg * p_degradation_mult):
            buffer = BytesIO()
            quality = random.randint(60, 90)
            img.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            img = Image.open(buffer).convert('RGB')

        # GEOMETRIC: Elastic deformation (paper warping/curling during scanning)
        # INCREASED for poor-condition docs (burnt pages can still curl!)
        # This simulates physical deformation during scanning, NOT document damage
        if random.random() < (self.p_elastic * p_elastic_mult):
            img = self._apply_elastic_transform(img, elastic_alpha_override)

        # DEGRADATION: Color jitter (paper color variance - brown/cream aging)
        # DISABLED for poor-condition docs (already discolored)
        # HUE/SATURATION ONLY - NO brightness/contrast (degrades quality)
        if random.random() < (self.p_color_jitter * p_color_mult):
            img = self._apply_color_jitter(img)

        # GEOMETRIC: Resolution jitter (prevents overfitting to fixed pixel budget)
        # INCREASED for poor-condition docs (scan quality varies independently of damage)
        # Proper implementation: jitter max_pixels, not downscale→upscale blur
        if random.random() < (self.p_resolution_jitter * p_resolution_mult):
            img = self._apply_resolution_jitter(img, min_pixels_override)

        return img

    def _apply_elastic_transform(self, img: Image.Image, alpha: float = None) -> Image.Image:
        """
        Apply elastic deformation to simulate paper warping/curling.

        Based on upgrades.txt: α≈25, σ≈6, bicubic resampling
        CRITICAL: Use bicubic, not bilinear (bilinear smears faded ink)

        Args:
            img: Input image
            alpha: Optional override for displacement strength (adaptive augmentation)
        """
        arr = np.array(img)
        h, w = arr.shape[:2]

        # Generate random displacement fields
        alpha_value = alpha if alpha is not None else self.elastic_alpha
        dx = np.random.randn(h, w) * alpha_value
        dy = np.random.randn(h, w) * alpha_value

        # Smooth the displacement fields (Gaussian filter)
        dx = gaussian_filter(dx, self.elastic_sigma, mode='constant', cval=0)
        dy = gaussian_filter(dy, self.elastic_sigma, mode='constant', cval=0)

        # Create meshgrid for remapping
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        indices = (y + dy).astype(np.float32), (x + dx).astype(np.float32)

        # Apply displacement with bicubic interpolation
        # cv2.INTER_CUBIC = bicubic (preserves faded ink better than bilinear)
        # REFINEMENT: Use BORDER_REPLICATE instead of white constant to avoid border artifacts
        warped = cv2.remap(arr, indices[1], indices[0],
                          interpolation=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)

        return Image.fromarray(warped)

    def _apply_color_jitter(self, img: Image.Image) -> Image.Image:
        """
        Apply color jitter (hue/saturation ONLY) to simulate paper color variance.

        Document condition analysis shows mean paper_color_variance = 27.9
        (brown → cream shifts from aging, different paper batches)

        IMPORTANT: NO brightness/contrast jitter - degrades already-faded docs
        """
        # Convert to HSV for hue/saturation adjustment
        arr = np.array(img)
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).astype(np.float32)

        # Hue jitter (brown ↔ cream color shifts)
        hue_shift = random.uniform(-self.hue_jitter, self.hue_jitter) * 180  # OpenCV hue is 0-180
        hsv[:, :, 0] = np.clip(hsv[:, :, 0] + hue_shift, 0, 180)

        # Saturation jitter (faded vs vibrant paper)
        sat_factor = random.uniform(1 - self.saturation_jitter, 1 + self.saturation_jitter)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_factor, 0, 255)

        # Convert back to RGB
        rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        return Image.fromarray(rgb)

    def _apply_resolution_jitter(self, img: Image.Image, min_pixels_ratio: float = None) -> Image.Image:
        """
        Apply resolution jitter (proper implementation).

        OLD approach (disabled): downscale→upscale = blur artifact
        NEW approach: genuinely resize to variable resolution

        From upgrades.txt:
        "Jitter max_pixels per sample (0.7×–1.0×) and let the vision tower
        genuinely see a smaller image. One honest resize, no blur."

        Args:
            img: Input image
            min_pixels_ratio: Optional override for minimum pixels ratio (adaptive augmentation)
        """
        w, h = img.size
        current_pixels = w * h

        # Random pixel budget between min and max ratio
        min_ratio = min_pixels_ratio if min_pixels_ratio is not None else self.min_pixels_ratio
        ratio = random.uniform(min_ratio, self.max_pixels_ratio)
        target_pixels = int(current_pixels * ratio)

        # Calculate new dimensions maintaining aspect ratio
        scale = (target_pixels / current_pixels) ** 0.5
        new_w = int(w * scale)
        new_h = int(h * scale)

        # One honest resize (LANCZOS for quality)
        # Vision tower will see this resolution (no upscale back)
        img = img.resize((new_w, new_h), Image.LANCZOS)

        return img


# ─────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────

def load_image(path: str, max_pixels: int = 2016000) -> Image.Image:
    """Load and downscale (LANCZOS) while preserving aspect ratio."""
    img = Image.open(path).convert("RGB")
    w, h = img.size

    if w * h > max_pixels:
        scale = (max_pixels / (w * h)) ** 0.5
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    return img


def build_dataset(df: pd.DataFrame, image_dir: Path, image_ext: str = ".jpg",
                  split_name: str = "") -> Dataset:
    """
    Build a HF dataset, reporting every dropped row rather than silently skipping.

    Includes condition_score if available in df for adaptive augmentation.
    """
    samples = []
    dropped = {"missing_image": 0, "empty_target": 0, "bad_id": 0}
    has_condition = "condition_score" in df.columns

    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc=f"Building {split_name or 'dataset'}", leave=False):
        img_id = str(row["ID"]).strip()
        if not img_id or img_id.lower() == "nan":
            dropped["bad_id"] += 1
            continue

        target = row.get("Target")
        # pd.isna catches NaN/None. str(np.nan) would otherwise yield "nan".
        if target is None or (isinstance(target, float) and pd.isna(target)):
            dropped["empty_target"] += 1
            continue

        target = str(target).strip()
        if not target:
            dropped["empty_target"] += 1
            continue

        img_path = image_dir / f"{img_id}{image_ext}"
        if not img_path.exists():
            dropped["missing_image"] += 1
            continue

        sample = {"id": img_id, "image_path": str(img_path), "text": target}

        # Add condition score if available (for adaptive augmentation)
        if has_condition:
            condition = row.get("condition_score")
            if condition is not None and not pd.isna(condition):
                sample["condition_score"] = float(condition)

        samples.append(sample)

    n_dropped = sum(dropped.values())
    label = split_name or "dataset"
    # Only print if samples were dropped
    if n_dropped:
        detail = ", ".join(f"{k}={v}" for k, v in dropped.items() if v)
        print(f"  {label}: {len(samples)} kept ({n_dropped} dropped: {detail})")

    if not samples:
        raise RuntimeError(
            f"{label} is empty. Check image_dir, image_ext, and the ID column."
        )

    return Dataset.from_list(samples)


# ─────────────────────────────────────────────────────────────
# COLLATOR
# ─────────────────────────────────────────────────────────────

class OCRCollator:
    """
    Collator for Qwen-VL OCR training.

    Loss is computed only on the assistant turn. The span is located by finding
    the assistant header token sequence, not by string-matching the target text
    (BPE merges whitespace differently in context, so the old approach could
    silently match nothing and train on the prompt).
    """

    def __init__(self, processor, max_pixels: int, augmenter: ImageAugmenter = None,
                 max_length: int = 2048, strict: bool = True):
        self.processor = processor
        self.max_pixels = max_pixels
        self.augmenter = augmenter
        self.max_length = max_length
        self.strict = strict
        self.mask_failures = 0

        tokenizer = processor.tokenizer
        self.header_ids = tokenizer.encode(ASSISTANT_HEADER, add_special_tokens=False)
        if not self.header_ids:
            raise RuntimeError(
                f"Assistant header {ASSISTANT_HEADER!r} tokenized to nothing. "
                "This chat template is not Qwen-style; adjust ASSISTANT_HEADER."
            )

    def __call__(self, examples: list) -> dict:
        images, texts = [], []

        for ex in examples:
            img = load_image(ex["image_path"], self.max_pixels)
            if self.augmenter is not None:
                # Pass condition score for adaptive augmentation (if available)
                condition_score = ex.get("condition_score", None)
                img = self.augmenter(img, condition_score=condition_score)
            images.append(img)
            texts.append(ex["text"])

        messages_batch = [
            [
                {"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": OCR_PROMPT},
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": label}]},
            ]
            for img, label in zip(images, texts)
        ]

        texts_formatted = [
            self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            for m in messages_batch
        ]

        batch = self.processor(
            text=texts_formatted,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()

        for i in range(labels.size(0)):
            seq = batch["input_ids"][i].tolist()
            start = self._rfind_subsequence(seq, self.header_ids)

            if start == -1:
                self.mask_failures += 1
                if self.strict:
                    raise RuntimeError(
                        "Could not locate the assistant header in the tokenized "
                        f"sequence for sample {i}. Target was:\n  {texts[i][:120]!r}\n"
                        "Set training.strict_masking=false to skip these instead."
                    )
                # Mask the whole sample rather than train on the prompt.
                labels[i, :] = -100
                continue

            # Mask everything up to and including the header; the target and the
            # trailing <|im_end|> stay supervised so the model learns to stop.
            labels[i, : start + len(self.header_ids)] = -100

        # Mask padding by position. Masking by pad_token_id would also wipe any
        # legitimate occurrence of that id, and breaks if pad_token == eos_token.
        if "attention_mask" in batch:
            labels[batch["attention_mask"] == 0] = -100

        batch["labels"] = labels
        return batch

    @staticmethod
    def _rfind_subsequence(sequence: list, subsequence: list) -> int:
        """Last occurrence - the assistant header is the final one in the turn."""
        n = len(subsequence)
        for i in range(len(sequence) - n, -1, -1):
            if sequence[i:i + n] == subsequence:
                return i
        return -1


# ─────────────────────────────────────────────────────────────
# LoRA+ TRAINER
# ─────────────────────────────────────────────────────────────

class LoRAPlusTrainer(Trainer):
    """
    Custom Trainer implementing LoRA+ (Hayou et al. 2024).

    LoRA+ uses different learning rates for A and B matrices:
    - B matrices (lora_B): Higher LR (base_lr * loraplus_lr_ratio)
    - A matrices (lora_A): Base LR

    This improves convergence quality at same nominal LR.
    Set loraplus_lr_ratio in config (typical: 4-16, recommended: 8).
    """

    def __init__(self, *args, loraplus_lr_ratio=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loraplus_lr_ratio = loraplus_lr_ratio

    def create_optimizer(self):
        """Override to implement LoRA+ parameter groups."""
        if self.loraplus_lr_ratio is None or self.loraplus_lr_ratio <= 1.0:
            # Standard optimizer if LoRA+ disabled
            return super().create_optimizer()

        opt_model = self.model
        if self.optimizer is None:
            decay_params = []
            decay_params_lora_b = []
            nodecay_params = []
            nodecay_params_lora_b = []

            for name, param in opt_model.named_parameters():
                if not param.requires_grad:
                    continue

                # Check if this is a LoRA B matrix
                is_lora_b = "lora_B" in name

                # Weight decay applied to most params except biases, layernorms, embeddings
                if param.ndim < 2 or "bias" in name or "norm" in name or "embed" in name:
                    if is_lora_b:
                        nodecay_params_lora_b.append(param)
                    else:
                        nodecay_params.append(param)
                else:
                    if is_lora_b:
                        decay_params_lora_b.append(param)
                    else:
                        decay_params.append(param)

            base_lr = self.args.learning_rate
            lora_b_lr = base_lr * self.loraplus_lr_ratio
            weight_decay = self.args.weight_decay

            optimizer_grouped_parameters = [
                {
                    "params": decay_params,
                    "lr": base_lr,
                    "weight_decay": weight_decay,
                },
                {
                    "params": nodecay_params,
                    "lr": base_lr,
                    "weight_decay": 0.0,
                },
                {
                    "params": decay_params_lora_b,
                    "lr": lora_b_lr,
                    "weight_decay": weight_decay,
                },
                {
                    "params": nodecay_params_lora_b,
                    "lr": lora_b_lr,
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            # Remove lr from kwargs since we set it per group
            optimizer_kwargs.pop("lr", None)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            # Log LoRA+ setup
            num_lora_b = len(decay_params_lora_b) + len(nodecay_params_lora_b)
            num_lora_a = len(decay_params) + len(nodecay_params) - num_lora_b
            print(f"\n{'='*60}")
            print(f"LoRA+ Optimizer Setup:")
            print(f"  Base LR (LoRA A): {base_lr:.2e}")
            print(f"  LoRA B LR: {lora_b_lr:.2e} ({self.loraplus_lr_ratio}x)")
            print(f"  LoRA A params: {num_lora_a}")
            print(f"  LoRA B params: {num_lora_b}")
            print(f"{'='*60}\n")

        return self.optimizer


# ─────────────────────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────────────────────

class CERCallback(TrainerCallback):
    """
    Greedy-decode a fixed val subset and inject `eval_cer` into the metrics dict.

    eval_loss is teacher-forced and only loosely tracks the leaderboard metric.
    Mutating the metrics dict here works because Trainer.evaluate() fires
    on_evaluate before returning, so `metric_for_best_model: eval_cer` is valid.
    """

    def __init__(self, processor, dataset, max_pixels: int, n_samples: int = 200,
                 max_new_tokens: int = 256, batch_size: int = 4, seed: int = 42,
                 is_kfold: bool = False, verbose: bool = False):
        self.processor = processor
        self.max_pixels = max_pixels
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.is_kfold = is_kfold
        self.verbose = verbose

        n = min(n_samples, len(dataset))
        idx = np.random.RandomState(seed).choice(len(dataset), n, replace=False)
        self.samples = [dataset[int(i)] for i in sorted(idx)]

    @torch.no_grad()
    def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
        if metrics is None or model is None:
            return

        was_training = model.training
        model.eval()

        cache_owner = getattr(model, "config", None)
        prev_cache = getattr(cache_owner, "use_cache", None) if cache_owner else None
        if cache_owner is not None:
            cache_owner.use_cache = True

        tokenizer = self.processor.tokenizer
        prev_side = tokenizer.padding_side
        tokenizer.padding_side = "left"  # required for batched generation

        preds, refs = [], []
        try:
            for i in range(0, len(self.samples), self.batch_size):
                chunk = self.samples[i:i + self.batch_size]
                images = [load_image(s["image_path"], self.max_pixels) for s in chunk]

                prompts = [
                    self.processor.apply_chat_template(
                        [{"role": "user", "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": OCR_PROMPT},
                        ]}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for img in images
                ]

                inputs = self.processor(
                    text=prompts, images=images, padding=True, return_tensors="pt"
                ).to(model.device)

                out = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    num_beams=1,  # FIXED: Match inference decoder (was greedy=1, caused checkpoint selection bias)
                    repetition_penalty=1.0,  # FIXED: Match inference (legal boilerplate has legitimate repetition)
                    eos_token_id=tokenizer.eos_token_id,  # Explicitly enforce stop token
                    pad_token_id=tokenizer.pad_token_id,
                )
                trimmed = out[:, inputs["input_ids"].shape[1]:]
                decoded = self.processor.batch_decode(trimmed, skip_special_tokens=True)

                preds.extend(d.strip() for d in decoded)
                refs.extend(s["text"] for s in chunk)
        finally:
            tokenizer.padding_side = prev_side
            if cache_owner is not None and prev_cache is not None:
                cache_owner.use_cache = prev_cache
            if was_training:
                model.train()

        # Compute both CER and WER (competition uses 0.5*WER + 0.5*CER)
        cer = corpus_cer(preds, refs)
        wer = corpus_wer(preds, refs)
        competition_score = 0.5 * wer + 0.5 * cer

        metrics["eval_cer"] = cer
        metrics["eval_wer"] = wer
        metrics["eval_score"] = competition_score  # Competition metric

        # Track best competition score (what matters for leaderboard)
        if not hasattr(self, 'best_score'):
            self.best_score = float('inf')
            self.best_cer = float('inf')
            self.best_wer = float('inf')

        # Always print eval_score (not just when it improves)
        if competition_score < self.best_score:
            self.best_score = competition_score
            self.best_cer = cer
            self.best_wer = wer
            print(f"eval_score={competition_score:.4f} (cer={cer:.4f}, wer={wer:.4f}) ✓ (new best)")
        else:
            print(f"eval_score={competition_score:.4f} (cer={cer:.4f}, wer={wer:.4f})")


class ProgressCallback(TrainerCallback):
    """Print clean progress updates instead of verbose tqdm bars."""

    def __init__(self, log_every_n_steps: int = 50):
        self.log_every_n_steps = log_every_n_steps
        self.last_log_step = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Print clean training progress."""
        if logs is None:
            return

        # Only log at intervals
        if state.global_step - self.last_log_step < self.log_every_n_steps:
            return

        self.last_log_step = state.global_step

        # Build compact log line
        parts = []
        if "loss" in logs:
            parts.append(f"loss={logs['loss']:.4f}")
        if "learning_rate" in logs:
            parts.append(f"lr={logs['learning_rate']:.2e}")
        if "epoch" in logs:
            parts.append(f"epoch={logs['epoch']:.2f}")

        if parts:
            progress = f"Step {state.global_step}/{state.max_steps}"
            print(f"{progress} | {' | '.join(parts)}")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Print evaluation results (eval_loss only - CER callback handles eval_cer)."""
        if metrics is None:
            return

        # Only print eval_loss (CER callback handles eval_cer printing)
        if "eval_loss" in metrics:
            print(f"Eval @ step {state.global_step} | eval_loss={metrics['eval_loss']:.4f}")


class EarlyStoppingCallback(TrainerCallback):
    """
    Stop training when eval_loss plateaus (no improvement for N evaluations).

    Critical for K-fold CV where each fold can take hours - don't waste compute
    on a fold that stopped improving.
    """

    def __init__(self, patience: int = 3, min_delta: float = 0.0001,
                 metric: str = "eval_loss", greater_is_better: bool = False,
                 verbose: bool = False):
        """
        Args:
            patience: Number of evaluations with no improvement before stopping
            min_delta: Minimum change to qualify as improvement
            metric: Metric to monitor (eval_loss or eval_cer)
            greater_is_better: True if higher is better (False for loss/cer)
            verbose: Print patience counter on every evaluation (False = only on stop)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric
        self.greater_is_better = greater_is_better
        self.verbose = verbose

        self.best_value = float('inf') if not greater_is_better else float('-inf')
        self.patience_counter = 0
        self.stopped_epoch = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None or self.metric not in metrics:
            return

        current_value = metrics[self.metric]

        # Check if improvement
        if self.greater_is_better:
            improved = current_value > (self.best_value + self.min_delta)
        else:
            improved = current_value < (self.best_value - self.min_delta)

        if improved:
            self.best_value = current_value
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        # Log progress (only if verbose)
        if self.verbose and self.patience_counter > 0:
            print(f"  Early stopping: {self.patience_counter}/{self.patience} "
                  f"(no improvement in {self.metric})")

        # Stop if patience exceeded
        if self.patience_counter >= self.patience:
            print(f"\n⚠️  Early stopping triggered!")
            print(f"    {self.metric} plateaued at {self.best_value:.4f}")
            print(f"    No improvement for {self.patience} evaluations")
            control.should_training_stop = True
            self.stopped_epoch = state.epoch


# ─────────────────────────────────────────────────────────────
# CHECKPOINT UTILITIES
# ─────────────────────────────────────────────────────────────

def get_last_checkpoint(output_dir: Path) -> str | None:
    """
    Find the last checkpoint in output directory for resuming training.

    Returns:
        Path to last checkpoint, or None if no checkpoints found.
    """
    if not output_dir.exists():
        return None

    # Look for checkpoint-* directories (created by Trainer during training)
    checkpoints = [d for d in output_dir.iterdir()
                   if d.is_dir() and d.name.startswith("checkpoint-")]

    if not checkpoints:
        return None

    # Sort by step number (checkpoint-100, checkpoint-200, etc.)
    def get_step_number(ckpt_path):
        try:
            return int(ckpt_path.name.split("-")[-1])
        except (ValueError, IndexError):
            return 0

    checkpoints.sort(key=get_step_number)
    last_checkpoint = checkpoints[-1]

    return str(last_checkpoint)


# ─────────────────────────────────────────────────────────────
# MODEL SETUP
# ─────────────────────────────────────────────────────────────

def setup_model(cfg: dict):
    """Load model + processor and attach LoRA."""
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    model_name = model_cfg["name"]

    model_class, family = resolve_model_class(model_name)
    model_short_name = model_name.split("/")[-1]  # Just "Qwen3-VL-8B-Instruct"
    print(f"Loading {model_short_name}...", end=" ", flush=True)

    if model_class.__name__ == "AutoModelForVision2Seq":
        raise RuntimeError(
            f"No dedicated class resolved for {model_name}. This usually means "
            "transformers is too old (Qwen3-VL needs >=4.57). Upgrade rather than "
            "falling back silently."
        )

    dtype = torch.bfloat16 if model_cfg.get("torch_dtype", "bfloat16") == "bfloat16" \
        else torch.float16

    # Explicit single-GPU placement. device_map="auto" installs accelerate
    # dispatch hooks and turns into naive pipeline parallelism on multi-GPU,
    # which conflicts with Trainer.
    model_kwargs = {
        "device_map": {"": 0} if torch.cuda.is_available() else None,
        "trust_remote_code": True,
    }

    # v5 renamed from_pretrained's torch_dtype -> dtype for every model. On v4
    # only Qwen3-VL takes `dtype`. Passing the wrong one is silently ignored
    # (it lands in **kwargs) and you get fp32 weights and an OOM.
    if TF_MAJOR >= 5 or family == "qwen3":
        model_kwargs["dtype"] = dtype
    else:
        model_kwargs["torch_dtype"] = dtype

    if model_cfg.get("use_flash_attention", True):
        try:
            import flash_attn  # noqa: F401
            model_kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            model_kwargs["attn_implementation"] = "sdpa"
    else:
        model_kwargs["attn_implementation"] = "sdpa"

    model = model_class.from_pretrained(model_name, **model_kwargs)
    model.config.use_cache = False

    # Processor. Passing max_pixels makes the visual token count deterministic
    # instead of relying on the processor's default cap.
    processor_kwargs = {"trust_remote_code": True}
    if train_cfg.get("max_pixels"):
        processor_kwargs["max_pixels"] = train_cfg["max_pixels"]
    if train_cfg.get("min_pixels"):
        processor_kwargs["min_pixels"] = train_cfg["min_pixels"]
    processor = AutoProcessor.from_pretrained(model_name, **processor_kwargs)

    # Gradient checkpointing must be enabled BEFORE the PEFT wrap, and needs
    # input grads enabled or nothing flows back through the frozen base.
    if train_cfg.get("gradient_checkpointing", True):
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    lora_config = LoraConfig(
        r=train_cfg["lora_r"],
        lora_alpha=train_cfg["lora_alpha"],
        target_modules=train_cfg["lora_target_modules"],
        modules_to_save=train_cfg.get("lora_modules_to_save"),
        lora_dropout=train_cfg["lora_dropout"],
        use_rslora=train_cfg.get("use_rslora", False),  # Rank-Stabilized LoRA
        # use_dora=train_cfg.get("use_dora", False),
        rank_pattern=train_cfg.get("rank_pattern"),  # Asymmetric ranks (e.g., vision=64, llm=16)
        alpha_pattern=train_cfg.get("alpha_pattern"),  # Matching alpha for asymmetric ranks
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # LoRA+ configuration (different LR for A and B matrices)
    loraplus_lr_ratio = train_cfg.get("loraplus_lr_ratio", None)
    if loraplus_lr_ratio is not None and loraplus_lr_ratio > 1.0:
        print(f"LoRA+ enabled: B matrices will learn {loraplus_lr_ratio}x faster than A matrices")

    # Verify LoRA targeting - check structure, not just total
    from peft.tuners.lora import LoraLayer
    lora_modules = [n for n, m in model.named_modules() if isinstance(m, LoraLayer)]
    vision_modules = sum(1 for n in lora_modules if ".visual." in n)
    llm_modules = sum(1 for n in lora_modules if ".language_model." in n)

    # Count trainable params
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    trainable_pct = 100 * n_trainable / n_total

    print(f"✓ ({n_trainable/1e6:.1f}M trainable / {n_total/1e6:.0f}M total = {trainable_pct:.2f}%)")
    print(f"  LoRA modules: {len(lora_modules)} total ({vision_modules} vision / {llm_modules} llm)")

    # Sanity checks
    if n_trainable == 0:
        raise RuntimeError("LoRA matched zero modules. Check lora_target_modules regex.")
    if vision_modules == 0:
        raise RuntimeError(f"LoRA matched 0 vision modules (expected ~116). Check target_modules regex.")

    return model, processor


# ─────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────

def train_single_fold(train_df, val_df, image_dir, fold_output_dir, cfg,
                      fold_num=None, n_folds=None):
    """Train one fold (or a single model when fold_num is None). Returns metrics dict."""
    # Suppress verbose library logs
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("transformers.trainer").setLevel(logging.WARNING)
    logging.getLogger("accelerate").setLevel(logging.ERROR)
    logging.getLogger("torch").setLevel(logging.ERROR)
    logging.getLogger("peft").setLevel(logging.WARNING)

    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    aug_cfg = cfg.get("augmentation", {})

    fold_output_dir.mkdir(parents=True, exist_ok=True)
    is_kfold = fold_num is not None
    fold_str = f"Fold {fold_num}/{n_folds}" if is_kfold else "Training"

    print(f"\n{'=' * 70}")
    print(f"{fold_str} | Train: {len(train_df)} | Val: {len(val_df)}")
    print(f"{'=' * 70}")

    image_ext = data_cfg.get("image_ext", ".jpg")
    train_dataset = build_dataset(train_df, image_dir, image_ext, "train")
    val_dataset = build_dataset(val_df, image_dir, image_ext, "val")

    model, processor = setup_model(cfg)

    max_pixels = train_cfg["max_pixels"]
    max_length = train_cfg.get("max_seq_length", 2048)
    strict = train_cfg.get("strict_masking", True)

    # Two collators. The eval one has no augmenter, so eval_loss is a fixed
    # measurement rather than a fresh random draw each time.
    train_collator = OCRCollator(processor, max_pixels, ImageAugmenter(aug_cfg),
                                 max_length, strict)
    eval_collator = OCRCollator(processor, max_pixels, None, max_length, strict)

    eval_bs = train_cfg.get("eval_batch_size", train_cfg["batch_size"])
    metric = train_cfg.get("metric_for_best_model", "eval_loss")

    # Express warmup as an explicit integer step count. transformers v5 removed
    # warmup_ratio; its warmup_steps accepts a float there but not on v4, so an
    # int is the only form that works on both. (The original script did this too.)
    steps_per_epoch, total_steps, warmup_steps = compute_schedule(
        n_samples=len(train_dataset),
        batch_size=train_cfg["batch_size"],
        grad_accum=train_cfg["gradient_accumulation_steps"],
        epochs=train_cfg["epochs"],
        warmup_ratio=train_cfg.get("warmup_ratio", 0.1),
    )
    print(f"Steps: {steps_per_epoch}/epoch × {train_cfg['epochs']} epochs = {total_steps} total")

    # Reduce logging frequency for non-kfold (less spam)
    log_steps = train_cfg.get("logging_steps", 50 if not is_kfold else 25)

    ta_kwargs = dict(
        output_dir=str(fold_output_dir),
        per_device_train_batch_size=train_cfg["batch_size"],
        per_device_eval_batch_size=eval_bs,
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        num_train_epochs=train_cfg["epochs"],
        learning_rate=train_cfg["learning_rate"],
        warmup_steps=warmup_steps,
        lr_scheduler_type=train_cfg["lr_scheduler"],
        weight_decay=train_cfg["weight_decay"],
        max_grad_norm=train_cfg["max_grad_norm"],
        bf16=True,
        logging_steps=log_steps,
        logging_strategy="steps",
        logging_first_step=False,
        logging_nan_inf_filter=True,  # Suppress NaN/inf warnings
        eval_strategy="steps",
        eval_steps=train_cfg["eval_steps"],
        save_strategy="steps",
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model=metric,
        greater_is_better=False,  # both eval_loss and eval_cer: lower is better
        dataloader_num_workers=train_cfg["num_workers"],
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        label_names=["labels"],  # PEFT wrappers hide this from Trainer's autodetect
        report_to="none",
        seed=data_cfg.get("seed", 42),
        data_seed=data_cfg.get("seed", 42),
        disable_tqdm=True,  # Disable all progress bars (too verbose)
        log_level="error",  # Suppress info logs from trainer
        log_level_replica="error",
    )
    args = TrainingArguments(**supported_kwargs(TrainingArguments, ta_kwargs))

    callbacks = []

    # Progress callback (replaces verbose tqdm with clean updates)
    if not is_kfold:
        callbacks.append(ProgressCallback(log_every_n_steps=log_steps))

    # Optional CER computation callback
    if train_cfg.get("compute_cer", True):
        callbacks.append(CERCallback(
            processor=processor,
            dataset=val_dataset,
            max_pixels=max_pixels,
            n_samples=train_cfg.get("cer_samples", 200),
            max_new_tokens=cfg.get("inference", {}).get("max_new_tokens", 256),
            batch_size=eval_bs,
            seed=data_cfg.get("seed", 42),
            is_kfold=is_kfold,
            verbose=False,  # Only print improvements
        ))
    elif metric == "eval_cer":
        raise ValueError("metric_for_best_model='eval_cer' requires compute_cer: true")

    # Early stopping callback (prevent wasting compute on plateaued folds)
    # IMPORTANT: Early stopping should monitor eval_loss (stable signal), while
    # metric_for_best_model can be eval_cer (what matters for leaderboard).
    early_stop_cfg = train_cfg.get("early_stopping", {})
    if early_stop_cfg.get("enabled", True):
        patience = early_stop_cfg.get("patience", 3)
        min_delta = early_stop_cfg.get("min_delta", 0.0001)

        # Use dedicated early stopping metric (defaults to eval_loss for stability)
        # eval_loss: computed on full val set, deterministic, low variance
        # eval_cer: computed on 200 samples, generative, high variance (bad for early stop)
        early_stop_metric = early_stop_cfg.get("metric", "eval_loss")

        callbacks.append(EarlyStoppingCallback(
            patience=patience,
            min_delta=min_delta,
            metric=early_stop_metric,
            greater_is_better=False,  # eval_loss and eval_cer: lower is better
            verbose=False,  # Don't print patience counter every eval
        ))

    # LoRA+ configuration (read from config for trainer instantiation)
    loraplus_lr_ratio = train_cfg.get("loraplus_lr_ratio", None)

    # Create trainer without default callbacks (too verbose)
    # Use LoRAPlusTrainer if loraplus_lr_ratio is set, otherwise standard Trainer
    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "data_collator": train_collator,
        "callbacks": callbacks,
    }
    if loraplus_lr_ratio:
        trainer_kwargs["loraplus_lr_ratio"] = loraplus_lr_ratio
        trainer = LoRAPlusTrainer(**trainer_kwargs)
    else:
        trainer = Trainer(**trainer_kwargs)

    # Remove default progress callback (replaced by our ProgressCallback)
    trainer.remove_callback(transformers.trainer_callback.ProgressCallback)
    # Trainer has no eval_data_collator arg; swap it in for the eval dataloader only.
    _base_get_eval = trainer.get_eval_dataloader

    def get_eval_dataloader(eval_dataset=None):
        saved, trainer.data_collator = trainer.data_collator, eval_collator
        try:
            return _base_get_eval(eval_dataset)
        finally:
            trainer.data_collator = saved

    trainer.get_eval_dataloader = get_eval_dataloader

    # Check for existing checkpoint to resume from
    resume_checkpoint = None
    if train_cfg.get("resume_from_checkpoint", True):
        resume_checkpoint = get_last_checkpoint(fold_output_dir)
        if resume_checkpoint:
            print(f"⏩ Resuming from {Path(resume_checkpoint).name}\n")
        else:
            print(f"Starting training...\n")
    else:
        print(f"Starting training...\n")

    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # Check if early stopping was triggered
    early_stop_callback = None
    for cb in callbacks:
        if isinstance(cb, EarlyStoppingCallback):
            early_stop_callback = cb
            break

    early_stopped = early_stop_callback.stopped_epoch if early_stop_callback else None

    # load_best_model_at_end=True means the in-memory model IS the best
    # checkpoint here. The original script also wrote a "final/" dir with these
    # same weights, which was misleading - dropped.
    best_dir = fold_output_dir / "best"
    trainer.save_model(str(best_dir))
    processor.save_pretrained(str(best_dir))

    history = [h for h in trainer.state.log_history if "eval_loss" in h]
    result = {
        "fold": fold_num,
        "best_metric": trainer.state.best_metric,
        "metric_name": metric,
        "best_eval_loss": min((h["eval_loss"] for h in history), default=None),
        "best_eval_cer": min((h["eval_cer"] for h in history if "eval_cer" in h), default=None),
        "best_eval_wer": min((h["eval_wer"] for h in history if "eval_wer" in h), default=None),
        "best_eval_score": min((h["eval_score"] for h in history if "eval_score" in h), default=None),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "mask_failures": train_collator.mask_failures + eval_collator.mask_failures,
        "early_stopped": early_stopped is not None,
        "stopped_epoch": early_stopped,
        "epochs_completed": trainer.state.epoch,
    }

    print(f"\n{'=' * 70}")
    status = f"Early stopped at epoch {early_stopped:.1f}" if early_stopped else f"Completed {result['epochs_completed']:.1f} epochs"
    print(f"✓ {fold_str} | {metric}={result['best_metric']:.4f} | {status}")

    # Print competition metrics
    if result["best_eval_score"] is not None:
        print(f"  Competition score (0.5*WER + 0.5*CER): {result['best_eval_score']:.4f}")
    if result["best_eval_cer"] is not None and result["best_eval_wer"] is not None:
        print(f"  CER: {result['best_eval_cer']:.4f} | WER: {result['best_eval_wer']:.4f}")

    if result["mask_failures"]:
        print(f"  ⚠️  {result['mask_failures']} samples failed label masking")
    print(f"  Saved: {best_dir}")
    print(f"{'=' * 70}\n")

    # Explicit teardown. Sequential folds otherwise fragment VRAM and OOM
    # around fold 4.
    del trainer, model, train_collator, eval_collator
    gc.collect()
    torch.cuda.empty_cache()

    return result


def compute_special_char_density(text: str) -> float:
    """
    Compute special character density as stratification proxy.

    Special characters indicate:
    - Document type (legal docs with £, dates, formal punctuation)
    - Scribe style (abbreviations, dashes)
    - Transcription complexity

    Returns: ratio of special chars to total chars (0.0 to 1.0)
    """
    if not text or pd.isna(text):
        return 0.0

    text = str(text)
    # Count non-alphanumeric, non-space characters
    special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
    return special_chars / max(len(text), 1)


def compute_digit_density(text: str) -> float:
    """
    Compute digit/number density as stratification proxy.

    Digits indicate:
    - Dates (1842, 15th)
    - Monetary amounts (£25-10-6)
    - Measurements (3 acres)
    - Different OCR challenge (digits often harder than letters)

    Returns: ratio of digits to total chars (0.0 to 1.0)
    """
    if not text or pd.isna(text):
        return 0.0

    text = str(text)
    digits = sum(1 for c in text if c.isdigit())
    return digits / max(len(text), 1)


def compute_uppercase_ratio(text: str) -> float:
    """
    Compute uppercase letter ratio as formality/emphasis indicator.

    Uppercase indicates:
    - Proper nouns (John Smith, London)
    - Formal language (WITNESSED, SEALED)
    - Emphasis and titles (Mr., Esq.)
    - Different capitalization patterns across document types

    Returns: ratio of uppercase letters to total letters (0.0 to 1.0)
    """
    if not text or pd.isna(text):
        return 0.0

    text = str(text)
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    uppercase = sum(1 for c in letters if c.isupper())
    return uppercase / len(letters)


def compute_lexical_diversity(text: str) -> float:
    """
    Compute lexical diversity (unique word ratio) as vocabulary complexity indicator.

    Lexical diversity indicates:
    - Repetitive/formulaic language (low diversity): "the said party... the said party"
    - Rich vocabulary (high diversity): "signed, sealed, witnessed, delivered, dated"
    - Document type (legal templates vs descriptive narratives)

    Returns: ratio of unique words to total words (0.0 to 1.0)
    """
    if not text or pd.isna(text):
        return 0.0

    words = str(text).lower().split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def compute_avg_word_length(text: str) -> float:
    """
    Compute average word length as vocabulary complexity indicator.

    Word length indicates:
    - Simple vocabulary (short words): "I see the man go"
    - Complex vocabulary (long words): "aforementioned beneficiary witnessed"
    - Different OCR challenge (longer words = more opportunities for errors)

    Returns: average characters per word
    """
    if not text or pd.isna(text):
        return 0.0

    words = [w for w in str(text).split() if w]
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)


def compute_ngram_jaccard(text1: str, text2: str, n: int = 3) -> float:
    """
    Compute character n-gram Jaccard similarity between two texts.

    Used for fuzzy boilerplate detection - historical legal documents often share
    90%+ boilerplate with only names/dates differing (e.g., "This Indenture made
    the [DATE] between [NAMES]...").

    Args:
        text1, text2: Texts to compare
        n: N-gram size (default 3-char for historical text)

    Returns:
        Jaccard similarity (0.0 to 1.0)
    """
    if not text1 or not text2:
        return 0.0

    # Normalize: lowercase, strip whitespace
    t1 = text1.lower().strip()
    t2 = text2.lower().strip()

    if t1 == t2:
        return 1.0

    # Generate character n-grams
    def get_ngrams(text, n):
        return set(text[i:i+n] for i in range(len(text) - n + 1))

    ngrams1 = get_ngrams(t1, n)
    ngrams2 = get_ngrams(t2, n)

    if not ngrams1 or not ngrams2:
        return 0.0

    # Jaccard similarity: |A ∩ B| / |A ∪ B|
    intersection = len(ngrams1 & ngrams2)
    union = len(ngrams1 | ngrams2)

    return intersection / union if union > 0 else 0.0


def make_splits(df: pd.DataFrame, data_cfg: dict):
    """Yield (train_df, val_df, fold_num). Stratified by semantic text properties from ground truth. Grouped by group_col when present."""
    df = df.copy()

    # STEP 1: Identify duplicate and near-duplicate texts to prevent train/val leakage
    print("Checking for duplicate and near-duplicate texts (fuzzy boilerplate detection)...")
    df["_text_clean"] = df["Target"].astype(str).str.lower().str.strip()

    # REFINEMENT: Fuzzy duplicate detection using character n-gram Jaccard similarity
    # Historical legal docs share 90%+ boilerplate with only names/dates differing
    # Example: "This Indenture made [DATE] between [NAMES]..." → 95% similarity
    # Config: data.fuzzy_duplicate_threshold (default 0.90, set to 0 to disable)
    fuzzy_threshold = data_cfg.get("fuzzy_duplicate_threshold", 0.90)

    # First pass: exact duplicates (fast)
    text_counts = df["_text_clean"].value_counts()
    exact_duplicates = text_counts[text_counts > 1]

    # Initialize duplicate groups
    df["_dup_group"] = -1  # -1 = not a duplicate
    next_group_id = 0

    # Group exact duplicates first
    if len(exact_duplicates) > 0:
        for dup_text in exact_duplicates.index:
            df.loc[df["_text_clean"] == dup_text, "_dup_group"] = next_group_id
            next_group_id += 1

    exact_grouped = (df["_dup_group"] >= 0).sum()
    print(f"  Exact matches: {len(exact_duplicates)} unique texts ({exact_grouped} samples)")

    # Second pass: fuzzy duplicates (slower, O(n²) - only on ungrouped samples)
    ungrouped_indices = df[df["_dup_group"] == -1].index.tolist()
    fuzzy_grouped = 0

    if len(ungrouped_indices) > 1 and fuzzy_threshold > 0:
        print(f"  Scanning {len(ungrouped_indices)} ungrouped samples for fuzzy duplicates (threshold={fuzzy_threshold})...")

        # Build list of ungrouped texts
        ungrouped_texts = [(idx, df.loc[idx, "_text_clean"]) for idx in ungrouped_indices]

        # Pairwise comparison (optimized: only compare each pair once)
        for i in range(len(ungrouped_texts)):
            idx_i, text_i = ungrouped_texts[i]

            # Skip if already grouped
            if df.loc[idx_i, "_dup_group"] >= 0:
                continue

            # Find all similar texts (including self)
            similar_group = [idx_i]

            for j in range(i + 1, len(ungrouped_texts)):
                idx_j, text_j = ungrouped_texts[j]

                # Skip if already grouped
                if df.loc[idx_j, "_dup_group"] >= 0:
                    continue

                # Compute fuzzy similarity
                similarity = compute_ngram_jaccard(text_i, text_j, n=3)

                if similarity >= fuzzy_threshold:
                    similar_group.append(idx_j)

            # If found similar texts, create a new group
            if len(similar_group) > 1:
                for idx in similar_group:
                    df.loc[idx, "_dup_group"] = next_group_id
                next_group_id += 1
                fuzzy_grouped += len(similar_group)

        print(f"  Fuzzy matches: {fuzzy_grouped} samples grouped into {next_group_id - len(exact_duplicates)} boilerplate clusters")
    else:
        print(f"  Fuzzy matching disabled (threshold={fuzzy_threshold})")

    total_grouped = (df["_dup_group"] >= 0).sum()
    if total_grouped > 0:
        print(f"  Total grouped: {total_grouped} samples ({exact_grouped} exact + {fuzzy_grouped} fuzzy)")

    # STEP 2: Load text difficulty scores (pre-computed from analysis)
    text_difficulty_path = REPO_ROOT / "dataset" / "text_difficulty.csv"

    if text_difficulty_path.exists():
        print(f"Loading text difficulty scores from {text_difficulty_path.name}...")
        text_diff_df = pd.read_csv(text_difficulty_path)
        df = df.merge(text_diff_df[["ID", "difficulty_score", "named_entity_score", "number_complexity"]],
                      on="ID", how="left")

        # Fill missing with median
        df["difficulty_score"] = df["difficulty_score"].fillna(df["difficulty_score"].median())
        df["named_entity_score"] = df["named_entity_score"].fillna(df["named_entity_score"].median())
        df["number_complexity"] = df["number_complexity"].fillna(df["number_complexity"].median())

        print(f"  Text difficulty range: {df['difficulty_score'].min():.1f} - {df['difficulty_score'].max():.1f}")
        print(f"  Using TEXT DIFFICULTY stratification (analysis-driven)")
    else:
        print(f"⚠️  Text difficulty CSV not found at {text_difficulty_path}")
        print("  Falling back to computing basic semantic features...")
        # Compute basic features as fallback
        df["_digit_density"] = df["Target"].apply(compute_digit_density)
        df["_uppercase_ratio"] = df["Target"].apply(compute_uppercase_ratio)
        df["_lexical_diversity"] = df["Target"].apply(compute_lexical_diversity)

        # Create proxy difficulty score
        df["difficulty_score"] = (
            df["_digit_density"] * 50 +  # Numbers are hard
            df["_uppercase_ratio"] * 30 +  # Names are hard
            df["_lexical_diversity"] * 20  # Diverse vocab is hard
        )
        df["named_entity_score"] = df["_uppercase_ratio"] * 100
        df["number_complexity"] = df["_digit_density"] * 100

    # Compute remaining text features for logging
    df["_digit_density"] = df["Target"].apply(compute_digit_density)
    df["_uppercase_ratio"] = df["Target"].apply(compute_uppercase_ratio)
    df["_lexical_diversity"] = df["Target"].apply(compute_lexical_diversity)
    df["_special_char_density"] = df["Target"].apply(compute_special_char_density)
    df["_avg_word_length"] = df["Target"].apply(compute_avg_word_length)

    # STRATIFICATION STRATEGY (REFINEMENT 2026-08-13):
    # Joint Visual-Linguistic Stratification to balance both text complexity AND visual degradation
    # Previous: Text difficulty (3) × Digits (2) × Names (2) = 12 bins (text-only)
    # Current: Condition (3) × Text difficulty (3) × Digits (2) × Names (2) = 36 bins (visual + text)

    # 1. Visual condition bins (Good/Medium/Poor) - NEW!
    has_condition = "condition_score" in df.columns and not df["condition_score"].isna().all()
    if has_condition:
        try:
            # Bin by visual degradation: good (<20), medium (20-35), poor (>35)
            # These thresholds match the adaptive augmentation tiers
            df["_condition_bin"] = pd.qcut(
                df["condition_score"],
                q=3,
                labels=["good_cond", "medium_cond", "poor_cond"],
                duplicates="drop"
            )
            print(f"  ✓ Visual condition stratification enabled (3 bins)")
        except ValueError:
            # Fallback: use fixed bins if qcut fails
            df["_condition_bin"] = pd.cut(
                df["condition_score"],
                bins=[0, 20, 35, 100],
                labels=["good_cond", "medium_cond", "poor_cond"],
                include_lowest=True
            )
            print(f"  ✓ Visual condition stratification enabled (3 fixed bins)")
    else:
        df["_condition_bin"] = "unknown_cond"
        print(f"  ⚠️  No condition_score found - using text-only stratification")

    # 2. Text difficulty bins (Easy/Medium/Hard)
    try:
        df["_text_diff_bin"] = pd.qcut(df["difficulty_score"], q=3, labels=["easy", "medium", "hard"], duplicates="drop")
    except ValueError:
        df["_text_diff_bin"] = "medium"

    # 3. Has digits (binary: yes/no)
    df["_has_digit"] = df["Target"].str.contains(r"\d", regex=True, na=False)
    df["_digit_bin"] = df["_has_digit"].map({True: "has_nums", False: "no_nums"})

    # 4. Has uppercase (binary: yes/no - indicates names)
    df["_has_upper"] = df["Target"].str.contains(r"[A-Z]", regex=True, na=False)
    df["_upper_bin"] = df["_has_upper"].map({True: "has_names", False: "no_names"})

    # Combine: condition (3) × text_difficulty (3) × digits (2) × names (2) = 36 bins
    # Ensures train and val have balanced distributions across:
    # - Visual degradation (good/medium/poor condition documents)
    # - Text complexity (easy/medium/hard difficulty)
    # - Numbers (harder to transcribe, not in language prior)
    # - Names (not in language prior, require visual fidelity)
    df["_bin"] = (df["_condition_bin"].astype(str) + "_" +
                  df["_text_diff_bin"].astype(str) + "_" +
                  df["_digit_bin"].astype(str) + "_" +
                  df["_upper_bin"].astype(str))

    # Keep text_len for logging
    df["_text_len"] = df["Target"].str.len()

    k_folds = data_cfg.get("k_folds", 1)
    seed = data_cfg.get("seed", 42)
    group_col = data_cfg.get("group_col")

    # Store original index for duplicate-aware splitting
    df["_orig_idx"] = df.index

    helper = ["_text_clean", "_dup_group", "_orig_idx", "_digit_density", "_uppercase_ratio", "_lexical_diversity",
              "_special_char_density", "_avg_word_length", "_has_digit", "_has_upper", "_text_len",
              "_condition_bin", "_text_diff_bin", "_digit_bin", "_upper_bin", "_bin",
              "difficulty_score", "named_entity_score", "number_complexity"]

    if group_col and group_col not in df.columns:
        raise ValueError(f"group_col '{group_col}' not in the CSV columns")

    if k_folds > 1:
        # K-fold with duplicate awareness
        has_duplicates = (df["_dup_group"] >= 0).any()

        if has_duplicates or group_col:
            # Use StratifiedGroupKFold to keep duplicates together
            if not GROUP_KFOLD_AVAILABLE:
                raise RuntimeError("K-fold with duplicates needs scikit-learn >= 1.0 for StratifiedGroupKFold")

            # Create synthetic group column combining duplicates + user group_col
            if has_duplicates and group_col:
                # Combine both: duplicate group + user-specified group
                df["_fold_group"] = df["_dup_group"].astype(str) + "_" + df[group_col].astype(str)
                print(f"K-fold with duplicate awareness + grouped on '{group_col}'")
            elif has_duplicates:
                # Use duplicate group as fold group
                # Assign unique group ID to non-duplicates
                max_dup_group = df["_dup_group"].max()
                df["_fold_group"] = df.apply(
                    lambda row: row["_dup_group"] if row["_dup_group"] >= 0 else max_dup_group + 1 + row.name,
                    axis=1
                )
                print(f"K-fold with duplicate awareness (keeps {(df['_dup_group'] >= 0).sum()} duplicate samples together)")
            else:
                # Only user-specified group
                df["_fold_group"] = df[group_col]
                print(f"K-fold grouped on '{group_col}' ({df[group_col].nunique()} groups)")

            helper.append("_fold_group")

            splitter = StratifiedGroupKFold(n_splits=k_folds, shuffle=True, random_state=seed)
            split_iter = splitter.split(df, df["_bin"], groups=df["_fold_group"])
        else:
            # Standard k-fold (no duplicates, no grouping)
            strat_desc = "condition (3) × difficulty (3) × digits (2) × names (2) = 36 bins" if has_condition else "difficulty (3) × digits (2) × names (2) = 12 bins"
            print(f"Stratified {k_folds}-fold: {strat_desc}")
            splitter = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
            split_iter = splitter.split(df, df["_bin"])

        for fold_num, (tr, va) in enumerate(split_iter, 1):
            train_df = df.iloc[tr].copy()
            val_df = df.iloc[va].copy()

            # Verify no duplicate leakage across folds
            if has_duplicates:
                train_texts = set(train_df["_text_clean"])
                val_texts = set(val_df["_text_clean"])
                leaked = train_texts & val_texts
                if leaked:
                    print(f"  ⚠️  Fold {fold_num}: {len(leaked)} duplicate texts leaked (should be 0)!")
                else:
                    print(f"  ✓ Fold {fold_num}: No duplicate leakage")

            yield (train_df.drop(columns=helper).copy(),
                   val_df.drop(columns=helper).copy(),
                   fold_num)
    else:
        # STEP 3: Split with duplicate AND document cluster awareness
        has_duplicates = (df["_dup_group"] >= 0).any()
        has_groups = group_col is not None

        if has_duplicates or has_groups:
            # Create unified group column combining duplicates + user group_col
            if has_duplicates and has_groups:
                # Combine both: duplicate group + document cluster
                df["_split_group"] = df["_dup_group"].astype(str) + "_" + df[group_col].astype(str)
                print(f"Splitting with duplicate awareness + document clustering ('{group_col}')...")
            elif has_duplicates:
                # Only duplicates - assign unique group ID to non-duplicates
                max_dup_group = df["_dup_group"].max()
                df["_split_group"] = df.apply(
                    lambda row: row["_dup_group"] if row["_dup_group"] >= 0 else max_dup_group + 1 + row.name,
                    axis=1
                )
                print("Splitting with duplicate-group awareness (keeps duplicate texts together)...")
            else:
                # Only user group_col (document clusters)
                df["_split_group"] = df[group_col]
                print(f"Splitting with document clustering ('{group_col}': {df[group_col].nunique()} clusters)...")

            helper.append("_split_group")

            # Strategy: For each group (duplicate/cluster), assign all members to train or val together
            # 1. Get one representative per group
            # 2. Split representatives with stratification
            # 3. Propagate split assignment to all group members

            # Get one representative per group (use first occurrence)
            group_representatives = df.groupby("_split_group", as_index=False).first()

            # Split representatives with stratification
            # Try full stratification, fall back if bins too small
            try:
                split_train, split_val = train_test_split(
                    group_representatives,
                    test_size=data_cfg["val_split"],
                    stratify=group_representatives["_bin"],
                    random_state=seed
                )
                print(f"  ✓ Using full stratification (12 bins)")
            except ValueError:
                # Some bins have <2 samples after grouping - fall back to simpler stratification
                print(f"  ⚠️  Full stratification failed (some bins too small after grouping)")

                # Try medium fallback: condition × digits × names (12 bins)
                if has_condition:
                    try:
                        group_representatives["_simple_bin"] = (
                            group_representatives["_condition_bin"].astype(str) + "_" +
                            group_representatives["_has_digit"].astype(str) + "_" +
                            group_representatives["_has_upper"].astype(str)
                        )
                        split_train, split_val = train_test_split(
                            group_representatives,
                            test_size=data_cfg["val_split"],
                            stratify=group_representatives["_simple_bin"],
                            random_state=seed
                        )
                        print(f"  ✓ Using medium stratification: condition (3) × digits (2) × names (2) = 12 bins")
                    except ValueError:
                        # Still too small, fall back to minimal
                        group_representatives["_simple_bin"] = (
                            group_representatives["_has_digit"].astype(str) + "_" +
                            group_representatives["_has_upper"].astype(str)
                        )
                        split_train, split_val = train_test_split(
                            group_representatives,
                            test_size=data_cfg["val_split"],
                            stratify=group_representatives["_simple_bin"],
                            random_state=seed
                        )
                        print(f"  ✓ Using minimal stratification: digits (2) × names (2) = 4 bins")
                else:
                    # No condition score, simplify to just has_digit × has_upper (4 bins)
                    group_representatives["_simple_bin"] = (
                        group_representatives["_has_digit"].astype(str) + "_" +
                        group_representatives["_has_upper"].astype(str)
                    )
                    split_train, split_val = train_test_split(
                        group_representatives,
                        test_size=data_cfg["val_split"],
                        stratify=group_representatives["_simple_bin"],
                        random_state=seed
                    )
                    print(f"  ✓ Using simplified stratification: digits (2) × names (2) = 4 bins")

            # Now propagate: which groups went to train vs val?
            train_groups = set(split_train["_split_group"])
            val_groups = set(split_val["_split_group"])

            # Build train/val by group assignment
            train_mask = df["_split_group"].isin(train_groups)
            val_mask = df["_split_group"].isin(val_groups)

            train_df = df[train_mask].copy()
            val_df = df[val_mask].copy()

            # Verify no leakage (for duplicates)
            if has_duplicates:
                train_texts = set(train_df["_text_clean"])
                val_texts = set(val_df["_text_clean"])
                leaked = train_texts & val_texts
                if leaked:
                    print(f"  ⚠️  WARNING: {len(leaked)} texts still leaked (should be 0)!")
                else:
                    print(f"  ✓ No duplicate text leakage - all copies kept together")

            # Verify group separation (for document clusters)
            if has_groups:
                train_clusters = set(train_df[group_col])
                val_clusters = set(val_df[group_col])
                leaked_clusters = train_clusters & val_clusters
                if leaked_clusters:
                    print(f"  ⚠️  WARNING: {len(leaked_clusters)} clusters leaked across train/val!")
                else:
                    print(f"  ✓ No cluster leakage - {len(train_clusters)} clusters in train, {len(val_clusters)} in val")

        else:
            # No duplicates or groups: standard stratified split
            train_df, val_df = train_test_split(
                df, test_size=data_cfg["val_split"], stratify=df["_bin"], random_state=seed
            )

        # Log distributions to verify stratification is working
        train_digits = train_df["_digit_density"]
        val_digits = val_df["_digit_density"]
        train_upper = train_df["_uppercase_ratio"]
        val_upper = val_df["_uppercase_ratio"]
        train_lex = train_df["_lexical_diversity"]
        val_lex = val_df["_lexical_diversity"]
        train_spec = train_df["_special_char_density"]
        val_spec = val_df["_special_char_density"]
        train_wlen = train_df["_avg_word_length"]
        val_wlen = val_df["_avg_word_length"]

        # Get difficulty scores for logging
        train_diff = train_df.get("difficulty_score", train_df["_lexical_diversity"])
        val_diff = val_df.get("difficulty_score", val_df["_lexical_diversity"])

        # Get condition scores for logging (if available)
        has_condition_logging = "condition_score" in train_df.columns and not train_df["condition_score"].isna().all()

        strat_desc = "condition (3) × difficulty (3) × digits (2) × names (2) = 36 bins" if has_condition else "difficulty (3) × digits (2) × names (2) = 12 bins"
        print(f"Stratified split: {strat_desc}")
        print(f"  Train: {len(train_df)} samples")

        # Visual condition distribution
        if has_condition_logging:
            train_cond = train_df["condition_score"]
            print(f"    Visual condition: min={train_cond.min():.1f}, median={train_cond.median():.1f}, max={train_cond.max():.1f}")

        print(f"    Text difficulty: min={train_diff.min():.1f}, median={train_diff.median():.1f}, max={train_diff.max():.1f}")
        print(f"    Digit density: min={train_digits.min():.3f}, median={train_digits.median():.3f}, max={train_digits.max():.3f}")
        print(f"    Uppercase ratio: min={train_upper.min():.3f}, median={train_upper.median():.3f}, max={train_upper.max():.3f}")
        print(f"    Lexical diversity: min={train_lex.min():.3f}, median={train_lex.median():.3f}, max={train_lex.max():.3f}")
        print(f"    Text length: min={train_df['_text_len'].min()}, median={train_df['_text_len'].median():.0f}, max={train_df['_text_len'].max()}")
        print(f"  Val:   {len(val_df)} samples")

        # Visual condition distribution
        if has_condition_logging:
            val_cond = val_df["condition_score"]
            print(f"    Visual condition: min={val_cond.min():.1f}, median={val_cond.median():.1f}, max={val_cond.max():.1f}")

        print(f"    Text difficulty: min={val_diff.min():.1f}, median={val_diff.median():.1f}, max={val_diff.max():.1f}")
        print(f"    Digit density: min={val_digits.min():.3f}, median={val_digits.median():.3f}, max={val_digits.max():.3f}")
        print(f"    Uppercase ratio: min={val_upper.min():.3f}, median={val_upper.median():.3f}, max={val_upper.max():.3f}")
        print(f"    Lexical diversity: min={val_lex.min():.3f}, median={val_lex.median():.3f}, max={val_lex.max():.3f}")
        print(f"    Text length: min={val_df['_text_len'].min()}, median={val_df['_text_len'].median():.0f}, max={val_df['_text_len'].max()}")

        yield train_df.drop(columns=helper), val_df.drop(columns=helper), None


def train(cfg: dict):
    data_cfg, train_cfg = cfg["data"], cfg["training"]

    train_csv = REPO_ROOT / data_cfg["train_csv"]
    image_dir = REPO_ROOT / data_cfg["image_dir"]
    base_output_dir = REPO_ROOT / train_cfg["output_dir"]

    df = pd.read_csv(train_csv)
    nan_count = df['Target'].isna().sum()
    print(f"Loaded {len(df)} samples from {train_csv.name}" + (f" ({nan_count} NaN targets)" if nan_count > 0 else ""))

    # Load document clusters if provided
    cluster_csv = data_cfg.get("cluster_csv")
    if cluster_csv:
        cluster_path = REPO_ROOT / cluster_csv
        if cluster_path.exists():
            cluster_df = pd.read_csv(cluster_path)
            df = df.merge(cluster_df, on="ID", how="left")
            group_col = data_cfg.get("group_col")
            if group_col and group_col in df.columns:
                print(f"Loaded document clusters from {cluster_path.name}")
                print(f"  {df[group_col].nunique()} unique clusters, avg {len(df)/df[group_col].nunique():.1f} samples/cluster")
            else:
                print(f"⚠️  Warning: cluster_csv provided but group_col '{group_col}' not found in merged data")
        else:
            print(f"⚠️  Warning: cluster_csv '{cluster_csv}' not found, proceeding without clustering")

    # Load document condition scores for adaptive augmentation (if available)
    condition_csv = REPO_ROOT / "dataset" / "document_condition.csv"
    if condition_csv.exists():
        condition_df = pd.read_csv(condition_csv)
        # Filter to successful analyses only
        condition_df = condition_df[condition_df["success"] == True]
        # Merge condition scores
        df = df.merge(condition_df[["ID", "condition_score"]], on="ID", how="left")
        # Fill missing with median (for any images that failed analysis)
        median_cond = df["condition_score"].median()
        n_missing = df["condition_score"].isna().sum()
        if n_missing > 0:
            df.loc[df["condition_score"].isna(), "condition_score"] = median_cond
        print(f"Loaded document condition scores from {condition_csv.name}")
        print(f"  Mean: {df['condition_score'].mean():.1f}, Median: {median_cond:.1f}, Std: {df['condition_score'].std():.1f}")
        if n_missing > 0:
            print(f"  Filled {n_missing} missing values with median")
    else:
        print(f"ℹ️  Document condition scores not found ({condition_csv.name}), proceeding without adaptive augmentation")

    k_folds = data_cfg.get("k_folds", 1)
    n_folds = k_folds if k_folds > 1 else None
    results = []

    for train_df, val_df, fold_num in make_splits(df, data_cfg):
        out_dir = base_output_dir / f"fold_{fold_num}" if fold_num else base_output_dir
        results.append(train_single_fold(
            train_df, val_df, image_dir, out_dir, cfg, fold_num, n_folds
        ))

    if len(results) > 1:
        print(f"\n{'=' * 70}")
        print(f"K-FOLD SUMMARY ({n_folds} folds)")
        print(f"{'=' * 70}")
        for r in results:
            stop_tag = f" [stopped @ {r['stopped_epoch']:.1f}]" if r.get("early_stopped") else ""
            score_str = f" | score={r['best_eval_score']:.4f}" if r.get("best_eval_score") is not None else ""
            cer_wer = ""
            if r.get("best_eval_cer") is not None and r.get("best_eval_wer") is not None:
                cer_wer = f" (cer={r['best_eval_cer']:.4f}, wer={r['best_eval_wer']:.4f})"
            print(f"  Fold {r['fold']}: loss={r['best_eval_loss']:.4f}{score_str}{cer_wer}{stop_tag}")

        losses = [r["best_eval_loss"] for r in results]
        scores = [r["best_eval_score"] for r in results if r.get("best_eval_score") is not None]
        cers = [r["best_eval_cer"] for r in results if r.get("best_eval_cer") is not None]
        wers = [r["best_eval_wer"] for r in results if r.get("best_eval_wer") is not None]

        print(f"\nAverage: loss={np.mean(losses):.4f}±{np.std(losses):.4f}", end="")
        if scores:
            print(f" | score={np.mean(scores):.4f}±{np.std(scores):.4f}", end="")
        if cers and wers:
            print(f" (cer={np.mean(cers):.4f}, wer={np.mean(wers):.4f})")
        else:
            print()

        print(f"{'=' * 70}\n")

    base_output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = base_output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({"config": cfg, "results": results}, f, indent=2, default=str)
    print(f"Summary -> {summary_path}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--folds", type=int, default=None,
                        help="Override data.k_folds (use 1 for a quick single-split run)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training.epochs")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start training from scratch, ignoring existing checkpoints")
    parser.add_argument("--no-early-stop", action="store_true",
                        help="Disable early stopping (train full epochs regardless of plateau)")
    args = parser.parse_args()

    config_path = SCRIPT_DIR / args.config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if args.folds is not None:
        cfg["data"]["k_folds"] = args.folds
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.no_resume:
        cfg["training"]["resume_from_checkpoint"] = False
    if args.no_early_stop:
        if "early_stopping" not in cfg["training"]:
            cfg["training"]["early_stopping"] = {}
        cfg["training"]["early_stopping"]["enabled"] = False

    seed = cfg["data"].get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    train(cfg)


if __name__ == "__main__":
    main()