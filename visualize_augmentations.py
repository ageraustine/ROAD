"""
Visualize augmentation effects for HTR training (train.py's ImageAugmenter).

Deliberately does NOT import train.py / torch / transformers / peft - it
vendors ImageAugmenter and load_image verbatim (same source, see NOTE below)
so you can iterate on augmentation config quickly on a laptop, without
touching the model stack. If you change ImageAugmenter in train.py, copy the
class body back into this file to keep them in sync.

Two views, side by side, saved as one PNG:

  1. ISOLATED  - each augmentation type forced to fire alone (p=1.0, all
     others 0), a few draws each, so you can see what each transform does
     in isolation before they're all mixed together.
  2. COMBINED  - N draws using a real augmentation config end-to-end
     (default probabilities, everything mixed), i.e. what a training batch
     actually sees. If document_condition.csv + an ID are given, this also
     runs the field-specific adaptive gating (see ImageAugmenter.__call__).

Usage:
    # single image, default (published) augmentation settings
    python visualize_augmentations.py --image /path/to/img.jpg

    # by ID, pulling image_dir/image_ext and augmentation: block from a
    # real config, plus condition metrics for that ID if available
    python visualize_augmentations.py \\
        --id abc123XYZ \\
        --config config_qwen3_8b_full.yaml \\
        --condition-csv dataset/document_condition.csv

    # no image available - generates a synthetic aged-document placeholder
    # so you can sanity-check the script itself
    python visualize_augmentations.py

    # only the isolated-effects panel, skip the combined-draws panel
    python visualize_augmentations.py --image img.jpg --skip-combined

Output: PNG at --output (default augmentation_preview.png).
"""

import argparse
import math
import random
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import yaml
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageEnhance
from scipy.ndimage import gaussian_filter


# ─────────────────────────────────────────────────────────────
# VENDORED FROM train.py - ImageAugmenter + load_image
# (kept byte-for-byte identical to train.py's implementation so what you
# preview here matches what training actually does; re-sync if train.py's
# ImageAugmenter changes)
# ─────────────────────────────────────────────────────────────

class ImageAugmenter:
    """
    Conservative augmentation for already-degraded historical documents.
    Training only - never applied to validation.
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

        # Document-condition augmentations (based on analysis)
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

        self.p_local_degradation = cfg.get("p_local_degradation", 0.0)
        self.local_width_ratio = cfg.get("local_width_ratio", (0.12, 0.28))
        self.local_height_ratio = cfg.get("local_height_ratio", (0.85, 1.0))
        self.local_max_regions = cfg.get("local_max_regions", 1)
        self.local_noise_std = cfg.get("local_noise_std", 20)
        self.local_shadow_strength = cfg.get("local_shadow_strength", 0.35)
        self.local_soft_edge_px = cfg.get("local_soft_edge_px", 8)

    def _sample_background_color(self, img: Image.Image) -> tuple:
        arr = np.array(img)
        h, w = arr.shape[:2]
        border_size = max(1, int(min(h, w) * 0.05))
        top = arr[:border_size, :].reshape(-1, 3)
        bottom = arr[-border_size:, :].reshape(-1, 3)
        left = arr[:, :border_size].reshape(-1, 3)
        right = arr[:, -border_size:].reshape(-1, 3)
        edge_pixels = np.vstack([top, bottom, left, right])
        mean_color = edge_pixels.mean(axis=0).astype(int)
        return tuple(mean_color)

    def __call__(self, img: Image.Image, condition_metrics: dict = None) -> Image.Image:
        if not self.enabled:
            return img

        if condition_metrics is not None:
            text_contrast = condition_metrics.get("text_contrast", 0)
            tears = condition_metrics.get("tears_and_holes", 0)
            stains = condition_metrics.get("stains", 0)
            paper_var = condition_metrics.get("paper_color_variance", 0)
            texture = condition_metrics.get("texture_degradation", 0)

            if text_contrast > 44:
                p_brightness_mult = 0.0
                p_contrast_mult = 0.0
                p_morphology_mult = 0.0
            else:
                p_brightness_mult = 1.0
                p_contrast_mult = 1.0
                p_morphology_mult = 1.0

            if tears > 7:
                p_elastic_mult = 1.5
                p_rotation_mult = 1.5
                p_resolution_mult = 1.5
                min_pixels_override = 0.65
                elastic_alpha_override = self.elastic_alpha * 1.4
                p_local_degradation_mult = 0.0
            else:
                p_elastic_mult = 1.0
                p_rotation_mult = 1.0
                p_resolution_mult = 1.0
                min_pixels_override = self.min_pixels_ratio
                elastic_alpha_override = self.elastic_alpha
                p_local_degradation_mult = 1.0

            if stains > 28 or paper_var > 19:
                p_color_mult = 0.0
            else:
                p_color_mult = 1.0

            hue_sat_mult = 1.0

            if texture > 18:
                p_degradation_mult = 0.0
            else:
                p_degradation_mult = 1.0

        else:
            p_degradation_mult = 1.0
            p_elastic_mult = 1.0
            p_resolution_mult = 1.0
            p_rotation_mult = 1.0
            min_pixels_override = self.min_pixels_ratio
            p_color_mult = 1.0
            p_brightness_mult = 1.0
            p_contrast_mult = 1.0
            p_morphology_mult = 1.0
            p_local_degradation_mult = 1.0
            hue_sat_mult = 1.0
            elastic_alpha_override = self.elastic_alpha

        if random.random() < (self.p_blur * p_degradation_mult):
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

        if random.random() < (self.p_noise * p_degradation_mult):
            arr = np.array(img).astype(np.float32)
            arr += np.random.normal(0, random.uniform(5, 15), arr.shape)
            img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

        if random.random() < (self.p_brightness * p_brightness_mult):
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))

        if random.random() < (self.p_contrast * p_contrast_mult):
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))

        if random.random() < (self.p_rotate * p_rotation_mult):
            angle = random.uniform(-self.max_rotation, self.max_rotation)
            bg_color = self._sample_background_color(img)
            img = img.rotate(angle, fillcolor=bg_color, expand=True, resample=Image.BICUBIC)

        if random.random() < (self.p_morphology * p_morphology_mult):
            arr = np.array(img)
            h, w = arr.shape[:2]
            kernel_size = max(3, int(h * 0.005))  # floored at 3, not 1 - see train.py fix notes
            kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            if random.random() < 0.5:
                arr = cv2.dilate(arr, kernel, iterations=1)
            else:
                arr = cv2.erode(arr, kernel, iterations=1)
            blur_kernel = max(3, kernel_size // 2)
            blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
            arr = cv2.GaussianBlur(arr, (blur_kernel, blur_kernel), 0.5)
            img = Image.fromarray(arr)

        if random.random() < self.p_shear:
            angle_deg = random.uniform(-self.max_shear, self.max_shear)
            angle_rad = np.deg2rad(angle_deg)
            w, h = img.size
            bg_color = self._sample_background_color(img)
            shear_factor = np.tan(angle_rad)
            offset = abs(shear_factor * h)
            new_w = int(w + offset)
            img = img.transform(
                (new_w, h),
                Image.AFFINE,
                (1, shear_factor, -shear_factor * h / 2, 0, 1, 0),
                fillcolor=bg_color,
                resample=Image.BICUBIC
            )

        if random.random() < (self.p_resolution * p_degradation_mult):
            w, h = img.size
            scale = random.uniform(0.6, 1.0)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.BILINEAR)
            img = img.resize((w, h), Image.BILINEAR)

        if random.random() < (self.p_jpeg * p_degradation_mult):
            buffer = BytesIO()
            quality = random.randint(60, 90)
            img.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            img = Image.open(buffer).convert('RGB')

        if random.random() < (self.p_elastic * p_elastic_mult):
            img = self._apply_elastic_transform(img, elastic_alpha_override)

        if random.random() < (self.p_color_jitter * p_color_mult):
            img = self._apply_color_jitter(img, hue_sat_mult)

        if random.random() < (self.p_resolution_jitter * p_resolution_mult):
            img = self._apply_resolution_jitter(img, min_pixels_override)

        if random.random() < (self.p_local_degradation * p_local_degradation_mult):
            img = self._apply_local_degradation(img)

        return img

    def _apply_elastic_transform(self, img: Image.Image, alpha: float = None) -> Image.Image:
        arr = np.array(img)
        h, w = arr.shape[:2]
        alpha_value = alpha if alpha is not None else self.elastic_alpha
        dx = np.random.randn(h, w) * alpha_value
        dy = np.random.randn(h, w) * alpha_value
        dx = gaussian_filter(dx, self.elastic_sigma, mode='constant', cval=0)
        dy = gaussian_filter(dy, self.elastic_sigma, mode='constant', cval=0)
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        indices = (y + dy).astype(np.float32), (x + dx).astype(np.float32)
        warped = cv2.remap(arr, indices[1], indices[0],
                          interpolation=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)
        return Image.fromarray(warped)

    def _apply_color_jitter(self, img: Image.Image, hue_sat_mult: float = 1.0) -> Image.Image:
        arr = np.array(img)
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).astype(np.float32)
        hue_range = self.hue_jitter * hue_sat_mult
        hue_shift = random.uniform(-hue_range, hue_range) * 180
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180  # circular, wrap not clip
        sat_range = self.saturation_jitter * hue_sat_mult
        sat_factor = random.uniform(1 - sat_range, 1 + sat_range)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_factor, 0, 255)
        rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return Image.fromarray(rgb)

    def _apply_resolution_jitter(self, img: Image.Image, min_pixels_ratio: float = None) -> Image.Image:
        w, h = img.size
        current_pixels = w * h
        min_ratio = min_pixels_ratio if min_pixels_ratio is not None else self.min_pixels_ratio
        ratio = random.uniform(min_ratio, self.max_pixels_ratio)
        target_pixels = int(current_pixels * ratio)
        scale = (target_pixels / current_pixels) ** 0.5
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        return img

    def _apply_local_degradation(self, img: Image.Image) -> Image.Image:
        arr = np.array(img).astype(np.float32)
        h, w = arr.shape[:2]

        n_regions = random.randint(1, max(1, self.local_max_regions))
        mode_choices = ["noise", "shadow"]
        width_budget = int(w * 0.4)  # cumulative cap across all regions

        for _ in range(n_regions):
            if width_budget <= 4:
                break
            patch_w = int(w * random.uniform(*self.local_width_ratio))
            patch_w = max(4, min(patch_w, int(w * 0.4), width_budget))
            width_budget -= patch_w
            patch_h = int(h * random.uniform(*self.local_height_ratio))
            patch_h = max(2, min(patch_h, h))

            x0 = random.randint(0, max(0, w - patch_w))
            y0 = random.randint(0, max(0, h - patch_h))

            mask = np.zeros((h, w), dtype=np.float32)
            mask[y0:y0 + patch_h, x0:x0 + patch_w] = 1.0
            if self.local_soft_edge_px > 0:
                mask = gaussian_filter(mask, self.local_soft_edge_px, mode="constant", cval=0)
            mask = mask[:, :, None]

            mode = random.choice(mode_choices)
            if mode == "noise":
                noise = np.random.normal(0, self.local_noise_std, arr.shape)
                arr = arr + noise * mask
            else:
                darken = 1.0 - self.local_shadow_strength * mask
                arr = arr * darken

        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)


def load_image(path: str, max_pixels: int = 2016000) -> Image.Image:
    """Load and downscale (LANCZOS) while preserving aspect ratio."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if w * h > max_pixels:
        scale = (max_pixels / (w * h)) ** 0.5
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def apply_shear_at(img: Image.Image, angle_deg: float) -> Image.Image:
    """
    Same shear transform as ImageAugmenter, but at an exact angle instead of
    a random draw - for directly comparing e.g. +8deg vs -8deg to check the
    canvas-expansion asymmetry (the translation offset is only compensated
    for shear_factor > 0 in the original code, so negative angles may clip
    instead of expanding cleanly).
    """
    angle_rad = np.deg2rad(angle_deg)
    w, h = img.size
    augmenter = ImageAugmenter({"enabled": True})  # only used for _sample_background_color
    bg_color = augmenter._sample_background_color(img)
    shear_factor = np.tan(angle_rad)
    offset = abs(shear_factor * h)
    new_w = int(w + offset)
    return img.transform(
        (new_w, h),
        Image.AFFINE,
        (1, shear_factor, -shear_factor * h / 2, 0, 1, 0),
        fillcolor=bg_color,
        resample=Image.BICUBIC,
    )


def diff_stats(original: Image.Image, augmented: Image.Image) -> str:
    """
    Quantify how much an augmented draw actually differs from the original,
    so subtle transforms (morphology, mild color jitter) can be confirmed to
    have fired even when the visual difference is too small to see at plot
    scale. Resizes to match if the canvas grew (rotate/shear expand).
    """
    a = np.array(original).astype(np.int16)
    b_img = augmented if augmented.size == original.size else augmented.resize(original.size, Image.BILINEAR)
    b = np.array(b_img).astype(np.int16)
    diff = np.abs(a - b)
    mean_abs = diff.mean()
    pct_changed = (diff.max(axis=-1) > 2).mean() * 100  # >2 ignores JPEG/round-trip noise
    return f"\u0394mean={mean_abs:.2f} \u0394px={pct_changed:.1f}%"


# ─────────────────────────────────────────────────────────────
# ISOLATED-EFFECT CONFIGS - one augmentation type forced on, rest off
# ─────────────────────────────────────────────────────────────

# Every probability explicitly zeroed - ImageAugmenter has non-zero built-in
# defaults for several (p_brightness=0.3, p_contrast=0.3, p_rotate=0.1), so
# without this baseline, "isolated" draws could have those sneaking in
# unnoticed on top of whatever type was actually being tested.
ZERO_PROB_CFG = {
    "p_blur": 0.0, "p_noise": 0.0, "p_brightness": 0.0, "p_contrast": 0.0,
    "p_rotate": 0.0, "p_morphology": 0.0, "p_shear": 0.0, "p_resolution": 0.0,
    "p_jpeg": 0.0, "p_elastic": 0.0, "p_color_jitter": 0.0,
    "p_resolution_jitter": 0.0, "p_local_degradation": 0.0,
}

ISOLATED_TYPES = {
    "blur":              {"p_blur": 1.0},
    "noise":             {"p_noise": 1.0},
    "brightness":        {"p_brightness": 1.0},
    "contrast":          {"p_contrast": 1.0},
    "rotate":            {"p_rotate": 1.0, "max_rotation": 3},
    "morphology":        {"p_morphology": 1.0},
    "shear":             {"p_shear": 1.0, "max_shear": 8},
    "resolution (old)":  {"p_resolution": 1.0},
    "jpeg":              {"p_jpeg": 1.0},
    "elastic":           {"p_elastic": 1.0, "elastic_alpha": 25, "elastic_sigma": 6},
    "color_jitter":      {"p_color_jitter": 1.0, "hue_jitter": 0.05, "saturation_jitter": 0.1},
    "resolution_jitter": {"p_resolution_jitter": 1.0, "min_pixels_ratio": 0.6, "max_pixels_ratio": 0.9},
    "local_degradation": {"p_local_degradation": 1.0, "local_max_regions": 1},
}

DEFAULT_COMBINED_CFG = {
    "enabled": True,
    "p_blur": 0.0,
    "p_noise": 0.0,
    "p_brightness": 0.3,
    "p_contrast": 0.3,
    "p_rotate": 0.1,
    "max_rotation": 1,
    "p_morphology": 0.0,
    "p_shear": 0.0,
    "max_shear": 8,
    "p_resolution": 0.0,
    "p_jpeg": 0.0,
    "p_elastic": 0.0,
    "elastic_alpha": 25,
    "elastic_sigma": 6,
    "p_color_jitter": 0.0,
    "hue_jitter": 0.05,
    "saturation_jitter": 0.1,
    "p_resolution_jitter": 0.0,
    "min_pixels_ratio": 0.7,
    "max_pixels_ratio": 1.0,
}


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def make_synthetic_document(w: int = 700, h: int = 500, seed: int = 0) -> Image.Image:
    """
    Aged-parchment-ish placeholder with fake handwriting-like scribbles, used
    only when no real image is given, so you can sanity-check this script
    without any manuscript images on hand.
    """
    rng = np.random.RandomState(seed)
    base = rng.uniform(205, 225)
    arr = np.full((h, w, 3), (base, base * 0.92, base * 0.75), dtype=np.uint8)

    # mottled paper texture
    noise = rng.normal(0, 6, (h, w, 1)).astype(np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    img = Image.fromarray(arr)
    pil_draw_img = img.copy()
    arr2 = np.array(pil_draw_img)

    # fake cursive "lines" of handwriting: wavy dark strokes
    for line_y in range(60, h - 40, 55):
        x = 40
        while x < w - 60:
            seg_len = rng.randint(15, 45)
            dy = rng.randint(-8, 8)
            cv2.line(arr2, (x, line_y + dy), (x + seg_len, line_y + dy + rng.randint(-5, 5)),
                      color=(60, 45, 35), thickness=2, lineType=cv2.LINE_AA)
            x += seg_len + rng.randint(4, 10)

    return Image.fromarray(arr2)


def find_config(config_arg: str) -> Path:
    """
    Resolve a config filename/path to an actual file, trying (in order):
      1. exactly what was given (as an absolute path, or relative to cwd)
      2. <cwd>/src/qwen2vl/configs/<name>
      3. <this script's dir>/src/qwen2vl/configs/<name>
      4. <cwd>/configs/<name>  and  <this script's dir>/configs/<name>
    This lets you pass just a filename ("config.yaml") regardless of whether
    you run this script from the repo root, from src/qwen2vl/, or anywhere
    else - matching how train.py/analyze_document_condition.py only ever
    take a bare filename for --config.
    """
    given = Path(config_arg)
    name = given.name
    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd()

    candidates = [given]
    for base in {cwd, script_dir}:
        candidates.append(base / "src" / "qwen2vl" / "configs" / name)
        candidates.append(base / "configs" / name)

    for c in candidates:
        if c.exists():
            return c.resolve()

    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Could not find config '{config_arg}'. Tried:\n  {tried}\n"
        "Pass a full/relative path with --config, or run from a directory "
        "where src/qwen2vl/configs/ is reachable."
    )


def resolve_image_and_metrics(args):
    """Return (PIL.Image, condition_metrics_dict_or_None, label_str)."""
    if args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            raise FileNotFoundError(f"--image path does not exist: {img_path}")
        return load_image(str(img_path)), None, img_path.name

    if args.id:
        if not args.config:
            raise ValueError("--id requires --config to resolve image_dir/image_ext")
        config_path = find_config(args.config)
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        data_cfg = cfg["data"]
        # config_path is .../repo_root/src/qwen2vl/configs/<file>.yaml, i.e.
        # repo_root/src/qwen2vl/configs/config.yaml -> 4 parents to repo_root
        # (configs -> qwen2vl -> src -> repo_root). dataset/ is a sibling of
        # src/, matching analyze_document_condition.py's REPO_ROOT.
        repo_root = config_path.parent.parent.parent.parent
        image_dir = repo_root / data_cfg["image_dir"]
        image_ext = data_cfg.get("image_ext", ".jpg")
        img_path = image_dir / f"{args.id}{image_ext}"
        if not img_path.exists():
            raise FileNotFoundError(
                f"Resolved image path does not exist: {img_path}\n"
                f"  (repo_root={repo_root}, from config={config_path})"
            )

        condition_metrics = None
        if args.condition_csv:
            import pandas as pd
            cond_df = pd.read_csv(args.condition_csv)
            row = cond_df[cond_df["ID"] == args.id]
            if len(row):
                r = row.iloc[0]
                condition_metrics = {
                    k: float(r[k]) for k in
                    ["text_contrast", "tears_and_holes", "stains",
                     "paper_color_variance", "texture_degradation"]
                    if k in r and not pd.isna(r[k])
                }
                print(f"  Loaded condition metrics for {args.id}: {condition_metrics}")
            else:
                print(f"  ⚠️  {args.id} not found in {args.condition_csv} - "
                      "running without condition-aware gating")

        return load_image(str(img_path)), condition_metrics, args.id

    print("  ℹ️  No --image or --id given - using a synthetic placeholder image")
    return make_synthetic_document(seed=args.seed), None, "synthetic"


def load_combined_cfg(args) -> dict:
    if not args.config:
        print("  ℹ️  No --config given - using DEFAULT_COMBINED_CFG (enabled forced True) "
              "rather than train.py's ImageAugmenter default of enabled=False")
        return dict(DEFAULT_COMBINED_CFG)

    config_path = find_config(args.config)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    aug_cfg = dict(cfg.get("augmentation", {}))
    if not aug_cfg:
        print("  ⚠️  Config has no 'augmentation:' block - falling back to DEFAULT_COMBINED_CFG")
        return dict(DEFAULT_COMBINED_CFG)
    if not aug_cfg.get("enabled", False):
        print("  ⚠️  augmentation.enabled is False/unset in this config - "
              "forcing True so the combined preview actually shows something. "
              "(Isolated panel always forces enabled=True per-type regardless.)")
        aug_cfg["enabled"] = True
    return aug_cfg


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Preview ImageAugmenter effects: isolated per-type + combined random draws.")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to a source image. Takes priority over --id.")
    parser.add_argument("--id", type=str, default=None,
                        help="Document ID to look up via --config's data.image_dir/image_ext.")
    parser.add_argument("--config", type=str, default=None,
                        help="Config filename (e.g. config.yaml) or path. If just a "
                             "filename, searched under src/qwen2vl/configs/ relative "
                             "to cwd and to this script's location.")
    parser.add_argument("--condition-csv", type=str, default=None,
                        help="document_condition.csv - if given with --id, the combined "
                             "panel uses field-specific adaptive gating for that document.")
    parser.add_argument("--shear-angles", type=str, default=None,
                        help="Comma-separated exact degrees for a deterministic shear "
                             "comparison (e.g. '-8,-4,0,4,8'), saved as "
                             "isolated_shear_deterministic.png. Bypasses randomness so "
                             "you can directly compare positive vs negative angles.")
    parser.add_argument("--types", type=str, default=None,
                        help="Comma-separated subset of isolated types to render "
                             f"(default: all). Choices: {', '.join(ISOLATED_TYPES)}")
    parser.add_argument("--n-isolated", type=int, default=3,
                        help="Draws per augmentation type in the isolated panel (default: 3)")
    parser.add_argument("--n-combined", type=int, default=8,
                        help="Number of combined random draws (default: 8)")
    parser.add_argument("--combined-cols", type=int, default=4,
                        help="Columns per row in the combined grid (default: 4)")
    parser.add_argument("--skip-isolated", action="store_true")
    parser.add_argument("--skip-combined", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="aug_preview",
                        help="Directory for output PNGs - one per augmentation type "
                             "plus one for the combined view (default: aug_preview/)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("Loading source image...")
    img, condition_metrics, label = resolve_image_and_metrics(args)
    print(f"  {label}: {img.size[0]}x{img.size[1]}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    if args.shear_angles:
        angles = [float(a.strip()) for a in args.shear_angles.split(",")]
        print(f"\nRendering deterministic shear comparison at {angles} degrees...")
        samples = [(f"{a:+.0f}°", apply_shear_at(img.copy(), a)) for a in angles]
        fname = out_dir / "isolated_shear_deterministic.png"
        save_row(img, "shear (deterministic angles)", samples, fname)
        saved.append(fname)
        print(f"  {fname}")

    if not args.skip_isolated:
        types = ISOLATED_TYPES
        if args.types:
            wanted = [t.strip() for t in args.types.split(",")]
            unknown = [t for t in wanted if t not in ISOLATED_TYPES]
            if unknown:
                raise ValueError(f"Unknown --types: {unknown}. "
                                  f"Choices: {list(ISOLATED_TYPES)}")
            types = {t: ISOLATED_TYPES[t] for t in wanted}

        print(f"\nRendering {len(types)} isolated-effect file(s), "
              f"{args.n_isolated} draws each...")
        for aug_name, overrides in types.items():
            cfg = {**ZERO_PROB_CFG, "enabled": True, **overrides}
            augmenter = ImageAugmenter(cfg)
            raw_draws = [augmenter(img.copy()) for _ in range(args.n_isolated)]
            samples = [(f"draw {i+1}\n{diff_stats(img, d)}", d)
                       for i, d in enumerate(raw_draws)]
            diffs = [diff_stats(img, d) for d in raw_draws]
            print(f"  {aug_name}: " + " | ".join(diffs))
            fname = out_dir / f"isolated_{aug_name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
            save_row(img, aug_name, samples, fname)
            saved.append(fname)
            print(f"  {fname}")

    if not args.skip_combined:
        print(f"\nRendering combined draws ({args.n_combined} samples)...")
        combined_cfg = load_combined_cfg(args)
        augmenter = ImageAugmenter(combined_cfg)
        samples = [(f"draw {i+1}", augmenter(img.copy(), condition_metrics=condition_metrics))
                   for i in range(args.n_combined)]
        title = "combined (training-realistic mix)"
        if condition_metrics:
            title += " - condition-aware gating active"
        fname = out_dir / "combined.png"
        save_grid(img, title, samples, fname, cols=args.combined_cols)
        saved.append(fname)
        print(f"  {fname}")

    print(f"\nDone. {len(saved)} file(s) in {out_dir}/")


def save_row(original: Image.Image, title: str, samples: list, output_path: Path):
    """One augmentation type: original + its draws, single row."""
    n = 1 + len(samples)
    fig_w = max(4, n * 2.4)
    fig, axes = plt.subplots(1, n, figsize=(fig_w, 2.8))

    axes[0].imshow(original)
    axes[0].set_title("original", fontsize=9)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    for c, (subtitle, aug_img) in enumerate(samples, start=1):
        axes[c].imshow(aug_img)
        axes[c].set_title(subtitle, fontsize=9)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_grid(original: Image.Image, title: str, samples: list, output_path: Path, cols: int = 4):
    """Combined view: original first, then draws wrapped into a grid."""
    cells = [("original", original)] + samples
    n = len(cells)
    cols = max(1, cols)
    rows = math.ceil(n / cols)

    fig_w = max(4, cols * 2.4)
    fig_h = max(2.8, rows * 2.6)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    axes = np.atleast_2d(axes)

    for ax in axes.flat:
        ax.axis("off")

    for i, (subtitle, cell_img) in enumerate(cells):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        ax.imshow(cell_img)
        ax.set_title(subtitle, fontsize=9)
        ax.axis("on")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()