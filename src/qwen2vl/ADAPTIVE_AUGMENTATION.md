# Adaptive Augmentation

## Overview

Adaptive augmentation automatically adjusts augmentation strength per document based on its physical condition score (from `document_condition.csv`). This maximizes data diversity while preserving readability on degraded documents.

## How It Works

### Condition Tiers

Documents are classified into three tiers based on their condition score (0-100, higher = worse):

1. **Good Condition** (score < 15): ~40% of dataset
   - Pristine documents with minimal degradation
   - Can handle aggressive augmentation to synthesize degradation

2. **Medium Condition** (15-25): ~50% of dataset
   - Moderate degradation (fading, stains, etc.)
   - Uses default augmentation settings

3. **Poor Condition** (> 25): ~10% of dataset
   - Heavily degraded (burnt edges, tears, severe fading)
   - Minimal augmentation to preserve readability

### Augmentation Adjustments

#### Good Condition (< 15)
```python
p_elastic_mult = 1.3           # 30% higher probability
p_color_mult = 1.2             # 20% higher probability
min_pixels_ratio = 0.75        # Can downsample to 75% (aggressive)
elastic_alpha = alpha * 1.2    # Stronger warping
```

**Why**: Pristine documents are rare in real scans. Aggressive augmentation synthesizes realistic degradation (warping, color shifts, lower resolution).

#### Medium Condition (15-25)
```python
p_elastic_mult = 1.0           # Default probability
p_color_mult = 1.0             # Default probability
min_pixels_ratio = 0.85        # Config default (conservative)
elastic_alpha = alpha          # Default warping
```

**Why**: Balanced approach - enough augmentation for robustness without degrading already-degraded text.

#### Poor Condition (> 25)
```python
p_elastic_mult = 0.65          # 35% lower probability
p_color_mult = 0.4             # 60% lower probability
p_resolution_mult = 0.5        # 50% lower probability
min_pixels_ratio = 0.9         # Very conservative (only 10% reduction)
elastic_alpha = alpha * 0.7    # Gentler warping
```

**Why**: Heavily degraded documents are already challenging. Minimal augmentation prevents text from becoming illegible.

## Expected Impact

- **+0.4-0.7%** improvement over fixed augmentation
- Better generalization across condition spectrum
- Reduced risk of degrading readability on poor-quality docs

## Implementation

### 1. Document Condition Loading

`train.py` automatically loads condition scores:

```python
# Load document condition scores for adaptive augmentation
condition_csv = REPO_ROOT / "dataset" / "document_condition.csv"
if condition_csv.exists():
    condition_df = pd.read_csv(condition_csv)
    condition_df = condition_df[condition_df["success"] == True]
    df = df.merge(condition_df[["ID", "condition_score"]], on="ID", how="left")
    # Fill missing with median
    median_cond = df["condition_score"].median()
    df.loc[df["condition_score"].isna(), "condition_score"] = median_cond
```

### 2. Adaptive Augmentation Logic

`ImageAugmenter.__call__()` adjusts parameters based on condition score:

```python
def __call__(self, img: Image.Image, condition_score: float = None) -> Image.Image:
    if condition_score is not None:
        if condition_score < 15:  # Good
            # Aggressive augmentation
        elif condition_score < 25:  # Medium
            # Default augmentation
        else:  # Poor
            # Minimal augmentation
```

### 3. Integration with Training

`OCRCollator` passes condition scores to the augmenter:

```python
for ex in examples:
    img = load_image(ex["image_path"], self.max_pixels)
    if self.augmenter is not None:
        condition_score = ex.get("condition_score", None)
        img = self.augmenter(img, condition_score=condition_score)
```

## Configuration

No changes needed to `config_qwen3_8b_full.yaml` - adaptive augmentation is automatic if `document_condition.csv` exists:

```yaml
augmentation:
  enabled: true  # Adaptive augmentation automatically enabled

  # Base parameters (Medium condition tier)
  p_elastic: 0.3
  p_color_jitter: 0.25
  p_resolution_jitter: 0.3
  min_pixels_ratio: 0.85
  elastic_alpha: 20
```

## Testing

```bash
# Test adaptive augmentation
python test_adaptive_aug.py --config config_qwen3_8b_full.yaml
```

This will:
1. Load condition scores
2. Find examples from each tier (good/medium/poor)
3. Apply augmentation 10 times per tier
4. Report statistics showing different augmentation strengths

## Verification

After training starts, you should see:

```
Loaded document condition scores from document_condition.csv
  Mean: 16.2, Median: 15.8, Std: 7.1
```

During training, the augmenter will automatically adjust strength per sample - no manual intervention needed.

## Fallback Behavior

If `document_condition.csv` is not found, training proceeds with default augmentation (no adaptation):

```
ℹ️  Document condition scores not found, proceeding without adaptive augmentation
```

All augmentation uses config defaults in this case.
