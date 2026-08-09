# K-Fold Ensemble Inference Guide

## Overview

After training 5 folds, ensemble their predictions to boost performance by +1.5-2%.

## Quick Start

### 1. Train 5 Folds (Already configured in config.yaml)

```bash
python train.py
```

This trains 5 models saved to:
```
/content/drive/MyDrive/prd/qwen3-8b-full_s7_kfold5/
├── fold_1/best/
├── fold_2/best/
├── fold_3/best/
├── fold_4/best/
└── fold_5/best/
```

### 2. Run Ensemble Inference (RECOMMENDED: rover_mbr)

```bash
python inference.py --kfold --ensemble-strategy rover_mbr
```

**Output:** `submission_full.csv` with ensembled predictions

## Ensemble Strategies

### rover_mbr (BEST - Default)
```bash
python inference.py --kfold --ensemble-strategy rover_mbr
```
- **What it does**: Aligns sequences using ROVER, votes character-by-character, uses MBR for ties
- **Best for**: Historical handwriting with ambiguous/faded characters
- **Expected gain**: +1.5-2% over single model
- **Use when**: You want maximum accuracy (worth the compute time)

### rover (Fast Alternative)
```bash
python inference.py --kfold --ensemble-strategy rover
```
- **What it does**: ROVER alignment + character voting (no MBR tiebreak)
- **Expected gain**: +1-1.5%
- **Use when**: You want speed without much accuracy loss

### mbr (Fastest Quality Option)
```bash
python inference.py --kfold --ensemble-strategy mbr
```
- **What it does**: Selects prediction with lowest average edit distance
- **Expected gain**: +0.5-1%
- **Use when**: You need fast ensemble with decent quality

## Caching & Resumability

Fold predictions are cached for fast re-ensemble:

```bash
# First run: Runs inference on all 5 folds (slow)
python inference.py --kfold --ensemble-strategy rover_mbr

# Try different strategy (fast - uses cached predictions)
python inference.py --kfold --ensemble-strategy mbr

# Clear cache and re-run everything
python inference.py --kfold --ensemble-strategy rover_mbr --clear-cache
```

**Cache location:** `/content/drive/MyDrive/prd/qwen3-8b-full_s7_kfold5/inference_cache/`

## Expected Performance

| Setup | Expected Score | Notes |
|-------|----------------|-------|
| Single model (rank 64, LR 3e-5) | 0.90 | Baseline |
| + 5-fold rover_mbr ensemble | 0.92-0.94 | +1.5-2% from ensemble |
| + Test-time augmentation (TTA) | 0.93-0.95 | Additional +0.5-1% |

## Troubleshooting

**Error: "No fold checkpoints found"**
```bash
# Check if folds were trained
ls /content/drive/MyDrive/prd/qwen3-8b-full_s7_kfold5/

# Should see: fold_1/, fold_2/, fold_3/, fold_4/, fold_5/
```

**Ensemble too slow?**
```bash
# Use faster strategy
python inference.py --kfold --ensemble-strategy mbr

# Or use fewer folds (edit config.yaml: k_folds: 3)
```

**Want to ensemble only best 3 folds?**
```bash
# Manually delete worse folds, then run ensemble
rm -rf fold_4/ fold_5/
python inference.py --kfold --ensemble-strategy rover_mbr
```

## Advanced: Compare All Strategies

```bash
# Generate submissions for all strategies
for strategy in rover_mbr rover mbr majority; do
  python inference.py --kfold --ensemble-strategy $strategy --output submission_${strategy}.csv
done

# Upload all to leaderboard to see which works best
```

## How ROVER Works

1. **Align sequences** from 5 models using dynamic programming
   ```
   Model 1: "recieved the s^d land"
   Model 2: "received the s^d land"
   Model 3: "received the said land"
   Model 4: "received the s^d land"
   Model 5: "received the sed land"
   ```

2. **Vote character-by-character**
   - Position 4: 'e' wins (4 votes) vs 'i' (1 vote)
   - Position 18: 's^d' wins (3 votes) vs 'said' (1) vs 'sed' (1)

3. **Tiebreak with MBR** (if rover_mbr)
   - If votes are tied, pick char from prediction with lowest edit distance to others

**Result:** `"received the s^d land"` (most agreed-upon consensus)

## Performance Tips

- **rover_mbr** is worth the extra compute for 0.90 → 0.94 push
- Cache means re-trying strategies is free (reuses same fold predictions)
- If inference OOMs, reduce `inference.batch_size` in config.yaml
