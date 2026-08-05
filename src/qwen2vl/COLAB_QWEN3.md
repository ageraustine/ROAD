# Qwen3-VL-2B Training in Colab

Copy and paste these cells into your Colab notebook to train Qwen3-VL-2B.

---

## Cell 1: Setup and Compatibility Test

```python
# Install bleeding-edge transformers for Qwen3-VL support
print("📦 Upgrading transformers for Qwen3-VL support...")
!pip install -q --upgrade git+https://github.com/huggingface/transformers

# Navigate to repo
%cd /content/ROAD/src/qwen2vl/

# Test Qwen3-VL compatibility
print("\n🧪 Testing Qwen3-VL-2B compatibility...\n")
!python test_qwen3_compatibility.py

print("\n" + "="*70)
print("If test passed, proceed to next cell to start training")
print("="*70)
```

---

## Cell 2: Train Qwen3-VL-2B with K-Fold CV

```python
%cd /content/ROAD/src/qwen2vl/
import time

start = time.time()

print("="*70)
print("🚀 Training Qwen3-VL-2B with 5-Fold CV")
print("="*70)
print("\nModel: Qwen/Qwen3-VL-2B-Instruct")
print("Architecture improvements:")
print("  ✓ Enhanced OCR (32 languages, blur/tilt robust)")
print("  ✓ Better ancient characters (1600s text)")
print("  ✓ DeepStack (fine-grained degraded text)")
print("\nConfig:")
print("  • K-Fold: 5 folds")
print("  • Batch size: 8 (effective=16)")
print("  • LoRA rank: 96")
print("  • Epochs: 7 per fold")
print("  • Resolution: 3M pixels (~1732x1732)")
print("\nExpected time: ~5-6 hours")
print("="*70 + "\n")

!python train.py --config config_qwen3_2b.yaml

elapsed = (time.time() - start) / 3600
print(f"\n✅ Training completed in {elapsed:.1f} hours")

# Show summary
print("\n📊 K-Fold Results:")
!cat /content/ROAD/outputs/qwen3-2b-kfold/kfold_summary.txt

%cd /content/ROAD
```

**Expected output:**
- 5 folds trained sequentially
- Each fold: ~1 hour
- Total: 5-6 hours
- Summary with average eval_loss

---

## Cell 3: Generate Ensemble Submission

```python
%cd /content/ROAD/src/qwen2vl/

print("="*70)
print("🔮 Generating ensemble predictions from 5 folds")
print("="*70)
print("\nEnsemble strategy: Character-level majority voting")
print("Expected improvement: 1-3% over single model\n")

!python inference.py --config config_qwen3_2b.yaml --kfold

print("\n✅ Submission generated: /content/ROAD/submission.csv")
print("\n📥 Download and submit to competition!")
print("="*70)

%cd /content/ROAD

# Show first few predictions
import pandas as pd
df = pd.read_csv('submission.csv')
print(f"\n✓ Generated {len(df)} predictions")
print("\nFirst 5 predictions:")
print(df.head())
```

---

## Cell 4: Quick Single Model Test (Optional)

Use this to test Qwen3-VL-2B quickly before committing to full K-Fold:

```python
%cd /content/ROAD/src/qwen2vl/
import time
import yaml

# Modify config for quick test
with open('config_qwen3_2b.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

# Override to single model, fewer epochs
cfg['data']['k_folds'] = 1
cfg['training']['epochs'] = 2
cfg['training']['output_dir'] = 'outputs/qwen3-2b-quicktest'

with open('config_qwen3_2b_test.yaml', 'w') as f:
    yaml.dump(cfg, f)

print("🧪 Quick test: Single model, 2 epochs (~20 minutes)\n")

start = time.time()
!python train.py --config config_qwen3_2b_test.yaml
elapsed = (time.time() - start) / 60

print(f"\n✅ Quick test completed in {elapsed:.1f} minutes")
print("\nIf eval_loss looks good (<0.65), proceed with full K-Fold training!")

%cd /content/ROAD
```

---

## Comparison: Qwen3-VL-2B vs Qwen2-VL-7B

| Metric | Qwen3-VL-2B (K-Fold) | Qwen2-VL-7B (K-Fold) |
|--------|----------------------|----------------------|
| **Training time** | ~6 hours | ~17 hours |
| **VRAM per fold** | ~25-30GB | ~50-55GB |
| **Speed** | 3x faster | Baseline |
| **OCR improvements** | ✓ Enhanced | Standard |
| **Expected WER** | 9-11% | 8-10% |
| **Expected CER** | 3-4% | 3-4% |

**Strategy:**
1. Train Qwen3-VL-2B first (6 hours) → submit
2. If competitive, you're done!
3. If not, train Qwen2-VL-7B (17 hours) → submit
4. Or ensemble both for maximum score

---

## Tips

**Monitor training:**
```python
# In a new cell, run periodically
!tail -20 /content/ROAD/outputs/qwen3-2b-kfold/fold_1/trainer_state.json
```

**Check VRAM usage:**
```python
!nvidia-smi
```

**Resume interrupted training:**
Training auto-resumes from last checkpoint if interrupted.

**Download checkpoints:**
```python
from google.colab import files

# Download best fold 1 model
!zip -r fold1.zip /content/ROAD/outputs/qwen3-2b-kfold/fold_1/best/
files.download('fold1.zip')
```
