# Troubleshooting Guide

## Common Errors and Solutions

### 1. `TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'warmup_ratio'`

**Error:**
```
TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'warmup_ratio'
```

**Cause:** Older transformers version doesn't support `warmup_ratio` parameter.

**Solution:** ✅ **FIXED** - Code now automatically converts `warmup_ratio` to `warmup_steps`.

**What changed:**
```python
# Before (doesn't work with some transformers versions)
warmup_ratio=train_cfg["warmup_ratio"]

# After (compatible with all versions)
warmup_steps = int(total_steps * warmup_ratio)
```

**If you still see this error:**
```bash
# Restart Colab runtime after installing transformers
Runtime > Restart Runtime

# Then re-run all cells
```

---

### 2. `ImportError: cannot import name 'Qwen3VLForConditionalGeneration'`

**Error:**
```
ImportError: cannot import name 'Qwen3VLForConditionalGeneration' from 'transformers'
```

**Cause:** Transformers not upgraded to latest version.

**Solution:**
```bash
# CRITICAL: Install from GitHub, not PyPI
pip install --upgrade git+https://github.com/huggingface/transformers

# After install, RESTART RUNTIME
Runtime > Restart Runtime
```

**Verify:**
```python
import transformers
print(transformers.__version__)  # Should be 4.47.0.dev0 or similar

from transformers import Qwen3VLForConditionalGeneration
print("✓ Qwen3-VL available")
```

---

### 3. CUDA Out of Memory

**Error:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**For Qwen3-VL-2B:**
```yaml
# Reduce in config_qwen3_2b.yaml
training:
  batch_size: 4  # down from 8
  max_pixels: 2500000  # down from 3M
```

**For Qwen2-VL-7B:**
```yaml
# Reduce in config.yaml
training:
  batch_size: 2  # down from 4
  gradient_accumulation_steps: 8  # up from 4 (keep effective=16)
  max_pixels: 1500000  # down from 2.5M
```

**Or clear memory:**
```python
import torch
torch.cuda.empty_cache()
```

---

### 4. `KeyError: 'warmup_ratio'` in config

**Error:**
```
KeyError: 'warmup_ratio'
```

**Cause:** Missing warmup_ratio in config file.

**Solution:** The code uses a default if missing:
```python
warmup_ratio = train_cfg.get("warmup_ratio", 0.1)  # defaults to 10%
```

No action needed - this is handled automatically.

---

### 5. Model downloads slowly or times out

**Error:**
```
ConnectionError: Couldn't reach https://huggingface.co/...
```

**Solution:**
```python
# Pre-download before training
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3-VL-2B-Instruct",
    cache_dir="/content/cache"
)
```

**Or use HF_HUB_OFFLINE mode:**
```bash
export HF_HUB_OFFLINE=1
```

---

### 6. Training hangs at "Building dataset"

**Symptom:** Stuck at 0% for >5 minutes

**Cause:** Image loading issue or corrupted files.

**Solution:**
```python
# Check for corrupted images
from PIL import Image
from pathlib import Path

image_dir = Path("/content/ROAD/dataset/images")
corrupted = []

for img_path in image_dir.glob("*.jpg"):
    try:
        img = Image.open(img_path)
        img.verify()
    except Exception as e:
        corrupted.append(str(img_path))
        print(f"Corrupted: {img_path}")

print(f"\nTotal corrupted: {len(corrupted)}")
```

---

### 7. `RuntimeError: Expected all tensors to be on the same device`

**Error:**
```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
```

**Cause:** Model parts on different devices.

**Solution:** Already handled by `device_map="auto"`, but if it persists:
```python
# In train.py, after model load
model = model.to("cuda")
```

---

### 8. `AttributeError: 'NoneType' object has no attribute 'input_ids'`

**Symptom:** Error during training step.

**Cause:** Qwen3-VL expects different processor output format.

**Solution:** ✅ **FIXED** - Code now uses Qwen3 official style:
```python
# Qwen3 style (what we use now)
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,  # Key: tokenize in template
    return_dict=True,
    return_tensors="pt"
)
```

---

### 9. K-Fold checkpoint not found during inference

**Error:**
```
FileNotFoundError: outputs/qwen3-2b-kfold/fold_1/best/ not found
```

**Cause:** Training didn't complete or crashed.

**Check:**
```bash
ls -la /content/ROAD/outputs/qwen3-2b-kfold/
```

**Solution:**
```bash
# Use single model inference instead
python inference.py --config config_qwen3_2b.yaml \
    --checkpoint outputs/qwen3-2b-kfold/fold_1/checkpoint-XXX
```

---

### 10. Flash Attention installation fails

**Error:**
```
ERROR: Failed building wheel for flash-attn
```

**Solution:** ✅ **Flash Attention is optional!**
```yaml
# In config file, disable it:
model:
  use_flash_attention: false
```

Training will work fine without it (20-30% slower, same accuracy).

---

## Quick Diagnostic Commands

### Check GPU
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
```

### Check Transformers Version
```python
import transformers
print(f"Transformers: {transformers.__version__}")

try:
    from transformers import Qwen3VLForConditionalGeneration
    print("✓ Qwen3-VL supported")
except:
    print("✗ Qwen3-VL NOT supported - upgrade transformers!")
```

### Check Dataset
```python
import pandas as pd
from pathlib import Path

df = pd.read_csv('/content/ROAD/dataset/Train.csv')
image_dir = Path('/content/ROAD/dataset/images')
images = list(image_dir.glob('*.jpg'))

print(f"CSV rows: {len(df)}")
print(f"Images: {len(images)}")
print(f"Match: {len(df) == len(images)}")
```

### Check Config
```python
import yaml

with open('/content/ROAD/src/qwen2vl/config_qwen3_2b.yaml') as f:
    cfg = yaml.safe_load(f)

print(f"Model: {cfg['model']['name']}")
print(f"K-Folds: {cfg['data']['k_folds']}")
print(f"Batch size: {cfg['training']['batch_size']}")
print(f"LoRA rank: {cfg['training']['lora_r']}")
```

---

## Still Having Issues?

1. **Restart Colab runtime** - Fixes 80% of problems
2. **Check TRAINING_NOTES.md** - Known issues and solutions
3. **Read error message carefully** - Often tells you exactly what's wrong
4. **Check VRAM usage** with `nvidia-smi`
5. **Verify files exist** before training/inference
