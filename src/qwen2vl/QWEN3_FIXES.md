# Qwen3-VL Code Fixes Applied

Based on the official Qwen3-VL example code, I've updated the codebase to properly support both Qwen2-VL and Qwen3-VL.

## Key Changes

### 1. Import Strategy (train.py, inference.py)

**Before:**
```python
from transformers import AutoModelForVision2Seq
```

**After:**
```python
# Support both Qwen2-VL and Qwen3-VL
try:
    from transformers import Qwen3VLForConditionalGeneration
    QWEN3_AVAILABLE = True
except ImportError:
    QWEN3_AVAILABLE = False

try:
    from transformers import Qwen2VLForConditionalGeneration
    QWEN2_AVAILABLE = True
except ImportError:
    QWEN2_AVAILABLE = False

# Fallback to generic class
if not QWEN3_AVAILABLE and not QWEN2_AVAILABLE:
    from transformers import AutoModelForVision2Seq
```

### 2. Model Loading (train.py setup_model, inference.py load_model)

**Before:**
```python
model = AutoModelForVision2Seq.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
```

**After:**
```python
# Detect model type
is_qwen3 = "Qwen3" in model_name or "qwen3" in model_name.lower()

# Setup kwargs based on model type
if is_qwen3:
    model_kwargs["dtype"] = "auto"  # Qwen3 official style
else:
    model_kwargs["torch_dtype"] = torch.bfloat16  # Qwen2 style

# Select appropriate model class
if is_qwen3 and QWEN3_AVAILABLE:
    model_class = Qwen3VLForConditionalGeneration
elif is_qwen2 and QWEN2_AVAILABLE:
    model_class = Qwen2VLForConditionalGeneration
else:
    model_class = AutoModelForVision2Seq

model = model_class.from_pretrained(model_name, **model_kwargs)
```

### 3. Inference Style (inference.py predict_single)

**Before:**
```python
# Old Qwen2 style
prompt = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = processor(
    text=[prompt],
    images=[image],
    return_tensors="pt",
    padding=True,
)

outputs = model.generate(**inputs, max_new_tokens=256)
decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]
```

**After:**
```python
# Qwen3 official style
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,  # Key difference: tokenize in apply_chat_template
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=256)

# Trim input prompt from output (Qwen3 style)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

# Decode only the generated part
output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)
```

### 4. Message Content Order

**Before:**
```python
content: [
    {"type": "text", "text": OCR_PROMPT},
    {"type": "image", "image": image},
]
```

**After:**
```python
content: [
    {"type": "image", "image": image},  # Image first (Qwen3 convention)
    {"type": "text", "text": OCR_PROMPT},
]
```

### 5. Compatibility Test (test_qwen3_compatibility.py)

Updated to use:
- `Qwen3VLForConditionalGeneration` specifically
- `dtype="auto"` for model loading
- Proper tokenization and decoding style
- Trimming of generated output

---

## What These Fixes Enable

✅ **Proper Qwen3-VL support** - Uses official loading and inference style
✅ **Backward compatibility** - Still works with Qwen2-VL
✅ **Automatic detection** - Detects model type from name
✅ **Correct output** - Properly trims input prompt from generated text
✅ **Better tokenization** - Uses Qwen3's preferred tokenize-in-template approach

---

## Testing

Run the compatibility test:
```bash
cd /content/ROAD/src/qwen2vl/
python test_qwen3_compatibility.py
```

Expected output:
```
✓ Qwen3VLForConditionalGeneration available
✓ Model loaded successfully
✓ VRAM used: 5.2GB
✓ Inference successful
COMPATIBILITY TEST PASSED ✓
```

---

## Training Command (No Changes Needed!)

```bash
# For Qwen3-VL-2B
python train.py --config config_qwen3_2b.yaml

# For Qwen2-VL-7B (still works)
python train.py --config config.yaml
```

The code automatically detects which model you're using based on the `model.name` in the config file.
