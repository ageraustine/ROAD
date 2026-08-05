"""
Quick compatibility test for Qwen3-VL-2B

Run this before starting full training to verify:
1. Transformers version supports Qwen3-VL
2. Model can be loaded
3. VRAM usage is acceptable
4. Processor works correctly

Usage:
    python test_qwen3_compatibility.py
"""

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import numpy as np

MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"

def test_compatibility():
    print("="*70)
    print("QWEN3-VL-2B COMPATIBILITY TEST")
    print("="*70)

    # Check CUDA
    print(f"\n✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    # Try loading model
    print(f"\nLoading model: {MODEL_NAME}")
    try:
        model = AutoModelForVision2Seq.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        print("✓ Model loaded successfully")

        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated() / 1e9
            print(f"✓ VRAM used: {vram_used:.1f}GB")
            print(f"✓ VRAM available for training: {(torch.cuda.get_device_properties(0).total_memory / 1e9) - vram_used:.1f}GB")
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False

    # Try loading processor
    print(f"\nLoading processor...")
    try:
        processor = AutoProcessor.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
        )
        print("✓ Processor loaded successfully")
    except Exception as e:
        print(f"✗ Processor loading failed: {e}")
        return False

    # Test inference on dummy image
    print(f"\nTesting inference on dummy image...")
    try:
        # Create dummy image (white background, black text-like shapes)
        dummy_img = Image.fromarray(
            (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {"type": "image", "image": dummy_img},
                ],
            }
        ]

        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = processor(
            text=[prompt],
            images=[dummy_img],
            return_tensors="pt",
            padding=True,
        )

        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
            )

        decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        print(f"✓ Inference successful")
        print(f"  Sample output: {decoded[:100]}...")

    except Exception as e:
        print(f"✗ Inference test failed: {e}")
        return False

    # Final summary
    print("\n" + "="*70)
    print("COMPATIBILITY TEST PASSED ✓")
    print("="*70)
    print("\nYou can now run training with Qwen3-VL-2B:")
    print("  python train.py --config config_qwen3_2b.yaml")
    print("\n" + "="*70)

    return True


if __name__ == "__main__":
    success = test_compatibility()
    exit(0 if success else 1)
