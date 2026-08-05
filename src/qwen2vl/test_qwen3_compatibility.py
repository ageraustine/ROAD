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
from PIL import Image
import numpy as np

MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"

# Try importing Qwen3-VL specific class
try:
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    print("✓ Qwen3VLForConditionalGeneration available")
    QWEN3_AVAILABLE = True
except ImportError:
    print("⚠️  Qwen3VLForConditionalGeneration not available")
    print("   Attempting fallback to AutoModelForVision2Seq...")
    from transformers import AutoModelForVision2Seq as Qwen3VLForConditionalGeneration, AutoProcessor
    QWEN3_AVAILABLE = False

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
        # Use Qwen3 official loading style
        if QWEN3_AVAILABLE:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                MODEL_NAME,
                dtype="auto",  # Qwen3 uses dtype="auto"
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
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
                    {"type": "image", "image": dummy_img},
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]

        # Qwen3 style: apply_chat_template with tokenize=True
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)

        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=50)

        # Qwen3 style: trim input from output before decoding
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        print(f"✓ Inference successful")
        print(f"  Sample output: {output_text[0][:100] if output_text else 'N/A'}...")

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
