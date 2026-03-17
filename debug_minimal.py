#!/usr/bin/env python3
"""
Minimal debug script to diagnose ROCm implementation.
Saves results to debug_results.txt
"""

import os
import sys
import traceback


def main():
    output = []

    output.append("=== ROCm Debug Script ===")
    output.append(f"Python: {sys.version}")
    output.append(f"Arguments: {sys.argv}")

    # Check environment
    output.append("\n=== Environment Variables ===")
    for var in ["HSA_OVERRIDE_GFX_VERSION", "PYTORCH_ROCM_ARCH", "ROCM_PATH"]:
        val = os.environ.get(var, "NOT SET")
        output.append(f"{var}: {val}")

    # Check PyTorch
    output.append("\n=== PyTorch Info ===")
    try:
        import torch

        output.append(f"Version: {torch.__version__}")
        output.append(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            output.append(f"Device: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        output.append(f"Error: {e}")
        output.append(f"Traceback:\n{traceback.format_exc()}")

    # Check ONNX Runtime
    output.append("\n=== ONNX Runtime Info ===")
    try:
        import onnxruntime as ort

        output.append(f"Version: {ort.__version__}")
        output.append(f"Available providers: {ort.get_available_providers()}")
    except Exception as e:
        output.append(f"Error: {e}")
        output.append(f"Traceback:\n{traceback.format_exc()}")

    output.append("\n=== Done ===")

    return "\n".join(output)


if __name__ == "__main__":
    output = main()
    print(output)
    with open("debug_results.txt", "w") as f:
        f.write(output)
    print("\nResults saved to debug_results.txt")
