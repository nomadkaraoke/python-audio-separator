import torch


def is_rocm():
    """Check if PyTorch is built with ROCm support."""
    return "+rocm" in torch.__version__
