import torch


def is_rocm():
    """Check if PyTorch is built with ROCm support."""
    return getattr(torch.version, "hip", None) is not None
