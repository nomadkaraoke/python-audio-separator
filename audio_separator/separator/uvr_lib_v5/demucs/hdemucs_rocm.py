"""ROCm entry point for the Demucs spectrogram and hybrid model.

Upstream ``HDemucs`` already keeps complex and spectral work on the GPU for cuda
devices, and PyTorch reports AMD GPUs under ROCm as cuda. There is no ROCm-specific
behaviour to fork, so ``HDemucsROCm`` is a thin subclass that preserves the name used
by the ROCm integration and inherits the upstream constructor and forward pass.
"""

from .hdemucs import HDemucs


class HDemucsROCm(HDemucs):
    """ROCm-named alias of the upstream HDemucs model (no behaviour changes)."""
