import torch
from rotary_embedding_torch.rotary_embedding_torch import apply_rotary_emb, rotate_half

from audio_separator.separator.uvr_lib_v5.device_utils import autocast_disabled


def _is_dml_device(device: torch.device) -> bool:
    """Return whether a device uses torch-directml's private backend slot."""
    return device.type == "privateuseone"


def _float32_frequencies(rotary_embed, *, seq_len: int, device: torch.device) -> torch.Tensor:
    """Build or retrieve rotary angles without allowing autocast to reduce precision."""
    # A compiled graph is shared by many regional Transformer instances. Their
    # time and frequency embeddings have different cache shapes, so reading or
    # mutating cached_freqs here creates instance-state guards and repeated
    # Dynamo recompilations. Angle construction is cheap relative to attention
    # and becomes part of the compiled graph, so skip the mutable cache while
    # Dynamo is tracing.
    is_compiling = torch.compiler.is_compiling()
    should_cache = (
        not is_compiling
        and rotary_embed.cache_if_possible
        and not rotary_embed.learned_freq
        and rotary_embed.freqs_for != "pixel"
    )
    cached_freqs = rotary_embed.cached_freqs if should_cache else None

    if (
        should_cache
        and cached_freqs is not None
        and cached_freqs.dtype == torch.float32
        and cached_freqs.device == device
        and seq_len <= cached_freqs.shape[0]
    ):
        return cached_freqs[:seq_len].detach()

    positions = rotary_embed.get_seq_pos(seq_len, device=device, dtype=torch.float32)
    base_frequencies = rotary_embed.freqs.to(dtype=torch.float32)
    frequencies = torch.einsum("..., f -> ... f", positions, base_frequencies)
    frequencies = torch.repeat_interleave(frequencies, 2, dim=-1)

    if should_cache:
        # Replace an older low-precision cache instead of allowing it to be
        # reused after leaving an autocast region.
        rotary_embed.cached_freqs = frequencies.detach()

    return frequencies


def rotate_queries_or_keys(rotary_embed, tensor: torch.Tensor) -> torch.Tensor:
    """Apply full-head rotary embeddings with float32 angles on every backend.

    rotary-embedding-torch 0.6.5 disables CUDA autocast only, so CPU and MPS
    autocast can otherwise lower the precision of its position/angle einsum.
    DirectML also rejects the empty edge tensors concatenated by the upstream
    helper when the full head dimension is rotated, so that case keeps the
    existing concat-free implementation.

    This module reads rotary-embedding-torch internals (freqs, cached_freqs,
    cache_if_possible, learned_freq, freqs_for, default_seq_dim, get_seq_pos)
    rather than a public API, which is why pyproject pins the dependency to
    the 0.6.x series these helpers are validated against.
    """
    input_dtype = tensor.dtype
    seq_dim = rotary_embed.default_seq_dim
    seq_len = tensor.shape[seq_dim]

    with autocast_disabled(tensor.device):
        frequencies = _float32_frequencies(rotary_embed, seq_len=seq_len, device=tensor.device)

        if seq_dim == -3:
            frequencies = frequencies[:, None, :]

        if _is_dml_device(tensor.device) and frequencies.shape[-1] == tensor.shape[-1]:
            rotated = tensor * frequencies.cos() + rotate_half(tensor) * frequencies.sin()
        else:
            rotated = apply_rotary_emb(frequencies, tensor, seq_dim=seq_dim)

    return rotated.to(dtype=input_dtype)
