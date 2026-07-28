from functools import wraps
from packaging import version
from collections import namedtuple

import torch
from torch import nn, einsum
from torch.nn.attention import SDPBackend, sdpa_kernel
import torch.nn.functional as F

from einops import rearrange, reduce

# constants

FlashAttentionConfig = namedtuple("FlashAttentionConfig", ["enable_flash", "enable_math", "enable_mem_efficient"])

# helpers


def _is_dml_device(device) -> bool:
    """torch-directml devices use torch's out-of-tree backend slot (privateuseone).

    F.scaled_dot_product_attention is not implemented by torch-directml (fails
    with a D3D12 'The parameter is incorrect.' error), so DML tensors must take
    the plain einsum attention path. Module-level so tests can patch it.
    """
    return device.type == "privateuseone"


def exists(val):
    return val is not None


def _sdpa_backends(config):
    """Translate the legacy SDPA flags to the Dynamo-compatible API."""
    backends = []
    if config.enable_flash:
        backends.append(SDPBackend.FLASH_ATTENTION)
    if config.enable_mem_efficient:
        backends.append(SDPBackend.EFFICIENT_ATTENTION)
    if config.enable_math:
        backends.append(SDPBackend.MATH)

    # torch.backends.cuda.sdp_kernel enabled cuDNN by default even though the
    # local config predates that fourth flag. Preserve that behavior.
    backends.append(SDPBackend.CUDNN_ATTENTION)
    return backends


def once(fn):
    called = False

    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)

    return inner


print_once = once(print)

# main class


class Attend(nn.Module):
    def __init__(self, dropout=0.0, flash=False, scale=None):
        super().__init__()
        self.scale = scale
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.flash = flash
        assert not (flash and version.parse(torch.__version__) < version.parse("2.0.0")), "in order to use flash attention, you must be using pytorch 2.0 or above"

        # determine efficient attention configs for cuda and cpu

        self.cpu_config = FlashAttentionConfig(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device("cuda"))

        if device_properties.major == 8 and device_properties.minor == 0:
            print_once("A100 GPU detected, using flash attention if input tensor is on cuda")
            self.cuda_config = FlashAttentionConfig(True, False, False)
        else:
            self.cuda_config = FlashAttentionConfig(False, True, True)

    def flash_attn(self, q, k, v):
        _, heads, q_len, _, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device

        # Check if there is a compatible device for flash attention

        config = self.cuda_config if is_cuda else self.cpu_config

        # sdpa_flash kernel only supports float16 on sm80+ architecture gpu
        if is_cuda and q.dtype != torch.float16:
            config = FlashAttentionConfig(False, True, True)

        # Keep SDPA backend selection inside the graphable PyTorch API.
        with sdpa_kernel(_sdpa_backends(config)):
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0.0,
                scale=self.scale,
            )

        return out

    def forward(self, q, k, v):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        q_len, k_len, device = q.shape[-2], k.shape[-2], q.device

        scale = self.scale if exists(self.scale) else q.shape[-1] ** -0.5

        # DML has no SDPA — fall through to the einsum path. Gated so every
        # other device keeps its exact existing behavior. (Issue #292)
        if self.flash and not _is_dml_device(device):
            return self.flash_attn(q, k, v)

        if _is_dml_device(device):
            # Use matmul (a real GEMM) instead of einsum: torch-directml
            # lowers einsum naively (broadcast multiply + reduce), whose
            # b×h×i×j×d intermediate is tens of GB at segment 801 — the
            # 'DML allocator out of memory' crash. Also slice the batch dim
            # to bound the materialized (b h i j) similarity tensor. The
            # result is mathematically identical to the einsum path below.
            outs = []
            step = 8
            for i in range(0, q.shape[0], step):
                qs, ks, vs = q[i : i + step], k[i : i + step], v[i : i + step]
                sim = torch.matmul(qs, ks.transpose(-1, -2)) * scale
                attn = sim.softmax(dim=-1)
                attn = self.attn_dropout(attn)
                outs.append(torch.matmul(attn, vs))
            return torch.cat(outs, dim=0)

        # similarity

        sim = einsum(f"b h i d, b h j d -> b h i j", q, k) * scale

        # attention

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # aggregate values

        out = einsum(f"b h i j, b h j d -> b h i d", attn, v)

        return out
