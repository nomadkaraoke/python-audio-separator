"""Unit tests for the ROCm MDX inference-path guard (no GPU or torch runtime needed)."""

import logging
import os
from unittest.mock import patch

import torch

from audio_separator.separator.architectures.mdx_separator import MDXSeparator

_NO_OVERRIDE = {"AUDIO_SEPARATOR_ROCM_ONNX": ""}


def _separator(provider, device="cuda"):
    """Build an MDXSeparator shell with the attributes the guard reads."""
    separator = MDXSeparator.__new__(MDXSeparator)
    separator.logger = logging.getLogger("test_mdx_rocm_inference")
    separator.onnx_execution_provider = [provider] if provider else None
    separator.torch_device = torch.device(device)
    return separator


def test_non_rocm_host_never_converts():
    """A host without ROCm PyTorch keeps the ONNX Runtime path."""
    separator = _separator("CUDAExecutionProvider")
    with patch.object(torch.version, "hip", None, create=True):
        assert separator._prefer_pytorch_inference_on_rocm() is False


def test_rocm_with_only_cuda_provider_converts():
    """On a ROCm GPU, a CUDA-named provider cannot activate, so convert to PyTorch."""
    separator = _separator("CUDAExecutionProvider")
    with patch.object(torch.version, "hip", "7.2", create=True), patch.dict(os.environ, _NO_OVERRIDE):
        assert separator._prefer_pytorch_inference_on_rocm() is True


def test_rocm_with_migraphx_keeps_onnx_runtime():
    """A real AMD ONNX GPU provider keeps the ONNX Runtime path."""
    separator = _separator("MIGraphXExecutionProvider")
    with patch.object(torch.version, "hip", "7.2", create=True), patch.dict(os.environ, _NO_OVERRIDE):
        assert separator._prefer_pytorch_inference_on_rocm() is False


def test_rocm_with_legacy_rocm_provider_keeps_onnx_runtime():
    """The legacy ROCm execution provider also keeps the ONNX Runtime path."""
    separator = _separator("ROCMExecutionProvider")
    with patch.object(torch.version, "hip", "7.2", create=True), patch.dict(os.environ, _NO_OVERRIDE):
        assert separator._prefer_pytorch_inference_on_rocm() is False


def test_rocm_conversion_can_be_opted_out():
    """AUDIO_SEPARATOR_ROCM_ONNX restores the ONNX Runtime path."""
    separator = _separator("CUDAExecutionProvider")
    with patch.object(torch.version, "hip", "7.2", create=True), patch.dict(os.environ, {"AUDIO_SEPARATOR_ROCM_ONNX": "ort"}):
        assert separator._prefer_pytorch_inference_on_rocm() is False


def test_missing_provider_defaults_to_conversion_on_rocm():
    """An unset provider list on a ROCm GPU is treated as no AMD ONNX provider."""
    separator = _separator(None)
    with patch.object(torch.version, "hip", "7.2", create=True), patch.dict(os.environ, _NO_OVERRIDE):
        assert separator._prefer_pytorch_inference_on_rocm() is True


def test_rocm_cpu_only_does_not_convert():
    """A ROCm PyTorch build running on CPU keeps the ONNX Runtime path."""
    separator = _separator("CUDAExecutionProvider", device="cpu")
    with patch.object(torch.version, "hip", "7.2", create=True), patch.dict(os.environ, _NO_OVERRIDE):
        assert separator._prefer_pytorch_inference_on_rocm() is False
        assert separator._should_convert_inactive_amd_provider_on_rocm("MIGraphXExecutionProvider") is False


def test_inactive_amd_provider_converts_on_rocm():
    """A requested AMD provider that did not activate falls back to PyTorch on ROCm."""
    separator = _separator("MIGraphXExecutionProvider")
    with patch.object(torch.version, "hip", "7.2", create=True), patch.dict(os.environ, _NO_OVERRIDE):
        assert separator._should_convert_inactive_amd_provider_on_rocm("MIGraphXExecutionProvider") is True
        assert separator._should_convert_inactive_amd_provider_on_rocm("ROCMExecutionProvider") is True


def test_inactive_amd_provider_not_converted_on_non_rocm():
    """Non-ROCm hosts keep the ONNX Runtime session even if the provider was inactive."""
    separator = _separator("MIGraphXExecutionProvider")
    with patch.object(torch.version, "hip", None, create=True):
        assert separator._should_convert_inactive_amd_provider_on_rocm("MIGraphXExecutionProvider") is False


def test_inactive_amd_provider_conversion_respects_override():
    """The explicit override keeps the ONNX Runtime path even for an inactive AMD provider."""
    separator = _separator("MIGraphXExecutionProvider")
    with patch.object(torch.version, "hip", "7.2", create=True), patch.dict(os.environ, {"AUDIO_SEPARATOR_ROCM_ONNX": "0"}):
        assert separator._should_convert_inactive_amd_provider_on_rocm("MIGraphXExecutionProvider") is False


def test_inactive_cuda_provider_does_not_trigger_amd_conversion():
    """The inactive-AMD-provider rule only applies to AMD provider names."""
    separator = _separator("CUDAExecutionProvider")
    with patch.object(torch.version, "hip", "7.2", create=True):
        assert separator._should_convert_inactive_amd_provider_on_rocm("CUDAExecutionProvider") is False


if __name__ == "__main__":
    test_non_rocm_host_never_converts()
    test_rocm_with_only_cuda_provider_converts()
    test_rocm_with_migraphx_keeps_onnx_runtime()
    test_rocm_with_legacy_rocm_provider_keeps_onnx_runtime()
    test_rocm_conversion_can_be_opted_out()
    test_missing_provider_defaults_to_conversion_on_rocm()
    test_rocm_cpu_only_does_not_convert()
    test_inactive_amd_provider_converts_on_rocm()
    test_inactive_amd_provider_not_converted_on_non_rocm()
    test_inactive_amd_provider_conversion_respects_override()
    test_inactive_cuda_provider_does_not_trigger_amd_conversion()
    print("all MDX ROCm guard checks passed")
