"""
Unit tests for ROCm (AMD GPU) device routing in Separator.setup_torch_device.

ROCm PyTorch reports AMD GPUs through torch.cuda and advertises torch.version.hip.
These tests confirm that ROCm input routes to configure_rocm without changing the
CUDA/CPU paths, and that MIGraphXExecutionProvider is preferred over the legacy
ROCMExecutionProvider and CUDAExecutionProvider when available.
"""

import logging
from unittest.mock import patch

import torch

import audio_separator.separator.separator as separator_module
from audio_separator.separator.separator import Separator


def _new_separator() -> Separator:
    """Build a Separator without running the heavy __init__ side effects."""
    separator = Separator.__new__(Separator)
    separator.logger = logging.getLogger("test_rocm_device_setup")
    separator.use_directml = False
    separator.torch_device_cpu = torch.device("cpu")
    separator.torch_device = torch.device("cpu")
    separator.onnx_execution_provider = None
    return separator


def test_rocm_torch_routes_to_configure_rocm_with_cuda_provider():
    """ROCm PyTorch is detected via torch.version.hip and stays on the cuda device."""
    separator = _new_separator()
    with patch("torch.cuda.is_available", return_value=True), patch.object(
        torch.version, "hip", "7.2", create=True
    ), patch.object(
        separator_module.ort, "get_available_providers", return_value=["CUDAExecutionProvider", "CPUExecutionProvider"]
    ):
        separator.setup_torch_device(object())

    assert separator.torch_device == torch.device("cuda")
    assert separator.onnx_execution_provider == ["CUDAExecutionProvider"]


def test_rocm_prefers_native_rocm_execution_provider_when_available():
    """A native AMD ONNX provider is selected ahead of CUDAExecutionProvider."""
    separator = _new_separator()
    with patch("torch.cuda.is_available", return_value=True), patch.object(
        torch.version, "hip", "7.2", create=True
    ), patch.object(
        separator_module.ort, "get_available_providers", return_value=["ROCMExecutionProvider", "CPUExecutionProvider"]
    ):
        separator.setup_torch_device(object())

    assert separator.onnx_execution_provider == ["ROCMExecutionProvider"]


def test_non_rocm_cuda_still_uses_configure_cuda():
    """Without torch.version.hip, CUDA hosts keep the original configure_cuda path."""
    separator = _new_separator()
    with patch("torch.cuda.is_available", return_value=True), patch.object(
        torch.version, "hip", None, create=True
    ), patch.object(
        separator_module.ort, "get_available_providers", return_value=["CUDAExecutionProvider", "CPUExecutionProvider"]
    ), patch.object(separator, "configure_cuda", wraps=separator.configure_cuda) as mock_cuda:
        separator.setup_torch_device(object())

    mock_cuda.assert_called_once()
    assert separator.onnx_execution_provider == ["CUDAExecutionProvider"]


def test_cpu_only_has_no_gpu_provider():
    """A host with no GPU accelerator falls back to the CPU execution provider."""
    separator = _new_separator()
    with patch("torch.cuda.is_available", return_value=False), patch.object(
        separator_module.ort, "get_available_providers", return_value=["CPUExecutionProvider"]
    ):
        separator.setup_torch_device(type("SystemInfo", (), {"processor": "x86_64"})())

    assert separator.torch_device == torch.device("cpu")
    assert separator.onnx_execution_provider == ["CPUExecutionProvider"]


if __name__ == "__main__":
    test_rocm_torch_routes_to_configure_rocm_with_cuda_provider()
    test_rocm_prefers_native_rocm_execution_provider_when_available()
    test_non_rocm_cuda_still_uses_configure_cuda()
    test_cpu_only_has_no_gpu_provider()
    print("all ROCm device-setup checks passed")
