import logging
import platform
from unittest.mock import MagicMock, patch

from audio_separator.separator import Separator

HINT = "DirectML packages detected but DirectML is not enabled"


def _run_setup(use_directml, dml_installed):
    """Construct a Separator without auto device-setup, then drive setup_torch_device
    with CUDA and MPS forced unavailable so the CPU-fallback path always runs."""
    sep = Separator(info_only=True)
    sep.use_directml = use_directml

    def fake_dist(name):
        if name in ("torch_directml", "onnxruntime-directml") and dml_installed:
            return MagicMock()
        return None

    with patch.object(sep, "get_package_distribution", side_effect=fake_dist), \
         patch("torch.cuda.is_available", return_value=False), \
         patch("torch.backends.mps.is_available", return_value=False), \
         patch("audio_separator.separator.separator.ort.get_available_providers", return_value=["CPUExecutionProvider"]):
        sep.setup_torch_device(platform.uname())
    return sep


def test_directml_hint_shown_when_packages_present_but_disabled(caplog):
    with caplog.at_level(logging.INFO):
        _run_setup(use_directml=False, dml_installed=True)
    assert any(HINT in r.message for r in caplog.records)


def test_directml_hint_absent_when_no_packages(caplog):
    with caplog.at_level(logging.INFO):
        _run_setup(use_directml=False, dml_installed=False)
    assert not any(HINT in r.message for r in caplog.records)
