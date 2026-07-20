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


# ---------------------------------------------------------------------------
# RoformerLoader map_location handling on DirectML (issue #292)
#
# torch-directml's deserialization hook expects integer device ids, so
# torch.load(map_location=<privateuseone device>) raises TypeError and the
# loader silently fell back to the legacy implementation. The fix loads the
# state dict on CPU (DML only) and moves the model to the device afterwards.
# ---------------------------------------------------------------------------

from audio_separator.separator.roformer.roformer_loader import RoformerLoader, _is_dml_device


class TestIsDmlDevice:
    def test_dml_device_strings(self):
        assert _is_dml_device("privateuseone")
        assert _is_dml_device("privateuseone:0")
        assert _is_dml_device("privateuseone:1")

    def test_non_dml_devices(self):
        assert not _is_dml_device("cpu")
        assert not _is_dml_device("cuda")
        assert not _is_dml_device("cuda:0")
        assert not _is_dml_device("mps")

    def test_torch_device_objects(self):
        import torch

        assert not _is_dml_device(torch.device("cpu"))
        assert _is_dml_device(torch.device("privateuseone", 0))


def _load_via_new_implementation(device):
    """Drive _load_with_new_implementation with mocked model + torch.load,
    returning (map_location_used, device_model_moved_to)."""
    loader = RoformerLoader()
    model = MagicMock(name="model")
    seen = {}

    def fake_torch_load(path, map_location=None):
        seen["map_location"] = map_location
        return {}

    with patch.object(loader, "_create_bs_roformer", return_value=model), \
         patch("torch.load", side_effect=fake_torch_load), \
         patch("os.path.exists", return_value=True):
        result = loader._load_with_new_implementation(
            model_path="/fake/model.ckpt",
            config={"dim": 1, "depth": 1, "freqs_per_bands": (2,)},
            model_type="bs_roformer",
            device=device,
        )

    assert result.success
    model.to.assert_called_once_with(device)
    return seen["map_location"]


def test_new_implementation_loads_on_cpu_for_dml_device():
    # State dict must be mapped to CPU; the model still moves to the DML device.
    assert _load_via_new_implementation("privateuseone:0") == "cpu"


def test_new_implementation_map_location_unchanged_for_cpu():
    assert _load_via_new_implementation("cpu") == "cpu"


def test_new_implementation_map_location_unchanged_for_cuda():
    assert _load_via_new_implementation("cuda:0") == "cuda:0"


def test_new_implementation_map_location_unchanged_for_mps():
    assert _load_via_new_implementation("mps") == "mps"


# ---------------------------------------------------------------------------
# MDXC-family DirectML CPU fallback (issue #292)
# ---------------------------------------------------------------------------

import torch as _torch

from audio_separator.separator.architectures.mdxc_separator import _mdxc_inference_device


class TestMdxcInferenceDevice:
    def test_cpu_and_cuda_pass_through(self):
        log = MagicMock()
        assert _mdxc_inference_device(_torch.device("cpu"), _torch.device("cpu"), log) == _torch.device("cpu")
        cuda = _torch.device("cuda", 0)
        assert _mdxc_inference_device(cuda, _torch.device("cpu"), log) == cuda
        log.warning.assert_not_called()

    def test_dml_falls_back_to_cpu_with_warning(self, monkeypatch):
        monkeypatch.delenv("AUDIO_SEPARATOR_FORCE_DML_MDXC", raising=False)
        log = MagicMock()
        dml = _torch.device("privateuseone", 0)
        result = _mdxc_inference_device(dml, _torch.device("cpu"), log)
        assert result == _torch.device("cpu")
        assert log.warning.call_count == 1
        assert "run on CPU under DirectML" in log.warning.call_args[0][0]

    def test_dml_fallback_without_cpu_device_configured(self, monkeypatch):
        monkeypatch.delenv("AUDIO_SEPARATOR_FORCE_DML_MDXC", raising=False)
        result = _mdxc_inference_device(_torch.device("privateuseone", 0), None, MagicMock())
        assert result == _torch.device("cpu")

    def test_env_override_keeps_dml(self, monkeypatch):
        monkeypatch.setenv("AUDIO_SEPARATOR_FORCE_DML_MDXC", "1")
        log = MagicMock()
        dml = _torch.device("privateuseone", 0)
        assert _mdxc_inference_device(dml, _torch.device("cpu"), log) == dml
        assert "attempting" in log.warning.call_args[0][0]
