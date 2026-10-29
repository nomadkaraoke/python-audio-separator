"""Demucs package files are pickles; loading one must never run code it carries."""

import importlib
import os
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import torch

UVR_LIB_PATH = Path(__file__).resolve().parents[2] / "audio_separator" / "separator" / "uvr_lib_v5"


@pytest.fixture
def demucs_modules(monkeypatch):
    # Checkpoints reference the top-level `demucs.*` names, as DemucsSeparator sets up.
    monkeypatch.syspath_prepend(str(UVR_LIB_PATH))
    return importlib.import_module("demucs.states"), importlib.import_module("demucs.hdemucs")


class _Payload:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return (os.system, (f"touch {self.marker}",))


def _save_package(path, klass, kwargs, state):
    torch.save({"klass": klass, "args": [], "kwargs": kwargs, "state": state}, path)


def test_malicious_package_is_refused_without_running(demucs_modules, tmp_path):
    states, _ = demucs_modules
    marker = tmp_path / "pwned"
    model_path = tmp_path / "evil.th"
    torch.save({"klass": _Payload(str(marker)), "args": [], "kwargs": {}, "state": {}}, model_path)

    with pytest.raises(Exception, match="not allowed"):
        states.load_model(model_path)

    assert not marker.exists()


def test_disallowed_class_reference_is_refused(demucs_modules, tmp_path):
    states, _ = demucs_modules
    model_path = tmp_path / "other_class.th"
    _save_package(model_path, torch.nn.Linear, {"in_features": 2, "out_features": 2}, {})

    with pytest.raises(Exception, match="torch.nn.modules.linear.Linear"):
        states.load_model(model_path)


def test_legitimate_package_round_trips(demucs_modules, tmp_path):
    states, hdemucs = demucs_modules
    kwargs = {"sources": ["drums", "bass"], "channels": 4, "depth": 2, "segment": Fraction(39, 5), "samplerate": np.int64(44100)}
    source = hdemucs.HDemucs(**kwargs)
    model_path = tmp_path / "tiny.th"
    _save_package(model_path, hdemucs.HDemucs, kwargs, {k: v.half() for k, v in source.state_dict().items()})

    model = states.load_model(model_path)

    assert isinstance(model, hdemucs.HDemucs)
    assert model.sources == ["drums", "bass"]
    for key, value in source.state_dict().items():
        assert torch.equal(model.state_dict()[key], value.half().to(value.dtype))
