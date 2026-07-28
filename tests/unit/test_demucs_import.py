import importlib
from pathlib import Path


def test_checkpoint_compatible_top_level_demucs_import(monkeypatch):
    """Demucs modules remain importable under checkpoint-compatible top-level names."""
    uvr_lib_path = Path(__file__).resolve().parents[2] / "audio_separator" / "separator" / "uvr_lib_v5"
    monkeypatch.syspath_prepend(str(uvr_lib_path))

    hdemucs = importlib.import_module("demucs.hdemucs")
    htdemucs = importlib.import_module("demucs.htdemucs")
    spec = importlib.import_module("demucs.spec")

    assert hdemucs.HDemucs is not None
    assert htdemucs.HTDemucs is not None
    assert spec.spectro is not None
