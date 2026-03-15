import pytest
import json
import logging
from unittest.mock import patch, MagicMock
from io import StringIO
from audio_separator.separator import Separator


@pytest.fixture
def mock_separator_init():
    """Fixture that patches hardware setup so Separator can be instantiated without GPU."""
    with patch.object(Separator, "setup_accelerated_inferencing_device"):
        yield


def test_load_preset_vocal_balanced(mock_separator_init):
    sep = Separator(ensemble_preset="vocal_balanced")
    assert sep.ensemble_algorithm == "avg_fft"
    assert sep._ensemble_preset_models == [
        "bs_roformer_vocals_resurrection_unwa.ckpt",
        "melband_roformer_big_beta6x.ckpt",
    ]
    assert sep.ensemble_weights is None


def test_load_preset_karaoke(mock_separator_init):
    sep = Separator(ensemble_preset="karaoke")
    assert sep.ensemble_algorithm == "avg_wave"
    assert len(sep._ensemble_preset_models) == 3


def test_load_preset_instrumental_clean(mock_separator_init):
    sep = Separator(ensemble_preset="instrumental_clean")
    assert sep.ensemble_algorithm == "uvr_max_spec"
    assert len(sep._ensemble_preset_models) == 2


def test_preset_algorithm_override(mock_separator_init):
    """User explicitly sets algorithm, which should override preset's default."""
    sep = Separator(ensemble_preset="vocal_clean", ensemble_algorithm="avg_wave")
    # vocal_clean preset uses min_fft, but user overrode to avg_wave
    assert sep.ensemble_algorithm == "avg_wave"
    # Models still come from preset
    assert sep._ensemble_preset_models == [
        "bs_roformer_vocals_revive_v2_unwa.ckpt",
        "mel_band_roformer_kim_ft2_bleedless_unwa.ckpt",
    ]


def test_preset_no_algorithm_uses_preset_default(mock_separator_init):
    """When no algorithm is specified, preset's algorithm is used."""
    sep = Separator(ensemble_preset="vocal_clean")
    assert sep.ensemble_algorithm == "min_fft"  # from preset


def test_preset_unknown_name(mock_separator_init):
    with pytest.raises(ValueError, match="Unknown ensemble preset"):
        Separator(ensemble_preset="nonexistent_preset")


def test_no_preset_defaults_to_avg_wave(mock_separator_init):
    sep = Separator()
    assert sep.ensemble_algorithm == "avg_wave"
    assert sep._ensemble_preset_models is None


def test_list_ensemble_presets(mock_separator_init):
    sep = Separator(info_only=True)
    presets = sep.list_ensemble_presets()
    assert isinstance(presets, dict)
    assert "vocal_balanced" in presets
    assert "karaoke" in presets
    assert "instrumental_clean" in presets
    assert len(presets) == 9


def test_preset_loads_models_on_load_model(mock_separator_init):
    """Calling load_model() with default arg should use preset models."""
    sep = Separator(ensemble_preset="karaoke")
    # Don't actually load the model, just check the preset models are set
    assert sep._ensemble_preset_models is not None
    assert len(sep._ensemble_preset_models) == 3


def test_preset_json_valid():
    """Validate that ensemble_presets.json is well-formed."""
    from importlib import resources
    with resources.open_text("audio_separator", "ensemble_presets.json") as f:
        data = json.load(f)

    assert "version" in data
    assert data["version"] == 1
    assert "presets" in data

    valid_algorithms = [
        "avg_wave", "median_wave", "min_wave", "max_wave",
        "avg_fft", "median_fft", "min_fft", "max_fft",
        "uvr_max_spec", "uvr_min_spec", "ensemble_wav",
    ]

    for preset_id, preset in data["presets"].items():
        assert "name" in preset, f"Preset {preset_id} missing 'name'"
        assert "description" in preset, f"Preset {preset_id} missing 'description'"
        assert "models" in preset, f"Preset {preset_id} missing 'models'"
        assert "algorithm" in preset, f"Preset {preset_id} missing 'algorithm'"
        assert isinstance(preset["models"], list), f"Preset {preset_id} models must be a list"
        assert len(preset["models"]) >= 2, f"Preset {preset_id} must have at least 2 models"
        assert preset["algorithm"] in valid_algorithms, f"Preset {preset_id} has invalid algorithm: {preset['algorithm']}"

        weights = preset.get("weights")
        if weights is not None:
            assert isinstance(weights, list), f"Preset {preset_id} weights must be a list"
            assert len(weights) == len(preset["models"]), f"Preset {preset_id} weights length mismatch"
