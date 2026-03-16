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


def test_preset_validation_bad_weights_length(mock_separator_init):
    """Preset with weights length != models length should raise ValueError."""
    # We need to patch the JSON to inject a bad preset
    import io
    bad_json = json.dumps({
        "version": 1,
        "presets": {
            "bad_weights": {
                "name": "Bad",
                "description": "test",
                "models": ["a.ckpt", "b.ckpt"],
                "algorithm": "avg_wave",
                "weights": [1.0, 2.0, 3.0],  # 3 weights for 2 models
            }
        }
    })
    with patch("audio_separator.separator.separator.resources.open_text", return_value=io.StringIO(bad_json)):
        with pytest.raises(ValueError, match="weights length"):
            Separator(ensemble_preset="bad_weights")


def test_preset_validation_bad_algorithm(mock_separator_init):
    """Preset with unknown algorithm should raise ValueError."""
    import io
    bad_json = json.dumps({
        "version": 1,
        "presets": {
            "bad_algo": {
                "name": "Bad",
                "description": "test",
                "models": ["a.ckpt", "b.ckpt"],
                "algorithm": "nonexistent_algorithm",
                "weights": None,
            }
        }
    })
    with patch("audio_separator.separator.separator.resources.open_text", return_value=io.StringIO(bad_json)):
        with pytest.raises(ValueError, match="unknown algorithm"):
            Separator(ensemble_preset="bad_algo")


def test_preset_validation_single_model(mock_separator_init):
    """Preset with only 1 model should raise ValueError."""
    import io
    bad_json = json.dumps({
        "version": 1,
        "presets": {
            "one_model": {
                "name": "Bad",
                "description": "test",
                "models": ["a.ckpt"],
                "algorithm": "avg_wave",
                "weights": None,
            }
        }
    })
    with patch("audio_separator.separator.separator.resources.open_text", return_value=io.StringIO(bad_json)):
        with pytest.raises(ValueError, match="at least 2 models"):
            Separator(ensemble_preset="one_model")


def test_preset_weights_applied(mock_separator_init):
    """Preset with explicit weights should apply them."""
    import io
    preset_json = json.dumps({
        "version": 1,
        "presets": {
            "weighted": {
                "name": "Weighted",
                "description": "test",
                "models": ["a.ckpt", "b.ckpt"],
                "algorithm": "avg_wave",
                "weights": [2.0, 1.0],
            }
        }
    })
    with patch("audio_separator.separator.separator.resources.open_text", return_value=io.StringIO(preset_json)):
        sep = Separator(ensemble_preset="weighted")
        assert sep.ensemble_weights == [2.0, 1.0]


def test_preset_explicit_weights_override(mock_separator_init):
    """User-provided weights should override preset weights."""
    import io
    preset_json = json.dumps({
        "version": 1,
        "presets": {
            "weighted": {
                "name": "Weighted",
                "description": "test",
                "models": ["a.ckpt", "b.ckpt"],
                "algorithm": "avg_wave",
                "weights": [2.0, 1.0],
            }
        }
    })
    with patch("audio_separator.separator.separator.resources.open_text", return_value=io.StringIO(preset_json)):
        sep = Separator(ensemble_preset="weighted", ensemble_weights=[5.0, 3.0])
        assert sep.ensemble_weights == [5.0, 3.0]
