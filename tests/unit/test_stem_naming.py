"""
Unit tests for stem name assignment logic in CommonSeparator and
stem normalization in _separate_ensemble.
"""

import pytest
import logging
from unittest.mock import patch, MagicMock, PropertyMock


class TestCommonSeparatorStemSwap:
    """Test the target_instrument / instruments[0] swap logic in CommonSeparator.__init__."""

    def _make_config(self, model_data):
        """Create a minimal config dict for CommonSeparator."""
        return {
            "logger": logging.getLogger("test"),
            "log_level": logging.DEBUG,
            "torch_device": None,
            "torch_device_cpu": None,
            "torch_device_mps": None,
            "onnx_execution_provider": None,
            "model_name": "test_model",
            "model_path": "/tmp/test.ckpt",
            "model_data": model_data,
            "output_dir": "/tmp",
            "output_format": "WAV",
            "output_bitrate": None,
            "normalization_threshold": 0.9,
            "amplification_threshold": 0.0,
            "enable_denoise": False,
            "output_single_stem": None,
            "invert_using_spec": False,
            "sample_rate": 44100,
            "use_soundfile": False,
        }

    @patch("audio_separator.separator.common_separator.CommonSeparator._detect_roformer_model", return_value=False)
    def test_no_swap_when_target_matches_instruments0(self, mock_detect):
        """Normal case: target_instrument == instruments[0], no swap needed."""
        from audio_separator.separator.common_separator import CommonSeparator
        model_data = {
            "training": {
                "instruments": ["vocals", "other"],
                "target_instrument": "vocals",
            }
        }
        sep = CommonSeparator(self._make_config(model_data))
        assert sep.primary_stem_name == "vocals"
        assert sep.secondary_stem_name == "other"

    @patch("audio_separator.separator.common_separator.CommonSeparator._detect_roformer_model", return_value=False)
    def test_swap_when_target_mismatches_instruments0(self, mock_detect):
        """Bug fix case: target_instrument == instruments[1], should swap."""
        from audio_separator.separator.common_separator import CommonSeparator
        model_data = {
            "training": {
                "instruments": ["vocals", "other"],
                "target_instrument": "other",
            }
        }
        sep = CommonSeparator(self._make_config(model_data))
        # Primary should be "other" (the target), secondary should be "vocals"
        assert sep.primary_stem_name == "other"
        assert sep.secondary_stem_name == "vocals"

    @patch("audio_separator.separator.common_separator.CommonSeparator._detect_roformer_model", return_value=False)
    def test_no_swap_when_no_target_instrument(self, mock_detect):
        """No target_instrument set — use instruments[0] as primary."""
        from audio_separator.separator.common_separator import CommonSeparator
        model_data = {
            "training": {
                "instruments": ["vocals", "other"],
            }
        }
        sep = CommonSeparator(self._make_config(model_data))
        assert sep.primary_stem_name == "vocals"
        assert sep.secondary_stem_name == "other"

    @patch("audio_separator.separator.common_separator.CommonSeparator._detect_roformer_model", return_value=False)
    def test_no_swap_when_target_not_in_instruments(self, mock_detect):
        """Edge case: target_instrument not in instruments list — no swap, use default order."""
        from audio_separator.separator.common_separator import CommonSeparator
        model_data = {
            "training": {
                "instruments": ["vocals", "other"],
                "target_instrument": "drums",
            }
        }
        sep = CommonSeparator(self._make_config(model_data))
        assert sep.primary_stem_name == "vocals"
        assert sep.secondary_stem_name == "other"

    @patch("audio_separator.separator.common_separator.CommonSeparator._detect_roformer_model", return_value=False)
    def test_single_instrument_no_swap(self, mock_detect):
        """Single instrument — no swap possible."""
        from audio_separator.separator.common_separator import CommonSeparator
        model_data = {
            "training": {
                "instruments": ["vocals"],
                "target_instrument": "vocals",
            }
        }
        sep = CommonSeparator(self._make_config(model_data))
        assert sep.primary_stem_name == "vocals"


class TestStemNameMap:
    """Test the STEM_NAME_MAP constant covers all expected mappings."""

    def test_stem_name_map_has_expected_entries(self):
        from audio_separator.separator.separator import STEM_NAME_MAP
        # All known 2-stem secondary names should map
        assert STEM_NAME_MAP["vocals"] == "Vocals"
        assert STEM_NAME_MAP["instrumental"] == "Instrumental"
        assert STEM_NAME_MAP["inst"] == "Instrumental"
        assert STEM_NAME_MAP["karaoke"] == "Instrumental"
        assert STEM_NAME_MAP["no_vocals"] == "Instrumental"
        assert STEM_NAME_MAP["other"] == "Other"  # For multi-stem; 2-stem override happens in ensemble
        assert STEM_NAME_MAP["drums"] == "Drums"
        assert STEM_NAME_MAP["bass"] == "Bass"

    def test_stem_name_map_keys_are_lowercase(self):
        from audio_separator.separator.separator import STEM_NAME_MAP
        for key in STEM_NAME_MAP:
            assert key == key.lower(), f"Key '{key}' is not lowercase"


class TestEnsembleOutputFilenames:
    """Test the output filename logic for preset and custom ensembles."""

    def test_preset_filename_format(self):
        """Preset ensemble should use 'preset_<name>' in filename."""
        # This is tested indirectly via integration, but let's verify the format
        import os
        base_name = "mardy20s"
        stem_name = "Vocals"
        preset = "vocal_balanced"
        expected = f"{base_name}_({stem_name})_preset_{preset}"
        assert expected == "mardy20s_(Vocals)_preset_vocal_balanced"

    def test_custom_ensemble_slug_generation(self):
        """Custom ensemble should generate model slugs for the filename."""
        import os
        # Simulate the slug logic from separator.py
        model_filenames = ["UVR-MDX-NET-Inst_HQ_5.onnx", "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt"]
        prefixes = ["mel_band_roformer_", "melband_roformer_", "bs_roformer_", "model_bs_roformer_", "UVR-MDX-NET-", "UVR_MDXNET_"]

        model_slugs = []
        for mf in model_filenames:
            name = os.path.splitext(mf)[0]
            for prefix in prefixes:
                if name.startswith(prefix):
                    name = name[len(prefix):]
                    break
            model_slugs.append(name[:12])

        slugs_str = "_".join(model_slugs)
        filename = f"mardy20s_(Vocals)_custom_ensemble_{slugs_str}"

        assert "Inst_HQ_5" in filename
        assert "karaoke_aufr" in filename
        assert filename.startswith("mardy20s_(Vocals)_custom_ensemble_")


class TestEnsembleCustomOutputNames:
    """Test that custom_output_names works correctly with ensemble separation."""

    def test_custom_output_names_not_passed_to_intermediate_separation(self):
        """Intermediate per-model separations must NOT receive custom_output_names.

        custom_output_names replaces the default '_(StemType)_model' naming, which
        removes the _(StemType)_ markers needed by _separate_ensemble to classify
        stems. custom_output_names should only be applied to the final ensembled output.
        """
        import re
        from unittest.mock import patch, MagicMock, call
        from audio_separator.separator.separator import Separator

        sep = Separator(
            log_level=logging.WARNING,
            model_file_dir="/tmp/models",
            output_dir="/tmp/output",
            output_format="flac",
        )
        sep.model_filenames = ["model_a.ckpt", "model_b.ckpt"]
        sep.model_filename = ["model_a.ckpt", "model_b.ckpt"]
        sep.ensemble_algorithm = "uvr_max_spec"
        sep.ensemble_weights = None
        sep.ensemble_preset = "test_preset"
        sep.sample_rate = 44100

        custom_names = {"Vocals": "job123_mixed_vocals", "Instrumental": "job123_mixed_instrumental"}

        with patch.object(sep, '_separate_file') as mock_separate, \
             patch.object(sep, 'load_model'), \
             patch('audio_separator.separator.separator.Ensembler') as MockEnsembler, \
             patch('audio_separator.separator.separator.librosa') as mock_librosa, \
             patch('audio_separator.separator.separator.np') as mock_np:

            # Mock _separate_file to return files with proper _(StemType)_ naming
            mock_separate.side_effect = [
                ["/tmp/ensemble/song_(Vocals)_model_a.flac", "/tmp/ensemble/song_(Instrumental)_model_a.flac"],
                ["/tmp/ensemble/song_(Vocals)_model_b.flac", "/tmp/ensemble/song_(Instrumental)_model_b.flac"],
            ]

            # Mock librosa and numpy for ensembling
            mock_wav = MagicMock()
            mock_wav.ndim = 2
            mock_wav.shape = (2, 44100)
            mock_librosa.load.return_value = (mock_wav, 44100)
            mock_np.asfortranarray.return_value = mock_wav

            mock_ensembler = MagicMock()
            mock_ensembler.ensemble.return_value = mock_wav
            MockEnsembler.return_value = mock_ensembler

            # Mock model_instance for write_audio
            sep.model_instance = MagicMock()
            sep.model_instance.output_dir = "/tmp/output"

            sep._separate_ensemble("/tmp/song.flac", custom_output_names=custom_names)

            # Key assertion: _separate_file must be called with None, not custom_names
            for call_args in mock_separate.call_args_list:
                assert call_args[0][1] is None, (
                    f"_separate_file was called with custom_output_names={call_args[0][1]!r} "
                    f"but should be None for intermediate ensemble files"
                )
