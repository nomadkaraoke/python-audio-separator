from itertools import combinations
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audio_separator.separator import Separator
from audio_separator.separator.architectures.demucs_separator import DemucsSeparator
from audio_separator.separator.architectures.mdx_separator import MDXSeparator
from audio_separator.separator.architectures.mdxc_separator import MDXCSeparator
from audio_separator.separator.architectures.vr_separator import VRSeparator

VOCALS_AND_INSTRUMENTAL = {"Vocals", "Instrumental"}
DEMUCS_6_STEMS = {"Vocals", "Drums", "Bass", "Other", "Guitar", "Piano"}

# The MDXC models cover both MDXC demix paths: MDX23C (TFC_TDF_net) with two
# trained stems, and RoFormer with a single target plus a residual stem.
# VR rebuilds waveforms from STFT frames, so its stems may end a few milliseconds early.
NUMPY_MODELS = [
    pytest.param("kuielab_b_vocals.onnx", MDXSeparator, False, 0.0, VOCALS_AND_INSTRUMENTAL, id="mdx"),
    pytest.param("MDX23C-8KFFT-InstVoc_HQ.ckpt", MDXCSeparator, False, 0.0, VOCALS_AND_INSTRUMENTAL, id="mdxc-mdx23c"),
    pytest.param("model_bs_roformer_ep_317_sdr_12.9755.ckpt", MDXCSeparator, True, 0.0, VOCALS_AND_INSTRUMENTAL, id="mdxc-roformer"),
    pytest.param("MGM_MAIN_v4.pth", VRSeparator, False, 0.05, VOCALS_AND_INSTRUMENTAL, id="vr"),
    pytest.param("htdemucs_6s.yaml", DemucsSeparator, False, 0.0, DEMUCS_6_STEMS, id="demucs"),
]


@pytest.mark.parametrize("model_filename, architecture, is_roformer, max_length_diff_seconds, expected_stems", NUMPY_MODELS)
def test_numpy_separation_returns_arrays_without_output_files(
    tmp_path, monkeypatch, model_filename, architecture, is_roformer, max_length_diff_seconds, expected_stems
):
    repo = Path(__file__).resolve().parents[2]
    audio, sample_rate = sf.read(repo / "tests/inputs/mardy20s.flac", dtype="float32")
    separator = Separator(output_dir=str(tmp_path))
    separator.load_model(model_filename=model_filename)
    assert type(separator.model_instance) is architecture
    assert getattr(separator.model_instance, "is_roformer", False) is is_roformer
    original_audio = audio.copy()

    def unexpected_audio_io(*args, **kwargs):
        raise AssertionError("NumPy separation must not read or write audio files")

    for method in ("load", "get_duration"):
        monkeypatch.setattr(f"audio_separator.separator.common_separator.librosa.{method}", unexpected_audio_io)
    monkeypatch.setattr("audio_separator.separator.common_separator.CommonSeparator.write_audio", unexpected_audio_io)

    stems = separator.separate_numpy(audio, sample_rate=sample_rate)

    np.testing.assert_array_equal(audio, original_audio)
    assert isinstance(stems, dict)
    assert set(stems) == expected_stems
    max_length_diff = int(max_length_diff_seconds * sample_rate)
    for stem in stems.values():
        assert isinstance(stem, np.ndarray)
        assert stem.ndim == 2 and stem.shape[1] == audio.shape[1]
        assert abs(stem.shape[0] - audio.shape[0]) <= max_length_diff
        assert np.issubdtype(stem.dtype, np.floating)
        assert np.isfinite(stem).all()
        assert np.any(stem)
        assert not np.shares_memory(stem, audio)
    assert not any(np.shares_memory(first, second) for first, second in combinations(stems.values(), 2))
    assert not any(tmp_path.iterdir())
