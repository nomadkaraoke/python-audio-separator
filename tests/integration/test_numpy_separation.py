from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audio_separator.separator import Separator
from audio_separator.separator.architectures.mdx_separator import MDXSeparator
from audio_separator.separator.architectures.mdxc_separator import MDXCSeparator

# The MDXC models cover both MDXC demix paths: MDX23C (TFC_TDF_net) with two
# trained stems, and RoFormer with a single target plus a residual stem.
NUMPY_MODELS = [
    pytest.param("kuielab_b_vocals.onnx", MDXSeparator, False, id="mdx"),
    pytest.param("MDX23C-8KFFT-InstVoc_HQ.ckpt", MDXCSeparator, False, id="mdxc-mdx23c"),
    pytest.param("model_bs_roformer_ep_317_sdr_12.9755.ckpt", MDXCSeparator, True, id="mdxc-roformer"),
]


@pytest.mark.parametrize("model_filename, architecture, is_roformer", NUMPY_MODELS)
def test_numpy_separation_returns_arrays_without_output_files(tmp_path, monkeypatch, model_filename, architecture, is_roformer):
    repo = Path(__file__).resolve().parents[2]
    audio, sample_rate = sf.read(repo / "tests/inputs/mardy20s.flac", dtype="float32")
    separator = Separator(
        model_file_dir=str(repo.parent / ".models"),
        output_dir=str(tmp_path),
    )
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
    assert set(stems) == {"Vocals", "Instrumental"}
    for stem in stems.values():
        assert isinstance(stem, np.ndarray)
        assert stem.shape == audio.shape
        assert np.issubdtype(stem.dtype, np.floating)
        assert np.isfinite(stem).all()
        assert np.any(stem)
        assert not np.shares_memory(stem, audio)
    assert not np.shares_memory(stems["Vocals"], stems["Instrumental"])
    assert not any(tmp_path.iterdir())
