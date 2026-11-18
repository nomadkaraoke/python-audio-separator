from pathlib import Path

import numpy as np
import soundfile as sf

from audio_separator.separator import Separator


def test_numpy_separation_returns_arrays_without_output_files(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[2]
    audio, sample_rate = sf.read(repo / "tests/inputs/mardy20s.flac", dtype="float32")
    separator = Separator(
        model_file_dir=str(repo.parent / ".models"),
        output_dir=str(tmp_path),
    )
    separator.load_model(model_filename="kuielab_b_vocals.onnx")
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
    assert not any(tmp_path.iterdir())
