"""Public separation error propagation and batch aggregation contracts."""

import logging
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from audio_separator.separator.exceptions import AudioExportError, BatchSeparationError, InvalidAudioDataError
from audio_separator.separator.separator import Separator


def separator_double():
    separator = Mock(spec=Separator)
    separator.torch_device = Mock()
    separator.model_instance = Mock()
    separator.model_filename = "model.ckpt"
    separator.logger = logging.getLogger(__name__)
    return separator


def test_single_file_separation_fails_fast_with_original_error():
    separator = separator_double()
    export_error = AudioExportError("writer failed", path="bad_(Vocals).wav", backend="soundfile")
    separator._separate_file.side_effect = export_error

    with pytest.raises(AudioExportError) as raised:
        Separator.separate(separator, "bad.wav")

    assert raised.value is export_error


def test_batch_processes_every_file_then_raises_aggregate_error():
    separator = separator_double()
    export_error = AudioExportError("writer failed", path="bad_(Vocals).wav", backend="soundfile")
    separator._separate_file.side_effect = [["good_(Vocals).wav"], export_error, ["later_(Vocals).wav"]]

    with pytest.raises(BatchSeparationError) as raised:
        Separator.separate(separator, ["good.wav", "bad.wav", "later.wav"])

    assert raised.value.successful_files == ["good_(Vocals).wav", "later_(Vocals).wav"]
    assert raised.value.failures == [("bad.wav", export_error)]
    assert "bad.wav: writer failed" in str(raised.value)
    assert separator._separate_file.call_count == 3


def test_directory_processes_accessible_files_before_raising_batch_error():
    separator = separator_double()
    export_error = AudioExportError("writer failed", path="bad_(Vocals).wav", backend="soundfile")
    separator._separate_file.side_effect = [["good_(Vocals).wav"], export_error]

    with patch("audio_separator.separator.separator.os.path.isdir", return_value=True), patch(
        "audio_separator.separator.separator.os.walk",
        return_value=[("/music", [], ["good.wav", "bad.wav"])],
    ):
        with pytest.raises(BatchSeparationError) as raised:
            Separator.separate(separator, "/music")

    assert raised.value.successful_files == ["good_(Vocals).wav"]
    assert raised.value.failures == [("/music/bad.wav", export_error)]
    assert separator._separate_file.call_count == 2


def test_cleanup_failures_do_not_mask_writer_error_or_skip_later_cleanup():
    separator = separator_double()
    separator.chunk_duration = None
    separator.use_autocast = False
    separator.normalization_threshold = 0.9
    separator.amplification_threshold = 0.0
    separator.print_uvr_vip_message = Mock()
    writer_error = AudioExportError("writer failed", path="bad_(Vocals).wav", backend="soundfile")
    separator.model_instance.separate.side_effect = writer_error
    separator.model_instance.clear_gpu_cache.side_effect = RuntimeError("cache cleanup failed")

    with pytest.raises(AudioExportError) as raised:
        Separator._separate_file(separator, "bad.wav")

    assert raised.value is writer_error
    separator.model_instance.clear_gpu_cache.assert_called_once_with()
    separator.model_instance.clear_file_specific_paths.assert_called_once_with()


@patch("audio_separator.separator.audio_chunking.AudioChunker")
def test_chunked_separation_rejects_duplicate_stem_outputs(chunker_class, tmp_path):
    separator = separator_double()
    separator.output_dir = str(tmp_path)
    separator.output_format = "WAV"
    separator.chunk_duration = 10.0
    separator.model_instance.output_dir = str(tmp_path)
    chunker = chunker_class.return_value
    chunker.split_audio.return_value = [str(tmp_path / "chunk_0000.wav")]
    separator._separate_file.return_value = ["first_(Vocals).wav", "second_(Vocals).wav"]

    with pytest.raises(InvalidAudioDataError, match="duplicate"):
        Separator._process_with_chunking(separator, str(tmp_path / "input.wav"))

    chunker.merge_chunks.assert_not_called()


def test_ensemble_batch_processes_every_file_then_raises_aggregate_error():
    separator = separator_double()
    separator.model_filename = ["model-a.ckpt", "model-b.ckpt"]
    export_error = AudioExportError("ensemble writer failed", path="bad_(Vocals).wav", backend="pydub")
    separator._separate_ensemble.side_effect = [["good_(Vocals).wav"], export_error, ["later_(Vocals).wav"]]

    with pytest.raises(BatchSeparationError) as raised:
        Separator.separate(separator, ["good.wav", "bad.wav", "later.wav"])

    assert raised.value.successful_files == ["good_(Vocals).wav", "later_(Vocals).wav"]
    assert raised.value.failures == [("bad.wav", export_error)]
    assert separator._separate_ensemble.call_count == 3


@patch("audio_separator.separator.separator.Ensembler")
@patch("audio_separator.separator.separator.librosa.load", return_value=(np.zeros((2, 32), dtype=np.float32), 44100))
def test_ensemble_fallback_writer_preserves_soundfile_error(load_audio, ensembler_class, tmp_path):
    separator = separator_double()
    separator.model_filename = ["model-a.ckpt", "model-b.ckpt"]
    separator.model_filenames = ["model-a.ckpt", "model-b.ckpt"]
    separator.model_instance = None
    separator.output_dir = str(tmp_path)
    separator.output_format = "WAV"
    separator.sample_rate = 44100
    separator.normalization_threshold = 0.9
    separator.amplification_threshold = 0.0
    separator.ensemble_algorithm = "avg_wave"
    separator.ensemble_weights = None
    separator.ensemble_preset = None
    separator._separate_file.return_value = ["input_(Vocals).wav"]
    ensembler_class.return_value.ensemble.return_value = np.zeros((2, 32), dtype=np.float32)
    backend_error = OSError("soundfile failed")

    with patch("soundfile.write", side_effect=backend_error):
        with pytest.raises(AudioExportError) as raised:
            Separator._separate_ensemble(separator, "input.wav")

    assert raised.value.backend == "soundfile"
    assert raised.value.__cause__ is backend_error
    assert raised.value.path.endswith(".wav")


@patch("audio_separator.separator.separator.Ensembler")
@patch("audio_separator.separator.separator.librosa.load", return_value=(np.zeros((2, 32), dtype=np.float32), 44100))
def test_ensemble_fallback_writer_uses_wav_when_requested_format_fails(load_audio, ensembler_class, tmp_path):
    separator = separator_double()
    separator.model_filename = ["model-a.ckpt", "model-b.ckpt"]
    separator.model_filenames = ["model-a.ckpt", "model-b.ckpt"]
    separator.model_instance = None
    separator.output_dir = str(tmp_path)
    separator.output_format = "M4A"
    separator.sample_rate = 44100
    separator.normalization_threshold = 0.9
    separator.amplification_threshold = 0.0
    separator.ensemble_algorithm = "avg_wave"
    separator.ensemble_weights = None
    separator.ensemble_preset = None
    separator._separate_file.return_value = ["input_(Vocals).wav"]
    ensembler_class.return_value.ensemble.return_value = np.zeros((2, 32), dtype=np.float32)

    def write_requested_then_wav(path, *_args, **_kwargs):
        if str(path).endswith(".m4a"):
            raise OSError("requested format is unsupported")
        Path(path).write_bytes(b"wav audio")

    with patch("soundfile.write", side_effect=write_requested_then_wav):
        outputs = Separator._separate_ensemble(separator, "input.wav")

    assert len(outputs) == 1
    assert outputs[0].endswith(".wav")
    assert Path(outputs[0]).read_bytes() == b"wav audio"


@patch("audio_separator.separator.separator.Ensembler")
@patch("audio_separator.separator.separator.librosa.load", return_value=(np.zeros((2, 32), dtype=np.float32), 44100))
def test_ensemble_model_writer_failure_is_not_returned_as_success(load_audio, ensembler_class, tmp_path):
    separator = separator_double()
    separator.model_filename = ["model-a.ckpt", "model-b.ckpt"]
    separator.model_filenames = ["model-a.ckpt", "model-b.ckpt"]
    separator.output_dir = str(tmp_path)
    separator.output_format = "WAV"
    separator.sample_rate = 44100
    separator.ensemble_algorithm = "avg_wave"
    separator.ensemble_weights = None
    separator.ensemble_preset = None
    separator._separate_file.return_value = ["input_(Vocals).wav"]
    ensembler_class.return_value.ensemble.return_value = np.zeros((2, 32), dtype=np.float32)
    writer_error = AudioExportError("ensemble writer failed", path=str(tmp_path / "output.wav"), backend="pydub")
    separator.model_instance.write_audio.side_effect = writer_error

    with pytest.raises(AudioExportError) as raised:
        Separator._separate_ensemble(separator, "input.wav")

    assert raised.value is writer_error
