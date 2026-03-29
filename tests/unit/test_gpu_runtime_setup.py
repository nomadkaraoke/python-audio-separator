from unittest.mock import patch

from audio_separator.separator.separator import Separator


def test_setup_accelerated_inferencing_device_preloads_onnxruntime_dependencies():
    separator = Separator(info_only=True)
    system_info = object()

    with patch.object(separator, "get_system_info", return_value=system_info), patch.object(separator, "check_ffmpeg_installed"), patch.object(
        separator, "log_onnxruntime_packages"
    ), patch("audio_separator.separator.separator.ort.preload_dlls", create=True) as mock_preload, patch.object(separator, "setup_torch_device") as mock_setup:
        separator.setup_accelerated_inferencing_device()

    mock_preload.assert_called_once_with()
    mock_setup.assert_called_once_with(system_info)


def test_setup_accelerated_inferencing_device_continues_when_preload_fails():
    separator = Separator(info_only=True)
    system_info = object()

    with patch.object(separator, "get_system_info", return_value=system_info), patch.object(separator, "check_ffmpeg_installed"), patch.object(
        separator, "log_onnxruntime_packages"
    ), patch("audio_separator.separator.separator.ort.preload_dlls", side_effect=RuntimeError("boom"), create=True), patch.object(
        separator, "setup_torch_device"
    ) as mock_setup, patch.object(separator.logger, "warning") as mock_warning:
        separator.setup_accelerated_inferencing_device()

    mock_setup.assert_called_once_with(system_info)
    mock_warning.assert_called_once()
