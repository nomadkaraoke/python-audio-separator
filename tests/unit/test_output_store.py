import pytest
from unittest.mock import MagicMock, patch, mock_open
from audio_separator.remote.output_store import GCSOutputStore


@pytest.fixture
def mock_storage_client():
    with patch("audio_separator.remote.output_store.storage.Client") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        yield mock_client


@pytest.fixture
def store(mock_storage_client):
    return GCSOutputStore(bucket_name="test-bucket", project="test-project")


class TestGCSOutputStore:
    def test_upload_directory(self, store, mock_storage_client):
        """Uploads all files from a local directory to GCS under task_id prefix."""
        import os
        with patch("os.listdir", return_value=["vocals.flac", "instrumental.flac"]):
            with patch("os.path.isfile", return_value=True):
                store.upload_task_outputs("task-123", "/tmp/outputs/task-123")

        bucket = mock_storage_client.bucket.return_value
        assert bucket.blob.call_count == 2
        blob = bucket.blob.return_value
        assert blob.upload_from_filename.call_count == 2

    def test_upload_builds_correct_gcs_paths(self, store, mock_storage_client):
        """GCS paths are {task_id}/{filename}."""
        with patch("os.listdir", return_value=["output.flac"]):
            with patch("os.path.isfile", return_value=True):
                store.upload_task_outputs("task-123", "/tmp/outputs/task-123")

        bucket = mock_storage_client.bucket.return_value
        bucket.blob.assert_called_with("task-123/output.flac")

    def test_download_file(self, store, mock_storage_client):
        """Downloads a specific file from GCS to a local path."""
        store.download_file("task-123", "vocals.flac", "/tmp/local/vocals.flac")

        bucket = mock_storage_client.bucket.return_value
        bucket.blob.assert_called_with("task-123/vocals.flac")
        blob = bucket.blob.return_value
        blob.download_to_filename.assert_called_with("/tmp/local/vocals.flac")

    def test_get_file_bytes(self, store, mock_storage_client):
        """Gets file content as bytes for streaming download responses."""
        bucket = mock_storage_client.bucket.return_value
        blob = bucket.blob.return_value
        blob.download_as_bytes.return_value = b"audio data"

        result = store.get_file_bytes("task-123", "vocals.flac")

        assert result == b"audio data"
        bucket.blob.assert_called_with("task-123/vocals.flac")

    def test_delete_task_outputs(self, store, mock_storage_client):
        """Deletes all files for a task from GCS."""
        bucket = mock_storage_client.bucket.return_value
        blob1 = MagicMock()
        blob2 = MagicMock()
        bucket.list_blobs.return_value = [blob1, blob2]

        deleted = store.delete_task_outputs("task-123")

        bucket.list_blobs.assert_called_with(prefix="task-123/")
        blob1.delete.assert_called_once()
        blob2.delete.assert_called_once()
        assert deleted == 2
