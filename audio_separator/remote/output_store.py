"""GCS-backed output file store for audio separation results.

Uploads separation output files to GCS so any Cloud Run instance can serve downloads.
"""
import logging
import os

from google.cloud import storage

logger = logging.getLogger("audio-separator-api")


class GCSOutputStore:
    """Manages separation output files in GCS."""

    def __init__(self, bucket_name: str = "nomadkaraoke-audio-separator-outputs", project: str = "nomadkaraoke"):
        self._client = storage.Client(project=project)
        self._bucket = self._client.bucket(bucket_name)

    def upload_task_outputs(self, task_id: str, local_dir: str) -> list[str]:
        """Upload all files in local_dir to GCS under {task_id}/ prefix.

        Returns list of uploaded filenames.
        """
        uploaded = []
        for filename in os.listdir(local_dir):
            local_path = os.path.join(local_dir, filename)
            if not os.path.isfile(local_path):
                continue
            gcs_path = f"{task_id}/{filename}"
            blob = self._bucket.blob(gcs_path)
            blob.upload_from_filename(local_path)
            uploaded.append(filename)
            logger.info(f"Uploaded {filename} to gs://{self._bucket.name}/{gcs_path}")
        return uploaded

    def get_file_bytes(self, task_id: str, filename: str) -> bytes:
        """Download file content as bytes (for HTTP responses)."""
        gcs_path = f"{task_id}/{filename}"
        blob = self._bucket.blob(gcs_path)
        return blob.download_as_bytes()

    def download_file(self, task_id: str, filename: str, local_path: str) -> str:
        """Download a file from GCS to a local path."""
        gcs_path = f"{task_id}/{filename}"
        blob = self._bucket.blob(gcs_path)
        blob.download_to_filename(local_path)
        return local_path

    def delete_task_outputs(self, task_id: str) -> int:
        """Delete all output files for a task. Returns count deleted."""
        deleted = 0
        for blob in self._bucket.list_blobs(prefix=f"{task_id}/"):
            blob.delete()
            deleted += 1
        if deleted:
            logger.info(f"Deleted {deleted} output file(s) for task {task_id}")
        return deleted
