"""Firestore-backed job status store for audio separation jobs.

Replaces the in-memory dict so any Cloud Run instance can read/write job status.
"""
import logging
import time
from typing import Optional

logger = logging.getLogger("audio-separator-api")

COLLECTION = "audio_separation_jobs"


class FirestoreJobStore:
    """Job status store backed by Firestore.

    Provides dict-like get/set interface for job status documents.
    """

    def __init__(self, project: str = "nomadkaraoke"):
        from google.cloud import firestore

        self._firestore = firestore
        self._db = firestore.Client(project=project)
        self._collection = self._db.collection(COLLECTION)

    def set(self, task_id: str, data: dict) -> None:
        """Create or overwrite a job status document."""
        data = {**data, "updated_at": self._firestore.SERVER_TIMESTAMP}
        if "created_at" not in data:
            data["created_at"] = self._firestore.SERVER_TIMESTAMP
        self._collection.document(task_id).set(data)

    def get(self, task_id: str) -> Optional[dict]:
        """Get job status. Returns None if not found."""
        doc = self._collection.document(task_id).get()
        if doc.exists:
            return doc.to_dict()
        return None

    def update(self, task_id: str, fields: dict) -> None:
        """Merge fields into an existing document."""
        fields = {**fields, "updated_at": self._firestore.SERVER_TIMESTAMP}
        self._collection.document(task_id).update(fields)

    def delete(self, task_id: str) -> None:
        """Delete a job status document."""
        self._collection.document(task_id).delete()

    def __contains__(self, task_id: str) -> bool:
        """Check if a task exists."""
        doc = self._collection.document(task_id).get()
        return doc.exists

    def cleanup_old_jobs(self, max_age_seconds: int = 3600) -> int:
        """Delete completed/errored jobs older than max_age_seconds. Returns count deleted."""
        cutoff = time.time() - max_age_seconds
        from datetime import datetime, timezone
        cutoff_dt = datetime.fromtimestamp(cutoff, tz=timezone.utc)

        deleted = 0
        query = (
            self._collection
            .where("status", "in", ["completed", "error"])
            .where("updated_at", "<", cutoff_dt)
        )
        for doc in query.stream():
            doc.reference.delete()
            deleted += 1

        if deleted:
            logger.info(f"Cleaned up {deleted} old job(s) from Firestore")
        return deleted
