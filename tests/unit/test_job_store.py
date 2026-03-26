import pytest
from unittest.mock import MagicMock, patch
from audio_separator.remote.job_store import FirestoreJobStore


@pytest.fixture
def mock_firestore_client():
    with patch("google.cloud.firestore.Client") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        yield mock_client


@pytest.fixture
def store(mock_firestore_client):
    return FirestoreJobStore(project="test-project")


class TestFirestoreJobStore:
    def test_set_creates_document(self, store, mock_firestore_client):
        """Setting a task_id writes to Firestore with timestamps."""
        store.set("task-123", {
            "task_id": "task-123",
            "status": "submitted",
            "progress": 0,
        })

        collection = mock_firestore_client.collection
        collection.assert_called_with("audio_separation_jobs")
        collection.return_value.document.assert_called_with("task-123")
        doc_ref = collection.return_value.document.return_value
        doc_ref.set.assert_called_once()

        written_data = doc_ref.set.call_args[0][0]
        assert written_data["task_id"] == "task-123"
        assert written_data["status"] == "submitted"
        assert "updated_at" in written_data

    def test_get_returns_document_data(self, store, mock_firestore_client):
        """Getting a task_id reads from Firestore."""
        doc_snapshot = MagicMock()
        doc_snapshot.exists = True
        doc_snapshot.to_dict.return_value = {
            "task_id": "task-123",
            "status": "processing",
            "progress": 50,
        }
        collection = mock_firestore_client.collection
        collection.return_value.document.return_value.get.return_value = doc_snapshot

        result = store.get("task-123")

        assert result["status"] == "processing"
        assert result["progress"] == 50

    def test_get_returns_none_for_missing_document(self, store, mock_firestore_client):
        """Getting a nonexistent task_id returns None."""
        doc_snapshot = MagicMock()
        doc_snapshot.exists = False
        collection = mock_firestore_client.collection
        collection.return_value.document.return_value.get.return_value = doc_snapshot

        result = store.get("nonexistent")
        assert result is None

    def test_contains_checks_existence(self, store, mock_firestore_client):
        """__contains__ checks if document exists in Firestore."""
        doc_snapshot = MagicMock()
        doc_snapshot.exists = True
        collection = mock_firestore_client.collection
        collection.return_value.document.return_value.get.return_value = doc_snapshot

        assert "task-123" in store

    def test_update_merges_fields(self, store, mock_firestore_client):
        """Updating a task merges fields without overwriting the whole doc."""
        store.update("task-123", {"status": "processing", "progress": 25})

        collection = mock_firestore_client.collection
        doc_ref = collection.return_value.document.return_value
        doc_ref.update.assert_called_once()
        updated_data = doc_ref.update.call_args[0][0]
        assert updated_data["status"] == "processing"
        assert updated_data["progress"] == 25
        assert "updated_at" in updated_data

    def test_delete_removes_document(self, store, mock_firestore_client):
        """Deleting a task_id removes the Firestore document."""
        store.delete("task-123")

        collection = mock_firestore_client.collection
        doc_ref = collection.return_value.document.return_value
        doc_ref.delete.assert_called_once()

    def test_cleanup_old_jobs(self, store, mock_firestore_client):
        """cleanup_old_jobs deletes documents older than max_age_seconds."""
        old_doc = MagicMock()
        old_doc.reference = MagicMock()
        query = MagicMock()
        query.stream.return_value = [old_doc]

        collection = mock_firestore_client.collection
        collection.return_value.where.return_value.where.return_value = query

        deleted = store.cleanup_old_jobs(max_age_seconds=3600)

        assert deleted == 1
        old_doc.reference.delete.assert_called_once()
