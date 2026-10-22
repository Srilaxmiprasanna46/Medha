"""Tests for the Medha AI HTTP API (Phase 1 + Phase 2).

Each test function gets a clean, isolated in-memory SQLite database so that
tests are independent and do not touch the real study library.
"""

import sqlite3
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import backend.main as main_module
from backend.main import app
from memory.document_store import DocumentStore

# A single TestClient is reused across tests; state is reset per-test via the
# fixture below.
client = TestClient(app, raise_server_exceptions=True)


@pytest.fixture(autouse=True)
def isolated_store(tmp_path):
    """Give every test its own temporary SQLite database.

    ``_reset_state`` clears the in-memory ``documents`` dict and ``FactStore``
    and installs the provided ``DocumentStore`` as the active store, so no
    test can pollute another.
    """
    store = DocumentStore(db_path=tmp_path / "test.db")
    main_module._reset_state(store)
    yield
    # Tear down: close the temp store and restore a blank slate.
    store.close()
    main_module._reset_state(None)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# Documents list
# ---------------------------------------------------------------------------

def test_list_documents_empty():
    response = client.get("/api/documents")
    assert response.status_code == 200
    assert response.json()["documents"] == []


# ---------------------------------------------------------------------------
# Upload validation
# ---------------------------------------------------------------------------

def test_upload_unsupported_extension_rejected():
    response = client.post(
        "/api/documents",
        files={"file": ("notes.docx", b"data", "application/octet-stream")},
    )
    assert response.status_code == 400
    assert "supported" in response.json()["detail"].lower()


def test_upload_empty_file_rejected():
    response = client.post(
        "/api/documents",
        files={"file": ("empty.txt", b"", "text/plain")},
    )
    assert response.status_code == 400


def test_upload_oversized_file_rejected():
    """A file just over 75 MB must be rejected with HTTP 413."""
    oversized = b"x" * (75 * 1024 * 1024 + 1)
    response = client.post(
        "/api/documents",
        files={"file": ("big.txt", oversized, "text/plain")},
    )
    assert response.status_code == 413
    detail = response.json()["detail"].lower()
    assert "75" in detail or "mb" in detail


def test_upload_file_at_limit_accepted():
    """A file exactly at 75 MB must be accepted."""
    at_limit = b"a" * (75 * 1024 * 1024)
    response = client.post(
        "/api/documents",
        files={"file": ("limit.txt", at_limit, "text/plain")},
    )
    assert response.status_code == 200


def test_upload_invalid_utf8_txt_rejected():
    response = client.post(
        "/api/documents",
        files={"file": ("bad.txt", b"\xff\xfe invalid", "text/plain")},
    )
    assert response.status_code == 400
    assert "utf-8" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Successful upload – response shape
# ---------------------------------------------------------------------------

_SAMPLE_CONTENT = (
    b"Overfitting occurs when a model learns training data too closely. "
    b"This can reduce performance on unseen examples."
)


def _upload_sample(name: str = "test_notes.txt") -> dict:
    response = client.post(
        "/api/documents",
        files={"file": (name, _SAMPLE_CONTENT, "text/plain")},
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_upload_returns_expected_fields():
    data = _upload_sample()
    assert "id" in data
    assert data["name"] == "test_notes.txt"
    assert data["pages"] == 1
    assert data["status"] == "indexed"
    assert "size_bytes" in data
    # content (raw page text) must NOT be returned to the client
    assert "content" not in data


def test_upload_appears_in_document_list():
    _upload_sample()
    response = client.get("/api/documents")
    assert response.status_code == 200
    docs = response.json()["documents"]
    assert len(docs) == 1
    assert docs[0]["name"] == "test_notes.txt"
    assert docs[0]["pages"] == 1
    assert docs[0]["size_bytes"] > 0


# ---------------------------------------------------------------------------
# Fact lookup
# ---------------------------------------------------------------------------

def test_fact_lookup_empty_store():
    response = client.post("/api/facts", json={"question": "What is overfitting?"})
    assert response.status_code == 200
    assert response.json()["count"] == 0
    assert response.json()["facts"] == []


def test_upload_and_fact_lookup():
    _upload_sample()

    response = client.post("/api/facts", json={"question": "What is overfitting?"})
    assert response.status_code == 200

    result = response.json()
    assert result["count"] > 0
    assert any("Overfitting" in fact["text"] for fact in result["facts"])
    assert all(fact["document"] == "test_notes.txt" for fact in result["facts"])


# ---------------------------------------------------------------------------
# /api/ask
# ---------------------------------------------------------------------------

def test_ask_empty_question_rejected():
    response = client.post("/api/ask", json={"question": "the and or"})
    assert response.status_code == 400


def test_ask_no_documents_returns_not_grounded():
    response = client.post("/api/ask", json={"question": "What is overfitting?"})
    assert response.status_code == 200
    data = response.json()
    assert data["grounded"] is False
    assert data["citations"] == []


def test_ask_after_upload_returns_citations():
    _upload_sample()

    response = client.post("/api/ask", json={"question": "What is overfitting?"})
    assert response.status_code == 200

    data = response.json()
    assert data["grounded"] is True
    assert len(data["citations"]) > 0
    citation = data["citations"][0]
    assert citation["document"] == "test_notes.txt"
    assert isinstance(citation["page"], int)
    assert isinstance(citation["relevance"], float)
    assert isinstance(citation["text"], str)


# ---------------------------------------------------------------------------
# Persistence across restart (Feature A / D)
# ---------------------------------------------------------------------------

def test_document_persists_across_restart(tmp_path):
    """Upload a document, simulate a restart, confirm it is still retrievable."""
    db_path = tmp_path / "persist_test.db"

    # --- First "run" ---
    store1 = DocumentStore(db_path=db_path)
    main_module._reset_state(store1)

    _upload_sample("ml_notes.txt")

    # Verify it is in the list before restart.
    list_before = client.get("/api/documents").json()["documents"]
    assert len(list_before) == 1

    store1.close()

    # --- Simulate restart: clear state and restore from the same DB ---
    store2 = DocumentStore(db_path=db_path)
    main_module._reset_state(store2)  # _reset_state calls _restore_from_db

    list_after = client.get("/api/documents").json()["documents"]
    assert len(list_after) == 1
    assert list_after[0]["name"] == "ml_notes.txt"

    store2.close()


def test_restored_document_queryable_via_ask(tmp_path):
    """After a simulated restart the document should still answer questions."""
    db_path = tmp_path / "ask_persist.db"

    store1 = DocumentStore(db_path=db_path)
    main_module._reset_state(store1)
    _upload_sample("notes.txt")
    store1.close()

    store2 = DocumentStore(db_path=db_path)
    main_module._reset_state(store2)

    response = client.post("/api/ask", json={"question": "What is overfitting?"})
    assert response.status_code == 200
    data = response.json()
    assert data["grounded"] is True
    assert any("overfitting" in c["text"].lower() for c in data["citations"])

    store2.close()


def test_restored_document_queryable_via_facts(tmp_path):
    """After a simulated restart facts should still be found."""
    db_path = tmp_path / "facts_persist.db"

    store1 = DocumentStore(db_path=db_path)
    main_module._reset_state(store1)
    _upload_sample("notes.txt")
    store1.close()

    store2 = DocumentStore(db_path=db_path)
    main_module._reset_state(store2)

    response = client.post("/api/facts", json={"question": "What is overfitting?"})
    assert response.status_code == 200
    assert response.json()["count"] > 0

    store2.close()


# ---------------------------------------------------------------------------
# Database failure isolation
# ---------------------------------------------------------------------------

def test_db_failure_does_not_create_inconsistent_state(tmp_path, monkeypatch):
    """If the DB write fails, the document must NOT appear in the list."""
    store = DocumentStore(db_path=tmp_path / "fail_test.db")
    main_module._reset_state(store)

    def raise_db_error(*args, **kwargs):
        raise sqlite3.Error("Simulated database failure")

    monkeypatch.setattr(store, "save_document", raise_db_error)

    response = client.post(
        "/api/documents",
        files={"file": ("notes.txt", _SAMPLE_CONTENT, "text/plain")},
    )
    assert response.status_code == 500

    # The document must not be in the in-memory list.
    list_response = client.get("/api/documents")
    assert list_response.json()["documents"] == []

    # The FactStore must not have indexed anything.
    facts_response = client.post("/api/facts", json={"question": "What is overfitting?"})
    assert facts_response.json()["count"] == 0

    store.close()
