"""Unit tests for memory.document_store (DocumentStore).

All tests use a temporary database so they are fully isolated from the real
study library and from each other.
"""

import sqlite3
import threading
from pathlib import Path

import pytest

from memory.document_store import DocumentStore

_SAMPLE_PAGES = [
    {"page": 1, "text": "Supervised learning uses labelled training data."},
    {"page": 2, "text": "Overfitting occurs when a model learns training data too closely."},
]


@pytest.fixture()
def store(tmp_path: Path) -> DocumentStore:
    """Provide a fresh DocumentStore backed by a temp database."""
    s = DocumentStore(db_path=tmp_path / "test.db")
    yield s
    s.close()


# ---------------------------------------------------------------------------
# Schema bootstrap
# ---------------------------------------------------------------------------

def test_schema_created_on_init(tmp_path):
    """Initialising the store must create the expected tables."""
    db_path = tmp_path / "schema.db"
    store = DocumentStore(db_path=db_path)
    conn = sqlite3.connect(str(db_path))
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    conn.close()
    store.close()
    assert "documents" in tables
    assert "document_pages" in tables


def test_schema_init_is_idempotent(tmp_path):
    """Opening the same database twice must not raise."""
    db_path = tmp_path / "idem.db"
    s1 = DocumentStore(db_path=db_path)
    s1.close()
    s2 = DocumentStore(db_path=db_path)
    s2.close()  # should not raise


# ---------------------------------------------------------------------------
# save_document
# ---------------------------------------------------------------------------

def test_save_document_is_stored(store):
    store.save_document("doc-1", "ml.txt", _SAMPLE_PAGES, 90, 500)
    rows = store.list_metadata()
    assert len(rows) == 1
    assert rows[0]["id"] == "doc-1"
    assert rows[0]["name"] == "ml.txt"
    assert rows[0]["pages"] == 2
    assert rows[0]["size_bytes"] == 500


def test_save_document_stores_page_content(store):
    store.save_document("doc-1", "ml.txt", _SAMPLE_PAGES, 90, 500)
    docs = store.load_all()
    assert len(docs) == 1
    content = docs[0]["content"]
    assert len(content) == 2
    assert content[0]["page"] == 1
    assert "Supervised" in content[0]["text"]
    assert content[1]["page"] == 2
    assert "Overfitting" in content[1]["text"]


def test_save_document_atomicity_on_duplicate_id(store):
    """Inserting a duplicate id must fail without partial data."""
    store.save_document("doc-1", "first.txt", _SAMPLE_PAGES, 90, 100)

    with pytest.raises(sqlite3.Error):
        store.save_document("doc-1", "second.txt", _SAMPLE_PAGES, 90, 100)

    # The second call must not have committed any data.
    rows = store.list_metadata()
    assert len(rows) == 1
    assert rows[0]["name"] == "first.txt"


# ---------------------------------------------------------------------------
# load_all
# ---------------------------------------------------------------------------

def test_load_all_empty(store):
    assert store.load_all() == []


def test_load_all_returns_full_shape(store):
    store.save_document("doc-1", "notes.txt", _SAMPLE_PAGES, 90, 400)
    docs = store.load_all()
    assert len(docs) == 1
    doc = docs[0]
    assert doc["id"] == "doc-1"
    assert doc["name"] == "notes.txt"
    assert doc["pages"] == 2
    assert doc["characters"] == 90
    assert doc["size_bytes"] == 400
    assert doc["status"] == "indexed"
    assert isinstance(doc["content"], list)


def test_load_all_multiple_documents(store):
    store.save_document("a", "a.txt", [{"page": 1, "text": "Alpha"}], 5, 10)
    store.save_document("b", "b.txt", [{"page": 1, "text": "Beta"}], 4, 10)
    docs = store.load_all()
    assert len(docs) == 2
    names = {d["name"] for d in docs}
    assert names == {"a.txt", "b.txt"}


# ---------------------------------------------------------------------------
# list_metadata
# ---------------------------------------------------------------------------

def test_list_metadata_excludes_page_text(store):
    store.save_document("doc-1", "ml.txt", _SAMPLE_PAGES, 90, 500)
    rows = store.list_metadata()
    assert len(rows) == 1
    assert "content" not in rows[0]
    assert "text" not in rows[0]


# ---------------------------------------------------------------------------
# exists
# ---------------------------------------------------------------------------

def test_exists_true_after_save(store):
    store.save_document("doc-1", "ml.txt", _SAMPLE_PAGES, 90, 500)
    assert store.exists("doc-1") is True


def test_exists_false_for_unknown_id(store):
    assert store.exists("no-such-id") is False


# ---------------------------------------------------------------------------
# Persistence across connections
# ---------------------------------------------------------------------------

def test_data_survives_new_connection(tmp_path):
    """Data saved through one connection must be readable via a new connection."""
    db_path = tmp_path / "persist.db"

    store1 = DocumentStore(db_path=db_path)
    store1.save_document("doc-1", "notes.txt", _SAMPLE_PAGES, 90, 300)
    store1.close()

    store2 = DocumentStore(db_path=db_path)
    docs = store2.load_all()
    store2.close()

    assert len(docs) == 1
    assert docs[0]["name"] == "notes.txt"
    assert len(docs[0]["content"]) == 2


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

def test_concurrent_writes_do_not_corrupt(tmp_path):
    """Multiple threads writing to the same store must not corrupt data."""
    db_path = tmp_path / "threads.db"
    store = DocumentStore(db_path=db_path)
    errors = []

    def write(doc_id: str, name: str):
        try:
            store.save_document(
                doc_id, name, [{"page": 1, "text": "Hello"}], 5, 10
            )
        except Exception as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=write, args=(f"doc-{i}", f"doc{i}.txt"))
        for i in range(10)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    store.close()

    # Re-open and count; no errors, all 10 documents persisted.
    assert errors == [], f"Thread errors: {errors}"
    store2 = DocumentStore(db_path=db_path)
    docs = store2.load_all()
    store2.close()
    assert len(docs) == 10
