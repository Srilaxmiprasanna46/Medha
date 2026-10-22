"""
Persistent SQLite-backed document store for Medha AI.

Documents (metadata + extracted page text) are saved to a local SQLite
database so they survive backend restarts.  The database file is stored on the
user's machine; it is never sent to any external service.

SQLite is part of Python's standard library – no extra dependencies needed.

Thread-safety model
-------------------
FastAPI runs synchronous endpoint functions in a thread-pool executor.
``sqlite3.connect`` is **not** thread-safe with a shared connection object by
default (raises ``ProgrammingError: SQLite objects created in a thread can
only be used in that same thread``).

This module resolves that by:
1. Passing ``check_same_thread=False`` to sqlite3.connect so that Python's
   per-thread check is lifted.
2. Holding a ``threading.Lock`` around every write operation so that only one
   thread can mutate the database at a time.
3. Using WAL (write-ahead log) journal mode so that concurrent readers do not
   block each other and do not block the writer.

Read operations (``load_all``, ``list_metadata``, ``exists``) are safe to run
concurrently in WAL mode without the lock because SQLite guarantees snapshot
isolation for reads.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default location: <project root>/data/medha_documents.db
# Callers can override this by passing a different path to DocumentStore().
_DEFAULT_DB_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "medha_documents.db"
)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    pages       INTEGER NOT NULL DEFAULT 0,
    characters  INTEGER NOT NULL DEFAULT 0,
    size_bytes  INTEGER NOT NULL DEFAULT 0,
    status      TEXT NOT NULL DEFAULT 'indexed',
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS document_pages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page        INTEGER NOT NULL,
    text        TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_pages_document_id
    ON document_pages(document_id);
"""


def _open_connection(db_path: Path) -> sqlite3.Connection:
    """Open (or create) a SQLite database file and configure it."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        str(db_path),
        check_same_thread=False,  # we handle thread safety ourselves via a Lock
    )
    conn.row_factory = sqlite3.Row
    # WAL mode: readers don't block writers; writers don't block readers.
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


class DocumentStore:
    """SQLite-backed store for uploaded document records.

    A single connection is shared across threads.  All write operations are
    serialised with ``self._lock``.  Read operations run without the lock
    because WAL mode provides snapshot-consistent reads.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path: Path = db_path or _DEFAULT_DB_PATH
        self._lock = threading.Lock()
        self._conn = _open_connection(self._db_path)
        self._initialise_schema()
        logger.info("DocumentStore ready: %s", self._db_path)

    # ------------------------------------------------------------------
    # Schema bootstrap
    # ------------------------------------------------------------------

    def _initialise_schema(self) -> None:
        """Create tables if they do not already exist (idempotent)."""
        with self._lock:
            # executescript commits any open transaction first, then runs the
            # script in its own implicit transaction.
            self._conn.executescript(_SCHEMA_SQL)

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def save_document(
        self,
        document_id: str,
        name: str,
        pages: list[dict[str, Any]],
        characters: int,
        size_bytes: int,
        status: str = "indexed",
    ) -> None:
        """Persist a document and its pages atomically.

        Raises ``sqlite3.Error`` on failure.  The caller must *not* update
        the in-memory state when this raises, so that in-memory and on-disk
        state remain consistent.
        """
        page_tuples = [
            (document_id, p["page"], p.get("text", "")) for p in pages
        ]

        with self._lock:
            # The connection's context manager issues BEGIN / COMMIT / ROLLBACK.
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO documents
                        (id, name, pages, characters, size_bytes, status)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (document_id, name, len(pages), characters, size_bytes, status),
                )
                self._conn.executemany(
                    """
                    INSERT INTO document_pages (document_id, page, text)
                    VALUES (?, ?, ?)
                    """,
                    page_tuples,
                )

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def load_all(self) -> list[dict[str, Any]]:
        """Return every saved document together with its page content.

        Shape of each item::

            {
                "id":         str,
                "name":       str,
                "pages":      int,
                "characters": int,
                "size_bytes": int,
                "status":     str,
                "content":    [{"page": int, "text": str}, ...]
            }

        This matches the in-memory ``documents`` dict values used by
        ``backend/main.py``.
        """
        doc_rows = self._conn.execute(
            """
            SELECT id, name, pages, characters, size_bytes, status
            FROM   documents
            ORDER  BY created_at
            """
        ).fetchall()

        result: list[dict[str, Any]] = []
        for row in doc_rows:
            doc_id = row["id"]
            page_rows = self._conn.execute(
                """
                SELECT page, text
                FROM   document_pages
                WHERE  document_id = ?
                ORDER  BY page
                """,
                (doc_id,),
            ).fetchall()
            content = [{"page": pr["page"], "text": pr["text"]} for pr in page_rows]
            result.append(
                {
                    "id": doc_id,
                    "name": row["name"],
                    "pages": row["pages"],
                    "characters": row["characters"],
                    "size_bytes": row["size_bytes"],
                    "status": row["status"],
                    "content": content,
                }
            )
        return result

    def list_metadata(self) -> list[dict[str, Any]]:
        """Return document metadata rows without page text."""
        rows = self._conn.execute(
            """
            SELECT id, name, pages, characters, size_bytes, status
            FROM   documents
            ORDER  BY created_at
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def exists(self, document_id: str) -> bool:
        """Return True if a document with this id is stored."""
        row = self._conn.execute(
            "SELECT 1 FROM documents WHERE id = ?", (document_id,)
        ).fetchone()
        return row is not None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()
