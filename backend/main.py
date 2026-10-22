"""Medha AI – FastAPI backend (v0.2.0).

What's new in Phase 2
---------------------
* Documents are persisted to a local SQLite database
  (``<project root>/data/medha_documents.db``) so they survive restarts.
* At startup the database is read back and the FactStore is rebuilt, so
  ``/api/ask`` and ``/api/facts`` work immediately without re-uploading.
* Maximum upload size raised to 75 MB per file.
* User-facing copy is natural and non-repetitive.

The database file is local-only and is excluded from Git via ``.gitignore``.
"""

from __future__ import annotations

import io
import logging
import re
import sqlite3
import sys
from pathlib import Path
from uuid import uuid4

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so `memory.*` imports work whether the
# file is run directly (uvicorn) or imported from the project root (pytest).
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader

from memory.document_store import DocumentStore
from memory.fact_store import FactStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Medha AI API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_FILE_SIZE: int = 75 * 1024 * 1024  # 75 MB per file
ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".txt", ".md"})

# ---------------------------------------------------------------------------
# Application state
#
# ``documents`` is the in-memory retrieval index (keyed by document id).
# ``fact_store`` is populated at startup from the database and updated on
# every successful upload.
# ``document_store`` owns the SQLite connection; tests replace it via
# ``_reset_state()``.
# ---------------------------------------------------------------------------

documents: dict = {}
fact_store = FactStore()
document_store: DocumentStore | None = None  # initialised lazily by _get_store()


def _get_store() -> DocumentStore:
    """Return the application-level DocumentStore, creating it on first call.

    Deferring construction to the first call (rather than at import time)
    means that test code can call ``_reset_state()`` *before* any store is
    created, avoiding touching the real production database during tests.
    """
    global document_store
    if document_store is None:
        document_store = DocumentStore()
    return document_store


def _reset_state(store: DocumentStore | None = None) -> None:
    """Replace the application state with a clean slate.

    Used by tests to inject a temporary database without touching the real one.

    Args:
        store: A ``DocumentStore`` backed by a temp database, or ``None`` to
               discard any existing store and trigger lazy creation on next use.
    """
    global document_store, documents, fact_store
    document_store = store
    documents.clear()
    fact_store.clear()
    if store is not None:
        _restore_from_db(store)


def _restore_from_db(store: DocumentStore) -> None:
    """Populate in-memory state from the database.

    Loads every document's metadata and page content, then re-indexes them
    into the FactStore so retrieval works immediately after a restart.
    """
    saved = store.load_all()
    for doc in saved:
        documents[doc["id"]] = doc
        fact_store.add_document(doc["name"], doc["content"])
    if saved:
        logger.info("Restored %d document(s) from the local database.", len(saved))


# ---------------------------------------------------------------------------
# Startup: restore persisted documents
#
# We use FastAPI's lifespan event rather than running at import time so that
# importing the module in tests does *not* immediately open the production DB.
# ---------------------------------------------------------------------------

from contextlib import asynccontextmanager  # noqa: E402 – after app definition


@asynccontextmanager
async def lifespan(application: FastAPI):  # noqa: ARG001
    # --- startup ---
    _restore_from_db(_get_store())
    yield
    # --- shutdown (nothing to do) ---


app.router.lifespan_context = lifespan


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    return {"status": "ok", "service": "Medha AI"}


@app.get("/api/documents")
def list_documents():
    """Return document metadata (no page text) for the frontend Knowledge Vault."""
    return {
        "documents": [
            {
                "id": doc["id"],
                "name": doc["name"],
                "pages": doc["pages"],
                "characters": doc["characters"],
                "size_bytes": doc.get("size_bytes", 0),
                "status": doc["status"],
            }
            for doc in documents.values()
        ]
    }


@app.post("/api/documents")
async def upload_document(file: UploadFile = File(...)):
    """Accept a PDF, TXT or MD file up to 75 MB and index it."""
    extension = Path(file.filename or "").suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only PDF, TXT, and MD files are supported.",
        )

    # Read one byte beyond the limit so we can detect oversized files without
    # loading arbitrarily large content into memory beyond MAX_FILE_SIZE + 1.
    content = await file.read(MAX_FILE_SIZE + 1)

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File exceeds 75 MB. Please upload a smaller file.",
        )

    if not content:
        raise HTTPException(status_code=400, detail="The file is empty.")

    size_bytes = len(content)

    # Parse the file content into pages.
    try:
        if extension == ".pdf":
            reader = PdfReader(io.BytesIO(content))

            if reader.is_encrypted:
                raise HTTPException(
                    status_code=400,
                    detail="Password-protected PDFs are not supported yet.",
                )

            pages = [
                {"page": index + 1, "text": page.extract_text() or ""}
                for index, page in enumerate(reader.pages)
            ]
        else:
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError:
                raise HTTPException(
                    status_code=400,
                    detail="Text files must use UTF-8 encoding.",
                )

            pages = [{"page": 1, "text": text}]

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not read this document.",
        )

    document_id = str(uuid4())
    characters = sum(len(p["text"]) for p in pages)
    document = {
        "id": document_id,
        "name": Path(file.filename or "document").name,
        "pages": len(pages),
        "characters": characters,
        "size_bytes": size_bytes,
        "status": "indexed",
        "content": pages,
    }

    # --- Persist first; only mutate in-memory state if persistence succeeds ---
    try:
        _get_store().save_document(
            document_id=document_id,
            name=document["name"],
            pages=pages,
            characters=characters,
            size_bytes=size_bytes,
            status=document["status"],
        )
    except sqlite3.Error as exc:
        logger.error("Database write failed for '%s': %s", document["name"], exc)
        raise HTTPException(
            status_code=500,
            detail="Could not save the document. Please try again.",
        )

    # Persistence succeeded – update the in-memory retrieval state.
    documents[document_id] = document
    fact_store.add_document(document["name"], pages)

    # Return metadata only (no page text).
    return {key: value for key, value in document.items() if key != "content"}


# ---------------------------------------------------------------------------
# Ask / Facts
# ---------------------------------------------------------------------------


class AskRequest(BaseModel):
    question: str


STOP_WORDS = {
    "the", "is", "a", "an", "and", "or", "of", "to", "in",
    "for", "on", "with", "what", "how", "why", "when", "where",
    "who", "which", "are", "was", "were", "do", "does", "did",
    "explain", "tell", "me", "about", "my", "from", "it",
}


def tokenize(text: str) -> set[str]:
    return {
        word
        for word in re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
        if word not in STOP_WORDS and len(word) > 1
    }


@app.post("/api/facts")
def lookup_facts(request: AskRequest):
    results = fact_store.lookup(request.question, limit=3)

    return {
        "method": "zero-llm-fact-lookup",
        "count": len(results),
        "facts": [
            {
                "text": item["text"],
                "document": item["document"],
                "page": item["page"],
                "relevance": item["relevance"],
            }
            for item in results
        ],
    }


@app.post("/api/ask")
def ask_question(request: AskRequest):
    query = tokenize(request.question)

    if not query:
        raise HTTPException(
            status_code=400,
            detail="Please ask a question with meaningful keywords.",
        )

    candidates = []

    for document in documents.values():
        for page in document["content"]:
            page_text = page["text"]

            for sentence in re.split(r"(?<=[.!?])\s+|\n+", page_text):
                sentence = sentence.strip()
                if not sentence:
                    continue

                words = tokenize(sentence)
                if not words:
                    continue

                overlap = query & words
                if not overlap:
                    continue

                # Simple lexical relevance score – not an AI confidence value.
                coverage = len(overlap) / len(query)
                precision = len(overlap) / len(words)

                # Prefer sentences that cover the question without being too broad.
                score = (0.7 * coverage) + (0.3 * precision)

                # Reduce score for short headings / fragments.
                if len(words) < 5:
                    score *= 0.5

                candidates.append(
                    {
                        "text": sentence,
                        "document": document["name"],
                        "page": page["page"],
                        "score": score,
                        "matched_terms": sorted(overlap),
                    }
                )

    candidates.sort(
        key=lambda item: (item["score"], len(item["matched_terms"])),
        reverse=True,
    )

    if not candidates or candidates[0]["score"] < 0.25:
        return {
            "answer": (
                "I couldn't find enough matching evidence in your uploaded "
                "documents. Try rephrasing the question or adding relevant notes."
            ),
            "grounded": False,
            "method": "zero-llm-lexical-retrieval",
            "citations": [],
        }

    best = candidates[:2]

    return {
        "answer": "Here are some useful passages from your notes.",
        "grounded": True,
        "method": "zero-llm-lexical-retrieval",
        "citations": [
            {
                "document": item["document"],
                "page": item["page"],
                "text": item["text"],
                "relevance": round(item["score"], 2),
            }
            for item in best
        ],
    }