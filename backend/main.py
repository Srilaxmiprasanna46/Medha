"""Medha AI – FastAPI backend (v0.3.0).

Phase 3 changes (retrieval quality)
-------------------------------------
* /api/ask now uses passage_chunker instead of a raw sentence split so that
  PDF broken lines are reassembled before scoring.
* Scoring uses coverage + phrase bonus + length bonus.  Precision is no
  longer in the primary score (it penalised long, information-rich passages).
* The top-scoring passage text is returned as the extractive answer, so the
  student sees the actual words from their notes instead of a generic
  "Here are some useful passages" intro.
* Near-duplicate passages are suppressed (Jaccard similarity > 0.65).
* Response shape is unchanged (answer / grounded / citations / method).

Phase 2 features retained
--------------------------
* SQLite persistence – documents survive backend restarts.
* FactStore rebuilt from DB at startup via lifespan.
* Lazy DocumentStore creation so tests can inject a temp store.
* 75 MB upload limit.
"""

from __future__ import annotations

import io
import logging
import math
import re
import sqlite3
import sys
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader

from memory.document_store import DocumentStore
from memory.fact_store import FactStore
from memory.passage_chunker import chunk_page, _count_meaningful_tokens

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Medha AI API", version="0.3.0")

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

MAX_FILE_SIZE: int = 75 * 1024 * 1024
ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".txt", ".md"})

# Retrieval thresholds
_MIN_SCORE: float = 0.20          # below this → not grounded
_DEDUP_JACCARD: float = 0.65      # suppress near-duplicates above this
_MAX_CITATIONS: int = 2           # number of supporting citations shown

# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

documents: dict = {}
fact_store = FactStore()
document_store: DocumentStore | None = None


def _get_store() -> DocumentStore:
    global document_store
    if document_store is None:
        document_store = DocumentStore()
    return document_store


def _reset_state(store: DocumentStore | None = None) -> None:
    """Replace application state with a clean slate (used by tests)."""
    global document_store, documents, fact_store
    document_store = store
    documents.clear()
    fact_store.clear()
    if store is not None:
        _restore_from_db(store)


def _restore_from_db(store: DocumentStore) -> None:
    saved = store.load_all()
    for doc in saved:
        documents[doc["id"]] = doc
        fact_store.add_document(doc["name"], doc["content"])
    if saved:
        logger.info("Restored %d document(s) from the local database.", len(saved))


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(application: FastAPI):  # noqa: ARG001
    _restore_from_db(_get_store())
    yield


app.router.lifespan_context = lifespan

# ---------------------------------------------------------------------------
# Routes – health / documents
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    return {"status": "ok", "service": "Medha AI"}


@app.get("/api/documents")
def list_documents():
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
    extension = Path(file.filename or "").suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only PDF, TXT, and MD files are supported.",
        )

    content = await file.read(MAX_FILE_SIZE + 1)

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File exceeds 75 MB. Please upload a smaller file.",
        )

    if not content:
        raise HTTPException(status_code=400, detail="The file is empty.")

    size_bytes = len(content)

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
        raise HTTPException(status_code=400, detail="Could not read this document.")

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

    documents[document_id] = document
    fact_store.add_document(document["name"], pages)

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


def _extract_bigrams(token_set: set[str], text: str) -> set[tuple[str, str]]:
    """Return all adjacent meaningful-word pairs in *text* (order-sensitive)."""
    words = [
        w for w in re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
        if w not in STOP_WORDS and len(w) > 1
    ]
    return {(words[i], words[i + 1]) for i in range(len(words) - 1)}


def _phrase_bonus(query_tokens: set[str], query_text: str, passage_text: str) -> float:
    """Return a 0–1 bonus for exact query bigrams found in the passage."""
    query_bigrams = _extract_bigrams(query_tokens, query_text)
    if not query_bigrams:
        return 0.0
    passage_bigrams = _extract_bigrams(set(), passage_text)
    matched = query_bigrams & passage_bigrams
    return len(matched) / len(query_bigrams)


def _length_bonus(passage_text: str) -> float:
    """Log-normalised bonus that rewards substantive passages.

    Returns a value in [0, 1].  A passage of ~200 tokens scores ~1.0;
    shorter passages score lower; single-sentence snippets score near 0.
    """
    token_count = _count_meaningful_tokens(passage_text)
    if token_count <= 0:
        return 0.0
    # log2(token_count) / log2(200) clamped to [0, 1]
    return min(1.0, math.log2(max(1, token_count)) / math.log2(200))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union)


def _deduplicate(
    candidates: list[dict],
    threshold: float = _DEDUP_JACCARD,
) -> list[dict]:
    """Remove passages that are near-duplicates of an already-selected result."""
    selected: list[dict] = []
    selected_terms: list[set[str]] = []

    for cand in candidates:
        cand_terms = set(cand["matched_terms"])
        if any(
            _jaccard(cand_terms, seen_terms) > threshold
            for seen_terms in selected_terms
        ):
            continue
        selected.append(cand)
        selected_terms.append(cand_terms)

    return selected


def _score_passage(
    query: set[str],
    query_text: str,
    passage_text: str,
) -> float:
    """Compute a relevance score for a single passage.

    Formula (all components in [0, 1]):
        score = 0.60 * coverage
              + 0.25 * phrase_bonus
              + 0.15 * length_bonus

    Coverage: fraction of query terms found in the passage.
    Phrase bonus: fraction of query bigrams found in the passage.
    Length bonus: log-normalised passage length (rewards substance).

    Heading penalty: passages with fewer than 8 meaningful tokens have their
    score halved, so they only win if nothing better matches.
    """
    passage_tokens = tokenize(passage_text)
    if not passage_tokens:
        return 0.0

    overlap = query & passage_tokens
    if not overlap:
        return 0.0

    coverage = len(overlap) / len(query)
    pb = _phrase_bonus(query, query_text, passage_text)
    lb = _length_bonus(passage_text)

    score = 0.60 * coverage + 0.25 * pb + 0.15 * lb

    # Penalise very short passages (headings, labels, fragments).
    if _count_meaningful_tokens(passage_text) < 8:
        score *= 0.4

    return score


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
            page_number = page["page"]
            page_text = page["text"]

            # Use the chunker so PDF broken lines are reassembled.
            for passage in chunk_page(page_text, page_number):
                score = _score_passage(query, request.question, passage.text)
                if score <= 0:
                    continue

                passage_tokens = tokenize(passage.text)
                overlap = query & passage_tokens

                candidates.append(
                    {
                        "text": passage.text,
                        "document": document["name"],
                        "page": passage.page,
                        "score": score,
                        "matched_terms": sorted(overlap),
                    }
                )

    # Sort by score descending, then by matched term count as tie-breaker.
    candidates.sort(
        key=lambda c: (c["score"], len(c["matched_terms"])),
        reverse=True,
    )

    # Remove near-duplicate passages.
    candidates = _deduplicate(candidates)

    if not candidates or candidates[0]["score"] < _MIN_SCORE:
        return {
            "answer": (
                "I couldn't find enough information in your notes to answer that. "
                "Try rephrasing, or add relevant study material."
            ),
            "grounded": False,
            "method": "zero-llm-lexical-retrieval",
            "citations": [],
        }

    best = candidates[0]
    # Supporting citations: up to _MAX_CITATIONS additional distinct passages.
    supporting = candidates[1 : _MAX_CITATIONS + 1]

    # The answer IS the best passage — extractive, not a generic intro.
    answer_text = best["text"]

    return {
        "answer": answer_text,
        "grounded": True,
        "method": "zero-llm-lexical-retrieval",
        "citations": [
            {
                "document": item["document"],
                "page": item["page"],
                "text": item["text"],
                "relevance": round(item["score"], 2),
            }
            for item in supporting
        ],
    }