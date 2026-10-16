from pathlib import Path
from uuid import uuid4
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memory.fact_store import FactStore
import re
from pydantic import BaseModel

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader

app = FastAPI(title="Medha AI API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}
documents = {}
fact_store = FactStore()

@app.get("/health")
def health():
    return {"status": "ok", "service": "Medha AI"}


@app.get("/api/documents")
def list_documents():
    return {"documents": list(documents.values())}


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
        raise HTTPException(status_code=413, detail="File exceeds 10 MB.")

    if not content:
        raise HTTPException(status_code=400, detail="The file is empty.")

    try:
        if extension == ".pdf":
            import io
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
    document = {
        "id": document_id,
        "name": Path(file.filename or "document").name,
        "pages": len(pages),
        "characters": sum(len(page["text"]) for page in pages),
        "status": "indexed",
        "content": pages,
    }

    documents[document_id] = document
    fact_store.add_document(document["name"], pages)

    return {
        key: value
        for key, value in document.items()
        if key != "content"
    }
class AskRequest(BaseModel):
    question: str


STOP_WORDS = {
    "the", "is", "a", "an", "and", "or", "of", "to", "in",
    "for", "on", "with", "what", "how", "why", "when", "where",
    "who", "which", "are", "was", "were", "do", "does", "did",
    "explain", "tell", "me", "about", "my", "from", "it"
}


def tokenize(text: str) -> set[str]:
    return {
        word for word in re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
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
            detail="Please ask a question with meaningful keywords."
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

                # Simple lexical relevance score, not AI confidence.
                coverage = len(overlap) / len(query)
                precision = len(overlap) / len(words)

                # Prefer sentences that cover the question without being too broad.
                score = (0.7 * coverage) + (0.3 * precision)

                # Reduce the score for short headings and fragments.
                if len(words) < 5:
                    score *= 0.5

                candidates.append({
                    "text": sentence,
                    "document": document["name"],
                    "page": page["page"],
                    "score": score,
                    "matched_terms": sorted(overlap),
                })

    candidates.sort(
        key=lambda item: (item["score"], len(item["matched_terms"])),
        reverse=True
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
        "answer": "Here are the most relevant passages from your documents.",
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