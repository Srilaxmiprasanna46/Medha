"""Retrieval quality tests for Medha AI Phase 3.

All tests use synthetic documents embedded in this file.  They do not depend
on any personal PDF files and will always produce deterministic results.

The tests verify that the improved chunker + scoring combination:
  1. Returns a meaningful definition rather than a heading for "What is Big Data?"
  2. Returns the 5Vs explanation for a 5Vs question.
  3. Handles PDF-style broken lines correctly.
  4. Penalises heading fragments below substantive passages.
  5. Returns not-grounded for unrelated questions.
  6. Preserves the correct document name and page number in citations.
  7. Suppresses near-duplicate passages.
"""

import pytest
from fastapi.testclient import TestClient

import backend.main as main_module
from backend.main import app
from memory.document_store import DocumentStore
from memory.fact_store import FactStore
from memory.passage_chunker import chunk_page

client = TestClient(app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# Fixture: isolated temp store for every test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_store(tmp_path):
    store = DocumentStore(db_path=tmp_path / "quality_test.db")
    main_module._reset_state(store)
    yield
    store.close()
    main_module._reset_state(None)


# ---------------------------------------------------------------------------
# Synthetic documents
# ---------------------------------------------------------------------------

# Mimics a realistic Big Data PDF with a one-line heading followed by
# paragraph body text split across multiple lines (as PDF extractors emit).
BIG_DATA_PAGE_1 = """\
Introduction to Big Data

Big Data refers to extremely large, diverse, and complex datasets that cannot
be processed or analysed using traditional data-processing tools and
techniques. These datasets are characterised by their volume, velocity,
and variety, requiring advanced technologies and architectures.

Big Data is often defined as:
data that is too large, too fast, or too varied
for conventional systems to handle efficiently.

Big Data technologies include distributed computing frameworks such as
Hadoop and Spark, which allow organisations to store and process petabytes
of data across clusters of commodity hardware.
"""

BIG_DATA_PAGE_2 = """\
Traditional Systems and Their Limitations

Why Traditional Systems Fail with Big Data

Relational databases such as MySQL were designed for structured data
and cannot horizontally scale to petabytes. They struggle with the
unstructured formats common in social media, sensor data, and logs.

The 5Vs of Big Data

Volume refers to the massive scale of data generated every second.
Velocity describes the speed at which new data is created and must be
processed in near real time. Variety captures the many different forms
data can take — structured tables, images, video, text, and sensor
readings. Veracity addresses the trustworthiness and accuracy of the
data. Value is the ultimate goal: extracting actionable insights that
justify the cost of collection and processing.
"""

UNRELATED_PAGE = """\
Chapter 4: Organic Chemistry

Alkanes are saturated hydrocarbons with the general formula CnH2n+2.
Methane, the simplest alkane, consists of one carbon atom bonded to four
hydrogen atoms. Ethane has two carbons, propane has three, and so on.
The IUPAC naming convention assigns systematic names to all organic
compounds based on their structure and functional groups.
"""


def _upload_text(name: str, text: str, page: int = 1) -> dict:
    content = text.encode("utf-8")
    response = client.post(
        "/api/documents",
        files={"file": (name, content, "text/plain")},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _ask(question: str) -> dict:
    response = client.post("/api/ask", json={"question": question})
    assert response.status_code == 200, response.text
    return response.json()


# ---------------------------------------------------------------------------
# Test 1 – Definition ranked above heading
# ---------------------------------------------------------------------------

def test_definition_ranked_above_heading():
    """'What is Big Data?' should return the substantive definition, not a
    short heading or sentence fragment like 'Big Data is often defined as:'."""
    _upload_text("big_data_notes.txt", BIG_DATA_PAGE_1)
    data = _ask("What is Big Data?")

    assert data["grounded"] is True, "Expected a grounded answer"
    answer = data["answer"].lower()

    # The answer should contain substance about what Big Data IS.
    # Either the definition paragraph or the 5Vs characteristics line.
    has_substance = (
        "extremely large" in answer
        or "cannot be processed" in answer
        or "datasets" in answer
        or "volume" in answer
    )
    assert has_substance, (
        f"Answer appears to be a heading or fragment rather than a definition.\n"
        f"Got: {data['answer']!r}"
    )

    # The answer must NOT be just the bare heading text.
    assert data["answer"].strip() != "Big Data is often defined as:", (
        "Answer is the bare heading, not a definition."
    )


# ---------------------------------------------------------------------------
# Test 2 – 5Vs retrieved correctly
# ---------------------------------------------------------------------------

def test_five_vs_retrieved():
    """A question about the 5Vs should return the passage that explains
    Volume, Velocity, Variety, Veracity and Value."""
    _upload_text("big_data_notes.txt", BIG_DATA_PAGE_2)
    data = _ask("What are the 5Vs of Big Data?")

    assert data["grounded"] is True, "Expected a grounded answer"
    answer = data["answer"].lower()

    # At least three of the five Vs must appear in the answer.
    vs_found = sum(
        1 for v in ["volume", "velocity", "variety", "veracity", "value"]
        if v in answer
    )
    assert vs_found >= 3, (
        f"Expected at least 3 of the 5Vs in the answer, found {vs_found}.\n"
        f"Got: {data['answer']!r}"
    )


# ---------------------------------------------------------------------------
# Test 3 – PDF broken-line text is reassembled
# ---------------------------------------------------------------------------

def test_pdf_linebreak_reconstruction():
    """Text with PDF-style line breaks should be reassembled into a
    meaningful passage by the chunker, not returned as fragments."""
    broken_text = (
        "Machine learning is a branch of artificial intelligence that\n"
        "enables computers to learn from data without being explicitly\n"
        "programmed. It uses statistical techniques to build models that\n"
        "generalise from training examples to unseen cases.\n"
        "\n"
        "ML\n"
    )
    _upload_text("ml_notes.txt", broken_text)
    data = _ask("What is machine learning?")

    assert data["grounded"] is True
    answer = data["answer"].lower()

    # The rejoined definition paragraph should be the answer.
    assert "artificial intelligence" in answer or "statistical" in answer or "computers" in answer, (
        f"Broken lines were not reassembled.\nGot: {data['answer']!r}"
    )
    # Must not return "ML" alone (a 1-token fragment).
    assert data["answer"].strip() != "ML", "Returned single-token fragment 'ML'"


# ---------------------------------------------------------------------------
# Test 4 – Heading fragment does not outrank substantive passage
# ---------------------------------------------------------------------------

def test_heading_fragment_does_not_outrank_passage():
    """When a document contains both a heading and a substantive paragraph,
    the paragraph should be returned as the primary answer.

    We ask about 'relational database scaling' which uses terms that appear
    in the substantive paragraph ('relational', 'scale', 'structured'),
    so it should return the paragraph, not a short heading-only fragment.
    """
    _upload_text("big_data_notes.txt", BIG_DATA_PAGE_2)
    data = _ask("How do relational databases struggle to scale with structured data?")

    assert data["grounded"] is True, (
        f"Expected grounded answer.\nGot: {data}"
    )
    answer = data["answer"]

    # The substantive paragraph about relational databases should win.
    # It must be longer than any heading fragment alone.
    assert len(answer) > 50, (
        f"Answer is too short to be the substantive paragraph.\nGot: {answer!r}"
    )
    answer_lower = answer.lower()
    has_substance = (
        "relational" in answer_lower
        or "scale" in answer_lower
        or "structured" in answer_lower
        or "mysql" in answer_lower
        or "petabytes" in answer_lower
    )
    assert has_substance, (
        f"Answer lacks substance about relational database limitations.\n"
        f"Got: {answer!r}"
    )



# ---------------------------------------------------------------------------
# Test 5 – Unrelated question returns not-grounded
# ---------------------------------------------------------------------------

def test_unrelated_question_returns_not_grounded():
    """A question about organic chemistry should not be answered from Big
    Data notes."""
    _upload_text("big_data_notes.txt", BIG_DATA_PAGE_1)
    data = _ask("What is an alkane in organic chemistry?")

    assert data["grounded"] is False
    assert data["citations"] == []


# ---------------------------------------------------------------------------
# Test 6 – Citations preserve correct document name and page number
# ---------------------------------------------------------------------------

def test_citations_preserve_document_and_page():
    """Every citation returned must carry the correct document filename
    and a valid page number."""
    _upload_text("big_data_notes.txt", BIG_DATA_PAGE_2)
    data = _ask("What are the 5Vs of Big Data?")

    assert data["grounded"] is True

    # The primary answer itself must reference the right document.
    # Supporting citations (if any) must also reference it.
    for citation in data.get("citations", []):
        assert citation["document"] == "big_data_notes.txt", (
            f"Citation has wrong document: {citation['document']!r}"
        )
        assert isinstance(citation["page"], int), (
            f"Citation page is not an int: {citation['page']!r}"
        )
        assert citation["page"] >= 1


# ---------------------------------------------------------------------------
# Test 7 – Duplicate passage suppression
# ---------------------------------------------------------------------------

def test_duplicate_passages_suppressed():
    """Uploading the same document twice should not produce duplicate
    citations for the same passage in a single response."""
    # Upload the same content twice under the same name (simulates re-index).
    _upload_text("notes_a.txt", BIG_DATA_PAGE_1)
    _upload_text("notes_b.txt", BIG_DATA_PAGE_1)

    data = _ask("What is Big Data?")
    assert data["grounded"] is True

    all_passages = [data["answer"]] + [c["text"] for c in data.get("citations", [])]

    # No two returned passages should be identical.
    assert len(all_passages) == len(set(all_passages)), (
        "Duplicate passages appear in the response."
    )


# ---------------------------------------------------------------------------
# Test 8 – Multi-document retrieval returns the most relevant document
# ---------------------------------------------------------------------------

def test_relevant_document_ranked_above_unrelated():
    """When two documents are uploaded, the answer to a Big Data question
    should come from the Big Data notes, not the chemistry notes."""
    _upload_text("big_data_notes.txt", BIG_DATA_PAGE_1)
    _upload_text("chemistry_notes.txt", UNRELATED_PAGE)

    data = _ask("What is Big Data?")
    assert data["grounded"] is True

    # The answer text itself should be from the Big Data document.
    answer_lower = data["answer"].lower()
    assert (
        "extremely large" in answer_lower
        or "datasets" in answer_lower
        or "volume" in answer_lower
        or "hadoop" in answer_lower
    ), f"Answer came from the wrong document.\nGot: {data['answer']!r}"


# ---------------------------------------------------------------------------
# Chunker unit tests
# ---------------------------------------------------------------------------

def test_chunker_rejoins_broken_pdf_lines():
    """Lines that continue across a newline should be joined."""
    text = (
        "Big Data refers to extremely large datasets that\n"
        "cannot be processed by traditional tools.\n"
        "\n"
        "Heading\n"
    )
    passages = chunk_page(text, page_number=1)
    assert passages, "Expected at least one passage"

    full_text = " ".join(p.text for p in passages)
    assert "extremely large datasets" in full_text
    assert "cannot be processed" in full_text


def test_chunker_drops_short_fragments():
    """Headings and short fragments should be filtered out."""
    text = (
        "Introduction\n"
        "\n"
        "Big Data refers to extremely large, diverse, and complex datasets "
        "that require advanced technologies for processing and analysis. "
        "These datasets grow beyond the capability of conventional tools.\n"
    )
    passages = chunk_page(text, page_number=1)
    texts = [p.text for p in passages]

    # "Introduction" is a 1-token heading — should be dropped.
    assert not any(t.strip() == "Introduction" for t in texts), (
        f"Heading 'Introduction' survived the filter: {texts}"
    )
    # The substantive paragraph should survive.
    assert any("extremely large" in t for t in texts), (
        f"Substantive paragraph was dropped: {texts}"
    )


def test_chunker_preserves_page_number():
    """Passages must carry the page number they were extracted from."""
    text = (
        "Supervised learning uses labelled training data to build predictive models. "
        "The model learns to map input features to output labels by minimising "
        "a loss function over the training set."
    )
    passages = chunk_page(text, page_number=5)
    assert passages
    assert all(p.page == 5 for p in passages)


def test_chunker_windows_long_paragraphs():
    """A very long paragraph should be split into multiple windows."""
    # Build a paragraph that is well over WINDOW_CHARS (600 chars).
    sentences = [
        f"Sentence {i}: This discusses concept number {i} in detail with several meaningful words."
        for i in range(1, 20)
    ]
    long_text = " ".join(sentences)
    passages = chunk_page(long_text, page_number=1)
    # Should produce more than one passage for a very long paragraph.
    assert len(passages) > 1, (
        f"Expected windowing for long paragraph, got {len(passages)} passage(s)."
    )
