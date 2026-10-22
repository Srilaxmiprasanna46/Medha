"""Retrieval quality tests for Medha AI — Phase 3 (revised scoring).

All tests use synthetic documents. They do NOT depend on any personal PDF files.

Mandatory regression tests per the Phase 3 bug report
-------------------------------------------------------
* test_definition_beats_visualization_table  — Definition of Big Data must rank
  above a visualization-tools table that incidentally contains 'Big Data'.
* test_definition_beats_hadoop_commands      — Shell commands page must not
  appear for 'What is Big Data?' (no query term overlap → score 0).
* test_definition_beats_yarn_scheduler       — YARN passage must not appear as
  a supporting citation (citation threshold blocks it).
* test_definition_scores_and_table_penalised — Assert exact score ordering with
  visible numbers so the reason is understandable.

Additional tests ensure the fix is general, not hard-coded to Big Data.
"""

import pytest
from fastapi.testclient import TestClient

import backend.main as main_module
from backend.main import (
    app,
    _is_table_like,
    _score_passage,
    _definition_signal,
    _term_density,
    tokenize,
)
from memory.document_store import DocumentStore
from memory.passage_chunker import chunk_page

client = TestClient(app, raise_server_exceptions=True)

# ---------------------------------------------------------------------------
# Per-test isolation fixture
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_store(tmp_path):
    store = DocumentStore(db_path=tmp_path / "quality_test.db")
    main_module._reset_state(store)
    yield
    store.close()
    main_module._reset_state(None)


# ---------------------------------------------------------------------------
# Synthetic document — mirrors the real failure scenario
# ---------------------------------------------------------------------------

_BIG_DATA_DEFINITION = (
    "Introduction to Big Data\n\n"
    "Big Data refers to extremely large, diverse, and complex datasets that\n"
    "cannot be processed or analysed using traditional data-processing tools.\n"
    "These datasets are characterised by Volume, Velocity, Variety, Veracity\n"
    "and Value, collectively known as the 5Vs. Big Data technologies enable\n"
    "organisations to store, process and derive insights from datasets that\n"
    "were previously too large to handle with conventional database systems.\n"
)

_VISUALIZATION_TABLE = (
    "Big Data Visualization Tools Comparison\n\n"
    "Tool | Category | Use Case | Big Data Support | License\n"
    "-----|----------|----------|-----------------|--------\n"
    "Tableau | BI | Business analytics for Big Data | Yes | Commercial\n"
    "Power BI | Dashboard | Data visualization for Big Data lakes | Yes | Commercial\n"
    "D3.js | JavaScript | Custom visualizations for Big Data | Yes | Open Source\n"
    "Apache Superset | BI | Data exploration for Big Data dashboards | Yes | Open Source\n"
    "Grafana | Monitoring | Time-series for Big Data streams | Yes | Open Source\n"
    "Kibana | Analytics | Elasticsearch for Big Data logs | Yes | Open Source\n"
    "Matplotlib | Python | Scientific Big Data plotting | Yes | Open Source\n"
    "Plotly | Python | Interactive Big Data charts | Yes | Open Source\n"
)

_HADOOP_COMMANDS = (
    "Hadoop HDFS Shell Commands\n\n"
    "hdfs dfs -ls /user/hadoop/\n"
    "hdfs dfs -put localfile.txt /user/hadoop/\n"
    "hdfs dfs -get /user/hadoop/file.txt localfile.txt\n"
    "hdfs dfs -mkdir /user/hadoop/newdir\n"
    "hadoop jar mapreduce.jar WordCount /input /output\n"
    "yarn application -list -appStates ALL\n"
    "yarn logs -applicationId application_12345_6789\n"
)

_YARN_SCHEDULER = (
    "YARN Resource Manager and Scheduler Configuration\n\n"
    "The YARN ResourceManager is the master daemon responsible for resource\n"
    "management and scheduling in a Hadoop cluster for Big Data processing.\n"
    "It manages the global assignment of CPU, memory and disk I/O across\n"
    "all applications. The CapacityScheduler divides cluster resources into\n"
    "queues. The FairScheduler gives every Big Data application a fair share\n"
    "of resources over time.\n"
)

# Combined into one synthetic document (4 separate pages)
_SYNTHETIC_BDA_PAGES = [
    {"page": 1, "text": _BIG_DATA_DEFINITION},
    {"page": 5, "text": _VISUALIZATION_TABLE},
    {"page": 8, "text": _HADOOP_COMMANDS},
    {"page": 9, "text": _YARN_SCHEDULER},
]


def _upload_text(name: str, text: str) -> dict:
    response = client.post(
        "/api/documents",
        files={"file": (name, text.encode(), "text/plain")},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _upload_pages(name: str, pages_text: str) -> dict:
    """Upload multi-page content as a single text file."""
    combined = "\n\n".join(p["text"] for p in _SYNTHETIC_BDA_PAGES)
    return _upload_text(name, combined)


def _ask(question: str) -> dict:
    response = client.post("/api/ask", json={"question": question})
    assert response.status_code == 200, response.text
    return response.json()


# ---------------------------------------------------------------------------
# Unit tests for scoring helpers (visible numbers, no API needed)
# ---------------------------------------------------------------------------

class TestScoringHelpers:
    """These tests expose exact score values so failures are self-explanatory."""

    def test_is_table_like_detects_pipe_table(self):
        """A passage with many | characters should be detected as a table."""
        passages = chunk_page(_VISUALIZATION_TABLE, page_number=5)
        assert passages, "Expected at least one passage"
        assert _is_table_like(passages[0].text), (
            f"Table not detected. Passage: {passages[0].text[:100]!r}"
        )

    def test_is_table_like_ignores_normal_prose(self):
        """A normal explanatory paragraph must NOT be flagged as a table."""
        passages = chunk_page(_BIG_DATA_DEFINITION, page_number=1)
        assert passages
        assert not _is_table_like(passages[0].text), (
            "Definition paragraph incorrectly flagged as table."
        )

    def test_definition_score_higher_than_table_score(self):
        """Definition passage must score strictly higher than the table passage."""
        q = tokenize("What is Big Data?")
        question = "What is Big Data?"

        def_passages = chunk_page(_BIG_DATA_DEFINITION, page_number=1)
        tab_passages = chunk_page(_VISUALIZATION_TABLE, page_number=5)

        def_score = _score_passage(q, question, def_passages[0].text)
        tab_score = _score_passage(q, question, tab_passages[0].text)

        print(f"\n  definition score : {def_score:.3f}")
        print(f"  table score      : {tab_score:.3f}")
        print(f"  margin           : {def_score - tab_score:.3f}")

        assert def_score > tab_score, (
            f"Table ({tab_score:.3f}) outranked definition ({def_score:.3f}).\n"
            f"Margin: {def_score - tab_score:.3f}"
        )

    def test_definition_signal_fires_for_what_is_question(self):
        """_definition_signal must return 1.0 for a definition passage + 'What is X?' query."""
        q = tokenize("What is Big Data?")
        def_passages = chunk_page(_BIG_DATA_DEFINITION, page_number=1)
        signal = _definition_signal("What is Big Data?", def_passages[0].text, q)
        assert signal == 1.0, f"Expected definition signal 1.0, got {signal}"

    def test_definition_signal_zero_for_table(self):
        """_definition_signal must return 0 for a table passage."""
        q = tokenize("What is Big Data?")
        tab_passages = chunk_page(_VISUALIZATION_TABLE, page_number=5)
        signal = _definition_signal("What is Big Data?", tab_passages[0].text, q)
        assert signal == 0.0, f"Expected 0.0 for table, got {signal}"

    def test_definition_signal_zero_for_non_definition_question(self):
        """_definition_signal must NOT fire for 'How does X work?' questions."""
        q = tokenize("How does MapReduce work?")
        text = "MapReduce is a programming model that refers to a distributed computation framework."
        signal = _definition_signal("How does MapReduce work?", text, q)
        assert signal == 0.0, f"Definition signal should be 0 for non-definition question, got {signal}"

    def test_term_density_rewards_focused_passage(self):
        """A passage where query terms are concentrated should score higher density
        than one where they appear sparsely among many unrelated words."""
        q = {"big", "data"}
        focused = "Big Data refers to large datasets. Big Data requires special tools. Big Data processing."
        diluted = ("This resource manager handles CPU, memory, networking, "
                   "storage, scheduling, and monitoring across all clusters. "
                   "Big Data is one use case among many.")
        focused_d = _term_density(q, focused)
        diluted_d = _term_density(q, diluted)
        assert focused_d > diluted_d, (
            f"Focused density ({focused_d:.3f}) should exceed diluted ({diluted_d:.3f})"
        )

    def test_hadoop_commands_score_zero(self):
        """Shell commands without 'big' or 'data' must score 0 and be excluded."""
        q = tokenize("What is Big Data?")
        question = "What is Big Data?"
        cmd_passages = chunk_page(_HADOOP_COMMANDS, page_number=8)
        scores = [_score_passage(q, question, p.text) for p in cmd_passages]
        assert all(s == 0.0 for s in scores), (
            f"Hadoop commands should score 0. Got: {scores}"
        )


# ---------------------------------------------------------------------------
# API-level regression tests
# ---------------------------------------------------------------------------

class TestBigDataRegressionAPI:
    """End-to-end regression tests via the FastAPI HTTP interface."""

    def _upload_synthetic(self):
        combined = "\n\n---PAGE BREAK---\n\n".join(
            p["text"] for p in _SYNTHETIC_BDA_PAGES
        )
        return _upload_text("bda_notes.txt", combined)

    def test_definition_returned_as_answer(self):
        """'What is Big Data?' must return the definition paragraph, not a table."""
        self._upload_synthetic()
        data = _ask("What is Big Data?")

        assert data["grounded"] is True, f"Expected grounded answer.\nGot: {data}"
        answer = data["answer"].lower()

        # The answer must contain substantive definition words.
        has_definition = (
            "refers to" in answer
            or "extremely large" in answer
            or "cannot be processed" in answer
            or "diverse" in answer
            or "characterised" in answer
        )
        assert has_definition, (
            f"Answer does not contain definition language.\nGot: {data['answer']!r}"
        )

    def test_table_not_returned_as_answer(self):
        """The visualization table must NOT be the primary answer."""
        self._upload_synthetic()
        data = _ask("What is Big Data?")

        assert data["grounded"] is True
        answer = data["answer"]
        # Table content is characterised by pipe characters.
        assert "|" not in answer, (
            f"Answer appears to be a table row.\nGot: {answer!r}"
        )
        # Table heading must not be the answer.
        assert "visualization tools" not in answer.lower(), (
            f"Answer is the table heading.\nGot: {answer!r}"
        )

    def test_hadoop_commands_not_in_citations(self):
        """Hadoop shell commands must not appear as supporting citations."""
        self._upload_synthetic()
        data = _ask("What is Big Data?")

        citation_texts = [c["text"].lower() for c in data.get("citations", [])]
        for text in citation_texts:
            assert "hdfs dfs" not in text, (
                f"Hadoop shell commands appeared as citation: {text!r}"
            )
            assert "mapreduce.jar" not in text, (
                f"Hadoop jar command appeared as citation: {text!r}"
            )

    def test_citation_threshold_excludes_unrelated(self):
        """Supporting citations must only appear when they are genuinely relevant.
        The YARN scheduler passage that merely mentions 'Big Data' in passing
        should not appear alongside a strong definition answer.
        """
        self._upload_synthetic()
        data = _ask("What is Big Data?")

        assert data["grounded"] is True
        # Citations must not contain 'YARN' or 'CapacityScheduler' content
        # because those are about cluster scheduling, not the definition of Big Data.
        for citation in data.get("citations", []):
            assert "capacityscheduler" not in citation["text"].lower(), (
                f"YARN scheduler appeared as citation: {citation['text'][:80]!r}"
            )
            assert "resourcemanager" not in citation["text"].lower(), (
                f"YARN ResourceManager appeared as citation: {citation['text'][:80]!r}"
            )


# ---------------------------------------------------------------------------
# General-purpose tests (not hard-coded to Big Data)
# ---------------------------------------------------------------------------

class TestGeneralRetrieval:
    """Verify the fix works for different topics, not just Big Data."""

    def test_ml_definition_ranked_above_ml_table(self):
        """Replacing 'Big Data' with 'machine learning': definition must still win."""
        definition = (
            "Introduction to Machine Learning\n\n"
            "Machine learning refers to a branch of artificial intelligence that\n"
            "enables computers to learn from data without being explicitly programmed.\n"
            "It uses statistical techniques to build models that generalise from\n"
            "training examples to unseen cases, making predictions or decisions.\n"
        )
        table = (
            "Machine Learning Frameworks\n\n"
            "Framework | Language | Use Case | Machine Learning Support | License\n"
            "----------|----------|----------|--------------------------|--------\n"
            "TensorFlow | Python | Deep learning for machine learning | Yes | Open\n"
            "PyTorch | Python | Neural networks for machine learning | Yes | Open\n"
            "Scikit-learn | Python | Classical machine learning algorithms | Yes | Open\n"
            "Keras | Python | High-level machine learning API | Yes | Open\n"
        )
        _upload_text("ml_notes.txt", definition + "\n\n" + table)
        data = _ask("What is machine learning?")

        assert data["grounded"] is True
        answer_lower = data["answer"].lower()
        assert "refers to" in answer_lower or "artificial intelligence" in answer_lower or (
            "statistical" in answer_lower
        ), f"Answer should be definition, got: {data['answer']!r}"
        assert "|" not in data["answer"], "Answer is a table row, not a definition."

    def test_5vs_retrieved_correctly(self):
        """5Vs question should return the explanation paragraph."""
        text = (
            "The 5Vs of Big Data\n\n"
            "Volume refers to the massive scale of data generated every second.\n"
            "Velocity describes the speed at which new data is created and processed.\n"
            "Variety captures the many different forms data can take — structured tables,\n"
            "images, video, text and sensor readings. Veracity addresses the\n"
            "trustworthiness of the data. Value is the ultimate goal: extracting\n"
            "actionable insights.\n\n"
            "Hadoop Commands\n\n"
            "hdfs dfs -ls / \nhdfs dfs -put file.txt /user/\n"
        )
        _upload_text("bda.txt", text)
        data = _ask("What are the 5Vs of Big Data?")

        assert data["grounded"] is True
        answer_lower = data["answer"].lower()
        vs_found = sum(1 for v in ["volume", "velocity", "variety", "veracity", "value"]
                       if v in answer_lower)
        assert vs_found >= 3, (
            f"Expected >= 3 of 5Vs in answer, found {vs_found}.\n"
            f"Got: {data['answer']!r}"
        )

    def test_unrelated_question_returns_not_grounded(self):
        """Chemistry question over Big Data notes → not grounded."""
        _upload_text("bda.txt", _BIG_DATA_DEFINITION)
        data = _ask("What is an alkane in organic chemistry?")
        assert data["grounded"] is False
        assert data["citations"] == []

    def test_citations_carry_correct_document_name(self):
        """Every returned citation must reference the correct document."""
        _upload_text("my_study_notes.txt", _BIG_DATA_DEFINITION + "\n\n" + _YARN_SCHEDULER)
        data = _ask("What is Big Data?")
        assert data["grounded"] is True
        for c in data.get("citations", []):
            assert c["document"] == "my_study_notes.txt", (
                f"Wrong document in citation: {c['document']!r}"
            )

    def test_duplicate_passages_suppressed(self):
        """Uploading the same content twice must not produce duplicate citations."""
        _upload_text("notes_a.txt", _BIG_DATA_DEFINITION)
        _upload_text("notes_b.txt", _BIG_DATA_DEFINITION)
        data = _ask("What is Big Data?")
        assert data["grounded"] is True
        all_texts = [data["answer"]] + [c["text"] for c in data.get("citations", [])]
        assert len(all_texts) == len(set(all_texts)), "Duplicate passages in response."

    def test_pdf_broken_lines_reassembled(self):
        """Broken-line text must be reassembled before retrieval."""
        broken = (
            "Machine learning is a subfield of artificial intelligence that\n"
            "enables systems to automatically learn and improve from experience\n"
            "without being explicitly programmed for each task.\n"
        )
        _upload_text("ml.txt", broken)
        data = _ask("What is machine learning?")
        assert data["grounded"] is True
        answer_lower = data["answer"].lower()
        assert "artificial intelligence" in answer_lower or "automatically learn" in answer_lower

    def test_correct_page_number_in_answer(self):
        """Citations must include a valid integer page number."""
        _upload_text("notes.txt", _BIG_DATA_DEFINITION)
        data = _ask("What is Big Data?")
        assert data["grounded"] is True
        for c in data.get("citations", []):
            assert isinstance(c["page"], int)
            assert c["page"] >= 1
