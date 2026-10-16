from memory.fact_store import FactStore


def make_store():
    store = FactStore()
    store.add_document(
        "machine_learning.txt",
        [
            {
                "page": 1,
                "text": (
                    "Supervised learning uses labelled training data. "
                    "Unsupervised learning discovers patterns in unlabelled data. "
                    "Overfitting occurs when a model learns training data too closely."
                ),
            }
        ],
    )
    return store


def test_add_document_stores_sentences():
    store = make_store()
    assert len(store) == 3


def test_lookup_returns_relevant_fact_and_citation():
    store = make_store()
    results = store.lookup("What is overfitting?")

    assert results
    assert "Overfitting" in results[0]["text"]
    assert results[0]["document"] == "machine_learning.txt"
    assert results[0]["page"] == 1


def test_unrelated_question_returns_no_results():
    store = make_store()
    results = store.lookup("What is quantum entanglement?")

    assert results == []


def test_empty_question_returns_no_results():
    store = make_store()
    assert store.lookup("") == []
