
from fastapi.testclient import TestClient
from backend.main import app, documents, fact_store

client = TestClient(app)


def setup_function():
    documents.clear()
    fact_store.clear()


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_fact_lookup_returns_no_results_when_empty():
    response = client.post(
        "/api/facts",
        json={"question": "What is overfitting?"},
    )

    assert response.status_code == 200
    assert response.json()["count"] == 0
    assert response.json()["facts"] == []


def test_upload_and_fact_lookup():
    content = (
        b"Overfitting occurs when a model learns training data too closely. "
        b"This can reduce performance on unseen examples."
    )

    upload_response = client.post(
        "/api/documents",
        files={
            "file": (
                "test_notes.txt",
                content,
                "text/plain",
            )
        },
    )

    assert upload_response.status_code == 200

    response = client.post(
        "/api/facts",
        json={"question": "What is overfitting?"},
    )

    assert response.status_code == 200

    result = response.json()
    assert result["count"] > 0
    assert any(
        "Overfitting" in fact["text"]
        for fact in result["facts"]
    )
    assert all(
        fact["document"] == "test_notes.txt"
        for fact in result["facts"]
    )
