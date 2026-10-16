import re

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


def score(question: str, passage: str) -> float:
    query_words = tokenize(question)
    passage_words = tokenize(passage)

    if not query_words or not passage_words:
        return 0.0

    overlap = query_words & passage_words
    coverage = len(overlap) / len(query_words)
    precision = len(overlap) / len(passage_words)
    result = 0.7 * coverage + 0.3 * precision

    if len(passage_words) < 5:
        result *= 0.5

    return result


def test_relevant_passage_ranks_above_unrelated_passage():
    question = "What are the algorithms used in unsupervised learning?"

    relevant = (
        "Common unsupervised learning algorithms include "
        "K-Means Clustering, Hierarchical Clustering, "
        "and Principal Component Analysis."
    )
    unrelated = (
        "Supervised learning uses labelled training data "
        "to predict outcomes."
    )

    assert score(question, relevant) > score(question, unrelated)


def test_exact_topic_has_positive_score():
    question = "What is overfitting?"
    passage = (
        "Overfitting occurs when a model learns training data "
        "too closely and performs poorly on new data."
    )

    assert score(question, passage) > 0
