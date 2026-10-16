
import re
from dataclasses import dataclass, asdict

STOP_WORDS = {
    "the", "is", "a", "an", "and", "or", "of", "to", "in",
    "for", "on", "with", "what", "how", "why", "when", "where",
    "who", "which", "are", "was", "were", "do", "does", "did",
    "tell", "me", "about", "my", "from", "it",
}


def tokenize(text: str) -> set[str]:
    return {
        word
        for word in re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
        if word not in STOP_WORDS and len(word) > 1
    }


@dataclass
class Fact:
    text: str
    document: str
    page: int
    terms: list[str]


class FactStore:
    def __init__(self):
        self._facts: list[Fact] = []

    def add_document(self, document: str, pages: list[dict]) -> int:
        added = 0

        for page in pages:
            text = page.get("text", "")
            page_number = page.get("page", 1)

            for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
                sentence = sentence.strip()
                terms = tokenize(sentence)

                if len(terms) < 3:
                    continue

                self._facts.append(
                    Fact(
                        text=sentence,
                        document=document,
                        page=page_number,
                        terms=sorted(terms),
                    )
                )
                added += 1

        return added

    def lookup(self, question: str, limit: int = 3) -> list[dict]:
        query_terms = tokenize(question)

        if not query_terms:
            return []

        results = []

        for fact in self._facts:
            fact_terms = set(fact.terms)
            overlap = query_terms & fact_terms

            if not overlap:
                continue

            coverage = len(overlap) / len(query_terms)
            precision = len(overlap) / len(fact_terms)
            score = 0.7 * coverage + 0.3 * precision

            if len(fact_terms) < 5:
                score *= 0.5

            if score >= 0.25:
                result = asdict(fact)
                result["relevance"] = round(score, 3)
                results.append(result)

        results.sort(
            key=lambda item: item["relevance"],
            reverse=True,
        )
        return results[:max(0, limit)]

    def clear(self):
        self._facts.clear()

    def __len__(self):
        return len(self._facts)
