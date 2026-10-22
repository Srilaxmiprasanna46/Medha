"""
FactStore — in-memory passage index for Medha AI.

Phase 3 changes (retrieval quality)
------------------------------------
* ``add_document`` now uses ``passage_chunker.chunk_page`` instead of a
  raw newline/sentence split.  This means PDF-style broken lines are
  rejoined into meaningful paragraphs before indexing.
* The minimum token threshold has been raised to match the chunker's
  MIN_TOKENS constant so that headings and short fragments are excluded at
  the indexing step, not just at query time.
* Everything else (``Fact`` dataclass, ``tokenize``, ``lookup``, ``clear``,
  ``__len__``) is unchanged.
"""

import re
from dataclasses import dataclass, asdict

from memory.passage_chunker import chunk_page, MIN_TOKENS

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
        """Index a document's pages into searchable passages.

        Each page's raw text is run through the passage chunker, which
        rejoins PDF broken lines and splits the result into paragraph-level
        (or sentence-window) passages.  Only passages with at least
        MIN_TOKENS meaningful tokens are stored.
        """
        added = 0

        for page in pages:
            text = page.get("text", "")
            page_number = page.get("page", 1)

            for passage in chunk_page(text, page_number):
                terms = tokenize(passage.text)
                # chunk_page already enforces MIN_TOKENS, but double-check
                # here in case callers construct Passage objects directly.
                if len(terms) < MIN_TOKENS:
                    continue

                self._facts.append(
                    Fact(
                        text=passage.text,
                        document=document,
                        page=passage.page,
                        terms=sorted(terms),
                    )
                )
                added += 1

        return added

    def lookup(self, question: str, limit: int = 3) -> list[dict]:
        """Return up to *limit* facts most relevant to *question*.

        Scoring is unchanged from Phase 2: 0.7 × coverage + 0.3 × precision,
        with a penalty for passages with fewer than 5 distinct terms.
        """
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
        return results[: max(0, limit)]

    def clear(self):
        self._facts.clear()

    def __len__(self):
        return len(self._facts)
