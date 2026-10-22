"""
Passage chunker for Medha AI.

Converts raw page text (including PDF-extracted text with frequent line breaks)
into meaningful, searchable passages.

Problem being solved
--------------------
PDF text extractors often emit text like:

    Big Data is often defined as:\n
    data that is too large,\n
    too fast, or too varied\n
    for traditional systems to handle.\n

A naive ``split("\\n")`` turns this into four tiny fragments, none of which
is a useful standalone answer.  A naive sentence splitter has the same problem
because ``"Big Data is often defined as:"`` is technically a sentence but is
almost meaningless without the lines that follow.

Algorithm
---------
Phase A – Rejoin broken lines
    Lines that do not end a sentence (no terminal ``. ! ?`` followed by a
    space or end-of-line, and no blank line after them) are joined to the next
    line.  A line ending with a lowercase letter, digit, comma or colon
    almost certainly continues on the next line.

Phase B – Split into paragraphs
    After rejoining, split on one or more blank lines.  Each block is a
    candidate passage.

Phase C – Window over long paragraphs
    If a paragraph exceeds WINDOW_CHARS (~600 chars) it is too long to be a
    clean citation.  Split it at sentence boundaries into overlapping windows
    of WINDOW_SENTENCES sentences with OVERLAP_SENTENCES overlap, so context
    at window boundaries is never lost.

Phase D – Filter
    Drop passages with fewer than MIN_TOKENS meaningful tokens (removes
    headings, page numbers, single-word fragments).
    Drop passages longer than MAX_CHARS (avoids returning entire chapters).

All output passages carry the original page number.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

# ---------------------------------------------------------------------------
# Tuning constants  (easy to adjust without touching calling code)
# ---------------------------------------------------------------------------

MIN_TOKENS: int = 8        # passages with fewer tokens are dropped (headings/fragments)
MAX_CHARS: int = 1200      # passages longer than this are windowed
WINDOW_CHARS: int = 600    # trigger windowing above this length
WINDOW_SENTENCES: int = 4  # sentences per window
OVERLAP_SENTENCES: int = 1 # sentences of overlap between consecutive windows

# Words that do not count toward the MIN_TOKENS threshold.
_STOP_WORDS: frozenset[str] = frozenset({
    "the", "is", "a", "an", "and", "or", "of", "to", "in",
    "for", "on", "with", "what", "how", "why", "when", "where",
    "who", "which", "are", "was", "were", "do", "does", "did",
    "tell", "me", "about", "my", "from", "it", "as", "by",
    "at", "be", "this", "that", "these", "those",
})

# Sentence-terminal pattern: period/!/? followed by whitespace or end.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

# A line that clearly continues onto the next (does not end a sentence).
# Ends with: lowercase letter, digit, comma, colon, semicolon, dash, or
# an opening parenthesis/bracket.
_CONTINUES_PATTERN = re.compile(
    r"[a-z0-9,;:\-–—\(\[\u2013\u2014]$",
    re.UNICODE,
)


@dataclass(frozen=True)
class Passage:
    text: str
    page: int


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chunk_page(raw_text: str, page_number: int) -> list[Passage]:
    """Return a list of meaningful passages extracted from *raw_text*.

    Args:
        raw_text:    The raw text of a single PDF/TXT page as returned by the
                     extractor.  May contain frequent line breaks.
        page_number: The 1-based page number to attach to every passage.

    Returns:
        A list of :class:`Passage` objects, each at most MAX_CHARS long and
        with at least MIN_TOKENS meaningful tokens.
    """
    if not raw_text or not raw_text.strip():
        return []

    rejoined = _rejoin_broken_lines(raw_text)
    paragraphs = _split_into_paragraphs(rejoined)
    passages: list[Passage] = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) > WINDOW_CHARS:
            windows = _window_sentences(para)
        else:
            windows = [para]

        for window in windows:
            window = window.strip()
            if not window:
                continue
            if _count_meaningful_tokens(window) < MIN_TOKENS:
                continue
            if len(window) > MAX_CHARS:
                window = window[:MAX_CHARS].rsplit(" ", 1)[0]
            passages.append(Passage(text=window, page=page_number))

    return passages


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _rejoin_broken_lines(text: str) -> str:
    """Join lines that clearly continue onto the next line.

    A line is considered to continue if it ends with a character that implies
    the sentence is incomplete (lowercase letter, digit, comma, colon, etc.)
    AND the next line is not blank.

    Blank lines (paragraph separators) are preserved as-is.
    """
    lines = text.splitlines()
    output: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.rstrip()

        # Preserve blank lines as paragraph separators.
        if not stripped:
            output.append("")
            i += 1
            continue

        # Look ahead: join to next non-blank line if this line continues.
        while (
            i + 1 < len(lines)
            and lines[i + 1].strip()              # next line is not blank
            and _CONTINUES_PATTERN.search(stripped)  # this line continues
        ):
            i += 1
            next_stripped = lines[i].strip()
            stripped = stripped + " " + next_stripped

        output.append(stripped)
        i += 1

    return "\n".join(output)


def _split_into_paragraphs(text: str) -> list[str]:
    """Split text on one or more blank lines into paragraph blocks."""
    return re.split(r"\n{2,}", text)


def _split_into_sentences(text: str) -> list[str]:
    """Split a paragraph into individual sentences."""
    raw = _SENTENCE_END.split(text)
    return [s.strip() for s in raw if s.strip()]


def _window_sentences(paragraph: str) -> list[str]:
    """Break a long paragraph into overlapping sentence windows.

    Each window contains WINDOW_SENTENCES sentences; consecutive windows share
    OVERLAP_SENTENCES sentences so no context is lost at boundaries.
    """
    sentences = _split_into_sentences(paragraph)
    if len(sentences) <= WINDOW_SENTENCES:
        return [paragraph]

    windows: list[str] = []
    step = max(1, WINDOW_SENTENCES - OVERLAP_SENTENCES)
    i = 0
    while i < len(sentences):
        chunk = sentences[i : i + WINDOW_SENTENCES]
        windows.append(" ".join(chunk))
        if i + WINDOW_SENTENCES >= len(sentences):
            break
        i += step

    return windows


def _count_meaningful_tokens(text: str) -> int:
    """Return the number of non-stop-word, length > 1 tokens in *text*."""
    words = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
    return sum(1 for w in words if w not in _STOP_WORDS and len(w) > 1)
