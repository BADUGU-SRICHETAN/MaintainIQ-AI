"""Citation validation.

What this checks:
  * every [E#] reference points at evidence that was actually retrieved,
  * each factual sentence carries at least one citation,
  * each cited sentence shares enough content words with its cited evidence.

What it does NOT check: semantic entailment or factual correctness. Lexical overlap can
pass a wrong claim and fail a correct paraphrase. Treat it as a guardrail, not a verdict.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_CITE = re.compile(r"\[(E\d+)\]")
_WORD = re.compile(r"[a-z]{3,}")
_STOPWORDS = frozenset(
    "the and for with that this from were was are been have has had into after before "
    "during over under their there which while when where what will would could should "
    "must also only than then them they these those such each both more most some".split()
)


@dataclass
class SentenceCheck:
    sentence: str
    citations: list[str]
    invalid_citations: list[str]
    uncited: bool
    lexical_support: float
    supported: bool


@dataclass
class ValidationReport:
    sentences: list[SentenceCheck] = field(default_factory=list)

    @property
    def claim_count(self) -> int:
        return len(self.sentences)

    @property
    def citation_coverage(self) -> float:
        if not self.sentences:
            return 0.0
        return sum(not s.uncited for s in self.sentences) / len(self.sentences)

    @property
    def invalid_citation_count(self) -> int:
        return sum(len(s.invalid_citations) for s in self.sentences)

    @property
    def lexically_supported_rate(self) -> float:
        cited = [s for s in self.sentences if not s.uncited]
        if not cited:
            return 0.0
        return sum(s.supported for s in cited) / len(cited)


_CITE_AFTER_PUNCT = re.compile(r"([.!?])\s*((?:\[E\d+\]\s*)+)")


def normalize_citations(text: str) -> str:
    """Move citations that follow a full stop in front of it: `claim. [E1]` -> `claim [E1].`
    so each citation stays attached to the sentence it supports."""
    def _move(match: re.Match) -> str:
        cites = " ".join(re.findall(r"\[E\d+\]", match.group(2)))
        return f" {cites}{match.group(1)} "
    return re.sub(r"\s+", " ", _CITE_AFTER_PUNCT.sub(_move, text)).strip()


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", normalize_citations(text.strip()))
    return [p.strip() for p in parts if p.strip()]


def content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS}


def validate(answer: str, evidence_texts: dict[str, str], threshold: float = 0.5) -> ValidationReport:
    """`evidence_texts` maps E-ids (E1, E2, ...) to the text that was shown to the model."""
    report = ValidationReport()
    for sentence in split_sentences(answer):
        if sentence.lower().startswith(("insufficient evidence", "no supported answer")):
            continue
        cites = _CITE.findall(sentence)
        invalid = [c for c in cites if c not in evidence_texts]
        valid = [c for c in cites if c in evidence_texts]
        body = _CITE.sub("", sentence)
        words = content_words(body)
        cited_words: set[str] = set()
        for c in valid:
            cited_words |= content_words(evidence_texts[c])
        support = len(words & cited_words) / len(words) if words else 0.0
        report.sentences.append(
            SentenceCheck(
                sentence=sentence,
                citations=valid,
                invalid_citations=invalid,
                uncited=not cites,
                lexical_support=round(support, 3),
                supported=bool(valid) and not invalid and support >= threshold,
            )
        )
    return report
