"""Cited investigation: retrieve evidence, generate with citations, validate, and refuse when weak."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from maintainiq.citations import ValidationReport, content_words, split_sentences, validate
from maintainiq.config import Settings
from maintainiq.llm import LLMBackend, build_backend
from maintainiq.retrieval import HybridRetriever, tokenize
from maintainiq.schemas import RetrievalResult

_NUMERIC_INTENT = re.compile(r"\b(cost|costs|price|how much|how many|exact|amount|value|number of)\b", re.I)
# A standalone number: not part of an identifier or date such as WO-0001 or 2026-01-05.
_NUMBER = re.compile(r"(?<![\w-])\d+(?:\.\d+)?(?![\w-])")

INSUFFICIENT = "Insufficient evidence in the retrieved records to answer this question."

# Words that appear in almost every question and record but do not identify a fault. They are
# ignored by the term-coverage gate, so "motor" or "repaired" cannot make a question look covered.
# Fault-specific words such as vibration, bearing, winding, overheat, and insulation are NOT listed.
GENERIC_TERMS = frozenset(tokenize(
    "motor motors induction maintenance technician technicians record records previous past last "
    "time case cases incident incidents problem problems reported work carried performed repair "
    "repaired repairs fixed resolved addressed steps taken action actions corrective happened ever "
    "anyone someone team were was did done what when how which who why there have has any"
))

PROMPT = """You are a maintenance investigation assistant for induction motors.

Question: {question}

Evidence records:
{evidence}

Instructions:
1. Find the record whose text states the cause, diagnosis, or action the question asks about.
2. Answer in one or two sentences. Each sentence must state a fact found in a record
   and end with that record's id in brackets, for example [E1]. Use one id per sentence.
3. Cite the record that actually contains the fact. Do not cite a manual section for a
   fact that appears only in a work order, and do not cite a work order for a fact that
   appears only in a manual.
4. Do not write summary or meta sentences, such as "All evidence points to..." or
   "The records show...". Every sentence must be a factual statement with a citation.
5. Use only facts stated in the evidence. Do not add causes the evidence does not state.
6. Only if no record says anything relevant to the question, reply exactly: "{insufficient}"

Answer:"""


@dataclass
class InvestigationResult:
    question: str
    answer: str
    evidence: list[RetrievalResult]
    validation: ValidationReport
    backend: str
    insufficient: bool
    evidence_ids: dict[str, str] = field(default_factory=dict)
    refusal_reason: str | None = None  # which check refused, for diagnosis
    raw_answer: str = ""               # model output before any post-check


class InvestigationEngine:
    def __init__(self, retriever: HybridRetriever, settings: Settings, backend: LLMBackend | None = None):
        self.retriever = retriever
        self.settings = settings
        self.backend = backend or build_backend(settings)

    def investigate(self, question: str, top_k: int = 5, equipment_id: str | None = None) -> InvestigationResult:
        filters = {"equipment_id": equipment_id} if equipment_id else None
        evidence = self.retriever.search(question, top_k=top_k, filters=filters)

        gate_failure = self._gate_failure(question, evidence)
        if gate_failure:
            return InvestigationResult(
                question, INSUFFICIENT, evidence, ValidationReport(), self.backend.name, True,
                refusal_reason=f"gate:{gate_failure}",
            )

        evidence_ids = {f"E{n}": r.document.text for n, r in enumerate(evidence, start=1)}
        prompt = PROMPT.format(
            insufficient=INSUFFICIENT,
            evidence=_format_evidence(evidence),
            question=question,
        )
        raw = self.backend.generate(prompt, evidence).strip()
        answer = _dedupe_sentences(raw)
        reason = None
        if not answer or answer.startswith(INSUFFICIENT):
            reason = "model:insufficient" if answer else "model:empty"
        elif _NUMERIC_INTENT.search(question) and not _NUMBER.search(_strip_citations(answer)):
            # Exact-value questions need a number in the answer itself, not just in the evidence.
            reason = "post:numeric"
        insufficient = reason is not None
        if insufficient:
            answer = INSUFFICIENT
        validation = validate(answer, evidence_ids) if not insufficient else ValidationReport()
        return InvestigationResult(
            question=question,
            answer=answer,
            evidence=evidence,
            validation=validation,
            backend=self.backend.name,
            insufficient=insufficient,
            evidence_ids=evidence_ids,
            refusal_reason=reason,
            raw_answer=raw,
        )

    def _gate_failure(self, question: str, evidence: list[RetrievalResult]) -> str | None:
        """Refuse before generation unless the evidence plausibly answers the question.

        1. Similarity: the top hit must be semantically close to the question.
        2. Term coverage: at least `min_term_coverage` of the question's content terms must
           appear in one of the top documents, ignoring GENERIC_TERMS. This catches questions
           about things the corpus never mentions, even when a shared word such as "sensor"
           pulls in a similar document.
        3. Numeric intent: "cost", "exact", "how much" questions need a number in the evidence.
        """
        if not evidence or evidence[0].dense_similarity < self.settings.min_dense_similarity:
            return "similarity"
        terms = set(tokenize(question)) - GENERIC_TERMS
        if not terms:
            return "empty_terms"
        best_coverage = max(
            len(terms & set(tokenize(r.document.text))) / len(terms) for r in evidence[:3]
        )
        if best_coverage < self.settings.min_term_coverage:
            return "term_coverage"
        if _NUMERIC_INTENT.search(question) and not any(_NUMBER.search(r.document.text) for r in evidence[:3]):
            return "numeric"
        return None


MAX_ANSWER_SENTENCES = 3
# A citation list such as [E1, E2] or [E1-E5]. Single citations like [E1] do not match.
_BRACKET_LIST = re.compile(r"\[E\d+(?:\s*(?:,|-|and)\s*E?\d+)+\]")
# A conjunction left at the start of a sentence by a fragment, optionally with its citation.
_LEADING_CONJUNCTION = re.compile(r"^(?:and|or|but)\s+(?:\[E\d+\]\s*)?")


def _dedupe_sentences(text: str) -> str:
    """Clean model output before any check runs.

    Keeps complete, cited sentences only. Drops repeats, citation lists (e.g. "[E1, E2, E3]"),
    uncited sentences, and fragments with fewer than three content words. Keeps at most
    MAX_ANSWER_SENTENCES. Small models produce all of these, so the cleanup is deterministic
    and does not depend on the model behaving.
    """
    # Remove citation lists before splitting: newlines are collapsed during splitting, so a list
    # on its own line would otherwise be glued onto the following sentence.
    text = _BRACKET_LIST.sub(" ", text)
    seen, kept = set(), []
    for sentence in split_sentences(text):
        sentence = _LEADING_CONJUNCTION.sub("", sentence)  # "and Shaft..." left by a fragment
        if not re.search(r"\[E\d+\]", sentence):
            continue  # no citation: nothing ties this sentence to a record, so it is not shown
        body = _strip_citations(sentence).strip()
        if not body.endswith((".", "!", "?")):
            continue  # truncated fragment, usually cut off by the output limit
        words = content_words(body)
        if len(words) < 3:
            continue  # a bare citation list or a fragment such as "and [E4]"
        key = frozenset(words)
        if key in seen:
            continue
        seen.add(key)
        kept.append(sentence)
        if len(kept) == MAX_ANSWER_SENTENCES:
            break
    return " ".join(kept)


def _strip_citations(text: str) -> str:
    return re.sub(r"\[E\d+\]", "", text)


def _format_evidence(evidence: list[RetrievalResult]) -> str:
    blocks = []
    for n, result in enumerate(evidence, start=1):
        doc = result.document
        blocks.append(f"[E{n}] {doc.title} ({doc.doc_id})\n{doc.text}")
    return "\n\n".join(blocks)