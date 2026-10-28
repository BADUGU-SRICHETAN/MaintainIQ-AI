"""Typed records shared across ingestion, retrieval, generation and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SOURCE_MANUAL = "manual"
SOURCE_WORK_ORDER = "work_order"
SOURCE_SENSOR_SUMMARY = "sensor_summary"


@dataclass(frozen=True)
class Document:
    """A retrievable chunk of evidence. `doc_id` is stable and citable."""

    doc_id: str
    text: str
    source_type: str
    title: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def equipment_id(self) -> str | None:
        return self.metadata.get("equipment_id")


@dataclass(frozen=True)
class RetrievalResult:
    document: Document
    rrf_score: float
    bm25_rank: int | None
    dense_rank: int | None
    dense_similarity: float


@dataclass(frozen=True)
class GoldenQuery:
    query_id: str
    question: str
    relevant_ids: frozenset[str]
    split: str  # "dev" | "test"
