"""Retrieval and citation metrics over the labelled golden set.

Metrics are computed per query, then averaged. A query whose relevant set is empty is rejected
rather than silently scored as zero.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

from maintainiq.retrieval import HybridRetriever
from maintainiq.schemas import GoldenQuery


def load_golden(path: Path) -> list[GoldenQuery]:
    queries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        relevant = frozenset(row["relevant_ids"])
        if not relevant:
            raise ValueError(f"query {row['query_id']} has no relevant ids")
        queries.append(GoldenQuery(row["query_id"], row["question"], relevant, row["split"]))
    return queries


def recall_at_k(ranked_ids: list[str], relevant: frozenset[str], k: int) -> float:
    """Standard recall@k: hits in the top k divided by |relevant|. Monotone in k."""
    return len(set(ranked_ids[:k]) & relevant) / len(relevant)


def max_recall_at_k(relevant: frozenset[str], k: int) -> float:
    """Best recall@k any ranking could reach. With ~10 relevant work orders per fault,
    recall@5 can be at most 5/|relevant|, so report this ceiling next to the score."""
    return min(k, len(relevant)) / len(relevant)


def precision_at_k(ranked_ids: list[str], relevant: frozenset[str], k: int) -> float:
    return len(set(ranked_ids[:k]) & relevant) / k


def reciprocal_rank(ranked_ids: list[str], relevant: frozenset[str]) -> float:
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


@dataclass(frozen=True)
class RetrievalReport:
    split: str
    queries: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    precision_at_5: float
    mrr: float
    hit_rate_at_5: float
    recall_at_5_ceiling: float

    def as_dict(self) -> dict:
        return asdict(self)


def evaluate_retrieval(retriever: HybridRetriever, queries: list[GoldenQuery], split: str,
                       depth: int = 10) -> RetrievalReport:
    selected = [q for q in queries if q.split == split]
    if not selected:
        raise ValueError(f"no queries in split {split!r}")
    per_query = []
    for q in selected:
        ranked = [r.document.doc_id for r in retriever.search(q.question, top_k=depth)]
        per_query.append(
            {
                "r1": recall_at_k(ranked, q.relevant_ids, 1),
                "r3": recall_at_k(ranked, q.relevant_ids, 3),
                "r5": recall_at_k(ranked, q.relevant_ids, 5),
                "c5": max_recall_at_k(q.relevant_ids, 5),
                "p5": precision_at_k(ranked, q.relevant_ids, 5),
                "rr": reciprocal_rank(ranked, q.relevant_ids),
                "hit5": float(bool(set(ranked[:5]) & q.relevant_ids)),
            }
        )
    return RetrievalReport(
        split=split,
        queries=len(selected),
        recall_at_1=round(mean(p["r1"] for p in per_query), 4),
        recall_at_3=round(mean(p["r3"] for p in per_query), 4),
        recall_at_5=round(mean(p["r5"] for p in per_query), 4),
        precision_at_5=round(mean(p["p5"] for p in per_query), 4),
        recall_at_5_ceiling=round(mean(p["c5"] for p in per_query), 4),
        mrr=round(mean(p["rr"] for p in per_query), 4),
        hit_rate_at_5=round(mean(p["hit5"] for p in per_query), 4),
    )
