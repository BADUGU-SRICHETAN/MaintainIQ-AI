import pytest

from maintainiq.evaluation import (
    evaluate_retrieval,
    load_golden,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_metric_math_on_known_ranking():
    ranked = ["x", "a", "y", "b", "z"]
    relevant = frozenset({"a", "b"})
    assert recall_at_k(ranked, relevant, 1) == 0.0
    assert recall_at_k(ranked, relevant, 5) == 1.0
    assert precision_at_k(ranked, relevant, 5) == 2 / 5
    assert reciprocal_rank(ranked, relevant) == 0.5


def test_recall_is_monotone_in_k():
    ranked = ["a", "x", "b", "y", "c"]
    relevant = frozenset({"a", "b", "c"})
    values = [recall_at_k(ranked, relevant, k) for k in range(1, 6)]
    assert values == sorted(values)


def test_golden_loader_rejects_empty_labels(tmp_path):
    path = tmp_path / "g.jsonl"
    path.write_text('{"query_id":"q","question":"x","relevant_ids":[],"split":"dev"}\n')
    with pytest.raises(ValueError, match="no relevant ids"):
        load_golden(path)


def test_retrieval_meets_regression_floor(system, settings):
    """Regression gate on the synthetic golden set. The floor sits below the current score so
    small changes do not flap, but a real drop fails CI. It is not evidence of real-world accuracy."""
    queries = load_golden(settings.eval_dir / "golden_set.jsonl")
    dev = evaluate_retrieval(system.retriever, queries, "dev")
    assert dev.mrr >= 0.80
    assert dev.precision_at_5 >= 0.65
    assert dev.hit_rate_at_5 >= 0.85
