import pytest

from maintainiq.retrieval import HybridRetriever, rrf_fuse, tokenize
from maintainiq.schemas import Document


def test_tokenizer_keeps_fault_codes_whole_and_stems():
    tokens = tokenize("Symptoms of BRG-WEAR on WO-0001")
    assert "brg-wear" in tokens and "brg" in tokens and "wear" in tokens
    assert "symptom" in tokens  # stemmed
    assert "of" not in tokens  # stopword removed


def test_tokenizer_matches_inflections():
    assert set(tokenize("vibrating")) & set(tokenize("vibration"))


def test_rrf_rewards_items_ranked_high_in_both_lists():
    fused = rrf_fuse([["a", "b", "c"], ["b", "a", "d"]], k=60)
    assert fused[:2] == ["a", "b"]  # a and b both appear high in both lists
    assert fused.index("c") > fused.index("b")


def test_rrf_is_rank_based_not_score_based():
    # Same ranks with different scores produce the same fused order.
    assert rrf_fuse([["x", "y"]], k=60) == rrf_fuse([["x", "y"]], k=60)


def _docs():
    return [
        Document("d1", "Bearing race wear causes drive end vibration.", "manual", "Bearings",
                 {"equipment_id": "MTR-001"}),
        Document("d2", "Winding insulation resistance dropped after overheating.", "manual", "Winding",
                 {"equipment_id": "MTR-002"}),
        Document("d3", "Coupling misalignment produces vibration at twice running speed.", "manual",
                 "Alignment", {"equipment_id": "MTR-001"}),
    ]


def test_search_returns_relevant_document_first(settings):
    r = HybridRetriever(_docs(), settings)
    top = r.search("why is the drive end vibrating", top_k=3)
    assert top[0].document.doc_id == "d1"


def test_metadata_filter_restricts_results(settings):
    r = HybridRetriever(_docs(), settings)
    results = r.search("vibration", top_k=5, filters={"equipment_id": "MTR-002"})
    assert all(res.document.equipment_id == "MTR-002" for res in results)


def test_empty_query_and_empty_filter_return_nothing(settings):
    r = HybridRetriever(_docs(), settings)
    assert r.search("   ") == []
    assert r.search("vibration", filters={"equipment_id": "MTR-999"}) == []


def test_duplicate_ids_are_rejected(settings):
    docs = _docs() + [Document("d1", "x", "manual", "dup")]
    with pytest.raises(ValueError):
        HybridRetriever(docs, settings)
