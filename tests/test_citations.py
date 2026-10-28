from maintainiq.citations import normalize_citations, split_sentences, validate

EVIDENCE = {
    "E1": "Shaft misalignment between motor and pump exceeded tolerance.",
    "E2": "Dust build-up on cooling fan blades caused rotor imbalance.",
}


def test_citation_after_period_stays_with_its_claim():
    assert normalize_citations("Claim one. [E1] Claim two. [E2]") == "Claim one [E1]. Claim two [E2]."
    assert split_sentences("Claim one. [E1]") == ["Claim one [E1]."]


def test_supported_claim_passes():
    report = validate("Shaft misalignment between motor and pump exceeded tolerance. [E1]", EVIDENCE)
    assert report.claim_count == 1
    assert report.sentences[0].supported
    assert report.citation_coverage == 1.0


def test_invalid_citation_id_is_reported():
    report = validate("Dust build-up caused rotor imbalance. [E9]", EVIDENCE)
    assert report.invalid_citation_count == 1
    assert not report.sentences[0].supported


def test_uncited_claim_is_flagged():
    report = validate("Dust build-up caused rotor imbalance.", EVIDENCE)
    assert report.sentences[0].uncited
    assert report.citation_coverage == 0.0


def test_unrelated_claim_is_not_lexically_supported():
    report = validate("Bearing grease costs forty dollars per tube. [E2]", EVIDENCE)
    assert not report.sentences[0].supported


def test_refusal_text_is_not_scored_as_a_claim():
    report = validate("Insufficient evidence in the retrieved records to answer this question.", EVIDENCE)
    assert report.claim_count == 0
