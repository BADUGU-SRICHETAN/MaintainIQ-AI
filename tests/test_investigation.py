from maintainiq.investigation import INSUFFICIENT


def test_grounded_question_gets_cited_answer(system):
    result = system.engine.investigate("Why did the bearing on the drive end fail?", top_k=5)
    assert not result.insufficient
    assert result.validation.citation_coverage == 1.0
    assert result.validation.invalid_citation_count == 0
    assert result.evidence_ids  # every cited id maps to a retrieved document


def test_unrelated_question_is_refused(system):
    result = system.engine.investigate("What is the price of coffee in Paris?", top_k=5)
    assert result.insufficient
    assert result.answer == INSUFFICIENT


def test_missing_exact_value_is_refused(system):
    result = system.engine.investigate("What is the replacement cost of the bearing?", top_k=5)
    assert result.insufficient


def test_causal_question_prefers_root_cause(system):
    result = system.engine.investigate("What caused the winding insulation to fail?", top_k=5)
    assert "overheating" in result.answer.lower()


def test_equipment_filter_keeps_shared_manuals_and_drops_other_motors(system):
    result = system.engine.investigate("bearing vibration", top_k=8, equipment_id="MTR-003")
    assert result.evidence, "expected evidence for MTR-003"
    for ev in result.evidence:
        assert ev.document.source_type == "manual" or ev.document.equipment_id == "MTR-003"
    assert any(ev.document.source_type == "manual" for ev in result.evidence)
