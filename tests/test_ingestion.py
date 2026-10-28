import pytest

from maintainiq.ingestion import load_manuals, split_manual, window_text
from maintainiq.schemas import SOURCE_MANUAL, SOURCE_WORK_ORDER


def test_split_manual_by_heading():
    text = "# Title\n\n## BRG-WEAR: Bearing wear\nBody one.\n\n## MISALIGN: Shaft misalignment\nBody two.\n"
    sections = split_manual(text, "bearings.md")
    assert [s[0] for s in sections] == ["bearings.md#BRG-WEAR", "bearings.md#MISALIGN"]
    assert sections[0][2] == "Body one."


def test_window_text_respects_size_and_overlap():
    text = "Sentence one is here. " * 80
    chunks = window_text(text, size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_window_text_rejects_bad_overlap():
    with pytest.raises(ValueError):
        window_text("abc " * 50, size=100, overlap=100)


def test_manual_chunks_carry_metadata(settings):
    docs = load_manuals(settings.data_dir / "manuals", settings)
    assert docs and all(d.source_type == SOURCE_MANUAL for d in docs)
    assert all(d.metadata["fault_code"] == d.doc_id.split("#")[1].split("-p")[0] for d in docs)


def test_work_orders_load_one_doc_each(settings, dataset_root):
    import pandas as pd

    from maintainiq.ingestion import load_work_orders

    path = dataset_root / "raw" / "work_orders" / "work_orders.csv"
    docs = load_work_orders(path)
    assert len(docs) == len(pd.read_csv(path))
    assert all(d.source_type == SOURCE_WORK_ORDER for d in docs)
    assert docs[0].equipment_id.startswith("MTR-")


def test_missing_work_order_columns_fail_loudly(tmp_path):
    from maintainiq.ingestion import load_work_orders

    bad = tmp_path / "wo.csv"
    bad.write_text("wo_id,equipment_id\nWO-1,MTR-001\n")
    with pytest.raises(ValueError, match="missing columns"):
        load_work_orders(bad)
