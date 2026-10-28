import json

import pandas as pd

from maintainiq.synthetic import EQUIPMENT_IDS, FAULTS, generate_sensor_readings, write_dataset


def test_generation_is_deterministic(tmp_path):
    a = write_dataset(tmp_path / "a", seed=7)
    b = write_dataset(tmp_path / "b", seed=7)
    assert a["work_orders"].read_text() == b["work_orders"].read_text()
    assert a["sensors"].read_text() == b["sensors"].read_text()


def test_work_orders_cover_every_fault(dataset_root):
    wo = pd.read_csv(dataset_root / "raw" / "work_orders" / "work_orders.csv")
    assert set(wo["fault_code"]) == set(FAULTS)
    assert wo["wo_id"].is_unique
    assert set(wo["equipment_id"]) <= set(EQUIPMENT_IDS)


def test_sensor_readings_have_expected_shape():
    import numpy as np

    df = generate_sensor_readings(np.random.default_rng(1), hours=48)
    assert df["equipment_id"].nunique() == len(EQUIPMENT_IDS)
    assert len(df) == 48 * len(EQUIPMENT_IDS)
    assert df[["vibration_rms_mm_s", "temperature_c", "phase_current_a", "rpm"]].notna().all().all()


def test_golden_labels_are_nonempty_and_reference_real_ids(dataset_root):
    wo_ids = set(pd.read_csv(dataset_root / "raw" / "work_orders" / "work_orders.csv")["wo_id"])
    manual_ids = {f"{f.manual_file}#{f.code}" for f in FAULTS.values()}
    rows = [json.loads(line) for line in (dataset_root.parent / "eval" / "golden_set.jsonl").read_text().splitlines()]
    assert rows
    for row in rows:
        assert row["relevant_ids"]
        for rid in row["relevant_ids"]:
            assert rid in wo_ids or rid in manual_ids, rid
        assert row["split"] in {"dev", "test"}
