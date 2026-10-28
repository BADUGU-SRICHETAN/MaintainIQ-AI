import numpy as np
import pandas as pd
import pytest

from maintainiq.sensors import load_readings, summarize_all, summarize_equipment
from maintainiq.synthetic import DEGRADING_MOTOR, generate_sensor_readings


@pytest.fixture(scope="module")
def readings():
    return generate_sensor_readings(np.random.default_rng(3), hours=168)


def test_degrading_motor_is_flagged_and_healthy_motors_are_not(readings, settings):
    summaries = {s.equipment_id: s for s in summarize_all(readings, settings)}
    assert "vibration_rms_mm_s" in summaries[DEGRADING_MOTOR].anomalous_metrics
    assert "temperature_c" in summaries[DEGRADING_MOTOR].anomalous_metrics
    for eq, s in summaries.items():
        if eq != DEGRADING_MOTOR:
            assert s.anomalous_metrics == [], eq


def test_trend_is_positive_for_degrading_vibration(readings, settings):
    s = summarize_equipment(readings, DEGRADING_MOTOR, settings)
    assert s.metrics["vibration_rms_mm_s"].slope_per_day > 0


def test_summary_text_mentions_anomaly_start(readings, settings):
    text = summarize_equipment(readings, DEGRADING_MOTOR, settings).to_text()
    assert "deviated from baseline starting" in text


def test_missing_columns_fail_loudly():
    with pytest.raises(ValueError, match="missing columns"):
        load_readings(pd.DataFrame({"equipment_id": ["MTR-001"]}))


def test_too_few_readings_is_an_error(settings):
    df = generate_sensor_readings(np.random.default_rng(0), hours=10)
    with pytest.raises(ValueError, match="too few readings"):
        summarize_equipment(df, "MTR-001", settings)
