"""Sensor analytics: baseline z-scores, rolling anomaly flags and linear trend per motor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from maintainiq.config import Settings

METRICS = ["vibration_rms_mm_s", "temperature_c", "phase_current_a", "rpm"]
REQUIRED_COLUMNS = {"equipment_id", "timestamp", *METRICS}
ROLLING_WINDOW = 6  # readings; with hourly data this is six hours


@dataclass(frozen=True)
class MetricSummary:
    metric: str
    baseline_mean: float
    baseline_std: float
    latest: float
    slope_per_day: float
    max_z: float
    first_anomaly: str | None


@dataclass(frozen=True)
class EquipmentSummary:
    equipment_id: str
    readings: int
    metrics: dict[str, MetricSummary]

    @property
    def anomalous_metrics(self) -> list[str]:
        return [m for m, s in self.metrics.items() if s.first_anomaly is not None]

    def to_text(self) -> str:
        lines = [f"Sensor summary for {self.equipment_id} over {self.readings} hourly readings."]
        if not self.anomalous_metrics:
            lines.append("No sustained deviation from baseline was detected.")
        for name in self.anomalous_metrics:
            s = self.metrics[name]
            lines.append(
                f"{name} deviated from baseline starting {s.first_anomaly} "
                f"(latest {s.latest:.2f}, baseline {s.baseline_mean:.2f}, "
                f"trend {s.slope_per_day:+.3f} per day, max z {s.max_z:.1f})."
            )
        return " ".join(lines)


def load_readings(path_or_df) -> pd.DataFrame:
    df = pd.read_csv(path_or_df) if not isinstance(path_or_df, pd.DataFrame) else path_or_df.copy()
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"sensor data is missing columns: {sorted(missing)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    before = len(df)
    df = df.dropna(subset=METRICS).sort_values(["equipment_id", "timestamp"])
    dropped = before - len(df)
    if dropped:
        import logging

        logging.getLogger(__name__).warning("dropped %d sensor rows with missing values", dropped)
    return df.reset_index(drop=True)


def summarize_equipment(df: pd.DataFrame, equipment_id: str, settings: Settings) -> EquipmentSummary:
    sub = df[df["equipment_id"] == equipment_id].sort_values("timestamp")
    if len(sub) < settings.baseline_readings + ROLLING_WINDOW:
        raise ValueError(f"{equipment_id} has too few readings for a baseline")
    base = sub.iloc[: settings.baseline_readings]
    hours = (sub["timestamp"] - sub["timestamp"].iloc[0]).dt.total_seconds().to_numpy() / 3600.0
    metrics: dict[str, MetricSummary] = {}
    for metric in METRICS:
        mean = float(base[metric].mean())
        std = float(base[metric].std(ddof=0))
        std = std if std > 1e-9 else 1e-9  # guard against flat baselines
        values = sub[metric].to_numpy(dtype=float)
        rolling = pd.Series(values).rolling(ROLLING_WINDOW).mean().to_numpy()
        z = (rolling - mean) / std
        flagged = np.flatnonzero(np.abs(np.nan_to_num(z)) >= settings.anomaly_z_threshold)
        first = str(sub["timestamp"].iloc[flagged[0]]) if flagged.size else None
        slope_per_hour = float(np.polyfit(hours[-72:], values[-72:], 1)[0]) if len(values) >= 3 else 0.0
        metrics[metric] = MetricSummary(
            metric=metric,
            baseline_mean=round(mean, 3),
            baseline_std=round(std, 3),
            latest=round(float(values[-1]), 3),
            slope_per_day=round(slope_per_hour * 24, 4),
            max_z=round(float(np.nanmax(np.abs(z))), 2),
            first_anomaly=first,
        )
    return EquipmentSummary(equipment_id=equipment_id, readings=len(sub), metrics=metrics)


def summarize_all(df: pd.DataFrame, settings: Settings) -> list[EquipmentSummary]:
    return [summarize_equipment(df, eq, settings) for eq in sorted(df["equipment_id"].unique())]
