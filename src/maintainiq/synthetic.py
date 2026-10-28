"""Deterministic synthetic maintenance data for three-phase induction motors.

All records are fabricated for development and evaluation. Nothing here is
real plant data. The generator also writes a golden query set whose relevance
labels come from construction (each query is generated from a known fault).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

EQUIPMENT_IDS = ["MTR-001", "MTR-002", "MTR-003", "MTR-004"]
DEGRADING_MOTOR = "MTR-003"  # carries a planted bearing/winding degradation trend


@dataclass(frozen=True)
class FaultSpec:
    code: str
    name: str
    manual_file: str
    symptoms: tuple[str, ...]
    paraphrases: tuple[str, ...]  # held-out style queries, never copied verbatim
    root_cause: str
    action: str


FAULTS: dict[str, FaultSpec] = {
    "BRG-WEAR": FaultSpec(
        code="BRG-WEAR",
        name="Bearing wear",
        manual_file="bearings.md",
        symptoms=(
            "High vibration at drive end bearing housing",
            "Metallic noise from bearing housing during operation",
            "Vibration RMS rising steadily over several shifts",
        ),
        paraphrases=(
            "why is the drive end shaking so much",
            "grinding sound coming from the motor bearing",
            "vibration keeps climbing on this motor for days",
        ),
        root_cause="Lubricant degradation and fatigue caused bearing race wear.",
        action="Replaced the drive-end bearing and re-greased the housing with the specified grease.",
    ),
    "MISALIGN": FaultSpec(
        code="MISALIGN",
        name="Shaft misalignment",
        manual_file="alignment.md",
        symptoms=(
            "Coupling vibration at twice running speed",
            "Axial vibration increased after coupling service",
            "Coupling bolts loosening and vibration after reassembly",
        ),
        paraphrases=(
            "shaking from the coupling side after maintenance",
            "why does the motor vibrate at double speed",
            "pump and motor seem out of line",
        ),
        root_cause="Shaft misalignment between motor and pump exceeded tolerance.",
        action="Realigned the shaft with laser alignment and re-torqued the coupling bolts.",
    ),
    "WIND-INS": FaultSpec(
        code="WIND-INS",
        name="Winding insulation degradation",
        manual_file="winding.md",
        symptoms=(
            "Insulation resistance dropped during megger test",
            "Winding temperature high at no load",
            "Tripped on earth fault after heating period",
        ),
        paraphrases=(
            "motor failing insulation test after long running",
            "stator running hot even when lightly loaded",
            "earth leakage after the motor overheated repeatedly",
        ),
        root_cause="Winding insulation degraded from sustained overheating.",
        action="Rewound the stator and applied new insulation varnish; retested insulation resistance.",
    ),
    "UNBAL": FaultSpec(
        code="UNBAL",
        name="Rotor unbalance",
        manual_file="cooling.md",
        symptoms=(
            "1x running speed vibration after fan service",
            "Vibration spike at running speed after cleaning",
            "Rotor vibration grew after dust accumulation",
        ),
        paraphrases=(
            "motor started vibrating after someone cleaned the fan",
            "why is the shaft wobbling at normal speed",
            "dust on the fan seems to be shaking the whole unit",
        ),
        root_cause="Dust build-up on cooling fan blades caused rotor imbalance.",
        action="Cleaned the cooling fan and rebalanced the rotor.",
    ),
    "COOL-FAIL": FaultSpec(
        code="COOL-FAIL",
        name="Cooling path blockage",
        manual_file="cooling.md",
        symptoms=(
            "Motor temperature rising steadily under load",
            "Frame temperature above alarm limit on hot days",
            "Reduced airflow over cooling fins",
        ),
        paraphrases=(
            "why is the motor body getting hotter over time",
            "temperature alarm keeps triggering under full load",
            "air is not moving across the fins anymore",
        ),
        root_cause="Debris blocked the cooling fan airflow, reducing heat removal.",
        action="Cleared the cooling fins and replaced the fan shroud.",
    ),
}

MANUAL_TEXT: dict[str, str] = {
    "bearings.md": """# Bearing Maintenance Manual: Induction Motors

## BRG-WEAR: Bearing wear
Procedure. Measure vibration RMS at the drive end and non-drive end with a handheld analyzer. Compare readings against the baseline recorded at commissioning.
Diagnosis. A steady rise in vibration over several shifts, together with metallic noise from the housing, indicates bearing race wear. Check lubricant condition and grease service interval.
Recommended action. Replace the drive-end bearing and re-grease the housing with the grade listed in the lubrication schedule. Record the new baseline after the run-in period.
""",
    "alignment.md": """# Alignment Manual: Coupled Pump and Motor Sets

## MISALIGN: Shaft misalignment
Procedure. Check coupling runout and axial vibration after any coupling service. Confirm that vibration at twice running speed is present.
Diagnosis. Vibration at twice running speed and rising axial vibration after reassembly point to shaft misalignment between motor and pump.
Recommended action. Realign the shaft with laser alignment equipment within tolerance. Re-torque the coupling bolts to specification and verify again after a warm-up run.
""",
    "winding.md": """# Winding Maintenance Manual: Stator Insulation

## WIND-INS: Winding insulation degradation
Procedure. Run an insulation resistance (megger) test at the specified voltage before returning the motor to service. Record winding temperature at no load.
Diagnosis. Low insulation resistance, high winding temperature at no load, or earth faults after heating periods indicate insulation degradation from sustained overheating.
Recommended action. Rewind the stator if resistance is below the limit. Apply new insulation varnish and retest insulation resistance before commissioning.
""",
    "cooling.md": """# Cooling and Rotor Manual: Induction Motors

## UNBAL: Rotor unbalance
Procedure. Inspect the cooling fan and rotor for dust build-up. Measure vibration at running speed before and after cleaning.
Diagnosis. A vibration increase at 1x running speed after fan service or dust accumulation indicates rotor imbalance.
Recommended action. Clean the cooling fan thoroughly and rebalance the rotor to the grade in the manual.

## COOL-FAIL: Cooling path blockage
Procedure. Check frame temperature under load and inspect the cooling fins and fan shroud for blockage.
Diagnosis. Frame temperature rising steadily under load, or reduced airflow over the fins, indicates the cooling path is blocked.
Recommended action. Clear the cooling fins and replace the fan shroud if it is damaged. Confirm temperature returns to baseline under load.
""",
}


def _work_order_text(wo: dict) -> str:
    return (
        f"Work order {wo['wo_id']} on {wo['equipment_id']} ({wo['date']}).\n"
        f"Fault code: {wo['fault_code']} ({FAULTS[wo['fault_code']].name}).\n"
        f"Symptoms: {wo['symptoms']}.\n"
        f"Root cause: {wo['root_cause']}\n"
        f"Repair action: {wo['repair_action']}\n"
        f"Outcome: {wo['outcome']}."
    )


def generate_work_orders(rng: random.Random, n: int = 40) -> pd.DataFrame:
    rows = []
    start = datetime(2026, 1, 5)
    for i in range(n):
        code = rng.choice(list(FAULTS))
        spec = FAULTS[code]
        symptom = rng.choice(spec.symptoms)
        recurred = rng.random() < 0.15
        rows.append(
            {
                "wo_id": f"WO-{i + 1:04d}",
                "equipment_id": rng.choice(EQUIPMENT_IDS),
                "date": (start + timedelta(days=rng.randint(0, 90))).date().isoformat(),
                "fault_code": code,
                "symptoms": symptom,
                "root_cause": spec.root_cause,
                "repair_action": spec.action,
                "outcome": "recurred within 30 days" if recurred else "resolved, no recurrence in 30 days",
                "downtime_hours": round(rng.uniform(2, 36), 1),
            }
        )
    df = pd.DataFrame(rows)
    df["text"] = [_work_order_text(r) for r in df.to_dict("records")]
    return df


def generate_sensor_readings(rng: np.random.Generator, hours: int = 168) -> pd.DataFrame:
    """Hourly readings for each motor over one week. MTR-003 degrades from day 4."""
    base_time = datetime(2026, 10, 20, 0, 0)
    frames = []
    for equipment in EQUIPMENT_IDS:
        t = np.arange(hours)
        degrade = np.where(t >= 96, (t - 96) / 72.0, 0.0) if equipment == DEGRADING_MOTOR else np.zeros(hours)
        vibration = 1.8 + 0.08 * rng.standard_normal(hours) + 1.6 * degrade
        temperature = 55.0 + 0.8 * rng.standard_normal(hours) + 14.0 * degrade
        current = 12.0 + 0.15 * rng.standard_normal(hours) + 0.9 * degrade
        rpm = 1480 + 6.0 * rng.standard_normal(hours) - 8.0 * degrade
        frames.append(
            pd.DataFrame(
                {
                    "equipment_id": equipment,
                    "timestamp": [base_time + timedelta(hours=int(h)) for h in t],
                    "vibration_rms_mm_s": vibration.round(3),
                    "temperature_c": temperature.round(2),
                    "phase_current_a": current.round(3),
                    "rpm": rpm.round(1),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def build_golden_queries(work_orders: pd.DataFrame, rng: random.Random) -> list[dict]:
    """Relevance labels from construction: each query relates to a fault family.

    A query is relevant to (a) the manual section for its fault and (b) work
    orders with the same fault code. Paraphrased queries are held out from the
    symptom text the index sees directly, so they test semantic matching.
    """
    queries = []
    for code, spec in FAULTS.items():
        relevant = {f"{spec.manual_file}#{code}"}
        relevant |= set(work_orders.loc[work_orders["fault_code"] == code, "wo_id"])
        for j, text in enumerate(spec.paraphrases):
            split = "test" if j == len(spec.paraphrases) - 1 else "dev"
            queries.append(
                {
                    "query_id": f"{code}-{j + 1}",
                    "question": text,
                    "relevant_ids": sorted(relevant),
                    "split": split,
                }
            )
    rng.shuffle(queries)
    return queries


def write_dataset(root: Path, seed: int = 42, n_work_orders: int = 40) -> dict[str, Path]:
    rng = random.Random(seed)
    nprng = np.random.default_rng(seed)
    manuals_dir = root / "raw" / "manuals"
    wo_dir = root / "raw" / "work_orders"
    sensor_dir = root / "raw" / "sensors"
    for d in (manuals_dir, wo_dir, sensor_dir):
        d.mkdir(parents=True, exist_ok=True)

    for name, text in MANUAL_TEXT.items():
        (manuals_dir / name).write_text(text, encoding="utf-8")

    work_orders = generate_work_orders(rng, n_work_orders)
    wo_path = wo_dir / "work_orders.csv"
    work_orders.to_csv(wo_path, index=False)

    sensor_path = sensor_dir / "motor_readings.csv"
    generate_sensor_readings(nprng).to_csv(sensor_path, index=False)

    golden_path = root.parent / "eval" / "golden_set.jsonl"
    golden_path.parent.mkdir(parents=True, exist_ok=True)
    with golden_path.open("w", encoding="utf-8") as fh:
        for row in build_golden_queries(work_orders, rng):
            fh.write(json.dumps(row) + "\n")

    return {"manuals": manuals_dir, "work_orders": wo_path, "sensors": sensor_path, "golden": golden_path}
