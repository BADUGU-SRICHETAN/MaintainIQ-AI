"""MaintenIQ AI dashboard: cited investigation, evidence trace, and sensor analytics."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from maintainiq.config import get_settings  # noqa: E402
from maintainiq.pipeline import build_system  # noqa: E402
from maintainiq.sensors import load_readings, summarize_all  # noqa: E402

st.set_page_config(page_title="MaintenIQ AI", page_icon="🛠️", layout="wide")


@st.cache_resource(show_spinner="Indexing maintenance records...")
def load_system():
    return build_system(get_settings())


@st.cache_data(show_spinner=False)
def load_sensor_frame() -> pd.DataFrame:
    return load_readings(get_settings().data_dir / "sensors" / "motor_readings.csv")


system = load_system()
settings = system.settings
readings = load_sensor_frame()
equipment_ids = sorted(readings["equipment_id"].unique())

with st.sidebar:
    st.title("MaintenIQ AI")
    st.caption("Hybrid retrieval over maintenance records, with cited investigation.")
    scope = st.selectbox("Equipment scope", ["All equipment", *equipment_ids])
    top_k = st.slider("Evidence items", 3, 10, 5)
    st.divider()
    st.markdown(f"**Dense backend:** `{system.retriever._dense.backend}`")
    st.markdown(f"**Documents indexed:** {len(system.documents)}")
    st.markdown(f"**Generation:** `{system.engine.backend.name}`")
    st.info("All maintenance records and sensor readings are synthetic. See the README for limits.")

investigate_tab, evidence_tab, sensors_tab = st.tabs(["Investigation", "Evidence trace", "Sensor analytics"])

with investigate_tab:
    question = st.text_input(
        "Investigation question",
        placeholder="e.g. Why is the motor vibrating at double running speed?",
    )
    if st.button("Investigate", type="primary", disabled=not question.strip()):
        equipment = None if scope == "All equipment" else scope
        with st.spinner("Retrieving evidence and generating a cited answer..."):
            st.session_state["result"] = system.engine.investigate(question, top_k=top_k, equipment_id=equipment)

    result = st.session_state.get("result")
    if result is None:
        st.caption("Ask a question about a fault, symptom, or repair.")
    else:
        if result.insufficient:
            st.warning(result.answer)
        else:
            st.markdown(result.answer)

        v = result.validation
        c1, c2, c3 = st.columns(3)
        c1.metric("Claims with citations", f"{v.citation_coverage:.0%}" if v.claim_count else "n/a")
        c2.metric("Invalid citation IDs", v.invalid_citation_count)
        c3.metric("Lexically supported claims", f"{v.lexically_supported_rate:.0%}" if v.claim_count else "n/a")
        st.caption(
            "Lexical support is word overlap with the cited evidence. It is a guardrail, "
            "not proof that a claim is correct. Check the cited records."
        )
        if result.evidence:
            st.subheader("Cited evidence")
            for n, ev in enumerate(result.evidence, start=1):
                with st.expander(f"[E{n}] {ev.document.title}  ·  {ev.document.doc_id}"):
                    st.write(ev.document.text)

with evidence_tab:
    result = st.session_state.get("result")
    if result is None or not result.evidence:
        st.caption("Run an investigation to see the retrieval trace.")
    else:
        rows = [
            {
                "ref": f"E{n}",
                "doc_id": ev.document.doc_id,
                "source": ev.document.source_type,
                "equipment": ev.document.equipment_id or "shared",
                "bm25_rank": ev.bm25_rank,
                "dense_rank": ev.dense_rank,
                "dense_similarity": round(ev.dense_similarity, 3),
                "rrf_score": round(ev.rrf_score, 4),
            }
            for n, ev in enumerate(result.evidence, start=1)
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(
            "BM25 catches exact terms and fault codes. Dense similarity catches paraphrases. "
            "Reciprocal Rank Fusion combines the two rankings, so their score scales never need to match."
        )

with sensors_tab:
    summaries = {s.equipment_id: s for s in summarize_all(readings, settings)}
    selected = st.selectbox("Motor", equipment_ids, key="sensor_motor")
    s = summaries[selected]
    if s.anomalous_metrics:
        st.error(f"{selected}: deviation from baseline in {', '.join(s.anomalous_metrics)}")
    else:
        st.success(f"{selected}: no sustained deviation from baseline")
    table = pd.DataFrame(
        [
            {
                "metric": m.metric,
                "baseline": m.baseline_mean,
                "latest": m.latest,
                "trend /day": m.slope_per_day,
                "max |z|": m.max_z,
                "first anomaly": m.first_anomaly or "-",
            }
            for m in s.metrics.values()
        ]
    )
    st.dataframe(table, hide_index=True, width="stretch")
    series = readings[readings["equipment_id"] == selected].set_index("timestamp")
    left, right = st.columns(2)
    left.line_chart(series[["vibration_rms_mm_s"]], y_label="mm/s", x_label=None)
    right.line_chart(series[["temperature_c"]], y_label="°C", x_label=None)
