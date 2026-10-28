"""One entry point that builds the corpus, the retriever and the investigation engine."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from maintainiq.config import Settings, get_settings
from maintainiq.ingestion import build_sensor_summary_documents, load_manuals, load_work_orders
from maintainiq.investigation import InvestigationEngine
from maintainiq.retrieval import HybridRetriever
from maintainiq.schemas import Document
from maintainiq.sensors import load_readings, summarize_all

logger = logging.getLogger(__name__)


@dataclass
class System:
    settings: Settings
    documents: list[Document]
    retriever: HybridRetriever
    engine: InvestigationEngine


def build_corpus(settings: Settings) -> list[Document]:
    raw = settings.data_dir
    docs: list[Document] = []
    docs += load_manuals(raw / "manuals", settings)
    docs += load_work_orders(raw / "work_orders" / "work_orders.csv")
    readings_path = raw / "sensors" / "motor_readings.csv"
    if readings_path.exists():
        summaries = summarize_all(load_readings(readings_path), settings)
        docs += build_sensor_summary_documents(
            [{"equipment_id": s.equipment_id, "text": s.to_text()} for s in summaries]
        )
    if not docs:
        raise FileNotFoundError(f"no documents found under {raw}; run scripts/generate_data.py")
    logger.info("corpus: %d documents", len(docs))
    return docs


def build_system(settings: Settings | None = None) -> System:
    settings = settings or get_settings()
    documents = build_corpus(settings)
    retriever = HybridRetriever(documents, settings)
    engine = InvestigationEngine(retriever, settings)
    return System(settings, documents, retriever, engine)
