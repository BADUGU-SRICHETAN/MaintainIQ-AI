"""Load manuals, work orders and sensor summaries into citable Documents."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from maintainiq.config import Settings
from maintainiq.schemas import (
    SOURCE_MANUAL,
    SOURCE_SENSOR_SUMMARY,
    SOURCE_WORK_ORDER,
    Document,
)

_HEADING = re.compile(r"^##\s+([A-Z0-9][A-Z0-9-]*):\s*(.+)$", re.MULTILINE)


def split_manual(text: str, file_name: str) -> list[tuple[str, str, str]]:
    """Split a manual on `## CODE: Title` headings. Returns (section_id, title, body)."""
    matches = list(_HEADING.finditer(text))
    sections = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        code, title = match.group(1), match.group(2).strip()
        body = text[match.end():end].strip()
        sections.append((f"{file_name}#{code}", title, body))
    return sections


def window_text(text: str, size: int, overlap: int) -> list[str]:
    """Character windows that prefer to end on a sentence boundary."""
    if len(text) <= size:
        return [text]
    if overlap >= size:
        raise ValueError("chunk overlap must be smaller than chunk size")
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind(". ", start + size // 2, end)
            if boundary != -1:
                end = boundary + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


def load_manuals(manual_dir: Path, settings: Settings) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(manual_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for section_id, title, body in split_manual(text, path.name):
            parts = window_text(body, settings.chunk_chars, settings.chunk_overlap)
            for n, part in enumerate(parts):
                suffix = f"-p{n + 1}" if len(parts) > 1 else ""
                docs.append(
                    Document(
                        doc_id=f"{section_id}{suffix}",
                        text=f"{title}. {part}",
                        source_type=SOURCE_MANUAL,
                        title=f"{path.name}: {title}",
                        metadata={
                            "section_id": section_id,
                            "fault_code": section_id.split("#", 1)[1],
                            "file": path.name,
                        },
                    )
                )
    return docs


def load_work_orders(csv_path: Path) -> list[Document]:
    df = pd.read_csv(csv_path)
    required = {"wo_id", "equipment_id", "fault_code", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"work order file is missing columns: {sorted(missing)}")
    docs = []
    for row in df.itertuples(index=False):
        docs.append(
            Document(
                doc_id=row.wo_id,
                text=row.text,
                source_type=SOURCE_WORK_ORDER,
                title=f"Work order {row.wo_id}",
                metadata={
                    "equipment_id": row.equipment_id,
                    "fault_code": row.fault_code,
                    "date": str(row.date),
                    "outcome": row.outcome,
                },
            )
        )
    return docs


def build_sensor_summary_documents(summaries: list[dict]) -> list[Document]:
    """Turn sensor analytics output into citable evidence (see sensors.py)."""
    docs = []
    for s in summaries:
        docs.append(
            Document(
                doc_id=f"SENSOR-{s['equipment_id']}",
                text=s["text"],
                source_type=SOURCE_SENSOR_SUMMARY,
                title=f"Sensor summary {s['equipment_id']}",
                metadata={"equipment_id": s["equipment_id"]},
            )
        )
    return docs
