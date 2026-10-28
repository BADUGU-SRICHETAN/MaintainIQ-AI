"""Runtime configuration loaded from environment variables (and optional .env)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


def _load_dotenv(path: Path) -> None:
    """Tiny .env reader so the project has no hard dependency on python-dotenv."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    data_dir: Path = PROJECT_ROOT / "data" / "raw"
    eval_dir: Path = PROJECT_ROOT / "eval"
    report_dir: Path = PROJECT_ROOT / "reports"

    # Retrieval
    embedding_backend: str = _env("EMBEDDING_BACKEND", "lsa")  # "lsa" | "sentence-transformers"
    embedding_model: str = _env("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    rrf_k: int = int(_env("RRF_K", "60"))
    candidate_k: int = int(_env("CANDIDATE_K", "20"))
    chunk_chars: int = int(_env("CHUNK_CHARS", "800"))
    chunk_overlap: int = int(_env("CHUNK_OVERLAP", "120"))
    min_dense_similarity: float = float(_env("MIN_DENSE_SIMILARITY", "0.10"))
    min_term_coverage: float = float(_env("MIN_TERM_COVERAGE", "0.5"))

    # Generation
    llm_provider: str = _env("LLM_PROVIDER", "ollama")  # "ollama" | "extractive"
    llm_model: str = _env("LLM_MODEL", "llama3.2:latest")
    ollama_base_url: str = _env("OLLAMA_BASE_URL", "http://localhost:11434")
    llm_timeout_s: float = float(_env("LLM_TIMEOUT_S", "120"))

    # Analytics
    anomaly_z_threshold: float = float(_env("ANOMALY_Z_THRESHOLD", "3.0"))
    baseline_readings: int = int(_env("BASELINE_READINGS", "24"))


def get_settings() -> Settings:
    return Settings()
