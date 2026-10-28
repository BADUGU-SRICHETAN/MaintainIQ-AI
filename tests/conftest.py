from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from maintainiq.config import Settings  # noqa: E402
from maintainiq.synthetic import write_dataset  # noqa: E402


@pytest.fixture(scope="session")
def dataset_root(tmp_path_factory) -> Path:
    """Generate a fresh dataset in a temp directory so tests never touch the repo's data."""
    root = tmp_path_factory.mktemp("data")
    write_dataset(root, seed=42)
    return root


@pytest.fixture(scope="session")
def settings(dataset_root: Path) -> Settings:
    return Settings(
        data_dir=dataset_root / "raw",
        eval_dir=dataset_root.parent / "eval",
        report_dir=dataset_root.parent / "reports",
        embedding_backend="lsa",
        llm_provider="extractive",
    )


@pytest.fixture(scope="session")
def system(settings: Settings):
    from maintainiq.pipeline import build_system

    return build_system(settings)
