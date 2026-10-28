"""Run retrieval evaluation on dev and held-out test splits and write a JSON report."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

from maintainiq.config import get_settings
from maintainiq.evaluation import evaluate_retrieval, load_golden
from maintainiq.pipeline import build_system


def main() -> int:
    settings = get_settings()
    system = build_system(settings)
    queries = load_golden(settings.eval_dir / "golden_set.jsonl")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dense_backend": system.retriever._dense.backend,
        "documents": len(system.documents),
        "retrieval": {split: evaluate_retrieval(system.retriever, queries, split).as_dict()
                      for split in ("dev", "test")},
    }
    settings.report_dir.mkdir(parents=True, exist_ok=True)
    out = settings.report_dir / "retrieval_eval.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
