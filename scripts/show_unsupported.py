"""Print every cited sentence that the lexical check did not mark as supported.

Use it to see WHY a claim scored low: the wording may be a fair paraphrase, or the claim may
go beyond the evidence. Only a human reading the cited record can tell which.
"""

from __future__ import annotations

import json

from maintainiq.config import get_settings
from maintainiq.pipeline import build_system


def main() -> int:
    settings = get_settings()
    system = build_system(settings)
    rows = [json.loads(line) for line in
            (settings.eval_dir / "answer_set.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    found = 0
    for row in rows:
        if not row["answerable"]:
            continue
        result = system.engine.investigate(row["question"], top_k=5)
        if result.insufficient:
            print(f"REFUSED (unexpected): {row['question']}")
            continue
        for check in result.validation.sentences:
            if not check.supported:
                found += 1
                print(f"\nQ: {row['question']}")
                print(f"  claim (lexical support {check.lexical_support}): {check.sentence}")
                for cite in check.citations:
                    print(f"  cited {cite}: {result.evidence_ids.get(cite, '')[:200]}")
    print(f"\n{found} unsupported claim(s) found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())