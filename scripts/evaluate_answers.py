"""Answer-level evaluation: refusal accuracy and citation validity over a labelled question set.

Usage:
    python scripts/evaluate_answers.py                       # tuned set (eval/answer_set.jsonl)
    python scripts/evaluate_answers.py --set eval/heldout_set.jsonl --out reports/heldout_answer_eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from maintainiq.config import get_settings
from maintainiq.pipeline import build_system


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", dest="question_set", type=Path, default=None,
                        help="JSONL file of questions with an 'answerable' flag")
    parser.add_argument("--out", type=Path, default=None, help="where to write the JSON report")
    args = parser.parse_args()

    settings = get_settings()
    system = build_system(settings)
    answer_set = args.question_set or settings.eval_dir / "answer_set.jsonl"
    set_name = answer_set.name
    rows = [json.loads(line) for line in answer_set.read_text(encoding="utf-8").splitlines() if line.strip()]

    cases, correct_refusals, invalid = [], 0, 0
    all_claims = cited_claims = supported_claims = uncited_claims = 0
    for row in rows:
        result = system.engine.investigate(row["question"], top_k=5)
        refused = result.insufficient
        expected_refusal = not row["answerable"]
        correct_refusals += refused == expected_refusal
        if not refused:
            # Every claim counts in the denominator, so an uncited claim cannot disappear from the metric.
            all_claims += len(result.validation.sentences)
            uncited_claims += sum(s.uncited for s in result.validation.sentences)
            cited_claims += sum(not s.uncited for s in result.validation.sentences)
            supported_claims += sum(s.supported for s in result.validation.sentences)
            invalid += result.validation.invalid_citation_count
        cases.append({
            "question": row["question"],
            "answerable": row["answerable"],
            "refused": refused,
            "correct": refused == expected_refusal,
            "answer": result.answer[:300],
            "refusal_reason": result.refusal_reason,
            "raw_model_output": result.raw_answer[:600],
            "top_evidence": [ev.document.doc_id for ev in result.evidence[:3]],
            "top_similarity": round(result.evidence[0].dense_similarity, 3) if result.evidence else None,
        })

    answerable = [c for c in cases if c["answerable"]]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "question_set": set_name,
        "generation_backend": system.engine.backend.name,
        "questions": len(cases),
        "refusal_accuracy": round(correct_refusals / len(cases), 4),
        "answerable_answered_rate": round(sum(not c["refused"] for c in answerable) / len(answerable), 4),
        "unanswerable_refused_rate": round(
            sum(c["refused"] for c in cases if not c["answerable"]) / sum(not c["answerable"] for c in cases), 4
        ),
        "claims_total": all_claims,
        "uncited_claims": uncited_claims,
        "claims_supported_lexically": round(supported_claims / all_claims, 4) if all_claims else None,
        "invalid_citation_ids": invalid,
        "cases": cases,
        "limitations": [
            "The tuned set (answer_set.jsonl) was used to develop the gates, so it is NOT a held-out result.",
            "A held-out set is only meaningful if it was written before tuning and never used to change code.",
            "Lexical support is word overlap, not semantic entailment.",
            "Sensor summaries contain numbers, so exact-value questions can pass the numeric gate "
            "when a related number exists.",
        ],
    }
    out = args.out or settings.report_dir / "answer_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = {k: v for k, v in report.items() if k not in ("cases", "limitations")}
    print(json.dumps(summary, indent=2))
    for c in cases:
        flag = "OK " if c["correct"] else "BAD"
        print(f"{flag} answerable={c['answerable']!s:<5} refused={c['refused']!s:<5} "
              f"reason={c['refusal_reason']!s:<20} {c['question']}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())