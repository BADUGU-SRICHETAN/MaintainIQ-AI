"""Ask one question from the command line and print the cited answer and validation."""

from __future__ import annotations

import sys

from maintainiq.pipeline import build_system


def main() -> int:
    question = " ".join(sys.argv[1:]) or "Why is the motor vibrating at double running speed?"
    system = build_system()
    result = system.engine.investigate(question, top_k=5)
    print(f"Backend: {result.backend}")
    print(f"Insufficient: {result.insufficient}")
    print(f"\nAnswer:\n{result.answer}\n")
    print("Evidence:")
    for n, ev in enumerate(result.evidence, start=1):
        print(f"  E{n} {ev.document.doc_id:<22} rrf={ev.rrf_score:.4f} dense={ev.dense_similarity:.3f}")
    print(f"\nCitation coverage: {result.validation.citation_coverage:.2f}  "
          f"invalid: {result.validation.invalid_citation_count}  "
          f"lexically supported: {result.validation.lexically_supported_rate:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
