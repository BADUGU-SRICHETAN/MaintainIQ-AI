"""Generate the synthetic dataset and golden evaluation set."""

from __future__ import annotations

import argparse
from pathlib import Path

from maintainiq.synthetic import write_dataset

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--work-orders", type=int, default=40)
    args = parser.parse_args()
    paths = write_dataset(ROOT / "data", seed=args.seed, n_work_orders=args.work_orders)
    for name, path in paths.items():
        print(f"{name:>12}: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
