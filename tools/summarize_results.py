#!/usr/bin/env python3
"""Summarize preserved ACT comparison CSVs (stdlib only; does not modify data)."""
from collections import defaultdict
import csv
from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / "reference_results"
NAMES = [
    "act_06_vs_07_5000_same_seed_30.csv",
    "act_07_08_30seeds.csv",
    "act_07_08_new50seeds.csv",
]


def summarize(path: Path):
    groups = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            groups[row["model"]].append(row)
    print(f"\n{path.name}")
    for model, rows in groups.items():
        count = len(rows)
        unique_seeds = {int(r["seed"]) for r in rows}
        if len(unique_seeds) != count:
            raise ValueError(f"duplicate seeds for {model}")
        once = sum(int(r["success_once"]) for r in rows)
        end = sum(int(r["success_at_end"]) for r in rows)
        print(f"  {model:19s}: once={once}/{count} ({100*once/count:.1f}%), "
              f"at_end={end}/{count} ({100*end/count:.1f}%)")


def main():
    for name in NAMES:
        summarize(BASE / name)


if __name__ == "__main__":
    main()
