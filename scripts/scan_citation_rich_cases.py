#!/usr/bin/env python3
"""Scan Multi-LexSum split for citation-rich cases (eyecite on gold summaries)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Rank cases by gold citation count")
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--min-citations", type=int, default=1)
    parser.add_argument(
        "--write-json",
        type=Path,
        default=ROOT / "config" / "citation_rich_validation_top10.json",
        help="Write top-N case_ids JSON artifact",
    )
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "hf")
    args = parser.parse_args()

    from data.case_selection import rank_cases_by_gold_citations, select_cases
    from data.load_multilexsum import get_split

    split = get_split(args.split, cache_dir=args.cache_dir)
    ranked = rank_cases_by_gold_citations(split)
    with_cites = [(cid, n) for cid, n in ranked if n >= args.min_citations]

    print(f"=== {args.split}: {len(with_cites)}/{len(split)} cases with >={args.min_citations} gold citation(s) ===\n")
    print(f"{'rank':>4}  {'case_id':<14}  {'gold_cites':>10}")
    print("-" * 32)
    for i, (cid, n) in enumerate(ranked[:50], start=1):
        if n < args.min_citations:
            break
        print(f"{i:4d}  {cid:<14}  {n:10d}")

    top_rows = select_cases(
        split,
        max_cases=args.top,
        min_gold_citations=args.min_citations,
    )
    top_ids = [row["id"] for row in top_rows]

    print(f"\nTop {args.top} case_ids:")
    for cid in top_ids:
        print(f"  - {cid}")

    if args.write_json:
        payload = {
            "split": args.split,
            "min_gold_citations": args.min_citations,
            "top_n": args.top,
            "case_ids": top_ids,
            "counts": {cid: n for cid, n in ranked if cid in top_ids},
        }
        args.write_json.parent.mkdir(parents=True, exist_ok=True)
        args.write_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote {args.write_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
