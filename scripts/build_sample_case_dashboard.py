#!/usr/bin/env python3
"""
Sample-case dashboard: one case (CJ-AZ-0002), all three generators.

Reads the same error_analysis_review.csv files compare_generators.py reads
from -- no hand-typed numbers. CJ-AZ-0002 is used because it is the only
case_id present in all three generators' review exports, so it is the one
case where PRIMERA, LLM zero-shot, and LLM+retrieval can be compared
directly against each other rather than across different cases.

Usage:
    cd code && source .venv/bin/activate
    python scripts/build_sample_case_dashboard.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASE_ID = "CJ-AZ-0002"

GENERATORS: list[tuple[str, str]] = [
    ("PRIMERA", "outputs/results_test100"),
    ("LLM zero-shot", "outputs/results_test100_llm"),
    ("LLM + retrieval", "outputs/results_test100_llm_rag"),
]


def _load_case_row(results_dir: Path, case_id: str) -> dict | None:
    csv_path = results_dir / "error_analysis_review.csv"
    if not csv_path.exists():
        print(f"  [skip] no review export at {csv_path}")
        return None
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["case_id"] == case_id:
                return row
    print(f"  [skip] {case_id} not found in {csv_path}")
    return None


def _bad_label_total(bad_labels_draft: str) -> int:
    """'Ambiguous:3; NotFound:23' -> 26"""
    total = 0
    for part in bad_labels_draft.split(";"):
        part = part.strip()
        if not part:
            continue
        _, count = part.split(":")
        total += int(count)
    return total


def main() -> int:
    rows = []
    for label, rel_dir in GENERATORS:
        row = _load_case_row(ROOT / rel_dir, CASE_ID)
        if row is None:
            print(f"Missing data for {label}; cannot build the dashboard.")
            return 1

        hr_draft = float(row["hr_draft"])
        hr_final = float(row["hr_final"])
        bad_draft = _bad_label_total(row["bad_labels_draft"])
        total = round(bad_draft / hr_draft) if hr_draft > 0 else bad_draft
        verified_draft = total - bad_draft
        still_bad_final = round(hr_final * total)
        corrected = bad_draft - still_bad_final

        rows.append(
            {
                "label": label,
                "total": total,
                "verified_draft": verified_draft,
                "corrected": corrected,
                "still_flagged": still_bad_final,
                "hr_draft": hr_draft,
                "hr_final": hr_final,
            }
        )
        print(
            f"{label}: total={total} verified_draft={verified_draft} "
            f"corrected={corrected} still_flagged={still_bad_final} "
            f"(HR {hr_draft:.1%} -> {hr_final:.1%})"
        )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [r["label"] for r in rows]
    verified = [r["verified_draft"] for r in rows]
    corrected = [r["corrected"] for r in rows]
    flagged = [r["still_flagged"] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = range(len(rows))
    ax.bar(x, verified, label="Verified at draft", color="#2a9d8f")
    ax.bar(x, corrected, bottom=verified, label="Corrected (redacted / repaired / re-anchored)", color="#e9c46a")
    bottom2 = [v + c for v, c in zip(verified, corrected)]
    ax.bar(x, flagged, bottom=bottom2, label="Still flagged after correction", color="#e76f51")

    for i, r in enumerate(rows):
        ax.text(
            i,
            r["total"] + 0.6,
            f"{r['total']} citations\nHR {r['hr_draft']:.0%} $\\rightarrow$ {r['hr_final']:.0%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Draft citations in this case")
    ax.set_title(f"Case {CASE_ID}: citation outcomes by generator")
    ax.set_ylim(0, max(r["total"] for r in rows) * 1.25)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()

    out_dir = ROOT.parent / "Documentation" / "figures" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sample_case_dashboard.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
