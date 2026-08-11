"""Qualitative error analysis sample selection and export."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from evaluation.conditions import CONDITION_SYMBOLIC
from evaluation.metrics import hr_from_citation_dicts


def _per_case_paired_hr(report: dict[str, Any]) -> float:
    return hr_from_citation_dicts(report.get("citations_draft", []))


def select_error_analysis_cases(
    reports: list[dict[str, Any]],
    *,
    n_corrected: int = 20,
    n_failed: int = 20,
) -> dict[str, list[dict[str, Any]]]:
    """
    Select cases for qualitative review.

    - corrected: HR dropped after symbolic correction
    - failed: HR unchanged or increased, or still has bad citations
    """
    corrected: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    for report in reports:
        hr_draft = _per_case_paired_hr(report)
        sym = report.get("conditions", {}).get(CONDITION_SYMBOLIC, {})
        hr_final = sym.get("hr_paired", sym.get("hr_final", hr_draft))
        delta = hr_draft - hr_final

        entry = {
            "case_id": report["case_id"],
            "hr_draft": hr_draft,
            "hr_final": hr_final,
            "hr_delta": delta,
            "correction_iterations": sym.get("correction_iterations", 0),
            "draft_summary_excerpt": report["draft_summary"][:500],
            "final_summary_excerpt": sym.get("final_summary", "")[:500],
            "n_citations_draft": len(report.get("citations_draft", [])),
            "bad_labels_draft": _bad_label_summary(report.get("citations_draft", [])),
        }

        if delta > 0 or (hr_draft > 0 and hr_final == 0):
            corrected.append(entry)
        elif hr_final > 0 or delta < 0:
            failed.append(entry)

    corrected.sort(key=lambda x: x["hr_delta"], reverse=True)
    failed.sort(key=lambda x: x["hr_final"], reverse=True)

    return {
        "corrected": corrected[:n_corrected],
        "failed": failed[:n_failed],
    }


def _bad_label_summary(citations: list[dict[str, Any]]) -> str:
    from collections import Counter

    bad = Counter(c.get("label") for c in citations if c.get("label") not in ("Verified",))
    return "; ".join(f"{k}:{v}" for k, v in bad.items())


def export_error_analysis_csv(
    reports: list[dict[str, Any]],
    path: str | Path,
    *,
    n_corrected: int = 20,
    n_failed: int = 20,
) -> Path:
    """Export qualitative review template CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = select_error_analysis_cases(
        reports, n_corrected=n_corrected, n_failed=n_failed
    )

    fieldnames = [
        "category",
        "case_id",
        "hr_draft",
        "hr_final",
        "hr_delta",
        "correction_iterations",
        "bad_labels_draft",
        "draft_summary_excerpt",
        "final_summary_excerpt",
        "failure_mode",
        "reviewer_notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in selected["corrected"]:
            row = {**row, "category": "corrected", "failure_mode": "", "reviewer_notes": ""}
            writer.writerow(row)
        for row in selected["failed"]:
            row = {**row, "category": "failed", "failure_mode": "", "reviewer_notes": ""}
            writer.writerow(row)
    return path


def save_error_analysis_json(
    reports: list[dict[str, Any]],
    path: str | Path,
    **kwargs: Any,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = select_error_analysis_cases(reports, **kwargs)
    path.write_text(json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
