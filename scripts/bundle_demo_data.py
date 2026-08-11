#!/usr/bin/env python3
"""
Bundle saved pipeline outputs into a compact JSON for the demo UI.

This is build-time only: the Canvas must embed data inline (no fetch calls).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _truncate(s: str, n: int) -> str:
    s = s or ""
    if len(s) <= n:
        return s
    return s[:n] + "\n\n[truncated]"


def _serialize_citation_row(v: dict[str, Any]) -> dict[str, Any]:
    # `v` is serialized from VerificationResult via `asdict()`.
    # Expected shape:
    # {
    #   label, cl_status, explanation, matched_case_name, matched_date_filed,
    #   citation: { raw_text, span, ... }
    # }
    c = v.get("citation") or {}
    return {
        "raw_text": c.get("raw_text"),
        "normalized": c.get("normalized"),
        "label": v.get("label"),
        "cl_status": v.get("cl_status"),
        "matched_case_name": v.get("matched_case_name"),
        "explanation": v.get("explanation"),
        "span": c.get("span"),
        "pin_cite": c.get("pin_cite"),
        "citation_id": c.get("citation_id"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Bundle demo data for canvas.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "full_pipeline_results",
        help="Folder containing results/, conditions/, generations/ (e.g. full_pipeline_results/).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "demo" / "demo_bundle.json",
    )
    parser.add_argument(
        "--case-order",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config" / "citation_rich_validation_top10.json",
        help="JSON artifact with pinned case_ids (written by scan_citation_rich_cases.py).",
    )
    parser.add_argument("--draft-chars", type=int, default=6500)
    parser.add_argument("--gold-chars", type=int, default=900)
    args = parser.parse_args()

    results_dir: Path = args.results_dir
    conditions_dir = results_dir / "conditions"
    gens_dir = results_dir / "generations"

    if not conditions_dir.exists():
        raise FileNotFoundError(f"Missing conditions dir: {conditions_dir}")
    if not (results_dir / "results" / "metrics_summary.json").exists():
        raise FileNotFoundError(f"Missing metrics_summary.json under {results_dir / 'results'}")

    pinned_case_ids: list[str] | None = None
    if args.case_order.exists():
        payload = _load_json(args.case_order)
        pinned_case_ids = payload.get("case_ids") or None

    metrics = _load_json(results_dir / "results" / "metrics_summary.json")
    rq_answers = _load_json(results_dir / "results" / "rq_answers.json")
    stats = _load_json(results_dir / "results" / "statistical_tests.json")
    batch_summary = _load_json(results_dir / "conditions" / "batch_summary.json")

    case_files = sorted(conditions_dir.glob("*.json"))
    reports_by_id: dict[str, dict[str, Any]] = {}
    for p in case_files:
        report = _load_json(p)
        cid = report.get("case_id")
        if cid:
            reports_by_id[cid] = report

    if pinned_case_ids:
        case_ids = [cid for cid in pinned_case_ids if cid in reports_by_id]
    else:
        case_ids = sorted(reports_by_id.keys())

    if not case_ids:
        raise RuntimeError(f"No cases found in {conditions_dir}")

    # Draft summaries are also in generations JSONL, but the condition report
    # already stores the draft text used for extraction/verification.
    # We'll trust `report["draft_summary"]` for consistency.
    cases: list[dict[str, Any]] = []
    for cid in case_ids:
        report = reports_by_id[cid]
        symbolic = (report.get("conditions") or {}).get("symbolic") or {}
        llm = (report.get("conditions") or {}).get("llm_rewrite") or {}

        citations_draft = [
            _serialize_citation_row(v) for v in (report.get("citations_draft") or [])
        ]
        citations_symbolic = [
            _serialize_citation_row(v)
            for v in (symbolic.get("citations_final") or [])
        ]
        citations_llm = [
            _serialize_citation_row(v)
            for v in (llm.get("citations_final") or [])
        ]

        cases.append(
            {
                "case_id": cid,
                "hr_draft": report.get("hr_draft"),
                "hr_symbolic": symbolic.get("hr_final"),
                "hr_symbolic_surviving": symbolic.get("hr_surviving"),
                "hr_llm_rewrite": llm.get("hr_final"),
                "hr_llm_rewrite_surviving": llm.get("hr_surviving"),
                "truncated": report.get("truncated"),
                "source_token_count": report.get("source_token_count"),
                "correction_iterations": symbolic.get("correction_iterations"),
                "correction_outcomes": symbolic.get("correction_outcomes", []),
                "gold_citation_count": None,
                # Excerpts for display (full text stays on disk).
                "draft_summary": _truncate(str(report.get("draft_summary") or ""), args.draft_chars),
                "gold_summary_excerpt": _truncate(
                    str(report.get("gold_summary") or ""), args.gold_chars
                ),
                "symbolic_summary": _truncate(
                    str(symbolic.get("final_summary") or ""), args.draft_chars
                ),
                "llm_summary": _truncate(
                    str(llm.get("final_summary") or ""), args.draft_chars
                ),
                "citations_draft": citations_draft,
                "citations_symbolic": citations_symbolic,
                "citations_llm_rewrite": citations_llm,
            }
        )

    # `gold_citation_count` isn't always present in artifacts; compute cheaply
    # from counts within bundle construction.
    # (We avoid importing eyecite here to keep bundling fast and offline.)
    # If you want exact counts, re-run with a richer bundler that precomputes.
    for c in cases:
        # Use the count of gold citations from pinned artifact as a proxy
        if pinned_case_ids and args.case_order.exists():
            payload = _load_json(args.case_order)
            counts = payload.get("counts") or {}
            c["gold_citation_count"] = counts.get(c["case_id"])

    presentation_label = (
        rq_answers.get("presentation_label")
        or "PRELIMINARY — n=10 citation-rich validation (D→L)"
    )

    bundle = {
        "meta": {
            "presentation_label": presentation_label,
            "n_cases": len(case_ids),
            "case_ids": case_ids,
        },
        "aggregate": {
            "metrics_by_condition": metrics,
            "rq": rq_answers,
            "stats": stats,
            "batch_summary": batch_summary,
        },
        "cases": cases,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote demo bundle: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

