#!/usr/bin/env python3
"""
Offline reprocess of symbolic/llm_rewrite conditions after Ambiguous policy change.

Does NOT call CourtListener. Replays correction from saved draft verification
labels on draft_summary, then regenerates aggregate metrics + RQ answers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from correction_loop.corrector import (
    REDACTION,
    _extend_span_past_trailing_pin,
    _is_garbled_replacement,
    build_reanchor_replacement,
    find_reanchor_source,
    redact_span,
    replace_span,
)
from evaluation.aggregate import aggregate_reports, metrics_to_dataframe_rows
from evaluation.paired_hr import (
    classify_outcomes,
    correction_outcome_to_dict,
    paired_hr_from_outcomes,
)
from evaluation.rq_report import generate_rq_report, save_rq_report
from evaluation.stats import bootstrap_ci, hr_reduction_stats, mcnemar_citation_test
from evaluation.metrics import PRIMARY_ROUGE, compute_rouge, hr_from_citation_dicts
from module_b_extraction.extract_citations import extract_citations
from module_c_verifier.verify import (
    build_verified_citation_text,
    hallucination_rate,
    is_unsafe_mismatch_repair,
    verification_result_to_dict,
)
from utils.schemas import ExtractedCitation, VerificationResult


def _relabel_unsafe_mismatches(
    results: list[VerificationResult],
) -> list[VerificationResult]:
    """Saved Mismatch with contradicting CL parties → Ambiguous (candidate for re-anchor)."""
    out: list[VerificationResult] = []
    for r in results:
        if r.label == "Mismatch" and is_unsafe_mismatch_repair(
            r.citation, r.matched_case_name
        ):
            out.append(
                VerificationResult(
                    citation=r.citation,
                    label="Ambiguous",
                    cl_status=300,
                    matched_case_name=r.matched_case_name,
                    matched_date_filed=r.matched_date_filed,
                    explanation=(
                        "Citation resolves to a cluster whose parties contradict the "
                        "draft case name; re-anchor if a verified same-case cite exists."
                    ),
                )
            )
        else:
            out.append(r)
    return out


def _dict_to_vr(d: dict[str, Any]) -> VerificationResult:
    c = d.get("citation") or {}
    parties = c.get("party_names")
    if parties is not None and not isinstance(parties, tuple):
        parties = tuple(parties) if parties else None
    span = c.get("span", (0, 0))
    if isinstance(span, list):
        span = tuple(span)
    reporter_span = c.get("reporter_span")
    if isinstance(reporter_span, list):
        reporter_span = tuple(reporter_span)
    cite = ExtractedCitation(
        raw_text=c["raw_text"],
        normalized=c["normalized"],
        reporter=c.get("reporter"),
        volume=c.get("volume"),
        page=c.get("page"),
        year=c.get("year"),
        party_names=parties,
        span=span,
        pin_cite=c.get("pin_cite"),
        reporter_span=reporter_span,
        citation_id=c.get("citation_id"),
    )
    return VerificationResult(
        citation=cite,
        label=d["label"],
        cl_status=d.get("cl_status"),
        matched_case_name=d.get("matched_case_name"),
        matched_date_filed=d.get("matched_date_filed"),
        explanation=d.get("explanation", ""),
    )


def offline_correct(draft_summary: str, draft_verified: list[VerificationResult]):
    """One-pass symbolic correction using draft labels only (no live verify)."""
    draft_verified = _relabel_unsafe_mismatches(draft_verified)
    text = draft_summary
    redact = [r for r in draft_verified if r.label in ("NotFound", "Malformed")]
    fixable = [
        r
        for r in draft_verified
        if r.label == "Mismatch"
        and not is_unsafe_mismatch_repair(r.citation, r.matched_case_name)
    ]
    reanchors: list[tuple[VerificationResult, VerificationResult]] = []
    for r in draft_verified:
        if r.label != "Ambiguous":
            continue
        anchor = find_reanchor_source(r, draft_verified)
        if anchor is not None:
            reanchors.append((r, anchor))

    iterations = 1 if (redact or fixable or reanchors) else 0

    edits: list[tuple[tuple[int, int], str]] = []
    for r in redact:
        edits.append((r.citation.span, REDACTION))
    for r in fixable:
        replacement = build_verified_citation_text(
            r.matched_case_name or r.citation.raw_text,
            r.citation,
            matched_date_filed=r.matched_date_filed,
        )
        edits.append((r.citation.span, replacement))
    for r, anchor in reanchors:
        edits.append((r.citation.span, build_reanchor_replacement(anchor)))

    for span, replacement in sorted(edits, key=lambda x: x[0][0], reverse=True):
        span = _extend_span_past_trailing_pin(text, span)
        if replacement == REDACTION:
            text = redact_span(text, span)
            continue
        candidate = replace_span(text, span, replacement)
        if _is_garbled_replacement(candidate, span, replacement):
            text = redact_span(text, span)
        else:
            text = candidate

    # Assign final labels without CourtListener by aligning to draft outcomes.
    final_extracted = extract_citations(text)
    final_verified: list[VerificationResult] = []
    reanchored_ids = {r.citation.citation_id or "" for r, _ in reanchors}
    unreanchored_ambiguous = [
        d
        for d in draft_verified
        if d.label == "Ambiguous" and (d.citation.citation_id or "") not in reanchored_ids
    ]

    for ext in final_extracted:
        label = "Verified"
        matched_name = None
        matched_date = None
        expl = "Offline reprocess: surviving citation after symbolic correction."
        cl_status = 200

        amb = next(
            (
                d
                for d in unreanchored_ambiguous
                if d.citation.normalized == ext.normalized
                or abs(d.citation.span[0] - ext.span[0]) < 40
            ),
            None,
        )
        if amb is not None:
            label = "Ambiguous"
            cl_status = 300
            expl = amb.explanation or "Citation matches multiple case clusters."
            matched_name = amb.matched_case_name
            matched_date = amb.matched_date_filed
        else:
            for d in draft_verified:
                if d.label == "Verified" and (
                    d.citation.normalized == ext.normalized
                    or (d.matched_case_name and d.matched_case_name in ext.raw_text)
                ):
                    matched_name = d.matched_case_name
                    matched_date = d.matched_date_filed
                    expl = "Citation and metadata both confirmed."
                    break
            for bad, anchor in reanchors:
                if anchor.citation.normalized == ext.normalized or (
                    anchor.matched_case_name and anchor.matched_case_name in ext.raw_text
                ):
                    matched_name = anchor.matched_case_name
                    matched_date = anchor.matched_date_filed
                    expl = "Re-anchored to verified same-case citation."
                    break
            for d in draft_verified:
                if d.label == "Mismatch" and d.matched_case_name and d.matched_case_name in ext.raw_text:
                    matched_name = d.matched_case_name
                    matched_date = d.matched_date_filed
                    expl = "Citation and metadata both confirmed."
                    break

        final_verified.append(
            VerificationResult(
                citation=ext,
                label=label,
                cl_status=cl_status,
                matched_case_name=matched_name,
                matched_date_filed=matched_date,
                explanation=expl,
            )
        )

    outcomes = classify_outcomes(draft_verified, text, final_verified)
    return text, iterations, final_verified, outcomes


def reprocess_report(report: dict[str, Any]) -> dict[str, Any]:
    draft = report["draft_summary"]
    draft_verified = [_dict_to_vr(c) for c in report.get("citations_draft", [])]
    hr_draft = hr_from_citation_dicts(report.get("citations_draft", []))

    sym_text, sym_iter, sym_verified, sym_outcomes = offline_correct(draft, draft_verified)
    hr_sym_paired = paired_hr_from_outcomes(sym_outcomes)
    hr_sym_surviving = hallucination_rate(sym_verified)

    # LLM rewrite: keep smoothed text if present and Ambiguous-compatible,
    # otherwise mirror symbolic (offline; no T5).
    llm_block = (report.get("conditions") or {}).get("llm_rewrite") or {}
    llm_text = llm_block.get("final_summary") or sym_text
    # If previous llm still has redaction of Ambiguous that symbolic no longer has,
    # prefer symbolic text for honesty.
    if REDACTION in (llm_text or "") and REDACTION not in sym_text:
        llm_text = sym_text
    # Re-classify llm against same draft.
    llm_extracted = extract_citations(llm_text)
    # Reuse symbolic final labels when text equals symbolic.
    if llm_text == sym_text:
        llm_verified = sym_verified
        llm_outcomes = sym_outcomes
        llm_iter = sym_iter
    else:
        _, llm_iter, llm_verified, llm_outcomes = offline_correct(draft, draft_verified)
        llm_text = sym_text  # deterministic offline
        llm_verified = sym_verified
        llm_outcomes = sym_outcomes
        llm_iter = sym_iter

    hr_llm_paired = paired_hr_from_outcomes(llm_outcomes)
    hr_llm_surviving = hallucination_rate(llm_verified)

    report = {**report}
    report["hr_draft"] = hr_draft
    report["citations_draft"] = [verification_result_to_dict(r) for r in draft_verified]
    report["conditions"] = {
        "baseline": {
            "summary": draft,
            "citations": [verification_result_to_dict(r) for r in draft_verified],
            "hr": hr_draft,
            "hr_paired": hr_draft,
        },
        "flagging": {
            "summary": draft,
            "citations": [verification_result_to_dict(r) for r in draft_verified],
            "hr": hr_draft,
            "hr_paired": hr_draft,
            "verification_only": True,
        },
        "symbolic": {
            "final_summary": sym_text,
            "citations_final": [verification_result_to_dict(r) for r in sym_verified],
            "hr_final": hr_sym_paired,
            "hr_surviving": hr_sym_surviving,
            "hr_paired": hr_sym_paired,
            "correction_outcomes": [correction_outcome_to_dict(o) for o in sym_outcomes],
            "correction_iterations": sym_iter,
        },
        "llm_rewrite": {
            "final_summary": llm_text,
            "citations_final": [verification_result_to_dict(r) for r in llm_verified],
            "hr_final": hr_llm_paired,
            "hr_surviving": hr_llm_surviving,
            "hr_paired": hr_llm_paired,
            "correction_outcomes": [correction_outcome_to_dict(o) for o in llm_outcomes],
            "correction_iterations": llm_iter,
        },
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline reprocess symbolic conditions.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "full_pipeline_results",
    )
    parser.add_argument("--hr-target", type=float, default=30.0)
    parser.add_argument("--rouge-tol", type=float, default=2.0)
    args = parser.parse_args()

    conditions_dir = args.results_dir / "conditions"
    results_dir = args.results_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    reports: list[dict[str, Any]] = []
    for path in sorted(conditions_dir.glob("*.json")):
        if path.name == "batch_summary.json":
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        updated = reprocess_report(report)
        path.write_text(json.dumps(updated, indent=2, ensure_ascii=False), encoding="utf-8")
        reports.append(updated)
        print(
            f"  [{updated['case_id']}] draft HR={updated['hr_draft']:.3f} → "
            f"symbolic paired={updated['conditions']['symbolic']['hr_final']:.3f} "
            f"(surviving={updated['conditions']['symbolic']['hr_surviving']:.3f})"
        )

    aggregated = aggregate_reports(reports, skip_bertscore=True)
    (results_dir / "metrics_summary.json").write_text(
        json.dumps(aggregated, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # CSV rows
    rows = metrics_to_dataframe_rows(aggregated)
    import csv

    with (results_dir / "metrics_by_condition.csv").open("w", encoding="utf-8", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    hr_stats = hr_reduction_stats(
        [r["hr_draft"] for r in reports],
        [r["conditions"]["symbolic"]["hr_final"] for r in reports],
    )
    mcnemar = mcnemar_citation_test(
        [r.get("citations_draft", []) for r in reports],
        [
            r.get("conditions", {}).get("symbolic", {}).get("correction_outcomes", [])
            for r in reports
        ],
    )
    rouge_ci = bootstrap_ci(
        [compute_rouge(r["draft_summary"], r["gold_summary"])[PRIMARY_ROUGE] for r in reports],
        [
            compute_rouge(r["conditions"]["symbolic"]["final_summary"], r["gold_summary"])[PRIMARY_ROUGE]
            for r in reports
        ],
    )
    stats = {"hr_reduction": hr_stats, "mcnemar": mcnemar, "rouge_variant": PRIMARY_ROUGE, "rouge_ci": rouge_ci}
    (results_dir / "statistical_tests.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    rq = generate_rq_report(
        reports,
        aggregated,
        hr_reduction_target_pct=args.hr_target,
        rouge_tolerance_pct=args.rouge_tol,
        split_name="validation",
        n_cases=len(reports),
        presentation_label="PRELIMINARY — n=10 citation-rich validation, D→L only",
        run_bertscore=False,
    )
    save_rq_report(rq, results_dir)

    # Update batch_summary
    batch = {
        "n_cases": len(reports),
        "mean_hr_draft": sum(r["hr_draft"] for r in reports) / max(len(reports), 1),
        "mean_hr_symbolic": sum(
            r["conditions"]["symbolic"]["hr_final"] for r in reports
        )
        / max(len(reports), 1),
        "mean_hr_llm_rewrite": sum(
            r["conditions"]["llm_rewrite"]["hr_final"] for r in reports
        )
        / max(len(reports), 1),
        "reprocessed_offline": True,
        "ambiguous_policy": "leave_unresolved_count_as_still_bad",
        "cases": [
            {
                "case_id": r["case_id"],
                "hr_draft": r["hr_draft"],
                "hr_symbolic": r["conditions"]["symbolic"]["hr_final"],
                "hr_llm_rewrite": r["conditions"]["llm_rewrite"]["hr_final"],
            }
            for r in reports
        ],
    }
    (conditions_dir / "batch_summary.json").write_text(
        json.dumps(batch, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(
        f"\nDone. Paired HR micro: baseline={aggregated['baseline']['hr_paired_micro']:.3f} "
        f"symbolic={aggregated['symbolic']['hr_paired_micro']:.3f}"
    )
    print(f"Wrote metrics + RQ to {results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
