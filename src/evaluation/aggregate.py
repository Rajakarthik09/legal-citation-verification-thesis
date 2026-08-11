"""Aggregate per-condition metrics across cases."""

from __future__ import annotations

from collections import Counter
from typing import Any

from evaluation.conditions import ALL_CONDITIONS, CONDITION_BASELINE, extract_condition_row
from evaluation.metrics import (
    compute_bertscore,
    compute_rouge,
    correction_efficiency_stats,
    hr_from_citation_dicts,
    macro_hr,
    micro_hr,
)
from evaluation.paired_hr import (
    micro_paired_hr,
    macro_paired_hr,
    outcome_rates,
    paired_hr_from_outcomes,
)
from utils.schemas import CorrectionOutcome


def _outcomes_from_dicts(rows: list[dict[str, Any]]) -> list[list[CorrectionOutcome]]:
    result: list[list[CorrectionOutcome]] = []
    for row in rows:
        lst = []
        for d in row.get("correction_outcomes", []):
            span = d.get("draft_span", (0, 0))
            if isinstance(span, list):
                span = tuple(span)
            lst.append(
                CorrectionOutcome(
                    citation_id=d["citation_id"],
                    draft_label=d["draft_label"],
                    final_status=d["final_status"],
                    draft_span=span,
                )
            )
        result.append(lst)
    return result


def aggregate_reports(
    reports: list[dict[str, Any]],
    *,
    conditions: tuple[str, ...] = ALL_CONDITIONS,
    bertscore_device: str | None = None,
    skip_bertscore: bool = False,
) -> dict[str, Any]:
    """
    Build per-condition aggregate metrics from condition reports.

    Returns dict keyed by condition name with summary statistics.
    """
    result: dict[str, Any] = {}
    for cond in conditions:
        rows = [extract_condition_row(r, cond) for r in reports]
        result[cond] = _aggregate_condition(
            rows,
            is_baseline=cond in (CONDITION_BASELINE, "flagging"),
            bertscore_device=bertscore_device,
            skip_bertscore=skip_bertscore,
        )
    return result


def _aggregate_condition(
    rows: list[dict[str, Any]],
    *,
    is_baseline: bool = False,
    bertscore_device: str | None = None,
    skip_bertscore: bool = False,
) -> dict[str, Any]:
    if not rows:
        return {"n_cases": 0}

    rouge1, rouge2, rougeL, rougeLsum = [], [], [], []
    bertscores: list[float] = []
    citation_lists = [r["citations"] for r in rows]
    label_counter: Counter[str] = Counter()
    iterations = [r["correction_iterations"] for r in rows]
    outcome_lists = _outcomes_from_dicts(rows)

    for row in rows:
        rouge = compute_rouge(row["summary"], row["gold_summary"])
        rouge1.append(rouge["rouge1"])
        rouge2.append(rouge["rouge2"])
        rougeL.append(rouge["rougeL"])
        rougeLsum.append(rouge["rougeLsum"])
        for label, count in Counter(c.get("label") for c in row["citations"]).items():
            label_counter[label] += count

    if not skip_bertscore:
        preds = [r["summary"] for r in rows]
        refs = [r["gold_summary"] for r in rows]
        bertscores = compute_bertscore(preds, refs, device=bertscore_device)

    per_case_hr = [hr_from_citation_dicts(c) for c in citation_lists]
    per_case_paired = [
        paired_hr_from_outcomes(o) if o else hr
        for o, hr in zip(outcome_lists, per_case_hr)
    ]

    all_outcomes = [o for lst in outcome_lists for o in lst]
    rates = outcome_rates(all_outcomes) if all_outcomes else outcome_rates([])

    agg: dict[str, Any] = {
        "n_cases": len(rows),
        "rouge1_mean": _mean(rouge1),
        "rouge2_mean": _mean(rouge2),
        "rougeL_mean": _mean(rougeL),
        "rougeLsum_mean": _mean(rougeLsum),
        "hr_micro": micro_hr(citation_lists),
        "hr_macro": macro_hr(citation_lists),
        "hr_per_case_mean": _mean(per_case_hr),
        "hr_paired_micro": micro_paired_hr(outcome_lists) if not is_baseline else micro_hr(citation_lists),
        "hr_paired_macro": macro_paired_hr(outcome_lists) if not is_baseline else macro_hr(citation_lists),
        "hr_paired_per_case_mean": _mean(per_case_paired),
        "hr_surviving_micro": micro_hr(citation_lists),
        "fix_rate": rates["fix_rate"],
        "removal_rate": rates["removal_rate"],
        "garbled_rate": rates["garbled_rate"],
        "still_bad_rate": rates["still_bad_rate"],
        "total_citations": sum(len(c) for c in citation_lists),
        "total_draft_citations": sum(len(o) for o in outcome_lists) if outcome_lists else sum(len(c) for c in citation_lists),
        "label_counts": dict(label_counter),
        "correction_efficiency": correction_efficiency_stats(iterations),
    }
    if bertscores:
        agg["bertscore_f1_mean"] = _mean(bertscores)
    return agg


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def metrics_to_dataframe_rows(aggregated: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten aggregated metrics for CSV export."""
    rows = []
    for cond, stats in aggregated.items():
        if stats.get("n_cases", 0) == 0:
            continue
        row = {
            "condition": cond,
            "n_cases": stats["n_cases"],
            "rouge1_mean": round(stats["rouge1_mean"], 4),
            "rouge2_mean": round(stats["rouge2_mean"], 4),
            "rougeL_mean": round(stats["rougeL_mean"], 4),
            "rougeLsum_mean": round(stats.get("rougeLsum_mean", 0.0), 4),
            "hr_micro": round(stats["hr_micro"], 4),
            "hr_macro": round(stats["hr_macro"], 4),
            "hr_paired_micro": round(stats.get("hr_paired_micro", stats["hr_micro"]), 4),
            "hr_surviving_micro": round(stats.get("hr_surviving_micro", stats["hr_micro"]), 4),
            "total_citations": stats["total_citations"],
        }
        if "bertscore_f1_mean" in stats:
            row["bertscore_f1_mean"] = round(stats["bertscore_f1_mean"], 4)
        ce = stats.get("correction_efficiency", {})
        row["mean_correction_iters"] = round(ce.get("mean_iterations", 0), 2)
        row["pct_corrected_within_2_iters"] = round(ce.get("pct_within_2", 0), 3)
        rows.append(row)
    return rows
