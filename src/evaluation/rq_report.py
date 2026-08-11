"""Research question synthesis report."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from evaluation.aggregate import aggregate_reports
from evaluation.conditions import (
    CONDITION_BASELINE,
    CONDITION_LLM_REWRITE,
    CONDITION_SYMBOLIC,
)
from evaluation.metrics import PRIMARY_ROUGE, hr_from_citation_dicts
from evaluation.stats import bootstrap_ci, hr_reduction_stats, mcnemar_citation_test


def generate_rq_report(
    reports: list[dict[str, Any]],
    aggregated: dict[str, Any] | None = None,
    *,
    hr_reduction_target_pct: float = 30.0,
    rouge_tolerance_pct: float = 2.0,
    split_name: str = "validation",
    n_cases: int = 50,
    partial_run: bool = False,
    presentation_label: str | None = None,
    run_bertscore: bool = False,
) -> dict[str, Any]:
    """Build structured RQ answers from condition reports."""
    if aggregated is None:
        aggregated = aggregate_reports(reports, skip_bertscore=not run_bertscore)

    rq1 = _answer_rq1(reports)
    rq2 = _answer_rq2(reports, aggregated)
    rq3 = _answer_rq3(reports, aggregated, hr_reduction_target_pct, rouge_tolerance_pct)
    limitations = _limitations_block(split_name, n_cases, partial_run, run_bertscore)
    if presentation_label:
        limitations.insert(0, f"Status: {presentation_label}")

    return {
        "presentation_label": presentation_label,
        "partial_run": partial_run,
        "run_bertscore": run_bertscore,
        "RQ1": rq1,
        "RQ2": rq2,
        "RQ3": rq3,
        "limitations": limitations,
        "aggregated_metrics": aggregated,
    }


def _per_case_paired_hr_draft(report: dict[str, Any]) -> float:
    return hr_from_citation_dicts(report.get("citations_draft", []))


def _per_case_paired_hr_final(report: dict[str, Any], condition: str) -> float:
    sym = report.get("conditions", {}).get(condition, {})
    return sym.get("hr_paired", sym.get("hr_final", 0.0))


def _answer_rq1(reports: list[dict[str, Any]]) -> dict[str, Any]:
    label_counter: Counter[str] = Counter()
    total_citations = 0
    for r in reports:
        for c in r.get("citations_draft", []):
            label_counter[c.get("label", "Unknown")] += 1
            total_citations += 1

    rates = {
        label: count / total_citations if total_citations else 0.0
        for label, count in label_counter.items()
    }
    bad_labels = {"NotFound", "Mismatch", "Ambiguous", "Malformed"}
    hr = sum(label_counter.get(l, 0) for l in bad_labels) / total_citations if total_citations else 0.0

    return {
        "question": "What types and frequencies of case-citation hallucinations do baseline summarizers produce?",
        "n_cases": len(reports),
        "total_citations": total_citations,
        "label_counts": dict(label_counter),
        "label_rates": {k: round(v, 4) for k, v in rates.items()},
        "hallucination_rate": round(hr, 4),
        "summary": (
            f"Across {len(reports)} cases, {total_citations} citations extracted from draft summaries. "
            f"Overall HR={hr:.1%}. "
            f"Most common labels: {', '.join(f'{k}({v})' for k, v in label_counter.most_common(5))}."
        ),
    }


def _answer_rq2(
    reports: list[dict[str, Any]],
    aggregated: dict[str, Any],
) -> dict[str, Any]:
    baseline_hr = aggregated.get(CONDITION_BASELINE, {}).get("hr_paired_micro", 0.0)
    symbolic_hr = aggregated.get(CONDITION_SYMBOLIC, {}).get("hr_paired_micro", 0.0)
    hr_surviving = aggregated.get(CONDITION_SYMBOLIC, {}).get("hr_surviving_micro", 0.0)
    sym_agg = aggregated.get(CONDITION_SYMBOLIC, {})
    reduction = hr_reduction_stats(
        [_per_case_paired_hr_draft(r) for r in reports],
        [_per_case_paired_hr_final(r, CONDITION_SYMBOLIC) for r in reports],
    )
    mcnemar = mcnemar_citation_test(
        [r.get("citations_draft", []) for r in reports],
        [
            r.get("conditions", {}).get(CONDITION_SYMBOLIC, {}).get("correction_outcomes", [])
            for r in reports
        ],
    )
    return {
        "question": "Does symbolic verification + correction reduce HR vs neural-only baseline?",
        "hr_baseline_paired_micro": round(baseline_hr, 4),
        "hr_symbolic_paired_micro": round(symbolic_hr, 4),
        "hr_symbolic_surviving_micro": round(hr_surviving, 4),
        "fix_rate": round(sym_agg.get("fix_rate", 0.0), 4),
        "removal_rate": round(sym_agg.get("removal_rate", 0.0), 4),
        "garbled_rate": round(sym_agg.get("garbled_rate", 0.0), 4),
        "reduction": {k: round(v, 4) if isinstance(v, float) else v for k, v in reduction.items()},
        "mcnemar_test": mcnemar,
        "summary": (
            f"Paired HR dropped from {baseline_hr:.1%} (baseline) to {symbolic_hr:.1%} "
            f"(symbolic correction), relative reduction {reduction['relative_reduction_pct']:.1f}%. "
            f"Surviving-citation HR={hr_surviving:.1%}. McNemar p={mcnemar.get('p_value')}."
        ),
    }


def _answer_rq3(
    reports: list[dict[str, Any]],
    aggregated: dict[str, Any],
    hr_target: float,
    rouge_tol: float,
) -> dict[str, Any]:
    base = aggregated.get(CONDITION_BASELINE, {})
    sym = aggregated.get(CONDITION_SYMBOLIC, {})
    llm = aggregated.get(CONDITION_LLM_REWRITE, {})

    hr_rel = hr_reduction_stats(
        [_per_case_paired_hr_draft(r) for r in reports],
        [_per_case_paired_hr_final(r, CONDITION_SYMBOLIC) for r in reports],
    )["relative_reduction_pct"]

    # rougeLsum is the primary quality gate (see evaluation.metrics notes);
    # plain rougeL is still reported alongside it for transparency.
    rouge_delta = sym.get("rougeLsum_mean", 0) - base.get("rougeLsum_mean", 0)
    rouge_delta_pct = rouge_delta * 100
    rougeL_delta_pct = (sym.get("rougeL_mean", 0) - base.get("rougeL_mean", 0)) * 100

    rouge_ci = bootstrap_ci(
        [_per_case_rouge(r, CONDITION_BASELINE) for r in reports],
        [_per_case_rouge(r, CONDITION_SYMBOLIC) for r in reports],
    )

    hr_pass = hr_rel >= hr_target
    rouge_pass = abs(rouge_delta_pct) <= rouge_tol

    return {
        "question": "Can the dual-system approach preserve quality while reducing HR by >=30%?",
        "hr_relative_reduction_pct": round(hr_rel, 2),
        "hr_target_pct": hr_target,
        "hr_target_met": hr_pass,
        "primary_rouge_variant": PRIMARY_ROUGE,
        "rougeLsum_delta_pct_points": round(rouge_delta_pct, 2),
        "rougeL_delta_pct_points": round(rougeL_delta_pct, 2),
        "rouge_tolerance_pct": rouge_tol,
        "rouge_target_met": rouge_pass,
        "rougeLsum_bootstrap_ci": rouge_ci,
        "symbolic_vs_llm_rewrite": {
            "hr_paired_symbolic": sym.get("hr_paired_micro"),
            "hr_paired_llm_rewrite": llm.get("hr_paired_micro"),
            "rougeLsum_symbolic": sym.get("rougeLsum_mean"),
            "rougeLsum_llm_rewrite": llm.get("rougeLsum_mean"),
            "rougeL_symbolic": sym.get("rougeL_mean"),
            "rougeL_llm_rewrite": llm.get("rougeL_mean"),
        },
        "summary": (
            f"Paired HR relative reduction {hr_rel:.1f}% (target {hr_target}%): {'PASS' if hr_pass else 'NOT MET'}. "
            f"ROUGE-Lsum delta {rouge_delta_pct:+.2f} pp (tolerance ±{rouge_tol}): {'PASS' if rouge_pass else 'NOT MET'}."
        ),
    }


def _per_case_rouge(report: dict[str, Any], condition: str) -> float:
    from evaluation.metrics import compute_rouge

    if condition == CONDITION_BASELINE:
        summary = report["draft_summary"]
    else:
        summary = report["conditions"][condition]["final_summary"]
    return compute_rouge(summary, report["gold_summary"])[PRIMARY_ROUGE]


def _limitations_block(
    split_name: str,
    n_cases: int,
    partial_run: bool = False,
    run_bertscore: bool = False,
) -> list[str]:
    items = [
        f"Evaluation on {split_name} split, n={n_cases} (not full test set).",
        "D→L (long summaries) only; short/tiny descoped after EDA.",
        "Primary HR is paired draft-citation HR; surviving-citation HR reported separately.",
        "Primary quality metric is ROUGE-Lsum (sentence-split union-LCS), the variant "
        "conventionally reported for multi-sentence summarization; plain ROUGE-L is also "
        "reported but is not comparable to published summarization baselines.",
        "CourtListener access level 10 req/min, 100/hr, 250/day; caching used; "
        "multi-day runs required at full scale.",
        "Source truncation to the generator's context budget (known limitation).",
        "Manual audit template exported separately; labels not included in automated RQ answers.",
    ]
    if partial_run:
        items.append("PARTIAL RUN: CourtListener daily cap hit before all cases completed.")
    if not run_bertscore:
        items.append(
            "BERTScore not computed in this run; set run_bertscore: true in experiment YAML "
            "and re-run Phase 4–5 for full RQ3 quality metrics."
        )
    return items


def format_rq_markdown(report: dict[str, Any]) -> str:
    """Render RQ report as markdown."""
    lines = ["# Research Question Answers\n"]
    if report.get("presentation_label"):
        lines.append(f"**{report['presentation_label']}**\n")
    if report.get("partial_run"):
        lines.append("> **Note:** Partial run — not all cases completed (API cap or session limit).\n")
    for key in ("RQ1", "RQ2", "RQ3"):
        block = report[key]
        lines.append(f"## {key}\n")
        lines.append(f"**Question:** {block['question']}\n")
        lines.append(f"**Answer:** {block['summary']}\n")
        for k, v in block.items():
            if k not in ("question", "summary"):
                lines.append(f"- **{k}:** {v}")
        lines.append("")
    lines.append("## Limitations\n")
    for lim in report["limitations"]:
        lines.append(f"- {lim}")
    return "\n".join(lines)


def save_rq_report(report: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "rq_answers.json"
    md_path = output_dir / "rq_answers.md"
    json_path.write_text(json.dumps(report, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(format_rq_markdown(report), encoding="utf-8")
    return json_path, md_path
