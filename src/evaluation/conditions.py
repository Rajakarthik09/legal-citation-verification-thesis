"""Map pipeline reports to evaluation conditions."""

from __future__ import annotations

from typing import Any

CONDITION_BASELINE = "baseline"
CONDITION_FLAGGING = "flagging"
CONDITION_SYMBOLIC = "symbolic"
CONDITION_LLM_REWRITE = "llm_rewrite"

ALL_CONDITIONS = (
    CONDITION_BASELINE,
    CONDITION_FLAGGING,
    CONDITION_SYMBOLIC,
    CONDITION_LLM_REWRITE,
)


def condition_summary_text(report: dict[str, Any], condition: str) -> str:
    """Return the summary text used for a given condition."""
    if condition in (CONDITION_BASELINE, CONDITION_FLAGGING):
        return report["draft_summary"]
    if condition == CONDITION_SYMBOLIC:
        return report["conditions"][CONDITION_SYMBOLIC]["final_summary"]
    if condition == CONDITION_LLM_REWRITE:
        return report["conditions"][CONDITION_LLM_REWRITE]["final_summary"]
    raise ValueError(f"Unknown condition: {condition!r}")


def condition_citations(report: dict[str, Any], condition: str) -> list[dict[str, Any]]:
    """Return citation verification results for a condition."""
    if condition in (CONDITION_BASELINE, CONDITION_FLAGGING):
        return report["citations_draft"]
    return report["conditions"][condition]["citations_final"]


def condition_correction_outcomes(report: dict[str, Any], condition: str) -> list[dict[str, Any]]:
    """Return per-draft-citation correction outcomes for a condition."""
    if condition in (CONDITION_BASELINE, CONDITION_FLAGGING):
        return []
    return report["conditions"][condition].get("correction_outcomes", [])


def condition_iterations(report: dict[str, Any], condition: str) -> int:
    if condition in (CONDITION_BASELINE, CONDITION_FLAGGING):
        return 0
    return report["conditions"][condition].get("correction_iterations", 0)


def extract_condition_row(report: dict[str, Any], condition: str) -> dict[str, Any]:
    """Flat row for aggregation tables."""
    cond_data = report.get("conditions", {}).get(condition, {})
    return {
        "case_id": report["case_id"],
        "granularity": report.get("granularity", "long"),
        "condition": condition,
        "summary": condition_summary_text(report, condition),
        "gold_summary": report["gold_summary"],
        "citations": condition_citations(report, condition),
        "correction_outcomes": condition_correction_outcomes(report, condition),
        "correction_iterations": condition_iterations(report, condition),
        "hr_paired": cond_data.get("hr_paired") if condition not in (CONDITION_BASELINE, CONDITION_FLAGGING) else report.get("hr_draft"),
        "hr_surviving": cond_data.get("hr_surviving") if condition not in (CONDITION_BASELINE, CONDITION_FLAGGING) else report.get("hr_draft"),
    }
