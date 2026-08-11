"""Evaluation harness (Phases 3–5).

Keep this package init lightweight so Streamlit / correction imports of
`evaluation.paired_hr` do not pull heavy deps like rouge_score.
"""

from __future__ import annotations

__all__ = [
    "ALL_CONDITIONS",
    "CONDITION_BASELINE",
    "CONDITION_SYMBOLIC",
    "aggregate_reports",
    "compute_case_metrics",
    "compute_rouge",
    "export_audit_template",
    "export_error_analysis_csv",
    "generate_rq_report",
    "load_examples_for_audit",
    "macro_hr",
    "metrics_to_dataframe_rows",
    "micro_hr",
    "save_rq_report",
    "select_error_analysis_cases",
]


def __getattr__(name: str):
    if name in ("ALL_CONDITIONS", "CONDITION_BASELINE", "CONDITION_SYMBOLIC"):
        from evaluation import conditions as _conditions

        return getattr(_conditions, name)
    if name in ("aggregate_reports", "metrics_to_dataframe_rows"):
        from evaluation import aggregate as _aggregate

        return getattr(_aggregate, name)
    if name in ("export_audit_template", "load_examples_for_audit"):
        from evaluation import audit_export as _audit

        return getattr(_audit, name)
    if name in ("export_error_analysis_csv", "select_error_analysis_cases"):
        from evaluation import error_analysis as _error

        return getattr(_error, name)
    if name in ("compute_case_metrics", "compute_rouge", "macro_hr", "micro_hr"):
        from evaluation import metrics as _metrics

        return getattr(_metrics, name)
    if name in ("generate_rq_report", "save_rq_report"):
        from evaluation import rq_report as _rq

        return getattr(_rq, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
