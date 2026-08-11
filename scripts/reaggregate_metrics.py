#!/usr/bin/env python3
"""
Recompute Phase 4-5 (metrics, statistics, RQ answers) from saved condition
reports -- WITHOUT re-running verification.

Every input this needs (draft summary, corrected summary, gold summary, and the
per-citation verification labels) is already persisted in the per-case JSON
reports written by Phase 3, so nothing here touches CourtListener and no API
budget is spent. Use it whenever an evaluation-side definition changes -- a new
metric, a changed target threshold, a fixed metric variant -- and the existing
verification results should be reused rather than recomputed.

Usage:
    python scripts/reaggregate_metrics.py --config experiment_test100_llm_rag.yaml
    python scripts/reaggregate_metrics.py --all
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ALL_CONFIGS = (
    "experiment_test100.yaml",
    "experiment_test100_llm.yaml",
    "experiment_test100_llm_rag.yaml",
)


def load_reports(conditions_dir: Path) -> list[dict]:
    """Load per-case condition reports, sorted by case_id for determinism."""
    files = sorted(p for p in conditions_dir.glob("*.json") if p.name != "batch_summary.json")
    reports = [json.loads(p.read_text()) for p in files]
    reports.sort(key=lambda r: r["case_id"])
    return reports


def reaggregate(config_name: str) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    from evaluation.aggregate import aggregate_reports, metrics_to_dataframe_rows
    from evaluation.error_analysis import export_error_analysis_csv
    from evaluation.metrics import PRIMARY_ROUGE, compute_rouge
    from evaluation.rq_report import generate_rq_report, save_rq_report
    from evaluation.stats import bootstrap_ci, hr_reduction_stats, mcnemar_citation_test
    from utils.config import load_yaml

    exp_cfg = load_yaml(config_name)
    conditions_dir = ROOT / exp_cfg["conditions_dir"]
    results_dir = ROOT / exp_cfg["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    reports = load_reports(conditions_dir)
    if not reports:
        print(f"FATAL: no condition reports in {conditions_dir}")
        return 1

    n_expected = exp_cfg["max_cases"]
    partial_run = len(reports) < n_expected
    run_bertscore = exp_cfg.get("run_bertscore", False)

    print(f"\n{'=' * 70}")
    print(f"{config_name}: {len(reports)}/{n_expected} reports"
          f"{' [PARTIAL]' if partial_run else ''} | bertscore={run_bertscore}")
    print(f"{'=' * 70}")

    # ---------- Phase 4: metrics ----------
    aggregated = aggregate_reports(reports, skip_bertscore=not run_bertscore, bertscore_device=None)
    df_metrics = pd.DataFrame(metrics_to_dataframe_rows(aggregated))
    df_metrics.to_csv(results_dir / "metrics_summary.csv", index=False)
    with open(results_dir / "metrics_summary.json", "w") as f:
        json.dump(aggregated, f, indent=2, default=str)
    print(df_metrics.to_string(index=False))

    label_counter: Counter[str] = Counter()
    for r in reports:
        for c in r.get("citations_draft", []):
            label_counter[c.get("label", "?")] += 1
    print("\nRQ1 - Draft citation labels:", dict(label_counter))

    # Plot the same pooled hr_paired_micro reported everywhere else (metrics_summary,
    # rq_answers, comparison table) -- NOT a mean of per-case HR. The two disagree
    # substantially (a case-count-weighted pool vs. an unweighted case average), and
    # plotting the latter here previously produced a chart that silently contradicted
    # the pipeline's own formally-defined primary metric (methodology.tex's
    # HR_paired = #{still_bad,garbled}/n_draft is a pooled definition).
    fig, ax = plt.subplots(figsize=(8, 4))
    vals = [
        aggregated["baseline"]["hr_paired_micro"],
        aggregated["symbolic"]["hr_paired_micro"],
        aggregated["llm_rewrite"]["hr_paired_micro"],
    ]
    ax.bar(["Baseline", "Symbolic", "LLM rewrite"], vals, color=["#e76f51", "#2a9d8f", "#264653"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Paired Hallucination Rate (pooled)")
    ax.set_title(f"HR by condition (n={len(reports)}, D->L{'  [PARTIAL]' if partial_run else ''})")
    plt.tight_layout()
    fig.savefig(results_dir / "hr_by_condition.png", dpi=150)
    plt.close(fig)

    # ---------- Phase 5: statistics ----------
    hr_stats = hr_reduction_stats(
        [r["hr_draft"] for r in reports],
        [r["conditions"]["symbolic"]["hr_final"] for r in reports],
    )
    mcnemar = mcnemar_citation_test(
        [r["citations_draft"] for r in reports],
        [r["conditions"]["symbolic"].get("correction_outcomes", []) for r in reports],
    )
    rouge_ci = bootstrap_ci(
        [compute_rouge(r["draft_summary"], r["gold_summary"])[PRIMARY_ROUGE] for r in reports],
        [compute_rouge(r["conditions"]["symbolic"]["final_summary"], r["gold_summary"])[PRIMARY_ROUGE]
         for r in reports],
    )
    print("\nHR reduction:", hr_stats)
    print("McNemar:", mcnemar)
    print(f"ROUGE ({PRIMARY_ROUGE}) bootstrap CI:", rouge_ci)
    with open(results_dir / "statistical_tests.json", "w") as f:
        json.dump(
            {"hr_reduction": hr_stats, "mcnemar": mcnemar,
             "rouge_variant": PRIMARY_ROUGE, "rouge_ci": rouge_ci},
            f, indent=2, default=str,
        )

    export_error_analysis_csv(
        reports,
        ROOT / exp_cfg["error_analysis_path"],
        n_corrected=min(20, len(reports)),
        n_failed=min(20, len(reports)),
    )

    rq = generate_rq_report(
        reports, aggregated,
        hr_reduction_target_pct=exp_cfg.get("hr_reduction_target_pct", 30.0),
        rouge_tolerance_pct=exp_cfg.get("rouge_tolerance_pct", 2.0),
        split_name=exp_cfg["split"],
        n_cases=len(reports),
        partial_run=partial_run,
        presentation_label=exp_cfg.get("presentation_label"),
        run_bertscore=run_bertscore,
    )
    json_path, md_path = save_rq_report(rq, results_dir)
    print(f"\nSaved: {json_path}, {md_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline Phase 4-5 re-aggregation")
    parser.add_argument("--config", default=None)
    parser.add_argument("--all", action="store_true", help=f"run all of: {', '.join(ALL_CONFIGS)}")
    args = parser.parse_args()

    if args.all:
        configs = list(ALL_CONFIGS)
    elif args.config:
        configs = [args.config]
    else:
        parser.error("pass --config <name>.yaml or --all")

    rc = 0
    for cfg in configs:
        rc |= reaggregate(cfg)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
