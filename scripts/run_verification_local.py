#!/usr/bin/env python3
"""
Module B/C + correction + evaluation driver — runs LOCALLY (no GPU needed).

This is Phases 3-5 of the full pipeline, extracted from
notebooks/thesis_full_pipeline_colab.ipynb into a plain CLI script, because
Module A (GPU generation, done separately on RunPod) and Module C
(CourtListener verification, rate-limited to 125 requests/day) have entirely
different resource profiles. Paying for a GPU pod to sit idle across the
3-5 days this phase needs would be wasteful; this runs for free on the
local M1 and picks up exactly where it left off each day.

Safe to re-run daily: skips case_ids already verified, stops cleanly (not an
error) when the CourtListener daily cap is hit, and regenerates the
evaluation report (Phase 4-5) on whatever reports exist so far each time, so
you always have current interim numbers.

Usage:
    cd code && source .venv/bin/activate
    python scripts/run_verification_local.py --config experiment_test100.yaml

Requires outputs/generations_test100/test_long.jsonl to already exist
(downloaded from the RunPod generation run).
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import json
import os
import sys
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# All experiment configs share one CourtListener rate-limiter state file, so
# two verification runs (any config, launched manually or by the scheduled
# task) hitting the API at the same time silently double-burn the shared
# daily/hourly budget -- this happened three times in one day. The lock is
# process-global (not per-config) for that reason. flock is held for the
# process's lifetime and released automatically by the OS on exit, crash, or
# kill -9, so there's no stale-lock cleanup to get wrong.
LOCK_PATH = ROOT / "data" / "processed" / "run_verification.lock"


@contextmanager
def single_instance_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            if e.errno not in (errno.EACCES, errno.EAGAIN):
                raise
            holder = os.read(fd, 256).decode("utf-8", "replace").strip()
            print(f"FATAL: another run_verification_local.py is already running "
                  f"(lock held by: {holder or 'unknown'}). Refusing to start a "
                  f"second one -- it would double-spend the shared CourtListener "
                  f"budget. Let it finish or kill it first.")
            raise SystemExit(1)
        os.ftruncate(fd, 0)
        os.write(fd, f"pid={os.getpid()} started={time.strftime('%Y-%m-%d %H:%M:%S')}".encode())
        os.fsync(fd)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(fd)


def main() -> int:
    parser = argparse.ArgumentParser(description="Local Module B/C + eval driver")
    parser.add_argument("--config", default="experiment_test100.yaml")
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    from correction_loop.llm_rewrite import create_rewriter_from_config
    from data.case_selection import select_cases
    from data.load_multilexsum import get_split
    from evaluation.aggregate import aggregate_reports, metrics_to_dataframe_rows
    from evaluation.audit_export import export_audit_template, load_examples_for_audit
    from evaluation.error_analysis import export_error_analysis_csv
    from evaluation.metrics import PRIMARY_ROUGE, compute_rouge
    from evaluation.rq_report import generate_rq_report, save_rq_report
    from evaluation.stats import bootstrap_ci, hr_reduction_stats, mcnemar_citation_test
    from module_a_summarizer import load_generations
    from module_c_verifier import Verifier
    from pipeline.run_conditions import run_conditions_batch
    from pipeline.run_pipeline import merge_configs
    from utils.config import load_yaml

    exp_cfg = load_yaml(args.config)
    mod_a = load_yaml("module_a.yaml")
    mod_c = load_yaml("module_c.yaml")

    split_name = exp_cfg["split"]
    granularity = exp_cfg["granularity"]
    run_bertscore = exp_cfg.get("run_bertscore", False)
    model_tag = exp_cfg.get("model_tag", "primera")

    gen_filename = exp_cfg.get("generations_filename") or f"{split_name}_{granularity}.jsonl"
    gen_path = ROOT / exp_cfg.get("generations_dir", "outputs/generations") / gen_filename
    if not gen_path.exists():
        print(f"FATAL: {gen_path} not found. Download it from the RunPod generation run first.")
        return 1
    generations = load_generations(gen_path)
    print(f"Loaded {len(generations)} generations from {gen_path} (model_tag={model_tag})")

    split = get_split(split_name)
    selected_rows = select_cases(
        split,
        case_ids=exp_cfg.get("case_ids"),
        max_cases=exp_cfg["max_cases"],
        min_gold_citations=exp_cfg.get("min_gold_citations", 1),
        seed=exp_cfg.get("seed", 42),
    )

    from transformers import AutoTokenizer

    # cases[] here only supplies case_id/gold_summary/truncation metadata for
    # the report -- verification reuses the already-generated draft text
    # (generations[case_id]), it never re-truncates sources for PRIMERA. But
    # for the LLM conditions the truncation stats in the report SHOULD
    # reflect what that generator actually saw (8192 tokens, Qwen tokenizer,
    # possibly citation-priority ordered) -- using PRIMERA's 4096/tokenizer
    # here for every condition would misreport those fields.
    if model_tag in ("llm", "llm_rag"):
        from data.preprocessing import build_case_input, build_case_input_retrieval

        mod_llm = load_yaml("module_llm.yaml")
        tokenizer = AutoTokenizer.from_pretrained(mod_llm["checkpoint"])
        max_tokens = mod_llm["max_input_tokens"]
        build_fn = build_case_input_retrieval if exp_cfg.get("use_retrieval") else build_case_input
    else:
        from data.preprocessing import GRANULARITY_CHECKPOINTS, build_case_input

        tokenizer = AutoTokenizer.from_pretrained(GRANULARITY_CHECKPOINTS[granularity])
        data_cfg = load_yaml("data.yaml")
        max_tokens = data_cfg["max_tokens"][granularity]
        build_fn = build_case_input

    cases = [c for row in selected_rows if (c := build_fn(row, granularity, tokenizer, max_tokens)) is not None]

    # ==================== Phase 3: verify + correct ====================
    runtime_cfg = merge_configs(exp_cfg, mod_a, mod_c)
    runtime_cfg["conditions"] = exp_cfg.get("conditions", [])

    verifier = Verifier(runtime_cfg)
    conditions_dir = ROOT / exp_cfg.get("conditions_dir", "outputs/conditions")
    gen_text = {cid: row["generated_summary"] for cid, row in generations.items()}

    print(f"\n=== Phase 3: verify + correct (CourtListener budget: "
          f"{verifier.rate_limiter.remaining_daily}/{verifier.rate_limiter.daily_cap} remaining today) ===")
    rewriter = create_rewriter_from_config(runtime_cfg)
    reports = run_conditions_batch(cases, verifier, runtime_cfg, conditions_dir, generations=gen_text, rewriter=rewriter)
    rewriter.unload()
    verifier.close()
    print(f"Condition reports: {len(reports)}/{len(cases)} cases done, in {conditions_dir}")

    batch_summary = json.loads((conditions_dir / "batch_summary.json").read_text())
    partial_run = batch_summary.get("stopped_early", False)
    if partial_run:
        reason = batch_summary.get("stop_reason", "")
        # "Daily cap reached" was previously printed unconditionally here,
        # even when the real stop_reason was a same-day 429 (minute/hour
        # throttle, or an unparseable rate-limit response) -- misleading
        # about whether resuming needs to wait until tomorrow or can be
        # retried sooner.
        is_daily = "daily" in reason.lower()
        retry_hint = "tomorrow" if is_daily else "shortly (check remaining budget first)"
        print(f"\n*** CourtListener stopped the run early: {reason} ***")
        print(f"*** {len(reports)}/{len(cases)} cases verified so far. Re-run this script {retry_hint}. ***\n")

    if not reports:
        print("No reports yet — nothing to evaluate.")
        return 0

    # ==================== Phase 4: evaluation metrics ====================
    results_dir = ROOT / exp_cfg.get("results_dir", "outputs/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Phase 4: evaluation metrics (n={len(reports)}, bertscore={run_bertscore}) ===")
    aggregated = aggregate_reports(reports, skip_bertscore=not run_bertscore, bertscore_device=None)
    df_metrics = pd.DataFrame(metrics_to_dataframe_rows(aggregated))
    df_metrics.to_csv(results_dir / "metrics_summary.csv", index=False)
    with open(results_dir / "metrics_summary.json", "w") as f:
        json.dump(aggregated, f, indent=2, default=str)
    print(df_metrics.to_string(index=False))

    label_counter = Counter()
    for r in reports:
        for c in r.get("citations_draft", []):
            label_counter[c.get("label", "?")] += 1
    print("\nRQ1 - Draft citation labels:", dict(label_counter))

    # Plot the pooled hr_paired_micro (same metric as metrics_summary/rq_answers/
    # comparison table -- and the pipeline's own formal HR_paired definition), not
    # a mean of per-case HR. The two disagree substantially since case citation
    # counts vary a lot; plotting the per-case mean here previously produced a
    # chart that silently contradicted the reported headline numbers.
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = ["Baseline", "Symbolic", "LLM rewrite"]
    vals = [
        aggregated["baseline"]["hr_paired_micro"],
        aggregated["symbolic"]["hr_paired_micro"],
        aggregated["llm_rewrite"]["hr_paired_micro"],
    ]
    ax.bar(labels, vals, color=["#e76f51", "#2a9d8f", "#264653"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Paired Hallucination Rate (pooled)")
    ax.set_title(f"HR by condition (n={len(reports)}, D->L{'  [PARTIAL]' if partial_run else ''})")
    plt.tight_layout()
    fig.savefig(results_dir / "hr_by_condition.png", dpi=150)
    plt.close(fig)

    audit_path = ROOT / exp_cfg.get("audit_template_path", "outputs/audit/manual_label_template_200.csv")
    if not audit_path.exists():
        ds_val = get_split("validation")
        ds_test = get_split("test")
        pool = load_examples_for_audit(ds_val, ds_test)
        export_audit_template(pool, audit_path, n_cases=200, seed=exp_cfg.get("seed", 42))
        print("Audit template:", audit_path)

    # ==================== Phase 5: stats + error analysis ====================
    print(f"\n=== Phase 5: statistical testing ===")
    hr_stats = hr_reduction_stats(
        [r["hr_draft"] for r in reports],
        [r["conditions"]["symbolic"]["hr_final"] for r in reports],
    )
    mcnemar = mcnemar_citation_test(
        [r["citations_draft"] for r in reports],
        [r["conditions"]["symbolic"].get("correction_outcomes", []) for r in reports],
    )
    rouge_baseline = [compute_rouge(r["draft_summary"], r["gold_summary"])[PRIMARY_ROUGE] for r in reports]
    rouge_symbolic = [compute_rouge(r["conditions"]["symbolic"]["final_summary"], r["gold_summary"])[PRIMARY_ROUGE] for r in reports]
    rouge_ci = bootstrap_ci(rouge_baseline, rouge_symbolic)

    print("HR reduction:", hr_stats)
    print("McNemar:", mcnemar)
    print(f"ROUGE ({PRIMARY_ROUGE}) bootstrap CI:", rouge_ci)
    with open(results_dir / "statistical_tests.json", "w") as f:
        json.dump(
            {"hr_reduction": hr_stats, "mcnemar": mcnemar,
             "rouge_variant": PRIMARY_ROUGE, "rouge_ci": rouge_ci},
            f, indent=2, default=str,
        )

    error_path = ROOT / exp_cfg.get("error_analysis_path", "outputs/results/error_analysis_review.csv")
    export_error_analysis_csv(reports, error_path, n_corrected=min(20, len(reports)), n_failed=min(20, len(reports)))

    rq = generate_rq_report(
        reports, aggregated,
        hr_reduction_target_pct=exp_cfg.get("hr_reduction_target_pct", 30.0),
        rouge_tolerance_pct=exp_cfg.get("rouge_tolerance_pct", 2.0),
        split_name=split_name,
        n_cases=len(reports),
        partial_run=partial_run,
        presentation_label=exp_cfg.get("presentation_label"),
        run_bertscore=run_bertscore,
    )
    json_path, md_path = save_rq_report(rq, results_dir)
    print("\n" + md_path.read_text())
    print(f"\nSaved: {json_path}, {md_path}")

    if partial_run:
        print(f"\n>>> {len(reports)}/{len(cases)} cases done. Run again tomorrow for the rest. <<<")
    else:
        print(f"\n>>> All {len(cases)} cases verified. Full-scale run complete. <<<")
    return 0


if __name__ == "__main__":
    with single_instance_lock(LOCK_PATH):
        raise SystemExit(main())
