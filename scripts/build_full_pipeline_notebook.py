#!/usr/bin/env python3
"""Generate thesis_full_pipeline_colab.ipynb — Phases 0–5 end-to-end."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from colab_install import INSTALL_DEPS_CELL

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "thesis_full_pipeline_colab.ipynb"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source.splitlines(keepends=True),
        "outputs": [],
        "execution_count": None,
    }


cells = [
    md(
        """# Thesis Full Pipeline — Phases 0–5

**Dual-System Hallucination-Free Legal Summarization** — Thursday professor demo

| Phase | Content |
|-------|---------|
| 0 | Setup, secrets, smoke tests (hard gate) |
| 1 | Data load — **10 citation-rich validation cases**, D→L long only |
| 2 | Module A — PRIMERA-long generation (resume-safe) |
| 3 | Conditions — baseline / flagging / symbolic / llm_rewrite |
| 4 | Evaluation + audit export |
| 5 | Statistics + **preliminary RQ1/RQ2/RQ3 answers** |

**Tonight:** `experiment_professor10.yaml` (n=10 citation-rich). **Later:** switch to `experiment_validation50.yaml` on your own GPU.

**Requirements:** T4 GPU, Colab Secrets (`HF_TOKEN`, `COURTLISTENER_API_TOKEN`), upload `full_pipeline/` to Drive."""
    ),
    md("## Phase 0 — Mount Drive & resolve WORKDIR"),
    code(
        """from google.colab import drive
from pathlib import Path

drive.mount("/content/drive")

DRIVE_BASE = Path("/content/drive/MyDrive/Thesis/full_pipeline")


def _resolve_workdir(base: Path) -> Path:
    candidates = [base / "colab_upload", base]
    for candidate in candidates:
        if (candidate / "src").exists() and (candidate / "requirements-colab.txt").exists():
            return candidate
    for candidate in candidates:
        if (candidate / "src").exists():
            return candidate
    return base if base.exists() else Path(".")


WORKDIR = _resolve_workdir(DRIVE_BASE)
print("WORKDIR:", WORKDIR.resolve())
print("src/:", (WORKDIR / "src").exists())
print("config/:", (WORKDIR / "config").exists())"""
    ),
    md("## Phase 0 — Install dependencies"),
    code(INSTALL_DEPS_CELL),
    code(
        """import os, sys, json
from pathlib import Path

os.chdir(WORKDIR)
sys.path.insert(0, str(WORKDIR / "src"))
print("cwd:", Path.cwd())"""
    ),
    md("## Phase 0 — Load secrets"),
    code(
        """from google.colab import userdata

os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
os.environ["COURTLISTENER_API_TOKEN"] = userdata.get("COURTLISTENER_API_TOKEN")
os.environ["HF_HOME"] = str(WORKDIR / "data" / "hf")
os.environ["MULTILEXSUM_CACHE"] = str(WORKDIR / "data" / "hf")
print("Secrets loaded.")"""
    ),
    md("## Phase 0 — Smoke tests (hard gate)"),
    code(
        """from smoke.colab_smoke import smoke_run_all, assert_smoke_gate

smoke_report = smoke_run_all(
    require_cuda=True,
    module_a_mode="lightweight",
    llm_rewrite_mode="load_only",
    include_live_api=True,
)
assert_smoke_gate(smoke_report)
print("\\n✓ Smoke gate passed — proceeding to Phase 1.")"""
    ),
    md(
        """## Phase 1 — Load cases (D→L long only)

**Config switch:** edit `EXPERIMENT_CONFIG` below.
- `experiment_professor10.yaml` — Thursday demo (10 citation-rich cases)
- `experiment_validation50.yaml` — full run later on your GPU"""
    ),
    code(
        """import torch
from transformers import AutoTokenizer

from utils.config import load_yaml
from data.load_multilexsum import load_multilexsum, clear_dataset_cache
from data.preprocessing import build_case_input, GRANULARITY_CHECKPOINTS
from data.case_selection import select_cases
from module_b_extraction.extract_citations import extract_citations

EXPERIMENT_CONFIG = "experiment_professor10.yaml"  # Thursday demo
# EXPERIMENT_CONFIG = "experiment_validation50.yaml"  # full run later

exp_cfg = load_yaml(EXPERIMENT_CONFIG)
data_cfg = load_yaml("data.yaml")
mod_a = load_yaml("module_a.yaml")
mod_c = load_yaml("module_c.yaml")

SPLIT = exp_cfg["split"]
GRANULARITY = exp_cfg["granularity"]
MAX_CASES = exp_cfg["max_cases"]
RUN_BERTSCORE = exp_cfg.get("run_bertscore", False)
print(f"Experiment: {EXPERIMENT_CONFIG} | {SPLIT} / {GRANULARITY} / max_cases={MAX_CASES}")
if exp_cfg.get("case_ids"):
    print(f"Citation-rich case_ids: {len(exp_cfg['case_ids'])} pinned")
assert GRANULARITY == "long", "This notebook is scoped to D→L (long) only."

clear_dataset_cache()
ds = load_multilexsum(cache_dir=WORKDIR / "data" / "hf")
split = ds[SPLIT]
print(f"Split {SPLIT}: {len(split)} cases")

tokenizer = AutoTokenizer.from_pretrained(GRANULARITY_CHECKPOINTS[GRANULARITY])
max_tokens = data_cfg["max_tokens"][GRANULARITY]

selected_rows = select_cases(
    split,
    case_ids=exp_cfg.get("case_ids"),
    max_cases=MAX_CASES,
    min_gold_citations=exp_cfg.get("min_gold_citations", 1),
    seed=exp_cfg.get("seed", 42),
)
cases = []
for row in selected_rows:
    case = build_case_input(row, GRANULARITY, tokenizer, max_tokens)
    if case is not None:
        cases.append(case)

print(f"\\nSelected {len(selected_rows)} cases (gold citations via eyecite):")
for row in selected_rows:
    n = len(extract_citations(row["summary/long"]))
    print(f"  {row['id']}: {n} gold citations")

print(f"\\nPrepared {len(cases)} CaseInput objects")
print(f"Truncated: {sum(1 for c in cases if c.truncated)}/{len(cases)}")
assert all(c.source_token_count <= max_tokens for c in cases)
assert len(cases) > 0, "No cases selected — check case_ids in experiment YAML"
print("First case:", cases[0].case_id)"""
    ),
    md(
        """## Phase 2 — Module A generation (resume-safe)

> **Resume:** If Colab disconnects, re-run **only this cell**. Already-written `case_id`s in the JSONL are skipped."""
    ),
    code(
        """from module_a_summarizer import Summarizer, generate_batch, load_generations

mod_a_run = {
    **mod_a,
    "granularity": GRANULARITY,
    "device": "cuda",
    "use_fp16": True,
    "output_dir": str(WORKDIR / exp_cfg.get("generations_dir", "outputs/generations")),
}

summarizer = Summarizer(granularity=GRANULARITY, device="cuda", use_fp16=True)
gen_path = generate_batch(cases, mod_a_run, split=SPLIT, summarizer=summarizer)
generations = load_generations(gen_path)
print(f"Generations: {len(generations)} cases → {gen_path}")

del summarizer
import gc; gc.collect()
torch.cuda.empty_cache()
print("PRIMERA unloaded.")"""
    ),
    md(
        """## Phase 2b — Pre-flight: gold vs draft citations

Compare eyecite extraction on gold vs PRIMERA drafts before Phase 3. Cases with 0 draft citations still run but weaken RQ1/RQ2."""
    ),
    code(
        """from module_b_extraction.extract_citations import extract_citations

print(f"{'case_id':<14} {'gold':>6} {'draft':>6}")
zero_draft = []
for case in cases:
    gold_n = len(extract_citations(case.gold_summary))
    draft = generations[case.case_id]["generated_summary"]
    draft_n = len(extract_citations(draft))
    print(f"{case.case_id:<14} {gold_n:>6} {draft_n:>6}")
    if draft_n == 0:
        zero_draft.append(case.case_id)

if zero_draft:
    print(f"\\n⚠ {len(zero_draft)} cases have 0 draft citations:", zero_draft)
    print("Pipeline will still run, but RQ1/RQ2 may be weak for those cases.")
    print("Show professor gold-vs-draft table + one case with draft_n > 0 if available.")
else:
    print("\\n✓ All cases have extractable draft citations.")"""
    ),
    md(
        """## Phase 3 — Run all conditions (verify + correct)

> **Resume:** Re-run this cell if CourtListener daily cap (125/day) stops early. Completed cases are skipped."""
    ),
    code(
        """from pipeline.run_pipeline import merge_configs
from pipeline.run_conditions import run_conditions_batch
from module_c_verifier import Verifier
from correction_loop.llm_rewrite import create_rewriter_from_config

runtime_cfg = merge_configs(exp_cfg, mod_a, mod_c)
runtime_cfg.update({
    "cache_path": str(WORKDIR / "data" / "processed" / "citation_cache.sqlite"),
    "conditions": exp_cfg.get("conditions", []),
})

verifier = Verifier(runtime_cfg)
conditions_dir = WORKDIR / exp_cfg.get("conditions_dir", "outputs/conditions")
gen_text = {cid: row["generated_summary"] for cid, row in generations.items()}

rewriter = create_rewriter_from_config(runtime_cfg)
reports = run_conditions_batch(
    cases, verifier, runtime_cfg, conditions_dir,
    generations=gen_text, rewriter=rewriter,
)
rewriter.unload()
verifier.close()
print(f"Condition reports: {len(reports)} cases in {conditions_dir}")

batch_summary = json.loads((conditions_dir / "batch_summary.json").read_text())
PARTIAL_RUN = batch_summary.get("stopped_early", False)
if PARTIAL_RUN:
    print("⚠ Partial run:", batch_summary.get("stop_reason"))
    print("Re-run this cell tomorrow to resume (CourtListener daily cap).")"""
    ),
    md(
        """## Reload from Drive (session restart safe)

Run this cell before Phase 4 if your Colab session disconnected after Phase 3."""
    ),
    code(
        """from pipeline.run_conditions import load_condition_reports
from module_a_summarizer import load_generations

conditions_dir = WORKDIR / exp_cfg.get("conditions_dir", "outputs/conditions")
gen_path = WORKDIR / exp_cfg.get("generations_dir", "outputs/generations") / f"{SPLIT}_{GRANULARITY}.jsonl"

reports = load_condition_reports(conditions_dir)
generations = load_generations(gen_path)
batch_summary_path = conditions_dir / "batch_summary.json"
PARTIAL_RUN = False
if batch_summary_path.exists():
    batch_summary = json.loads(batch_summary_path.read_text())
    PARTIAL_RUN = batch_summary.get("stopped_early", False)

if len(reports) == 0:
    raise RuntimeError(
        "No condition reports found. Run Phase 3 first, or check conditions_dir: "
        f"{conditions_dir}"
    )
print(f"Loaded {len(reports)} reports from disk")
if PARTIAL_RUN:
    print("⚠ Partial run detected — RQ answers will be labeled preliminary/partial.")"""
    ),
    md(
        """## Phase 4 — Evaluation metrics

> **BERTScore:** skipped by default (`run_bertscore: false`). To enable later, set `run_bertscore: true` in your experiment YAML and re-run Phase 4–5 only."""
    ),
    code(
        """import pandas as pd
import matplotlib.pyplot as plt

from evaluation.aggregate import aggregate_reports, metrics_to_dataframe_rows
from evaluation.audit_export import export_audit_template, load_examples_for_audit

results_dir = WORKDIR / exp_cfg.get("results_dir", "outputs/results")
results_dir.mkdir(parents=True, exist_ok=True)

aggregated = aggregate_reports(
    reports,
    skip_bertscore=not RUN_BERTSCORE,
    bertscore_device="cuda" if RUN_BERTSCORE else None,
)
df_metrics = pd.DataFrame(metrics_to_dataframe_rows(aggregated))
display(df_metrics)

df_metrics.to_csv(results_dir / "metrics_summary.csv", index=False)
with open(results_dir / "metrics_summary.json", "w") as f:
    json.dump(aggregated, f, indent=2, default=str)

from collections import Counter
label_counter = Counter()
for r in reports:
    for c in r.get("citations_draft", []):
        label_counter[c.get("label", "?")] += 1
print("\\nRQ1 — Draft citation labels:", dict(label_counter))

fig, ax = plt.subplots(figsize=(8, 4))
labels = ["Baseline", "Symbolic", "LLM rewrite"]
vals = [
    sum(r["hr_draft"] for r in reports) / max(len(reports), 1),
    sum(r["conditions"]["symbolic"]["hr_final"] for r in reports) / max(len(reports), 1),
    sum(r["conditions"]["llm_rewrite"]["hr_final"] for r in reports) / max(len(reports), 1),
]
ax.bar(labels, vals, color=["#e76f51", "#2a9d8f", "#264653"])
ax.set_ylabel("Hallucination Rate (micro)")
ax.set_title(f"HR by condition (n={len(reports)}, D→L)")
plt.tight_layout()
fig.savefig(results_dir / "hr_by_condition.png", dpi=150)
plt.show()"""
    ),
    md("## Phase 4 — Showcase case (for professor slides)"),
    code(
        """showcase = next((r for r in reports if r.get("citations_draft")), reports[0])
print("CASE:", showcase["case_id"])
print("\\n--- DRAFT (excerpt) ---")
print(showcase["draft_summary"][:1200])
print("\\n--- FINAL symbolic (excerpt) ---")
print(showcase["conditions"]["symbolic"]["final_summary"][:1200])
print("\\n--- Citation labels (draft) ---")
if not showcase["citations_draft"]:
    print("  (none — pick another case or note PRIMERA did not emit reporter citations)")
for c in showcase["citations_draft"]:
    cit = c["citation"]
    print(f"  [{c['label']}] {cit.get('raw_text', '')[:80]}")
print("\\nHR draft:", round(showcase["hr_draft"], 3),
      "→ symbolic:", round(showcase["conditions"]["symbolic"]["hr_final"], 3),
      "| iters:", showcase["conditions"]["symbolic"]["correction_iterations"])"""
    ),
    md("## Phase 4 — Export manual audit template (200 cases)"),
    code(
        """audit_path = WORKDIR / exp_cfg.get("audit_template_path", "outputs/audit/manual_label_template_200.csv")
pool = load_examples_for_audit(ds["validation"], ds["test"])
export_audit_template(pool, audit_path, n_cases=200, seed=exp_cfg.get("seed", 42))
print("Audit template:", audit_path)"""
    ),
    md("## Phase 5 — Statistical testing"),
    code(
        """from evaluation.stats import bootstrap_ci, hr_reduction_stats, mcnemar_citation_test
from evaluation.metrics import compute_rouge

hr_stats = hr_reduction_stats(
    [r["hr_draft"] for r in reports],
    [r["conditions"]["symbolic"]["hr_final"] for r in reports],
)
mcnemar = mcnemar_citation_test(
    [r["citations_draft"] for r in reports],
    [r["conditions"]["symbolic"].get("correction_outcomes", []) for r in reports],
)
rouge_baseline = [compute_rouge(r["draft_summary"], r["gold_summary"])["rougeL"] for r in reports]
rouge_symbolic = [compute_rouge(r["conditions"]["symbolic"]["final_summary"], r["gold_summary"])["rougeL"] for r in reports]
rouge_ci = bootstrap_ci(rouge_baseline, rouge_symbolic)

print("HR reduction:", hr_stats)
print("McNemar:", mcnemar)
print("ROUGE-L bootstrap CI:", rouge_ci)

with open(results_dir / "statistical_tests.json", "w") as f:
    json.dump({"hr_reduction": hr_stats, "mcnemar": mcnemar, "rougeL_ci": rouge_ci}, f, indent=2, default=str)"""
    ),
    md("## Phase 5 — Error analysis export"),
    code(
        """from evaluation.error_analysis import export_error_analysis_csv

error_path = WORKDIR / exp_cfg.get("error_analysis_path", "outputs/results/error_analysis_review.csv")
export_error_analysis_csv(reports, error_path, n_corrected=min(20, len(reports)), n_failed=min(20, len(reports)))
print("Error analysis:", error_path)"""
    ),
    md("## Phase 5 — Research Question Answers (preliminary)"),
    code(
        """from evaluation.rq_report import generate_rq_report, save_rq_report

rq = generate_rq_report(
    reports, aggregated,
    hr_reduction_target_pct=exp_cfg.get("hr_reduction_target_pct", 30.0),
    rouge_tolerance_pct=exp_cfg.get("rouge_tolerance_pct", 2.0),
    split_name=SPLIT,
    n_cases=len(reports),
    partial_run=PARTIAL_RUN,
    presentation_label=exp_cfg.get("presentation_label"),
    run_bertscore=RUN_BERTSCORE,
)
json_path, md_path = save_rq_report(rq, results_dir)
print(open(md_path).read())
print("\\nSaved:", json_path, md_path)"""
    ),
    md("## Export results zip"),
    code(
        """import zipfile
from google.colab import files

zip_path = WORKDIR / "outputs" / "full_pipeline_results.zip"
zip_path.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in results_dir.glob("*"):
        zf.write(p, arcname=f"results/{p.name}")
    for p in conditions_dir.glob("*.json"):
        zf.write(p, arcname=f"conditions/{p.name}")
    if gen_path.exists():
        zf.write(gen_path, arcname=f"generations/{gen_path.name}")
    hr_png = results_dir / "hr_by_condition.png"
    if hr_png.exists():
        zf.write(hr_png, arcname="figures/hr_by_condition.png")
print("Wrote:", zip_path)
files.download(str(zip_path))"""
    ),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "colab": {"provenance": []},
    },
    "cells": cells,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("Wrote", OUT)
