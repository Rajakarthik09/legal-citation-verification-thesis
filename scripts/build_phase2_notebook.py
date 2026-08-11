#!/usr/bin/env python3
"""Generate phase2_demo_colab.ipynb from template cells."""

from __future__ import annotations

import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from colab_install import INSTALL_DEPS_CELL

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "phase2_demo_colab.ipynb"


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
        """# Phase 2 Demo — Dual-System Legal Summarization

**Thesis pipeline:** Module A (PRIMERA) → Module B (eyecite) → Module C (CourtListener) → symbolic correction loop.

- **Batch:** 10 validation cases, long granularity
- **GPU:** T4 required (`Runtime → Change runtime type → GPU`)
- **Secrets:** `HF_TOKEN`, `COURTLISTENER_API_TOKEN` in Colab Secrets"""
    ),
    code(
        """# Mount Google Drive and resolve WORKDIR (flat or colab_upload/ layout)
from google.colab import drive
from pathlib import Path

drive.mount("/content/drive")

# ── Adjust if your Drive path differs ──
DRIVE_BASE = Path("/content/drive/MyDrive/Thesis/phase2")


def _resolve_workdir(base: Path) -> Path:
    candidates = [base / "colab_upload", base]
    for candidate in candidates:
        if (candidate / "src").exists() and (candidate / "requirements-colab.txt").exists():
            return candidate
    for candidate in candidates:
        if (candidate / "src").exists():
            return candidate
    if base.exists():
        return base
    fallback = Path("/content/phase2")
    return fallback if fallback.exists() else Path(".")


WORKDIR = _resolve_workdir(DRIVE_BASE)
print("WORKDIR:", WORKDIR.resolve())
print("requirements-colab.txt:", (WORKDIR / "requirements-colab.txt").exists())
print("src/:", (WORKDIR / "src").exists())"""
    ),
    code(INSTALL_DEPS_CELL),
    code(
        """import os
import sys
import json
import zipfile
from pathlib import Path

os.chdir(WORKDIR)
sys.path.insert(0, str(WORKDIR / "src"))
print("cwd:", Path.cwd())
print("src exists:", (WORKDIR / "src").exists())"""
    ),
    code(
        """# Load secrets from Colab user secrets
from google.colab import userdata

os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
os.environ["COURTLISTENER_API_TOKEN"] = userdata.get("COURTLISTENER_API_TOKEN")
os.environ["HF_HOME"] = str(WORKDIR / "data" / "hf")
os.environ["MULTILEXSUM_CACHE"] = str(WORKDIR / "data" / "hf")
print("Secrets loaded (tokens not printed).")"""
    ),
    code(
        """import torch
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

from utils.config import load_yaml
from data.load_multilexsum import load_multilexsum, clear_dataset_cache
from data.preprocessing import build_case_input, GRANULARITY_CHECKPOINTS
from module_a_summarizer import Summarizer, generate_batch, load_generations
from module_c_verifier import Verifier
from pipeline.run_pipeline import merge_configs, run_batch

print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")

exp_cfg = load_yaml("experiment_demo.yaml")
mod_a = load_yaml("module_a.yaml")
mod_c = load_yaml("module_c.yaml")
runtime_cfg = merge_configs(exp_cfg, mod_a, mod_c)

SPLIT = exp_cfg["split"]
GRANULARITY = exp_cfg["granularity"]
MAX_CASES = exp_cfg["max_cases"]
print(f"Experiment: {SPLIT} / {GRANULARITY} / max_cases={MAX_CASES}")"""
    ),
    code(
        """# Load Multi-LexSum (cached on Drive after first run)
clear_dataset_cache()
ds = load_multilexsum(cache_dir=WORKDIR / "data" / "hf")
split = ds[SPLIT]
print(f"Split {SPLIT}: {len(split)} cases")

tokenizer = AutoTokenizer.from_pretrained(GRANULARITY_CHECKPOINTS[GRANULARITY])
max_tokens = load_yaml("data.yaml")["max_tokens"][GRANULARITY]

cases = []
for i in range(len(split)):
    if len(cases) >= MAX_CASES:
        break
    case = build_case_input(split[i], GRANULARITY, tokenizer, max_tokens)
    if case is not None:
        cases.append(case)
print(f"Prepared {len(cases)} CaseInput objects")
print("First case:", cases[0].case_id, "| truncated:", cases[0].truncated)"""
    ),
    code(
        """# Module A — PRIMERA generation (resume-safe JSONL on Drive)
mod_a_run = {**mod_a, "granularity": GRANULARITY, "device": "cuda", "use_fp16": True}
mod_a_run["output_dir"] = str(WORKDIR / exp_cfg.get("generations_dir", "outputs/generations"))

summarizer = Summarizer(granularity=GRANULARITY, device="cuda", use_fp16=True)
gen_path = generate_batch(cases, mod_a_run, split=SPLIT, summarizer=summarizer)
generations = load_generations(gen_path)
print(f"Generations: {len(generations)} cases → {gen_path}")"""
    ),
    code(
        """# Full pipeline: verify + correct (Modules B, C, correction loop)
mod_c_run = {**mod_c, "cache_path": str(WORKDIR / "data" / "processed" / "citation_cache.sqlite")}
runtime_cfg["module_a"] = mod_a_run
runtime_cfg.update(mod_c_run)

verifier = Verifier(mod_c_run)
reports_dir = WORKDIR / exp_cfg.get("reports_dir", "outputs/verification_reports")
gen_text = {cid: row["generated_summary"] for cid, row in generations.items()}

reports = run_batch(
    cases,
    summarizer,
    verifier,
    runtime_cfg,
    reports_dir,
    generations=gen_text,
)
verifier.close()
print(f"Pipeline complete: {len(reports)} reports in {reports_dir}")"""
    ),
    code(
        """# Professor demo — summary table
rows = []
for r in reports:
    rows.append({
        "case_id": r["case_id"],
        "n_citations_draft": len(r["citations_draft"]),
        "hr_draft": round(r["hr_draft"], 3),
        "hr_final": round(r["hr_final"], 3),
        "hr_reduction": round(r["hr_draft"] - r["hr_final"], 3),
        "correction_iters": r["correction_iterations"],
    })
df = pd.DataFrame(rows)
display(df)

print("\\n=== Aggregate ===")
print(f"Mean HR draft:  {df['hr_draft'].mean():.3f}")
print(f"Mean HR final:  {df['hr_final'].mean():.3f}")
print(f"Mean reduction: {df['hr_reduction'].mean():.3f}")"""
    ),
    code(
        """# Showcase case — draft vs corrected with citation labels
showcase = reports[0]
print("CASE:", showcase["case_id"])
print("\\n--- DRAFT (excerpt) ---")
print(showcase["draft_summary"][:1200], "…" if len(showcase["draft_summary"]) > 1200 else "")
print("\\n--- FINAL (excerpt) ---")
print(showcase["final_summary"][:1200], "…" if len(showcase["final_summary"]) > 1200 else "")
print("\\n--- Citation labels (draft) ---")
for c in showcase["citations_draft"]:
    cit = c["citation"]
    print(f"  [{c['label']}] {cit['raw_text'][:80]}")"""
    ),
    code(
        """# HR before/after bar chart
fig, ax = plt.subplots(figsize=(10, 4))
x = range(len(df))
ax.bar([i - 0.2 for i in x], df["hr_draft"], width=0.4, label="HR draft", color="#e76f51")
ax.bar([i + 0.2 for i in x], df["hr_final"], width=0.4, label="HR final", color="#2a9d8f")
ax.set_xticks(list(x))
ax.set_xticklabels(df["case_id"], rotation=45, ha="right")
ax.set_ylabel("Hallucination Rate")
ax.set_title("Citation HR before vs after symbolic correction (demo batch)")
ax.legend()
plt.tight_layout()
fig.savefig(WORKDIR / "outputs" / "hr_comparison.png", dpi=150)
plt.show()"""
    ),
    code(
        """# Export demo_results.zip for download / sharing with professor
zip_path = WORKDIR / "outputs" / "demo_results.zip"
zip_path.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
  for p in reports_dir.glob("*.json"):
    zf.write(p, arcname=f"reports/{p.name}")
  if gen_path.exists():
    zf.write(gen_path, arcname=f"generations/{gen_path.name}")
  hr_png = WORKDIR / "outputs" / "hr_comparison.png"
  if hr_png.exists():
    zf.write(hr_png, arcname="figures/hr_comparison.png")
print("Wrote:", zip_path)
from google.colab import files
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
