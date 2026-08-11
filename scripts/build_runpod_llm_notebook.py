#!/usr/bin/env python3
"""Generate notebooks/runpod_llm_generation.ipynb -- LLM zero-shot + LLM+RAG conditions.

Companion to build_runpod_generation_notebook.py (PRIMERA). Loads Qwen2.5-7B-
Instruct ONCE and generates both conditions in one session (naive-truncation
then citation-priority-truncation), since reloading a 7B model twice would
waste billed GPU time for no reason.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "runpod_llm_generation.ipynb"


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
        """# RunPod Generation -- LLM zero-shot + LLM+RAG (cross-model comparison)

Generates two more conditions against the SAME 100 test-split cases already
used for PRIMERA: (1) Qwen2.5-7B-Instruct, naive truncation, larger context
than PRIMERA's fixed 4096 tokens; (2) the same model with citation-density
source-document retrieval instead of naive truncation. One model load, two
generation passes -- reloading a 7B model twice would waste billed GPU time.

**Before running:** rebuild and re-upload `code_bundle.zip` (it must include
`src/module_a_llm/`, the updated `src/data/preprocessing.py`, and the new
`config/experiment_test100_llm*.yaml` / `config/module_llm.yaml` files --
these are new since the PRIMERA run). This notebook **overwrites** any
existing `code/` folder next to it, so a stale previous copy can't cause
confusing bugs.

| Cell | Does |
|---|---|
| 1 | Force-unzips code (overwrites stale copy), sets up paths |
| 2 | Confirms GPU is visible |
| 3 | Installs dependencies (same requirements-colab.txt, no new packages) |
| 4 | Sets `HF_TOKEN` (masked prompt) |
| 5 | Pre-flight smoke checks |
| 6 | Loads the 100 cases TWICE: naive truncation + citation-priority truncation (both at 8192 tokens, Qwen tokenizer) |
| 7 | Loads Qwen2.5-7B-Instruct once |
| 8 | **Generates condition 1** (llm, naive) -- resume-safe |
| 9 | **Generates condition 2** (llm_rag, citation-priority) -- resume-safe, reuses the loaded model |
| 10 | Wrap-up: what to download, remember to terminate the pod |
"""
    ),
    md("## 1. Unzip code (force-overwrite) + set up paths"),
    code(
        """import shutil
import zipfile
from pathlib import Path

HERE = Path.cwd()
ZIP_PATH = HERE / "code_bundle.zip"
if not ZIP_PATH.exists():
    raise FileNotFoundError(
        f"{ZIP_PATH} not found. Upload the freshly rebuilt code_bundle.zip into this notebook's folder first."
    )

CODE_DIR = HERE / "code"
if CODE_DIR.exists():
    shutil.rmtree(CODE_DIR)  # force-overwrite -- code changed since the PRIMERA run
with zipfile.ZipFile(ZIP_PATH) as zf:
    zf.extractall(HERE)
print("Unzipped (fresh) to", CODE_DIR)

WORKDIR = CODE_DIR
import os, sys
os.chdir(WORKDIR)
sys.path.insert(0, str(WORKDIR / "src"))
assert (WORKDIR / "src" / "module_a_llm").exists(), "module_a_llm missing -- rebuild code_bundle.zip locally first"
print("cwd:", Path.cwd())
print("module_a_llm/ present:", (WORKDIR / "src" / "module_a_llm").exists())"""
    ),
    md("## 2. Confirm GPU"),
    code(
        """import subprocess
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)

import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    raise RuntimeError("No CUDA device visible -- check the pod's GPU template before continuing.")"""
    ),
    md("## 3. Install dependencies"),
    code(
        """import subprocess

req = WORKDIR / "requirements-colab.txt"
subprocess.run(["pip", "install", "-q", "-r", str(req)], check=True)

import numpy as np, datasets, pandas, transformers
print("numpy:", np.__version__)
print("datasets:", datasets.__version__)
if np.__version__.startswith("2."):
    raise RuntimeError(f"numpy 2.x may break pyarrow/datasets (got {np.__version__}). Re-run this cell.")
print("Dependency stack OK.")"""
    ),
    md("## 4. Set HF_TOKEN (masked, not saved into this file)"),
    code(
        """import os
from getpass import getpass

os.environ["HF_TOKEN"] = getpass("HF_TOKEN: ")
os.environ["HF_HOME"] = str(WORKDIR / "data" / "hf")
os.environ["MULTILEXSUM_CACHE"] = str(WORKDIR / "data" / "hf")
print("HF_TOKEN set:", bool(os.environ["HF_TOKEN"]))"""
    ),
    md("## 5. Pre-flight smoke checks"),
    code(
        """from smoke.colab_smoke import smoke_environment, smoke_hf_token_only, smoke_multilexsum, smoke_module_b

for result in (
    smoke_environment(require_cuda=True),
    smoke_hf_token_only(),
    smoke_multilexsum(),
    smoke_module_b(),
):
    status = "PASS" if result.passed else "FAIL"
    print(f"[{status}] {result.name} -- {result.message}")
    if not result.passed:
        raise RuntimeError(f"Smoke check failed: {result.name} -- {result.message}")

print("\\nAll smoke checks passed.")"""
    ),
    md(
        """## 6. Load the 100 cases -- naive truncation AND citation-priority truncation

Both at 8192 tokens (vs PRIMERA's fixed 4096), Qwen's own tokenizer.
`build_case_input_retrieval` ranks each case's source documents by a fast
citation-pattern heuristic before truncating -- see
`src/data/preprocessing.py` for why this uses a regex proxy instead of full
eyecite parsing (eyecite's cross-reference resolution over 100 cases' full
raw dockets, some 100k+ tokens each, is too slow to be worth it for a coarse
ranking signal -- it would sit here burning GPU-pod time before generation
even starts)."""
    ),
    code(
        """from transformers import AutoTokenizer

from utils.config import load_yaml
from data.load_multilexsum import load_multilexsum, clear_dataset_cache
from data.preprocessing import build_case_input, build_case_input_retrieval
from data.case_selection import select_cases

llm_cfg = load_yaml("experiment_test100_llm.yaml")
rag_cfg = load_yaml("experiment_test100_llm_rag.yaml")
mod_llm = load_yaml("module_llm.yaml")
assert llm_cfg["case_ids"] == rag_cfg["case_ids"], "both conditions must use the same 100 case_ids"

SPLIT = llm_cfg["split"]
GRANULARITY = llm_cfg["granularity"]
CASE_IDS = llm_cfg["case_ids"]
MAX_TOKENS = mod_llm["max_input_tokens"]
print(f"Split={SPLIT} granularity={GRANULARITY} max_cases={len(CASE_IDS)} max_input_tokens={MAX_TOKENS}")

clear_dataset_cache()
ds = load_multilexsum(cache_dir=WORKDIR / "data" / "hf")
split = ds[SPLIT]
print(f"Split {SPLIT!r}: {len(split)} cases total")

tokenizer = AutoTokenizer.from_pretrained(mod_llm["checkpoint"])
selected_rows = select_cases(split, case_ids=CASE_IDS, max_cases=len(CASE_IDS), min_gold_citations=1)
missing = set(CASE_IDS) - {r["id"] for r in selected_rows}
if missing:
    print(f"WARNING: {len(missing)} pinned case_ids not found: {sorted(missing)[:10]}")

naive_cases = [c for r in selected_rows if (c := build_case_input(r, GRANULARITY, tokenizer, MAX_TOKENS)) is not None]
rag_cases = [c for r in selected_rows if (c := build_case_input_retrieval(r, GRANULARITY, tokenizer, MAX_TOKENS)) is not None]
print(f"naive cases: {len(naive_cases)} | truncated: {sum(1 for c in naive_cases if c.truncated)}")
print(f"rag cases:   {len(rag_cases)} | truncated: {sum(1 for c in rag_cases if c.truncated)}")
diffs = sum(1 for a, b in zip(naive_cases, rag_cases) if a.sources_text != b.sources_text)
print(f"cases where retrieval reordering changed the truncated text: {diffs}/{len(naive_cases)}")
assert len(naive_cases) > 0 and len(rag_cases) > 0"""
    ),
    md("## 7. Load Qwen2.5-7B-Instruct (once, reused for both conditions)"),
    code(
        """import time
from module_a_llm import LLMSummarizer

t0 = time.time()
llm = LLMSummarizer(
    checkpoint=mod_llm["checkpoint"],
    device="cuda",
    use_bf16=mod_llm.get("use_bf16", True),
)
print(f"Loaded {mod_llm['checkpoint']} in {time.time() - t0:.1f}s, context={llm.max_context}")"""
    ),
    md(
        """## 8. Generate condition 1 -- LLM zero-shot (naive truncation)

Resume-safe: checkpoints to `outputs/generations_test100/test_llm.jsonl`
after every case. If interrupted, re-run this cell -- completed case_ids are
skipped."""
    ),
    code(
        """from module_a_llm import generate_batch_llm, load_generations_llm

llm_run_cfg = {**mod_llm, "output_dir": str(WORKDIR / llm_cfg.get("generations_dir", "outputs/generations"))}

t0 = time.time()
gen_path_llm = generate_batch_llm(naive_cases, llm_run_cfg, split=SPLIT, model_tag="llm", summarizer=llm)
elapsed = time.time() - t0
gens_llm = load_generations_llm(gen_path_llm)
print(f"\\nCondition 1 (llm) done in {elapsed/60:.1f} min: {len(gens_llm)}/{len(naive_cases)} -> {gen_path_llm}")"""
    ),
    md(
        """## 9. Generate condition 2 -- LLM + citation-priority retrieval

Same loaded model, different pre-truncated source text. Resume-safe the same
way, writes to `outputs/generations_test100/test_llm_rag.jsonl`."""
    ),
    code(
        """rag_run_cfg = {**mod_llm, "output_dir": str(WORKDIR / rag_cfg.get("generations_dir", "outputs/generations"))}

t0 = time.time()
gen_path_rag = generate_batch_llm(rag_cases, rag_run_cfg, split=SPLIT, model_tag="llm_rag", summarizer=llm)
elapsed = time.time() - t0
gens_rag = load_generations_llm(gen_path_rag)
print(f"\\nCondition 2 (llm_rag) done in {elapsed/60:.1f} min: {len(gens_rag)}/{len(rag_cases)} -> {gen_path_rag}")

llm.unload()
print("Qwen2.5-7B-Instruct unloaded.")"""
    ),
    md("## 10. Done -- next steps"),
    code(
        """print(f'''
Generation complete:
  llm:     {len(gens_llm)}/{len(naive_cases)} cases -> {gen_path_llm}
  llm_rag: {len(gens_rag)}/{len(rag_cases)} cases -> {gen_path_rag}

NEXT STEPS:
1. Download BOTH files via the Jupyter file browser:
   {gen_path_llm.relative_to(WORKDIR)}
   {gen_path_rag.relative_to(WORKDIR)}
2. Save them locally to the same relative paths under code/outputs/generations_test100/
3. Terminate this RunPod pod now.
4. Locally: python scripts/run_verification_local.py --config experiment_test100_llm.yaml
            python scripts/run_verification_local.py --config experiment_test100_llm_rag.yaml
   (CourtListener-rate-limited, shares the same daily cap/cache as the PRIMERA run.)
''')"""
    ),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": cells,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("Wrote", OUT)
