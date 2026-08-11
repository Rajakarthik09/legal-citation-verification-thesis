#!/usr/bin/env python3
"""Generate notebooks/runpod_generation.ipynb — Phase 0-2 only, plain Jupyter (no Colab deps).

Companion to run_generation_runpod.py: same logic, notebook cells instead of
a terminal script, for driving generation interactively from the pod's
Jupyter UI. Module B/C/correction/eval (Phase 3-5) are NOT here — those run
locally afterward (scripts/run_verification_local.py), CourtListener-rate-
limited, no GPU needed.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "runpod_generation.ipynb"


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
        """# RunPod Generation — Phase 0-2 (Module A only)

Generates PRIMERA summaries for the 100-case full-scale run. GPU-bound only —
Module B/C (CourtListener verification, correction loop) runs **locally**
afterward over several days (rate-limited to 125 requests/day), not here.

**Before running:** upload `code_bundle.zip` (built locally, see
`docs/RUNPOD_SETUP.md`) into the same folder as this notebook via the Jupyter
file browser.

| Cell | Does |
|---|---|
| 1 | Unzip code, set up paths |
| 2 | Confirm GPU is visible |
| 3 | Install dependencies |
| 4 | Set `HF_TOKEN` (typed via masked prompt, not saved to this file) |
| 5 | Pre-flight smoke checks |
| 6 | Load the 100 selected cases |
| 7 | **Generate** (resume-safe — re-running skips completed case_ids) |
| 8 | Wrap-up: what to download, remember to terminate the pod |
"""
    ),
    md("## 1. Unzip code + set up paths"),
    code(
        """import zipfile
from pathlib import Path

HERE = Path.cwd()
ZIP_PATH = HERE / "code_bundle.zip"
if not ZIP_PATH.exists():
    raise FileNotFoundError(
        f"{ZIP_PATH} not found. Upload code_bundle.zip into this notebook's folder first."
    )

if not (HERE / "code").exists():
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(HERE)
    print("Unzipped to", HERE / "code")
else:
    print("code/ already present, skipping unzip.")

WORKDIR = HERE / "code"
import os, sys
os.chdir(WORKDIR)
sys.path.insert(0, str(WORKDIR / "src"))
print("cwd:", Path.cwd())
print("src/ exists:", (WORKDIR / "src").exists())"""
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
    raise RuntimeError("No CUDA device visible — check the pod's GPU template before continuing.")"""
    ),
    md("## 3. Install dependencies\n\nNumpy pinned `<2.0` — see `requirements-colab.txt` header for why."),
    code(
        """import subprocess

req = WORKDIR / "requirements-colab.txt"
subprocess.run(["pip", "install", "-q", "-r", str(req)], check=True)

import numpy as np, datasets, pandas, transformers
print("numpy:", np.__version__)
print("datasets:", datasets.__version__)
if np.__version__.startswith("2."):
    raise RuntimeError(f"numpy 2.x may break pyarrow/datasets (got {np.__version__}). Re-run this cell.")
if not datasets.__version__.startswith(("2.", "3.")):
    raise RuntimeError(f"Expected datasets<4, got {datasets.__version__}.")
print("Dependency stack OK.")"""
    ),
    md(
        """## 4. Set HF_TOKEN

Typed via a masked prompt so it never gets written into this notebook file.
Only `HF_TOKEN` is needed here — `COURTLISTENER_API_TOKEN` stays on your
local machine, since Module C runs there."""
    ),
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
    smoke_hf_token_only(),  # not smoke_secrets() -- this pod never holds COURTLISTENER_API_TOKEN
    smoke_multilexsum(),
    smoke_module_b(),
):
    status = "PASS" if result.passed else "FAIL"
    print(f"[{status}] {result.name} — {result.message}")
    if not result.passed:
        raise RuntimeError(f"Smoke check failed: {result.name} — {result.message}")

print("\\nAll smoke checks passed.")"""
    ),
    md("## 6. Load the 100 selected cases (test split, D→L)"),
    code(
        """from transformers import AutoTokenizer

from utils.config import load_yaml
from data.load_multilexsum import load_multilexsum, clear_dataset_cache
from data.preprocessing import build_case_input, GRANULARITY_CHECKPOINTS
from data.case_selection import select_cases

EXPERIMENT_CONFIG = "experiment_test100.yaml"

exp_cfg = load_yaml(EXPERIMENT_CONFIG)
data_cfg = load_yaml("data.yaml")
mod_a = load_yaml("module_a.yaml")

SPLIT = exp_cfg["split"]
GRANULARITY = exp_cfg["granularity"]
MAX_CASES = exp_cfg["max_cases"]
assert GRANULARITY == "long", "This notebook is scoped to D→L (long) only."
print(f"Experiment: {EXPERIMENT_CONFIG} | {SPLIT}/{GRANULARITY} | max_cases={MAX_CASES}")

clear_dataset_cache()
ds = load_multilexsum(cache_dir=WORKDIR / "data" / "hf")
split = ds[SPLIT]
print(f"Split {SPLIT!r}: {len(split)} cases total")

tokenizer = AutoTokenizer.from_pretrained(GRANULARITY_CHECKPOINTS[GRANULARITY])
max_tokens = data_cfg["max_tokens"][GRANULARITY]

selected_rows = select_cases(
    split,
    case_ids=exp_cfg.get("case_ids"),
    max_cases=MAX_CASES,
    min_gold_citations=exp_cfg.get("min_gold_citations", 1),
    seed=exp_cfg.get("seed", 42),
)
missing = set(exp_cfg.get("case_ids") or []) - {r["id"] for r in selected_rows}
if missing:
    print(f"WARNING: {len(missing)} pinned case_ids not found: {sorted(missing)[:10]}")

cases = []
for row in selected_rows:
    case = build_case_input(row, GRANULARITY, tokenizer, max_tokens)
    if case is not None:
        cases.append(case)

print(f"Prepared {len(cases)}/{len(selected_rows)} CaseInput objects")
print(f"Truncated to {max_tokens} tokens: {sum(1 for c in cases if c.truncated)}/{len(cases)}")
assert len(cases) > 0, "No cases selected — check case_ids in the experiment YAML."
print("First case:", cases[0].case_id)"""
    ),
    md(
        """## 7. Generate (resume-safe)

Checkpoints to `outputs/generations_test100/test_long.jsonl` after every case.
If this cell is interrupted (kernel restart, pod hiccup), just **re-run it** —
already-written `case_id`s are skipped, so nothing already done is repeated.

Expect roughly 1-3 hours for 100 D→L cases on an RTX 4090 (beam=5, ~850 output
tokens/case). Progress prints per case via the tqdm bar below."""
    ),
    code(
        """import gc
from module_a_summarizer import Summarizer, generate_batch, load_generations

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
print(f"\\nGenerations: {len(generations)}/{len(cases)} cases → {gen_path}")

del summarizer
gc.collect()
torch.cuda.empty_cache()
print("PRIMERA unloaded.")"""
    ),
    md("## 8. Done — next steps"),
    code(
        """print(f'''
Generation complete: {len(generations)}/{len(cases)} cases.

NEXT STEPS:
1. In the Jupyter file browser, right-click {gen_path.relative_to(WORKDIR)} -> Download.
2. Save it locally to: code/{gen_path.relative_to(WORKDIR)}
   (create the outputs/generations_test100/ folder locally if it doesn't exist yet)
3. Terminate this RunPod pod now -- generation is the only phase that needs the GPU.
4. Locally: python scripts/run_verification_local.py --config experiment_test100.yaml
   (Module B/C + correction + evaluation, CourtListener-rate-limited, runs over
   the next few days -- no GPU needed for this part.)
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
