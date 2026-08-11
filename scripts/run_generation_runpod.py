#!/usr/bin/env python3
"""
Module A generation driver for RunPod (Phase 1-2 only — no Drive/Colab deps).

Runs the same resume-safe generate_batch() the Colab notebook uses, so
outputs/generations_test100/test_long.jsonl is a drop-in replacement for
whatever the notebook would have produced. Designed to run under tmux/nohup
on a RunPod pod: safe to disconnect and reattach, safe to re-run (skips
case_ids already written to the output JSONL).

CourtListener is never called here — Module C (verification/correction) runs
locally afterward, over multiple days, to respect the 125/day cap without
paying for idle GPU time.

Usage (on the pod, inside tmux):
    cd /workspace/code
    source .venv/bin/activate
    python scripts/run_generation_runpod.py --config experiment_test100.yaml
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description="RunPod Module A batch generation")
    parser.add_argument("--config", default="experiment_test100.yaml")
    parser.add_argument("--require-cuda", action="store_true", default=True)
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")

    import torch
    from transformers import AutoTokenizer

    from data.case_selection import select_cases
    from data.load_multilexsum import clear_dataset_cache, load_multilexsum
    from data.preprocessing import GRANULARITY_CHECKPOINTS, build_case_input
    from module_a_summarizer import Summarizer, generate_batch, load_generations
    from smoke.colab_smoke import (
        smoke_environment,
        smoke_hf_token_only,
        smoke_module_b,
        smoke_multilexsum,
    )
    from utils.config import load_yaml

    print("=== Pre-flight smoke checks ===")
    # smoke_hf_token_only(), not smoke_secrets() -- this pod never holds
    # COURTLISTENER_API_TOKEN (Module C runs locally instead).
    for result in (smoke_environment(require_cuda=args.require_cuda), smoke_hf_token_only()):
        status = "PASS" if result.passed else "FAIL"
        print(f"  [{status}] {result.name} — {result.message}")
        if not result.passed:
            print("\nAborting: fix the failure above before burning GPU time.")
            return 1

    if not torch.cuda.is_available():
        print("FATAL: CUDA not available on this pod. Check the RunPod GPU template.")
        return 1
    print(f"  GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")

    exp_cfg = load_yaml(args.config)
    data_cfg = load_yaml("data.yaml")
    mod_a = load_yaml("module_a.yaml")

    split_name = exp_cfg["split"]
    granularity = exp_cfg["granularity"]
    assert granularity == "long", "This driver is scoped to D->L (long) only."

    print(f"\n=== Experiment: {args.config} | {split_name}/{granularity} | max_cases={exp_cfg['max_cases']} ===")

    clear_dataset_cache()
    ds = load_multilexsum(cache_dir=ROOT / "data" / "hf")
    split = ds[split_name]
    print(f"Split {split_name!r}: {len(split)} cases total")

    # Confirms the dataset loader is healthy before spending GPU time.
    smoke_ds = smoke_multilexsum()
    print(f"  [{'PASS' if smoke_ds.passed else 'FAIL'}] multilexsum — {smoke_ds.message}")
    smoke_b = smoke_module_b()
    print(f"  [{'PASS' if smoke_b.passed else 'FAIL'}] module_b — {smoke_b.message}")
    if not (smoke_ds.passed and smoke_b.passed):
        return 1

    tokenizer = AutoTokenizer.from_pretrained(GRANULARITY_CHECKPOINTS[granularity])
    max_tokens = data_cfg["max_tokens"][granularity]

    selected_rows = select_cases(
        split,
        case_ids=exp_cfg.get("case_ids"),
        max_cases=exp_cfg["max_cases"],
        min_gold_citations=exp_cfg.get("min_gold_citations", 1),
        seed=exp_cfg.get("seed", 42),
    )
    if exp_cfg.get("case_ids"):
        missing = set(exp_cfg["case_ids"]) - {r["id"] for r in selected_rows}
        if missing:
            print(f"WARNING: {len(missing)} pinned case_ids not found in split {split_name!r}: {sorted(missing)[:10]}...")

    cases = []
    for row in selected_rows:
        case = build_case_input(row, granularity, tokenizer, max_tokens)
        if case is not None:
            cases.append(case)

    print(f"Prepared {len(cases)}/{len(selected_rows)} CaseInput objects "
          f"(dropped = missing/empty gold summary)")
    print(f"Truncated to {max_tokens} tokens: {sum(1 for c in cases if c.truncated)}/{len(cases)}")
    if not cases:
        print("FATAL: no cases selected — check case_ids in the experiment YAML.")
        return 1

    mod_a_run = {
        **mod_a,
        "granularity": granularity,
        "device": "cuda",
        "use_fp16": True,
        "output_dir": str(ROOT / exp_cfg.get("generations_dir", "outputs/generations")),
    }

    print(f"\n=== Loading {mod_a['checkpoints'][granularity]} ===")
    t0 = time.time()
    summarizer = Summarizer(granularity=granularity, device="cuda", use_fp16=True)
    print(f"Loaded in {time.time() - t0:.1f}s")

    print(f"\n=== Generating ({len(cases)} cases, resume-safe JSONL) ===")
    t0 = time.time()
    gen_path = generate_batch(cases, mod_a_run, split=split_name, summarizer=summarizer)
    elapsed = time.time() - t0
    generations = load_generations(gen_path)
    print(f"\nDone in {elapsed / 60:.1f} min. {len(generations)} generations -> {gen_path}")
    print(f"({elapsed / max(len(cases), 1):.1f}s/case average)")

    del summarizer
    gc.collect()
    torch.cuda.empty_cache()

    print("\n=== Next step ===")
    print(f"Download {gen_path} to your local machine, e.g.:")
    print(f"  scp -P <port> root@<pod-host>:{gen_path} code/outputs/generations_test100/")
    print("Then run Module B/C + correction loop LOCALLY (no GPU needed, CourtListener-rate-limited).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
