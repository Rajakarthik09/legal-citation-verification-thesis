#!/usr/bin/env python3
"""
Cross-model comparison: PRIMERA vs. LLM zero-shot vs. LLM+retrieval.

Reads the per-generator aggregate outputs already produced by
run_verification_local.py (evaluation.aggregate.aggregate_reports) -- no new
aggregation logic, just assembly across the three parallel runs. Safe to run
at any point: a generator whose verification hasn't started/finished yet is
skipped with a clear message rather than crashing, and partial runs are
labeled as such rather than silently presented as complete.

Usage:
    cd code && source .venv/bin/activate
    python scripts/compare_generators.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# (label, results_dir) -- the three generators being compared, all run
# against the identical 100 citation-rich test-split case_ids.
GENERATORS: list[tuple[str, str]] = [
    ("PRIMERA", "outputs/results_test100"),
    ("LLM zero-shot", "outputs/results_test100_llm"),
    ("LLM + retrieval", "outputs/results_test100_llm_rag"),
]

CONDITIONS_TO_COMPARE = ("baseline", "symbolic")
METRIC_FIELDS = (
    "n_cases",
    "total_citations",
    "hr_paired_micro",
    "rougeLsum_mean",
    "rougeL_mean",
    "bertscore_f1_mean",
    "fix_rate",
    "removal_rate",
)


def _load_generator(label: str, results_dir: Path) -> dict[str, Any] | None:
    metrics_path = results_dir / "metrics_summary.json"
    if not metrics_path.exists():
        print(f"  [skip] {label}: no results yet at {metrics_path}")
        return None

    metrics = json.loads(metrics_path.read_text())

    rq_path = results_dir / "rq_answers.json"
    partial = False
    if rq_path.exists():
        partial = bool(json.loads(rq_path.read_text()).get("partial_run", False))

    display_label = label
    n_cases = metrics.get("baseline", {}).get("n_cases")
    if partial:
        display_label = f"{label} (partial, n={n_cases})"

    rows = []
    for cond in CONDITIONS_TO_COMPARE:
        if cond not in metrics:
            continue
        row = {"generator": display_label, "condition": cond}
        row.update({f: metrics[cond].get(f) for f in METRIC_FIELDS})
        rows.append(row)
    return {"label": display_label, "raw_label": label, "partial": partial, "rows": rows, "metrics": metrics}


def _to_markdown_table(df) -> str:
    """Minimal dependency-free markdown table (avoids adding tabulate)."""
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def build_comparison(results_root: Path = ROOT) -> list[dict[str, Any]]:
    loaded = []
    for label, rel_dir in GENERATORS:
        result = _load_generator(label, results_root / rel_dir)
        if result is not None:
            loaded.append(result)
    return loaded


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare PRIMERA/LLM/LLM+RAG verification results")
    parser.add_argument("--out-dir", default="outputs/results_comparison")
    args = parser.parse_args()

    print("=== Cross-model comparison ===")
    loaded = build_comparison()
    if not loaded:
        print("\nNo generator results available yet -- nothing to compare.")
        return 0

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    all_rows = [row for entry in loaded for row in entry["rows"]]
    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "comparison_table.csv", index=False)
    with open(out_dir / "comparison_table.md", "w") as f:
        f.write(_to_markdown_table(df))
    print(f"\n{df.to_string(index=False)}")
    print(f"\nWrote {out_dir / 'comparison_table.csv'} and comparison_table.md")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [entry["label"] for entry in loaded]

    # Figure 1: HR by generator, baseline vs symbolic
    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(loaded))
    width = 0.35
    baseline_hr = [next((r["hr_paired_micro"] for r in e["rows"] if r["condition"] == "baseline"), None) for e in loaded]
    symbolic_hr = [next((r["hr_paired_micro"] for r in e["rows"] if r["condition"] == "symbolic"), None) for e in loaded]
    ax.bar([i - width / 2 for i in x], baseline_hr, width, label="Baseline", color="#e76f51")
    ax.bar([i + width / 2 for i in x], symbolic_hr, width, label="Symbolic (corrected)", color="#2a9d8f")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Hallucination Rate (paired)")
    ax.set_title("HR by generator: baseline vs. symbolic correction")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "hr_by_generator.png", dpi=150)
    plt.close(fig)

    # Figure 2: citations attempted per generator (baseline draft) -- the
    # number that motivated running LLM/RAG conditions in the first place.
    fig, ax = plt.subplots(figsize=(8, 5))
    attempted = [next((r["total_citations"] for r in e["rows"] if r["condition"] == "baseline"), None) for e in loaded]
    ax.bar(labels, attempted, color="#264653")
    ax.set_ylabel("Citations attempted (draft, extracted)")
    ax.set_title("Citation-attempt volume by generator")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    fig.savefig(out_dir / "citations_attempted_by_generator.png", dpi=150)
    plt.close(fig)

    print(f"Wrote {out_dir / 'hr_by_generator.png'} and citations_attempted_by_generator.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
