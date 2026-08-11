"""Export manual audit labeling template (200 cases)."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

from module_b_extraction.extract_citations import extract_citations


def _citation_density(text: str) -> int:
    return len(extract_citations(text or ""))


def export_audit_template(
    examples: list[dict[str, Any]],
    path: str | Path,
    *,
    n_cases: int = 200,
    seed: int = 42,
    summary_key: str = "summary/long",
) -> Path:
    """
    Export stratified manual labeling template.

    Stratifies by citation density tertiles in gold long summaries.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pool: list[dict[str, Any]] = []
    for ex in examples:
        gold = ex.get(summary_key) or ""
        if not str(gold).strip():
            continue
        pool.append({
            "case_id": ex["id"],
            "gold_summary_excerpt": str(gold)[:800],
            "citation_density": _citation_density(str(gold)),
            "split_hint": ex.get("_split", ""),
        })

    if not pool:
        path.write_text("", encoding="utf-8")
        return path

    pool.sort(key=lambda x: x["citation_density"])
    n = len(pool)
    tertile_size = max(n // 3, 1)
    low = pool[:tertile_size]
    mid = pool[tertile_size : 2 * tertile_size]
    high = pool[2 * tertile_size :]

    rng = random.Random(seed)
    per_bucket = max(n_cases // 3, 1)
    selected: list[dict[str, Any]] = []
    for bucket in (low, mid, high):
        rng.shuffle(bucket)
        selected.extend(bucket[:per_bucket])
    if len(selected) < n_cases:
        remaining = [p for p in pool if p not in selected]
        rng.shuffle(remaining)
        selected.extend(remaining[: n_cases - len(selected)])
    selected = selected[:n_cases]

    fieldnames = [
        "case_id",
        "citation_density",
        "gold_summary_excerpt",
        "draft_summary_excerpt",
        "citation_text",
        "auto_label",
        "manual_label",
        "hallucination_type",
        "reviewer_notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected:
            writer.writerow({
                "case_id": row["case_id"],
                "citation_density": row["citation_density"],
                "gold_summary_excerpt": row["gold_summary_excerpt"],
                "draft_summary_excerpt": "",
                "citation_text": "",
                "auto_label": "",
                "manual_label": "",
                "hallucination_type": "",
                "reviewer_notes": "Fill after running pipeline; compare auto labels to manual review",
            })

    readme = path.parent / "AUDIT_README.txt"
    readme.write_text(
        "Manual Audit Template (200 cases)\n"
        "================================\n"
        "1. After Phase 3 completes, paste draft summary excerpts from condition reports.\n"
        "2. For each citation in the draft, record citation_text and auto_label from Module C.\n"
        "3. Assign manual_label: Verified | NotFound | Mismatch | Ambiguous | Malformed\n"
        "4. Assign hallucination_type if manual_label differs from auto_label.\n"
        f"Template: {path.name}\n",
        encoding="utf-8",
    )
    return path


def load_examples_for_audit(
    validation_split,
    test_split,
) -> list[dict[str, Any]]:
    """Combine validation + test splits for audit pool."""
    examples: list[dict[str, Any]] = []
    for row in validation_split:
        ex = dict(row)
        ex["_split"] = "validation"
        examples.append(ex)
    for row in test_split:
        ex = dict(row)
        ex["_split"] = "test"
        examples.append(ex)
    return examples
