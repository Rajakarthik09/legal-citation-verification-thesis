"""Batch generation with resume-safe JSONL checkpointing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from module_a_summarizer.model import Summarizer
from utils.schemas import CaseInput


def _output_path(output_dir: Path, split: str, granularity: str) -> Path:
    return output_dir / f"{split}_{granularity}.jsonl"


def _load_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            done.add(row["case_id"])
    return done


def _append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def generate_batch(
    cases: list[CaseInput],
    config: dict[str, Any],
    *,
    split: str = "validation",
    summarizer: Summarizer | None = None,
) -> Path:
    """
    Generate summaries for ``cases``, skipping ``case_id``s already in the output JSONL.

    Returns path to ``{split}_{granularity}.jsonl``.
    """
    granularity = config.get("granularity", cases[0].granularity if cases else "long")
    output_dir = Path(config.get("output_dir", "outputs/generations"))
    out_path = _output_path(output_dir, split, granularity)
    done = _load_completed_ids(out_path)

    if summarizer is None:
        summarizer = Summarizer(
            granularity=granularity,
            device=config.get("device"),
            use_fp16=config.get("use_fp16"),
        )

    max_new_tokens = config["max_new_tokens"][granularity]
    min_new_tokens = config.get("min_new_tokens", {}).get(granularity)
    length_penalty = config.get("length_penalty", 2.0)
    early_stopping = config.get("early_stopping", False)
    num_beams = config.get("num_beams", 5)
    flush_every = config.get("checkpoint_flush_every", 1)

    pending = [c for c in cases if c.case_id not in done]
    for i, case in enumerate(tqdm(pending, desc=f"ModuleA {split}/{granularity}")):
        generated = summarizer.generate(
            case.sources_text,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            min_new_tokens=min_new_tokens,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
        )
        _append_row(
            out_path,
            {
                "case_id": case.case_id,
                "granularity": case.granularity,
                "generated_summary": generated,
                "gold_summary": case.gold_summary,
                "truncated": case.truncated,
                "source_token_count": case.source_token_count,
            },
        )
        if flush_every and (i + 1) % flush_every == 0:
            pass  # already flushed in _append_row

    return out_path


def load_generations(path: Path) -> dict[str, dict[str, Any]]:
    """Load generations JSONL into a dict keyed by case_id."""
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[row["case_id"]] = row
    return rows
