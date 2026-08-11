"""Batch generation for the LLM summarizer, resume-safe (mirrors module_a_summarizer.generate)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tqdm import tqdm

from module_a_llm.llm_summarizer import LLMSummarizer
from module_a_summarizer.generate import _append_row, _load_completed_ids
from utils.schemas import CaseInput


def generate_batch_llm(
    cases: list[CaseInput],
    config: dict[str, Any],
    *,
    split: str = "test",
    model_tag: str = "llm",
    summarizer: LLMSummarizer | None = None,
) -> Path:
    """
    Generate summaries for ``cases`` with the LLM summarizer, skipping
    ``case_id``s already in the output JSONL.

    ``model_tag`` distinguishes output files between conditions run against
    the same case set, e.g. "llm" (naive truncation) vs "llm_rag"
    (citation-priority truncation) -- both use this same function, differing
    only in which CaseInput.sources_text they were built with.

    Returns path to ``{split}_{model_tag}.jsonl``.
    """
    output_dir = Path(config.get("output_dir", "outputs/generations"))
    out_path = output_dir / f"{split}_{model_tag}.jsonl"
    done = _load_completed_ids(out_path)

    if summarizer is None:
        summarizer = LLMSummarizer(
            checkpoint=config.get("checkpoint", "Qwen/Qwen2.5-7B-Instruct"),
            device=config.get("device"),
            use_bf16=config.get("use_bf16"),
        )

    max_new_tokens = config.get("max_new_tokens", 2000)
    target_words = config.get("target_words", 1500)
    temperature = config.get("temperature", 0.3)

    pending = [c for c in cases if c.case_id not in done]
    for case in tqdm(pending, desc=f"ModuleA-LLM {split}/{model_tag}"):
        generated = summarizer.generate(
            case.sources_text,
            max_new_tokens=max_new_tokens,
            target_words=target_words,
            temperature=temperature,
        )
        _append_row(
            out_path,
            {
                "case_id": case.case_id,
                "granularity": case.granularity,
                "model_tag": model_tag,
                "checkpoint": summarizer.checkpoint,
                "generated_summary": generated,
                "gold_summary": case.gold_summary,
                "truncation_strategy": case.truncation_strategy,
                "truncated": case.truncated,
                "source_token_count": case.source_token_count,
            },
        )

    return out_path


def load_generations_llm(path: Path) -> dict[str, dict[str, Any]]:
    """Load LLM generations JSONL into a dict keyed by case_id."""
    import json

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
