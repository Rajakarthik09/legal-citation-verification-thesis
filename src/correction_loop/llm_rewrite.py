"""LLM-based sentence smoothing after symbolic citation edits."""

from __future__ import annotations

import re
from typing import Any

DEFAULT_MODEL = "google/flan-t5-base"


class LLMRewriter:
    """Lazy-loaded flan-t5 rewriter for post-symbolic smoothing."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        max_new_tokens: int = 128,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None
        self._device = device

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        device = self._device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(device)
        self._model.eval()
        self._device = device

    def rewrite_sentence(self, sentence: str) -> str:
        """Rewrite one sentence for fluency without adding citations."""
        self._ensure_loaded()
        import torch

        prompt = (
            "Rewrite the following legal summary sentence for clarity and fluency. "
            "Do not add new case citations or legal references. "
            "Keep all factual claims. Sentence: "
            f"{sentence.strip()}"
        )
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self._device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_beams=2,
                do_sample=False,
            )
        rewritten = self._tokenizer.decode(out[0], skip_special_tokens=True).strip()
        return rewritten if rewritten else sentence

    def smooth_text(self, text: str, *, redaction_marker: str = "[unverifiable citation removed]") -> str:
        """
        Smooth sentences that contain redactions or were edited.

        Only rewrites sentences containing the redaction marker or obvious edit artifacts.
        """
        if not text.strip():
            return text

        sentences = _split_sentences(text)
        out: list[str] = []
        for sent in sentences:
            if redaction_marker in sent or _needs_smoothing(sent):
                out.append(self.rewrite_sentence(sent))
            else:
                out.append(sent)
        return " ".join(out)

    def unload(self) -> None:
        """Free GPU memory."""
        self._model = None
        self._tokenizer = None
        try:
            import gc
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]


def _needs_smoothing(sentence: str) -> bool:
    """Heuristic: double spaces or awkward bracket placement."""
    return "  " in sentence or sentence.count("[") != sentence.count("]")


def create_rewriter_from_config(config: dict[str, Any]) -> LLMRewriter:
    return LLMRewriter(
        model_name=config.get("llm_rewrite_model", DEFAULT_MODEL),
        device=config.get("llm_rewrite_device"),
        max_new_tokens=config.get("llm_rewrite_max_tokens", 128),
    )
