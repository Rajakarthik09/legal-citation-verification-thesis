"""
Module A (LLM variant) -- instruction-tuned open LLM summarizer.

Cross-model comparison arm alongside PRIMERA (module_a_summarizer): same
verify -> correct -> evaluate pipeline downstream, different generator.
Two things PRIMERA cannot do regardless of GPU budget, both real ablation
variables (not just "swap the model"):
  1. Much larger input context (this checkpoint: 32768 tokens vs PRIMERA's
     architecturally fixed 4096 -- see config/module_a.yaml for the measured
     evidence PRIMERA's own output length, not input truncation, was the
     dominant constraint on citation count).
  2. Explicit citation-preservation instructions PRIMERA cannot receive
     (it's a fine-tuned seq2seq model, not instruction-following).
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_CHECKPOINT = "Qwen/Qwen2.5-7B-Instruct"  # Apache-2.0, ungated, 32768-token context

SYSTEM_PROMPT = (
    "You are a legal analyst producing case summaries for a civil-rights litigation "
    "database. Write a thorough, accurate summary of the case based only on the "
    "provided source documents. Preserve EVERY case citation you encounter in the "
    "source material that supports the legal reasoning, procedural history, or "
    "outcome (format: Case Name, Volume Reporter Page (Year)). Do not invent "
    "citations that are not present in the source text -- if you are not sure a "
    "citation is accurate, omit it rather than guess."
)

USER_TEMPLATE = (
    "Source documents for this case (may be excerpted for length):\n\n{sources}\n\n"
    "Write a comprehensive summary of this case, approximately {target_words} words, "
    "preserving every real case citation from the source material above."
)


def resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class LLMSummarizer:
    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        device: str | None = None,
        use_bf16: bool | None = None,
    ):
        self.checkpoint = checkpoint
        self.device = device or resolve_device()
        use_bf16 = self.device == "cuda" if use_bf16 is None else use_bf16
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        dtype = torch.bfloat16 if use_bf16 else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=dtype).to(self.device)
        self.model.eval()
        self.max_context = getattr(self.model.config, "max_position_embeddings", 32768)

    def generate(
        self,
        sources_text: str,
        max_new_tokens: int = 2000,
        target_words: int = 1500,
        temperature: float = 0.3,
    ) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(sources=sources_text, target_words=target_words)},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # sources_text should already be pre-truncated to the intended input
        # budget (build_case_input / build_case_input_retrieval) -- this cap
        # is a safety net against the template pushing over max_context, not
        # the primary truncation step.
        max_input_len = max(self.max_context - max_new_tokens, 256)
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=max_input_len
        ).to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else 1.0,
                top_p=0.9 if temperature > 0 else 1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated_ids = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def unload(self) -> None:
        del self.model
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
