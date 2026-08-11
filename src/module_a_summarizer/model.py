"""Module A — PRIMERA neural summarizer."""

from __future__ import annotations

import torch
from transformers import AutoTokenizer, LEDForConditionalGeneration

CHECKPOINTS = {
    "long": "allenai/primera-multi_lexsum-source-long",
    "short": "allenai/primera-multi_lexsum-source-short",
}


def resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Summarizer:
    def __init__(
        self,
        granularity: str,
        device: str | None = None,
        use_fp16: bool | None = None,
        checkpoint: str | None = None,
    ):
        if granularity not in CHECKPOINTS:
            raise ValueError(f"granularity must be one of {list(CHECKPOINTS)}, got {granularity!r}")
        ckpt = checkpoint or CHECKPOINTS[granularity]
        self.granularity = granularity
        self.device = device or resolve_device()
        use_fp16 = self.device == "cuda" if use_fp16 is None else use_fp16
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt)
        model = LEDForConditionalGeneration.from_pretrained(ckpt).to(self.device)
        self.model = model.half() if use_fp16 else model
        self.max_encoder_length = self.model.config.max_encoder_position_embeddings

    def generate(
        self,
        sources_text: str,
        max_new_tokens: int,
        num_beams: int = 5,
        min_new_tokens: int | None = None,
        length_penalty: float = 2.0,
        early_stopping: bool = False,
    ) -> str:
        inputs = self.tokenizer(
            sources_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_encoder_length,
        ).to(self.device)
        global_attention_mask = torch.zeros_like(inputs["input_ids"])
        global_attention_mask[:, 0] = 1
        # Explicit overrides, not the checkpoint's baked-in generation_config
        # (length_penalty=0 there favors short outputs in beam search -- see
        # config/module_a.yaml for the measured evidence this was truncating
        # summaries well before max_new_tokens).
        out = self.model.generate(
            **inputs,
            global_attention_mask=global_attention_mask,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            no_repeat_ngram_size=3,
        )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)
