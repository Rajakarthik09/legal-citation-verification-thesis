#!/usr/bin/env python3
"""
Local smoke test: NOT the thesis baseline.
Requires: pip install -e ".[train]" and HF dataset cache (or network).
"""
from __future__ import annotations

import os

import torch
from datasets import load_dataset
from transformers import AdamW, AutoModelForSeq2SeqLM, AutoTokenizer


def main() -> None:
    if os.environ.get("HF_TOKEN"):
        print("HF_TOKEN is set")

    ds = load_dataset(
        "allenai/multi_lexsum",
        "v20230518",
        split="train[:256]",
        trust_remote_code=True,
    )
    model_name = "facebook/bart-base"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    batch = ds[:32]
    texts = ["\n".join(s or "" for s in src)[:8000] for src in batch["sources"]]
    summaries = [t or "" for t in batch["summary/long"]]
    enc = tok(texts, truncation=True, max_length=512, padding=True, return_tensors="pt")
    dec = tok(summaries, truncation=True, max_length=128, padding=True, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    dec = {k: v.to(device) for k, v in dec.items()}
    labels = dec["input_ids"].clone()
    labels[labels == tok.pad_token_id] = -100
    out = model(**enc, labels=labels)
    print("forward loss:", float(out.loss))

    opt = AdamW(model.parameters(), lr=1e-5)
    out.loss.backward()
    opt.step()
    opt.zero_grad()
    print("one optimizer step OK (subset batch 32 of 256 loaded)")


if __name__ == "__main__":
    main()
