# Design memo: Multi-LexSum (Week 1)

## Dataset pin

- **HF:** [`allenai/multi_lexsum`](https://huggingface.co/datasets/allenai/multi_lexsum)
- **Config:** `v20230518`
- **Loader:** `multilexsum.load.load_multilexsum()` — `trust_remote_code=True`, **`datasets>=2.16,<4`**

## Schema (expected)

| Field | Description |
|--------|-------------|
| `id` | Case id |
| `sources` | `List[str]` multi-document inputs |
| `summary/long`, `summary/short`, `summary/tiny` | Expert summaries (short/tiny optional) |
| `case_metadata` | Dict when present |

**Splits:** `train`, `validation`, `test` (paper “dev” ≈ `validation`).

## Split policy

Use official HF splits only until evaluation protocol parity is decided.

## Out of scope (until advisor gate)

- Final summarization architecture (Legal-BERT vs long-context encoder–decoder).
- Full fine-tunes on M1; use Colab/GPU for training scale.
- CourtListener pipeline (later milestone).

## Limitations

- Very long sources; EDA documents truncation risk.
- Summaries/metadata **CC BY-NC** — keep `appendix_samples.md` internal to coursework.
- Dataset loading script + version pin — watch HF breaking changes.

## Reference

Shen et al., *Multi-LexSum*, 2022.
