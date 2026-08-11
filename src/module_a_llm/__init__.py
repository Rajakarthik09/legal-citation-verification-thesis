"""Module A (LLM variant) -- instruction-tuned open LLM summarizer, cross-model comparison arm."""

from module_a_llm.generate import generate_batch_llm, load_generations_llm
from module_a_llm.llm_summarizer import LLMSummarizer, resolve_device

__all__ = ["LLMSummarizer", "resolve_device", "generate_batch_llm", "load_generations_llm"]
