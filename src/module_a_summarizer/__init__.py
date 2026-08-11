"""Module A — neural summarizer."""

from module_a_summarizer.generate import generate_batch, load_generations
from module_a_summarizer.model import Summarizer, resolve_device

__all__ = ["Summarizer", "resolve_device", "generate_batch", "load_generations"]
