#!/usr/bin/env python3
"""Print environment and GPU/compute sanity checks."""

from __future__ import annotations

import platform
import sys
from pathlib import Path

_CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_ROOT / "src"))


def main() -> None:
    print("=== Environment ===")
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Machine: {platform.machine()}")

    try:
        import torch

        print(f"\n=== PyTorch {torch.__version__} ===")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"MPS available: {torch.backends.mps.is_available()}")
    except ImportError:
        print("\nPyTorch: NOT INSTALLED")

    packages = [
        "transformers",
        "datasets",
        "eyecite",
        "requests",
        "yaml",
        "dotenv",
        "pytest",
        "tenacity",
    ]
    print("\n=== Package versions ===")
    for pkg in packages:
        try:
            mod = __import__(pkg if pkg != "yaml" else "yaml")
            ver = getattr(mod, "__version__", "unknown")
            print(f"  {pkg}: {ver}")
        except ImportError:
            print(f"  {pkg}: NOT INSTALLED")


if __name__ == "__main__":
    main()
