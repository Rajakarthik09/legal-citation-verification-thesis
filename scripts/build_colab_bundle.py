#!/usr/bin/env python3
"""Sync src/ + config/ + notebooks into colab_upload/ for Google Drive."""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "colab_upload"
SRC = ROOT / "src"
CONFIG = ROOT / "config"

NOTEBOOKS = [
    ("notebooks/thesis_full_pipeline_colab.ipynb", "notebooks/thesis_full_pipeline_colab.ipynb"),
    ("notebooks/phase2_demo_colab.ipynb", "notebooks/phase2_demo_colab.ipynb"),
]


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(
        src,
        dst,
        ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", ".pytest_cache", "*.egg-info"
        ),
    )


def main() -> None:
    BUNDLE.mkdir(parents=True, exist_ok=True)
    _copy_tree(SRC, BUNDLE / "src")
    _copy_tree(CONFIG, BUNDLE / "config")

    (BUNDLE / "notebooks").mkdir(parents=True, exist_ok=True)
    for src_rel, dst_rel in NOTEBOOKS:
        src = ROOT / src_rel
        dst = BUNDLE / dst_rel
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  notebook: {dst_rel}")
        else:
            print(f"  skip (missing): {src_rel}")

    req_src = ROOT / "requirements-colab.txt"
    if req_src.exists():
        shutil.copy2(req_src, BUNDLE / "requirements-colab.txt")

    readme_src = ROOT / "colab_upload" / "README_COLAB.md"
    readme_dst = BUNDLE / "README_COLAB.md"
    if readme_src.exists() and readme_src.resolve() != readme_dst.resolve():
        pass  # README updated in place below

    print(f"\nColab bundle ready: {BUNDLE}")
    print("Upload to Google Drive as MyDrive/Thesis/full_pipeline/")
    print("Open notebooks/thesis_full_pipeline_colab.ipynb in Colab (T4 GPU).")


if __name__ == "__main__":
    main()
