"""Shared Colab dependency install cells for notebook builders."""

from __future__ import annotations

# Pin numpy<2 before pyarrow/datasets — Colab's numpy 2.x stack often breaks with
# force-reinstalled pyarrow (ImportError: cannot import name '_center' from numpy._core.umath).
INSTALL_DEPS_CELL = '''import subprocess

REQ = WORKDIR / "requirements-colab.txt"
if not REQ.exists():
    raise FileNotFoundError(f"Missing {REQ}. Upload full_pipeline/ to Drive.")

# 1) Pin numpy stack first (must stay on 1.x for current pyarrow/datasets on Colab)
subprocess.run(
    [
        "pip", "install", "-q",
        "numpy>=1.26.0,<2.0.0",
        "pyarrow>=14.0.0,<18.0.0",
        "pandas>=2.0.0,<2.3.0",
    ],
    check=True,
)

# 2) Colab pre-installs datasets 4.x; force downgrade for multi_lexsum loading scripts
subprocess.run(
    [
        "pip", "install", "-q",
        "datasets>=2.16.0,<4.0.0",
        "huggingface_hub",
        "--force-reinstall",
        "--no-deps",
    ],
    check=True,
)
subprocess.run(
    ["pip", "install", "-q", "datasets>=2.16.0,<4.0.0", "huggingface_hub"],
    check=True,
)

# 3) Remaining thesis deps (respect numpy pin above)
subprocess.run(["pip", "install", "-q", "-r", str(REQ)], check=True)

import datasets
import numpy as np
import pyarrow as pa

print("numpy:", np.__version__)
print("pyarrow:", pa.__version__)
print("datasets:", datasets.__version__)

if not datasets.__version__.startswith(("2.", "3.")):
    raise RuntimeError(
        f"Expected datasets<4, got {datasets.__version__}. "
        "Use Runtime → Restart session, then re-run from the top."
    )
if np.__version__.startswith("2."):
    raise RuntimeError(
        f"numpy 2.x breaks pyarrow on Colab (got {np.__version__}). "
        "Restart runtime and re-run the install cell."
    )

# Smoke-import stack used by Phases 0–2 (fail fast with a clear error)
import pandas  # noqa: F401
import torch  # noqa: F401
import transformers  # noqa: F401

print("Dependency stack OK.")'''
