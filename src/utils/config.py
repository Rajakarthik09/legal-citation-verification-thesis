"""Configuration loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_CODE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = _CODE_ROOT / "config"


def load_yaml(name: str, *, config_dir: Path | None = None) -> dict[str, Any]:
    """Load a YAML config file from ``config/``."""
    config_dir = config_dir or DEFAULT_CONFIG_DIR
    path = config_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)
