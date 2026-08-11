#!/bin/bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a
source "$HERE/../.env"
set +a
exec "$HERE/../.venv/bin/python3" -m streamlit run "$HERE/live_showcase.py" --server.headless true --server.port 8501
