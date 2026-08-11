#!/usr/bin/env python3
"""Phase 0 smoke test: Multi-LexSum load + CourtListener citation lookup."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure code/ is on path and .env is loaded
_CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CODE_ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(_CODE_ROOT / ".env")

import requests
from data.load_multilexsum import get_split

OBERGEFELL_TEXT = (
    "Obergefell v. Hodges (576 U.S. 644) established the right to marriage"
)
COURTLISTENER_URL = "https://www.courtlistener.com/api/rest/v4/citation-lookup/"


def smoke_multilexsum() -> bool:
    print("=== Multi-LexSum ===")
    split = get_split("validation")
    if len(split) == 0:
        print("FAIL: validation split is empty")
        return False
    row = split[0]
    required = {"sources", "summary/long", "summary/short", "summary/tiny", "case_metadata", "id"}
    missing = required - set(row.keys())
    if missing:
        print(f"FAIL: missing keys {missing}")
        return False
    print(f"OK: validation has {len(split)} cases")
    print(f"  Example id: {row['id']}")
    print(f"  case_metadata: {row['case_metadata']}")
    return True


def smoke_courtlistener() -> bool:
    print("\n=== CourtListener ===")
    token = os.environ.get("COURTLISTENER_API_TOKEN", "")
    if not token or token.startswith("your-"):
        print("FAIL: COURTLISTENER_API_TOKEN not set in .env")
        return False
    resp = requests.post(
        COURTLISTENER_URL,
        headers={"Authorization": f"Token {token}"},
        data={"text": OBERGEFELL_TEXT},
        timeout=30,
    )
    print(f"  HTTP status: {resp.status_code}")
    try:
        body = resp.json()
    except Exception:
        print(f"FAIL: non-JSON response: {resp.text[:500]}")
        return False
    print(f"  Response (first item): {body[0] if isinstance(body, list) and body else body}")
    if resp.status_code != 200:
        print("FAIL: expected HTTP 200")
        return False
    return True


def main() -> int:
    ok_ml = smoke_multilexsum()
    ok_cl = smoke_courtlistener()
    if ok_ml and ok_cl:
        print("\nAll smoke tests passed.")
        return 0
    print("\nSmoke tests FAILED.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
