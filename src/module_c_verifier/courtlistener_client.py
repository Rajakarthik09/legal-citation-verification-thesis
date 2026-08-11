"""CourtListener HTTP client with retry."""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from module_c_verifier.rate_limiter import RateLimiter

DEFAULT_URL = "https://www.courtlistener.com/api/rest/v4/citation-lookup/"

# We've now hit unparseable 429s twice (no Retry-After header, no matching
# DRF detail message) despite local pacing that should have stayed under the
# modeled 5/min, 50/hr tiers (daily_cap confirmed at 250/day, not the
# 125/day originally assumed) -- meaning the server enforces something we
# haven't fully modeled yet. Log full response details on any unparseable
# 429 so the next occurrence gives real evidence instead of another guess,
# and default to a conservative wait rather than none.
_DIAGNOSTIC_LOG = Path(
    os.environ.get(
        "COURTLISTENER_429_LOG",
        Path(__file__).resolve().parents[2] / "data" / "processed" / "courtlistener_429_log.jsonl",
    )
)
_FALLBACK_WAIT_SECONDS = 60.0


def _retry_after_seconds(resp: requests.Response) -> float | None:
    """
    Parse how long to wait before retrying a 429.

    Confirmed from real captured responses (data/processed/courtlistener_429_log.jsonl):
    CourtListener's actual body is {"error_message": "...", "wait_until":
    "<ISO 8601 timestamp>"} -- a custom format, not vanilla DRF. Two earlier
    versions of this parser both guessed wrong: one assumed `wait_until` was
    already a relative seconds count (always got None, logged "Rate limited
    until None"), the other assumed a DRF-style {"detail": "...available in
    N seconds."} message (also always None against the real format). Falls
    back to the Retry-After header and the DRF detail pattern in case a
    different throttle scope ever uses either of those instead.
    """
    header = resp.headers.get("Retry-After")
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    if not resp.content:
        return None
    try:
        body = resp.json()
    except ValueError:
        return None

    wait_until = body.get("wait_until")
    if wait_until:
        try:
            target = datetime.fromisoformat(str(wait_until))
            if target.tzinfo is None:
                target = target.replace(tzinfo=timezone.utc)
            delta = (target - datetime.now(timezone.utc)).total_seconds()
            if delta > 0:
                return delta
        except (ValueError, TypeError):
            pass

    detail = str(body.get("detail", ""))
    m = re.search(r"available in (\d+(?:\.\d+)?) second", detail)
    if m:
        return float(m.group(1))
    return None


def _log_unparseable_429(resp: requests.Response) -> None:
    try:
        body = resp.text[:2000]
    except Exception:
        body = "<unreadable>"
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status_code": resp.status_code,
        "headers": dict(resp.headers),
        "body": body,
    }
    try:
        _DIAGNOSTIC_LOG.parent.mkdir(parents=True, exist_ok=True)
        with _DIAGNOSTIC_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass  # diagnostics are best-effort, never let logging break the run
    print(f"  [courtlistener] unparseable 429 -- logged raw response to {_DIAGNOSTIC_LOG}", flush=True)


@retry(wait=wait_exponential(multiplier=1, min=2, max=60), stop=stop_after_attempt(5))
def lookup_citations(
    text: str,
    *,
    url: str | None = None,
    token: str | None = None,
    rate_limiter: RateLimiter | None = None,
) -> list[dict]:
    # rate_limiter.acquire() belongs INSIDE the retried function body, not
    # just once in the caller before the first attempt -- tenacity's retries
    # on a 429 fire real HTTP requests against the server too, and a caller-
    # side acquire() never saw those, silently undercounting real request
    # volume during exactly the bursts most likely to trip the server's
    # actual 60/min limit.
    if rate_limiter is not None:
        rate_limiter.acquire()
    token = token or os.environ.get("COURTLISTENER_API_TOKEN", "")
    if not token:
        raise RuntimeError("COURTLISTENER_API_TOKEN not set")
    resp = requests.post(
        url or DEFAULT_URL,
        headers={"Authorization": f"Token {token}"},
        data={"text": text},
        timeout=60,
    )
    if resp.status_code == 429:
        wait_s = _retry_after_seconds(resp)
        if wait_s is None:
            _log_unparseable_429(resp)
            wait_s = _FALLBACK_WAIT_SECONDS
        # Honor the server's real reset window (or the conservative
        # fallback above) before tenacity's next attempt, instead of hoping
        # a fixed exponential backoff happens to be long enough (it wasn't
        # -- this is what caused the crash on the first full-scale run: 5
        # fast attempts within ~30s against a throttle that needed longer).
        time.sleep(min(wait_s, 90) + 0.5)
        raise RuntimeError(f"Rate limited (retry after {wait_s}s)")
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list):
        return data
    return data.get("results", data)
