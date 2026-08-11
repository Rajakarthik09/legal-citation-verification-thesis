"""Token-bucket rate limiter for CourtListener API calls.

Limits (10/min, 100/hr, 250/day) are the account's own confirmed access
level, read directly off its CourtListener dashboard -- the authoritative
source, after two earlier guesses both turned out wrong: a "plain free
account" assumption from THESIS_ROADMAP.md (125/day), then a 60/min figure
misread out of a captured 429 error message that described a different
throttle scope than this per-account one. An earlier version of this
limiter also only tracked minute+day and missed the hourly dimension
entirely -- it hit a real 429 on the very first call of a run despite the
daily counter showing budget left, because retries-on-429 were firing real
HTTP requests the rate limiter never counted (see
courtlistener_client.lookup_citations) on top of an unmodeled hourly tier.
All tiers are now derived from one persisted list of call timestamps, so
restarting the process (new RunPod/local session) can't lose track of any
of them.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

_DAY_SECONDS = 86400
_HOUR_SECONDS = 3600
_MINUTE_SECONDS = 60


@dataclass
class RateLimiter:
    per_minute: int = 10
    per_hour: int = 100
    daily_cap: int = 250
    state_path: Path | str | None = None
    _call_timestamps: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.state_path is not None:
            self.state_path = Path(self.state_path)
            self._load_state()

    def _load_state(self) -> None:
        if self.state_path is None or not self.state_path.exists():
            return
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            if "call_timestamps" in data:
                self._call_timestamps = [float(t) for t in data["call_timestamps"]]
            elif "day_start" in data and "day_count" in data:
                # Migrate the old {day_start, day_count} format: approximate
                # every historical call as having happened at day_start.
                # Still enforces the daily cap correctly; the minute/hour
                # windows will look empty for migrated state, which is safe
                # (under-throttles briefly rather than over-throttling
                # forever) since those windows are short-lived anyway.
                day_start = float(data["day_start"])
                day_count = int(data["day_count"])
                self._call_timestamps = [day_start] * day_count
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            pass  # corrupt/missing state — start fresh rather than crash the run

    def _save_state(self) -> None:
        if self.state_path is None:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"call_timestamps": self._call_timestamps}),
            encoding="utf-8",
        )
        tmp.replace(self.state_path)

    def _prune(self, now: float) -> None:
        cutoff = now - _DAY_SECONDS  # daily is the longest window we track
        self._call_timestamps = [t for t in self._call_timestamps if t > cutoff]

    def _count_since(self, now: float, window_seconds: float) -> int:
        cutoff = now - window_seconds
        return sum(1 for t in self._call_timestamps if t > cutoff)

    def acquire(self) -> None:
        now = time.time()
        self._prune(now)

        if self._count_since(now, _DAY_SECONDS) >= self.daily_cap:
            raise RuntimeError(
                f"Daily CourtListener cap reached ({self.daily_cap}). Stop and resume tomorrow."
            )

        while True:
            now = time.time()
            minute_n = self._count_since(now, _MINUTE_SECONDS)
            hour_n = self._count_since(now, _HOUR_SECONDS)
            if minute_n < self.per_minute and hour_n < self.per_hour:
                break
            waits = []
            if minute_n >= self.per_minute:
                oldest = min(t for t in self._call_timestamps if t > now - _MINUTE_SECONDS)
                waits.append(("minute", _MINUTE_SECONDS - (now - oldest) + 0.1))
            if hour_n >= self.per_hour:
                oldest = min(t for t in self._call_timestamps if t > now - _HOUR_SECONDS)
                waits.append(("hour", _HOUR_SECONDS - (now - oldest) + 0.5))
            window, sleep_for = max(waits, key=lambda w: w[1])
            sleep_for = min(sleep_for, _HOUR_SECONDS)  # safety cap per loop iteration
            print(f"  [rate limiter] {window}ly cap hit, sleeping {sleep_for:.0f}s...", flush=True)
            time.sleep(sleep_for)

        self._call_timestamps.append(time.time())
        self._save_state()

    @property
    def day_count(self) -> int:
        now = time.time()
        self._prune(now)
        return self._count_since(now, _DAY_SECONDS)

    @property
    def hour_count(self) -> int:
        now = time.time()
        self._prune(now)
        return self._count_since(now, _HOUR_SECONDS)

    @property
    def remaining_daily(self) -> int:
        return max(0, self.daily_cap - self.day_count)

    @property
    def remaining_hourly(self) -> int:
        return max(0, self.per_hour - self.hour_count)
