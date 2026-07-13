"""API observability: per-request ID tagging, in-memory rate limiting, structured access logging."""

from __future__ import annotations

import json
import logging
import os
import stat
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Optional

from fastapi import Request

log = logging.getLogger("gpp.access")

RATE_LIMIT_RPM = 60

# Upper bound on distinct client_keys tracked at once — without it, a client that can vary its apparent source IP (trivial over IPv6) has an unbounded memory-exhaustion vector; matches this codebase's `lru_cache(maxsize=...)` bounding pattern elsewhere.
_MAX_TRACKED_CLIENTS = 100_000

# Throttles state persistence to at most once per this many seconds, bounding disk I/O under many distinct clients; a crash between persists loses at most this much accounting, not the whole bucket set.
_PERSIST_INTERVAL_S = 1.0

# A rejection still forces a persist sooner than the interval above (see allow()), but not unconditionally: write cost scales with the *total* tracked-client count (every persist serializes the whole dict), not just the rejected client, so a client retrying immediately after a 429 — with no backoff, malicious or just buggy — would otherwise force a full-dict write on every single request. Measured: ~2.5ms/request at 5,000 tracked clients, ~66ms/request at the 100k cap, run synchronously on the single event-loop thread (allow() is called with no `await` before it in `async def observability`), so it stalls every other visitor too. This interval bounds that to a fixed worst-case rate regardless of how fast rejections arrive.
_FORCE_PERSIST_INTERVAL_S = 0.2

# Generously covers _MAX_TRACKED_CLIENTS entries even with long IPv6 keys; matches manifest.py/gpu_spec_db.py's convention of capping untrusted-file reads before parsing them.
_MAX_STATE_BYTES = 10 * 1024 * 1024  # 10 MB


def new_request_id() -> str:
    return uuid.uuid4().hex


class RateLimiter:
    """In-memory token bucket, one bucket per client IP, LRU-bounded to `_MAX_TRACKED_CLIENTS` entries. Soft abuse prevention, not a hard SLA guarantee. State is per-process by default (`state_path=None`, same as the module-level spec/model caches elsewhere in this codebase); passing `state_path` opts into surviving a process restart by persisting to a symlink-guarded, atomically-written JSON file, matching `train_final.py`'s convention — `time.monotonic()` values stay valid to compare across the old and new process since CLOCK_MONOTONIC is boot-relative, not process-relative."""

    def __init__(
        self,
        rpm: int = RATE_LIMIT_RPM,
        max_clients: int = _MAX_TRACKED_CLIENTS,
        state_path: Path | None = None,
    ) -> None:
        self._capacity = float(rpm)
        self._refill_per_sec = rpm / 60.0
        self._max_clients = max_clients
        self._state_path = state_path
        self._last_persist = 0.0
        # client_key -> (tokens, last_refill_monotonic); insertion order = LRU order, moved to the end on every access so the evicted entry is genuinely the least-recently-seen client.
        self._buckets: OrderedDict[str, tuple[float, float]] = OrderedDict()
        if self._state_path is not None:
            self._load_state()

    def allow(self, client_key: str) -> bool:
        now = time.monotonic()
        # A never-seen client starts with a full bucket and zero elapsed time — seeding via `defaultdict(time.monotonic)` instead would call time.monotonic() *after* `now` is already captured, making `elapsed` spuriously negative and shaving a fraction of a token off every client's first request.
        tokens, last = self._buckets.get(client_key, (self._capacity, now))
        elapsed = now - last
        tokens = min(self._capacity, tokens + elapsed * self._refill_per_sec)
        allowed = tokens >= 1.0
        if allowed:
            tokens -= 1.0

        self._buckets[client_key] = (tokens, now)
        self._buckets.move_to_end(client_key)
        if len(self._buckets) > self._max_clients:
            self._buckets.popitem(last=False)

        if self._state_path is not None:
            # A rejection is forced through the throttle sooner (see _FORCE_PERSIST_INTERVAL_S), not unconditionally: a rapid burst can otherwise exhaust the bucket in-memory well before the next throttled write reaches disk, so a crash right after the burst would recover a stale, under-depleted snapshot and grant free requests — the one state a crash most needs to preserve is "this client is already being blocked." Bounded, not immediate, so a client retrying rejections in a tight loop can't force a full-dict write on every single request.
            self._persist_state(now, force=not allowed)
        return allowed

    def _load_state(self) -> None:
        # A single lstat() call refuses both a symlinked path and an oversized file before this reads/parses it — matching manifest.py's _refuse_symlink_and_oversize convention for any file this codebase treats as untrusted external state, not just MLPerf sources.
        try:
            st = os.lstat(self._state_path)
        except FileNotFoundError:
            return
        except OSError:
            log.warning("Could not stat persisted rate-limiter state at %s; starting fresh", self._state_path)
            return
        if stat.S_ISLNK(st.st_mode):
            log.warning("Refusing to load rate-limiter state through a symlink: %s", self._state_path)
            return
        if st.st_size > _MAX_STATE_BYTES:
            log.warning("Persisted rate-limiter state at %s is too large (%d bytes); starting fresh", self._state_path, st.st_size)
            return
        # The whole parse-and-shape-check lives in one try/except, not just json.loads(): a file that's valid JSON but the wrong shape (a list, or values that aren't a [tokens, last] pair of numbers) must degrade to "start fresh," never raise past this constructor — this runs at src/api/main.py's module import time, so an uncaught exception here would crash the app on every single startup, turning a supervisord crash-restart into a permanent crash loop instead of recovering it. Demonstrated exploitable before this fix: a state file containing a JSON list, or a dict with a non-numeric or wrong-length value, each crashed __init__ with an unhandled AttributeError/ValueError.
        try:
            raw = json.loads(self._state_path.read_text())
            if not isinstance(raw, dict):
                raise ValueError(f"expected a JSON object at the top level, got {type(raw).__name__}")
            loaded = {str(k): (float(tokens), float(last)) for k, (tokens, last) in raw.items()}
        except (OSError, ValueError, TypeError):
            log.warning("Could not parse persisted rate-limiter state from %s; starting fresh", self._state_path)
            return
        self._buckets.update(loaded)
        while len(self._buckets) > self._max_clients:
            self._buckets.popitem(last=False)

    def _persist_state(self, now: float, force: bool = False) -> None:
        interval = _FORCE_PERSIST_INTERVAL_S if force else _PERSIST_INTERVAL_S
        if now - self._last_persist < interval:
            return
        self._last_persist = now
        try:
            # The tmp-file + rename() below already can't write *through* a symlink at this path — POSIX rename() replaces the destination directory entry itself, it never dereferences it, unlike a bare open("w") (the risk train_final.py/03_model_training.ipynb's atomic-write helpers guard against for that reason). What this check protects against instead: silently clobbering an unexpected pre-existing symlink at this path into a plain file. Refusing and warning is more honest than overwriting something that isn't supposed to be here.
            if self._state_path.is_symlink():
                log.warning("Refusing to persist rate-limiter state through a symlink: %s", self._state_path)
                return
            tmp_path = self._state_path.with_name(self._state_path.name + ".tmp")
            tmp_path.write_text(json.dumps({k: list(v) for k, v in self._buckets.items()}))
            tmp_path.replace(self._state_path)
        except OSError:
            log.warning("Could not persist rate-limiter state to %s", self._state_path)


def log_access(
    request_id: str,
    request: Request,
    status_code: int,
    latency_ms: float,
) -> None:
    """One structured JSON line per request: request_id, ts, method, route, status, latency_ms, client_ip."""
    client_ip: Optional[str] = request.client.host if request.client else None
    log.info(json.dumps({
        "request_id": request_id,
        "ts": time.time(),
        "method": request.method,
        "route": request.url.path,
        "status": status_code,
        "latency_ms": round(latency_ms, 2),
        "client_ip": client_ip,
    }))
