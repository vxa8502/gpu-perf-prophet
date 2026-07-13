"""API observability: per-request ID tagging, in-memory rate limiting, structured access logging."""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections import OrderedDict
from typing import Optional

from fastapi import Request

log = logging.getLogger("gpp.access")

RATE_LIMIT_RPM = 60

# Upper bound on distinct client_keys tracked at once — without it, a client that can vary its apparent source IP (trivial over IPv6) has an unbounded memory-exhaustion vector; matches this codebase's `lru_cache(maxsize=...)` bounding pattern elsewhere.
_MAX_TRACKED_CLIENTS = 100_000


def new_request_id() -> str:
    return uuid.uuid4().hex


class RateLimiter:
    """In-memory token bucket, one bucket per client IP, LRU-bounded to `_MAX_TRACKED_CLIENTS` entries. Soft abuse prevention, not a hard SLA guarantee: state is per-process, so it resets on restart and isn't shared across worker processes — acceptable for this project's single-process deployment (same single-process assumption as the module-level spec/model caches elsewhere in this codebase)."""

    def __init__(self, rpm: int = RATE_LIMIT_RPM, max_clients: int = _MAX_TRACKED_CLIENTS) -> None:
        self._capacity = float(rpm)
        self._refill_per_sec = rpm / 60.0
        self._max_clients = max_clients
        # client_key -> (tokens, last_refill_monotonic); insertion order = LRU order, moved to the end on every access so the evicted entry is genuinely the least-recently-seen client.
        self._buckets: OrderedDict[str, tuple[float, float]] = OrderedDict()

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
        return allowed


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
