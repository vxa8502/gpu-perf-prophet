"""Unit tests for src/api/middleware.py: RateLimiter token-bucket behavior in isolation from the live app."""

from __future__ import annotations

import pytest

from src.api.middleware import RateLimiter, new_request_id


class TestNewRequestId:
    def test_returns_unique_hex_strings(self):
        ids = {new_request_id() for _ in range(100)}
        assert len(ids) == 100
        assert all(len(i) == 32 for i in ids)  # uuid4().hex


class TestRateLimiter:
    def test_allows_up_to_capacity(self):
        limiter = RateLimiter(rpm=5)
        results = [limiter.allow("1.2.3.4") for _ in range(5)]
        assert all(results)

    def test_rejects_beyond_capacity_within_the_same_instant(self):
        limiter = RateLimiter(rpm=5)
        for _ in range(5):
            assert limiter.allow("1.2.3.4")
        assert limiter.allow("1.2.3.4") is False

    def test_buckets_are_independent_per_client(self):
        limiter = RateLimiter(rpm=1)
        assert limiter.allow("1.2.3.4") is True
        assert limiter.allow("1.2.3.4") is False
        # A different IP has its own untouched bucket.
        assert limiter.allow("5.6.7.8") is True

    def test_refills_over_time(self, monkeypatch):
        import src.api.middleware as mw

        t = [1000.0]
        monkeypatch.setattr(mw.time, "monotonic", lambda: t[0])

        limiter = RateLimiter(rpm=60)  # 1 token/sec
        for _ in range(60):
            assert limiter.allow("1.2.3.4")
        assert limiter.allow("1.2.3.4") is False

        t[0] += 5.0  # 5 seconds later -> 5 tokens refilled
        allowed = [limiter.allow("1.2.3.4") for _ in range(5)]
        assert all(allowed)
        assert limiter.allow("1.2.3.4") is False

    def test_distinct_clients_are_capped_not_retained_forever(self):
        # Without a bound, a client that can vary its apparent source IP (trivial over IPv6) has an unbounded memory-exhaustion vector: one dict entry per distinct IP ever seen, forever.
        limiter = RateLimiter(rpm=5, max_clients=10)
        for i in range(1000):
            limiter.allow(f"10.0.0.{i}")
        assert len(limiter._buckets) == 10

    def test_least_recently_seen_client_is_evicted_first(self):
        limiter = RateLimiter(rpm=5, max_clients=2)
        limiter.allow("a")
        limiter.allow("b")
        limiter.allow("a")  # touches "a" again, so "b" is now the least-recently-seen
        limiter.allow("c")  # forces an eviction
        assert set(limiter._buckets) == {"a", "c"}

    def test_partially_drained_bucket_is_retained_across_calls(self):
        limiter = RateLimiter(rpm=5)
        limiter.allow("1.2.3.4")
        limiter.allow("1.2.3.4")
        assert "1.2.3.4" in limiter._buckets
        tokens, _ = limiter._buckets["1.2.3.4"]
        assert tokens == pytest.approx(3.0)
