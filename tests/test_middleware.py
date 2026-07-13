"""Unit tests for src/api/middleware.py: RateLimiter token-bucket behavior in isolation from the live app."""

from __future__ import annotations

import json

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


class TestRateLimiterPersistence:
    """state_path=None (every test above) is unaffected by any of this — persistence is opt-in, needed because supervisord's autorestart silently resets rate-limit state on every crash-restart."""

    def test_no_state_path_means_no_persistence(self, tmp_path):
        limiter = RateLimiter(rpm=5)
        limiter.allow("1.2.3.4")
        assert list(tmp_path.iterdir()) == []

    def test_persists_state_and_reloads_across_instances(self, tmp_path, monkeypatch):
        import src.api.middleware as mw

        t = [1000.0]
        monkeypatch.setattr(mw.time, "monotonic", lambda: t[0])

        state_path = tmp_path / "state.json"
        limiter1 = RateLimiter(rpm=5, state_path=state_path)
        assert limiter1.allow("1.2.3.4")
        t[0] += 2.0  # past the persist-throttle interval, so each call below actually hits disk
        assert limiter1.allow("1.2.3.4")
        t[0] += 2.0
        assert limiter1.allow("1.2.3.4")

        # A fresh instance sharing state_path should see the drained bucket (2 of 5 tokens left), not the full 5 a reset-on-restart would give it.
        limiter2 = RateLimiter(rpm=5, state_path=state_path)
        assert limiter2.allow("1.2.3.4")
        assert limiter2.allow("1.2.3.4")
        assert limiter2.allow("1.2.3.4") is False

    def test_persistence_is_throttled_within_the_interval(self, tmp_path, monkeypatch):
        import src.api.middleware as mw

        t = [1000.0]
        monkeypatch.setattr(mw.time, "monotonic", lambda: t[0])

        state_path = tmp_path / "state.json"
        limiter = RateLimiter(rpm=5, state_path=state_path)
        limiter.allow("1.2.3.4")
        first_write = state_path.read_text()

        limiter.allow("1.2.3.4")  # same instant — still inside the throttle window
        assert state_path.read_text() == first_write

        t[0] += 2.0
        limiter.allow("1.2.3.4")
        assert state_path.read_text() != first_write

    def test_rejection_forces_a_persist_sooner_than_the_normal_throttle_window(self, tmp_path, monkeypatch):
        # Verified against a real burst against a live uvicorn process first: a rapid burst can exhaust a bucket in-memory well before the next throttled write reaches disk, so a crash right after would recover a stale, under-depleted snapshot and hand out free requests. A rejection reaches disk within _FORCE_PERSIST_INTERVAL_S, well inside the normal 1s window — but still throttled, not unconditional (see the bounded-write-rate test below for why: cost scales with total tracked clients, not just the rejected one).
        import src.api.middleware as mw

        t = [1000.0]
        monkeypatch.setattr(mw.time, "monotonic", lambda: t[0])

        state_path = tmp_path / "state.json"
        limiter = RateLimiter(rpm=1, state_path=state_path)
        assert limiter.allow("1.2.3.4") is True  # first-ever persist, unconditional
        first_write = state_path.read_text()

        t[0] += 0.5  # past the force interval, still well inside the normal 1s throttle window
        assert limiter.allow("1.2.3.4") is False  # rejected — must still reach disk well before the normal window
        assert state_path.read_text() != first_write

    def test_rapid_repeated_rejections_do_not_each_force_a_write(self, tmp_path, monkeypatch):
        # A client retrying immediately after a 429 with no backoff — malicious or just buggy — must not force a full-dict write on every single request: write cost scales with the *total* tracked-client count (every persist serializes the whole dict), not just the rejected client. Measured against a real RateLimiter before this fix: ~2.5ms/request at 5,000 tracked clients, ~66ms/request at the 100k cap, run synchronously on the single event-loop thread (allow() is called with no `await` before it in `async def observability`) — an unthrottled force write per rejection stalls every other visitor too.
        import src.api.middleware as mw

        t = [1000.0]
        monkeypatch.setattr(mw.time, "monotonic", lambda: t[0])

        state_path = tmp_path / "state.json"
        limiter = RateLimiter(rpm=1, state_path=state_path)
        limiter.allow("1.2.3.4")  # consumes the only token

        write_calls = []
        real_replace = mw.Path.replace
        def counting_replace(self, target):
            write_calls.append(1)
            return real_replace(self, target)
        monkeypatch.setattr(mw.Path, "replace", counting_replace)

        for _ in range(500):
            assert limiter.allow("1.2.3.4") is False  # every call rejected, same instant — no time advances

        assert len(write_calls) <= 1  # at most the throttle allows within a single instant, never one per rejection

    def test_missing_state_file_starts_fresh(self, tmp_path):
        limiter = RateLimiter(rpm=5, state_path=tmp_path / "does_not_exist.json")
        assert limiter._buckets == {}

    def test_corrupted_state_file_starts_fresh_instead_of_raising(self, tmp_path):
        state_path = tmp_path / "state.json"
        state_path.write_text("not valid json{{{")
        limiter = RateLimiter(rpm=5, state_path=state_path)
        assert limiter._buckets == {}

    def test_refuses_to_persist_through_a_symlink(self, tmp_path):
        # `not target.exists()` alone is a false-confidence assertion here: the write is a temp-file + rename(), which POSIX guarantees never dereferences a symlink at the destination — target would stay untouched even with the is_symlink() guard deleted entirely (verified by mutation). What the guard actually prevents is silently clobbering state_path's own symlink into a plain file; that's the assertion that actually distinguishes guarded from unguarded behavior.
        target = tmp_path / "real_target.json"
        state_path = tmp_path / "state.json"
        state_path.symlink_to(target)

        limiter = RateLimiter(rpm=5, state_path=state_path)
        limiter.allow("1.2.3.4")

        assert state_path.is_symlink()
        assert not target.exists()

    def test_refuses_to_load_through_a_symlink(self, tmp_path):
        target = tmp_path / "real_target.json"
        target.write_text('{"1.2.3.4": [0.0, 1000.0]}')
        state_path = tmp_path / "state.json"
        state_path.symlink_to(target)

        limiter = RateLimiter(rpm=5, state_path=state_path)

        assert limiter._buckets == {}

    def test_oversized_state_file_starts_fresh_instead_of_loading(self, tmp_path):
        # Content must be valid JSON, not just large bytes: a giant invalid-JSON blob (e.g. "x" * N) gets rejected by the parse-error path regardless of the size guard, so that version of this test passed even with the size check deleted entirely (found by mutation). Building real, valid, oversized JSON is the only way to prove the size guard itself is load-bearing.
        import src.api.middleware as mw

        entries = {str(i): [1.0, 2.0] for i in range(600_000)}
        content = json.dumps(entries)
        assert len(content) > mw._MAX_STATE_BYTES

        state_path = tmp_path / "state.json"
        state_path.write_text(content)

        limiter = RateLimiter(rpm=5, state_path=state_path)

        assert limiter._buckets == {}

    @pytest.mark.parametrize(
        "content",
        [
            "[1, 2, 3]",  # valid JSON, but a list, not an object
            '{"1.2.3.4": "oops"}',  # value isn't a [tokens, last] pair at all
            '{"1.2.3.4": ["not-a-number", "also-not"]}',  # pair present, values aren't numeric
            '{"1.2.3.4": [1.0]}',  # pair present but wrong length
        ],
    )
    def test_malformed_but_valid_json_state_file_starts_fresh_instead_of_crashing(self, tmp_path, content):
        # Demonstrated exploitable before this guard: each of these crashed __init__ with an unhandled AttributeError/ValueError/TypeError. Because `_rate_limiter = RateLimiter(state_path=...)` runs at src/api/main.py's module import time, that crash takes down the whole app on every single startup — turning a supervisord crash-restart into a permanent crash loop instead of recovering it, the opposite of what persistence was built for.
        state_path = tmp_path / "state.json"
        state_path.write_text(content)

        limiter = RateLimiter(rpm=5, state_path=state_path)

        assert limiter._buckets == {}
