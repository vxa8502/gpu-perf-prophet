"""
Unit tests for src/recommend/recommender.py.
"""

from __future__ import annotations

import pytest

from src.models.predictor import GpuPredictor
from src.recommend.recommender import (
    GpuRecommender,
    _pareto_frontier,
    _RANKING_FIELDS,
    VALID_RANKING_OBJECTIVES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def predictor() -> GpuPredictor:
    return GpuPredictor()


@pytest.fixture(scope="module")
def recommender(predictor: GpuPredictor) -> GpuRecommender:
    return GpuRecommender(predictor)


# ---------------------------------------------------------------------------
# _pareto_frontier (pure function)
# ---------------------------------------------------------------------------

class TestParetoFrontier:
    def _c(self, throughput, price, watts, cost_efficiency=None) -> dict:
        """Build a candidate dict for the Pareto objective vector.

        Dominance uses (throughput ↑, price ↓, watts ↓) directly. cost_efficiency
        defaults to throughput/price (None if price is None) since it's still
        read by the default "tokens_per_dollar" ranking_objective.
        """
        if cost_efficiency is None and price:
            cost_efficiency = throughput / price
        return {
            "throughput": throughput,
            "price_per_gpu_hr": price,
            "watts": watts,
            "cost_efficiency": cost_efficiency,
            "tokens_per_watt": (throughput / watts) if watts else None,
            "cost_per_million_tokens": None,
            "gpu_id": "test",
        }

    def test_all_on_frontier_when_no_domination(self):
        # A: best throughput; B: best price; C: best watts — no one dominates
        candidates = [
            self._c(1000, 200, 500),   # A
            self._c(500,  50,  500),   # B
            self._c(600,  150, 100),   # C
        ]
        frontier, dominated = _pareto_frontier(candidates)
        assert len(frontier) == 3
        assert len(dominated) == 0

    def test_dominated_candidate_excluded_from_frontier(self):
        # D is strictly worse than A on all objectives → dominated
        candidates = [
            self._c(1000, 50,  100),   # A — best on all
            self._c(500,  200, 500),   # D — dominated by A
        ]
        frontier, dominated = _pareto_frontier(candidates)
        assert len(frontier) == 1
        assert frontier[0]["throughput"] == 1000
        assert len(dominated) == 1

    def test_frontier_sorted_by_default_ranking_objective(self):
        # Default ranking_objective is tokens_per_dollar (cost_efficiency desc).
        # All three mutually non-dominated on (throughput, price, watts).
        candidates = [
            self._c(800, 8,   200),  # cost_efficiency 100
            self._c(900, 3,   300),  # cost_efficiency 300
            self._c(700, 3.5, 100),  # cost_efficiency 200
        ]
        frontier, _ = _pareto_frontier(candidates)
        assert len(frontier) == 3
        efficiencies = [c["cost_efficiency"] for c in frontier]
        assert efficiencies == sorted(efficiencies, reverse=True)

    def test_frontier_sorted_by_explicit_ranking_objective(self):
        candidates = [
            self._c(800, 8,   200),
            self._c(900, 3,   300),
            self._c(700, 3.5, 100),
        ]
        frontier, _ = _pareto_frontier(candidates, ranking_objective="tokens_per_second")
        throughputs = [c["throughput"] for c in frontier]
        assert throughputs == sorted(throughputs, reverse=True)

    def test_invalid_ranking_objective_raises(self):
        with pytest.raises(KeyError):
            _pareto_frontier([self._c(500, 100, 200)], ranking_objective="bogus")

    def test_empty_input(self):
        frontier, dominated = _pareto_frontier([])
        assert frontier == []
        assert dominated == []

    def test_single_candidate_on_frontier(self):
        frontier, dominated = _pareto_frontier([self._c(500, 100, 200)])
        assert len(frontier) == 1
        assert len(dominated) == 0

    def test_none_ranking_field_sorts_last(self):
        # An unpriced GPU (price=None → cost_efficiency=None) must sort last
        # under the default ranking_objective — and must not crash dominance,
        # which reads price_per_gpu_hr directly, not cost_efficiency. Neither
        # candidate dominates the other: #1 has the real price, #2 has better
        # throughput/watts.
        candidates = [
            self._c(500, 50,   300),   # priced
            self._c(800, None, 100),   # unpriced; better throughput & watts
        ]
        frontier, dominated = _pareto_frontier(candidates)
        assert len(frontier) == 2
        assert len(dominated) == 0
        assert frontier[-1]["cost_efficiency"] is None  # None sorts last

    def test_none_price_dominated_by_priced(self):
        # An unpriced GPU (price=None, treated as worst-case for dominance) is
        # dominated by any GPU that also matches or beats it on throughput/watts —
        # the missing price alone already supplies the required strict inequality.
        candidates = [
            self._c(1000, 100, 100),   # beats unpriced on all objectives
            self._c(500,  None, 300),  # unpriced; worse on throughput & watts too
        ]
        frontier, dominated = _pareto_frontier(candidates)
        assert len(frontier) == 1
        assert frontier[0]["price_per_gpu_hr"] == 100
        assert len(dominated) == 1
        assert dominated[0]["price_per_gpu_hr"] is None

    def test_objective_vector_computed_once_per_candidate(self):
        # _dominates() used to take raw candidate dicts and recompute
        # _obj_vector() (3 field reads) on every pairwise comparison — O(n)
        # recomputations per candidate, O(n^2) total across the full sweep.
        # Measured 336 field accesses for 8 mutually non-dominated candidates
        # against a theoretical minimum of 24. Fixed by
        # precomputing each candidate's vector once before the O(n^2) loop.
        class _CountingDict(dict):
            access_count = 0

            def __getitem__(self, key):
                if key in ("throughput", "price_per_gpu_hr", "watts"):
                    _CountingDict.access_count += 1
                return super().__getitem__(key)

        # 8 mutually non-dominated candidates: throughput rises while price
        # and watts also rise, so no candidate dominates another.
        candidates = [
            _CountingDict(self._c(1000 + i * 37, 2.0 + i * 0.3, 100 + i * 90))
            for i in range(8)
        ]
        _CountingDict.access_count = 0
        frontier, dominated = _pareto_frontier(candidates)
        assert len(frontier) == 8, "fixture must produce zero domination to hit the worst case"
        assert len(dominated) == 0
        assert _CountingDict.access_count == 8 * 3, (
            f"expected exactly 24 field accesses (8 candidates x 3 objective fields, "
            f"each read once), got {_CountingDict.access_count} — _obj_vector is being "
            "recomputed per comparison again"
        )


# ---------------------------------------------------------------------------
# GpuRecommender.recommend
# ---------------------------------------------------------------------------

class TestGpuRecommender:
    def test_workload_echoed(self, recommender):
        result = recommender.recommend(
            model_name="gptj",
            scenario="Server",
            accuracy_tier="base",
            framework="tensorrt",
        )
        wl = result["workload"]
        assert wl["model_name"] == "gptj"
        assert wl["scenario"] == "Server"
        assert wl["accuracy_tier"] == "base"
        assert wl["framework"] == "tensorrt"

    def test_model_size_in_workload(self, recommender):
        result = recommender.recommend(model_name="llama2-70b", accuracy_tier="99")
        # 70B × 1 byte (FP8) = 70 GB
        assert result["workload"]["model_size_gb"] == pytest.approx(70.0)

    def test_vram_filter_splits_correctly(self, recommender):
        # llama2-70b at tier 99.9, default batch=32/in=2048/out=256: FP16
        # weights for NVIDIA (140 GB), FP8 for AMD (70 GB), plus a KV cache
        # (~24 GB FP16 / ~12 GB FP8) and 10% overhead. This is
        # the killer-demo result: once KV cache + overhead is realistically
        # accounted for, no in-scope NVIDIA GPU — not even H200 SXM's 141 GB —
        # can serve Llama-2-70B FP16 single-GPU, while all three AMD GPUs
        # (which run this tier at FP8) comfortably fit.
        # Fit: MI355X(288), MI325X(256), MI300X(192) — AMD, ~77 GB total need
        # Doesn't fit: H200 SXM(141), H100 SXM(80), A100 SXM 80GB(80),
        #              L4(24), RTX4090(24) — NVIDIA, ~181 GB total need
        result = recommender.recommend(model_name="llama2-70b", accuracy_tier="99.9")
        candidate_ids = {r["gpu_id"] for r in result["frontier"] + result["dominated"]}
        vram_filtered_ids = {
            f["gpu_id"] for f in result["filtered"]
            if f.get("memory_fit_verdict") == "does_not_fit"
        }
        for gid in ("mi355x", "mi325x", "mi300x"):
            assert gid in candidate_ids, f"{gid} should fit but is absent from candidates"
        for gid in ("h200_sxm", "h100_sxm", "a100_sxm_80gb", "l4", "rtx4090"):
            assert gid in vram_filtered_ids, f"{gid} should be memory-filtered but is missing"
        # The two sets must be disjoint — no GPU can be both candidate and filtered
        assert candidate_ids.isdisjoint(vram_filtered_ids)

    def test_405b_fp16_all_filtered(self, recommender):
        # 405B × 2 bytes = 810 GB weights alone; largest GPU is MI355X at
        # 288 GB — none fit even before KV cache/overhead are added.
        result = recommender.recommend(
            model_name="llama3.1-405b",
            accuracy_tier="99.9",
        )
        assert len(result["frontier"]) == 0
        assert len(result["dominated"]) == 0
        assert len(result["filtered"]) > 0
        for f in result["filtered"]:
            assert f["memory_fit_verdict"] == "does_not_fit"

    def test_tight_verdict_included_not_filtered(self, recommender):
        # Golden "tight" case (default batch=32/in=2048/out=256): llama2-70b
        # base tier is FP16 on AMD too, so MI300X needs weights=140 GB +
        # kv≈24 GB + 10% overhead ≈ 180.6 GB against its 192 GB — util≈0.94,
        # inside the tight band. tight is a disclosure flag, not a
        # hard exclusion — MI300X must still appear as a real candidate.
        result = recommender.recommend(model_name="llama2-70b", accuracy_tier="base")
        candidates = result["frontier"] + result["dominated"]
        mi300x = next((r for r in candidates if r["gpu_id"] == "mi300x"), None)
        assert mi300x is not None, "MI300X should be a candidate (tight, not excluded)"
        assert mi300x["memory_fit_verdict"] == "tight"
        assert mi300x["vram_fits"] is True
        assert not any(f["gpu_id"] == "mi300x" for f in result["filtered"])

    def test_candidate_memory_fit_matches_standalone_predict(self, predictor, recommender):
        # recommend() passes its own precomputed memory-fit result into
        # predict_batch() instead of letting it recompute (perf fix — see
        # predict_batch()'s docstring). This is the end-to-end guard that the
        # precomputed values recommender hands over are actually correct: a
        # candidate's memory-fit fields must match what a standalone
        # predictor.predict() call for the same (gpu, workload) computes from
        # scratch, not just be internally consistent with themselves.
        result = recommender.recommend(model_name="llama2-70b", accuracy_tier="base")
        candidates = result["frontier"] + result["dominated"]
        mi300x = next((r for r in candidates if r["gpu_id"] == "mi300x"), None)
        assert mi300x is not None

        standalone = predictor.predict(gpu_id="mi300x", model_name="llama2-70b", accuracy_tier="base")
        for key in ("memory_fit_verdict", "kv_cache_gb", "memory_total_gb", "vram_utilization"):
            assert mi300x[key] == standalone[key], (
                f"{key}: recommend()={mi300x[key]!r} standalone predict()={standalone[key]!r}"
            )

    def test_budget_filter_respected(self, recommender):
        # Budget $1.50/hr — should filter out expensive GPUs.
        # All in-scope GPUs have pricing (validated at startup), so
        # price_per_gpu_hr is never None — the is-None arm was dead code.
        #
        # batch_size=8 (not the default 32): at the default batch size, gptj's
        # real KV cache (MHA, 16 kv-heads x 256 head_dim) needs ~16.9 GB, which
        # pushes weights+KV+overhead past RTX4090/L4's 24 GB — so at batch=32
        # the only GPU left within budget and "fits" was a100_sxm_80gb, priced
        # exactly at the $1.50 boundary, and only because of an unrelated bug:
        # A100 has no native FP8 (accuracy_tier="99"), and before
        # precision-support enforcement was added, the system silently substituted FP16's peak
        # TFLOPS instead of raising, producing a normal-looking but physically
        # inconsistent prediction. batch_size=8 shrinks gptj's KV cache enough
        # that RTX4090/L4 genuinely fit, so this test's "keeps a cheap GPU"
        # case no longer accidentally depends on that now-fixed bug.
        result = recommender.recommend(
            model_name="gptj",
            accuracy_tier="99",
            budget_per_gpu_hr=1.50,
            batch_size=8,
        )
        candidates = result["frontier"] + result["dominated"]
        # RTX4090 ($0.39) and L4 ($0.44) are both priced under $1.50 and fit
        # gptj (6 GB) at this batch size. Without this guard the loop below
        # fires zero assertions if pricing changes so all GPUs exceed budget.
        assert len(candidates) > 0, (
            "Expected at least one GPU within $1.50/hr — RTX4090 ($0.39) and "
            "L4 ($0.44) are currently below this threshold"
        )
        for r in candidates:
            assert r["price_per_gpu_hr"] <= 1.50
        candidate_ids = {r["gpu_id"] for r in candidates}
        assert candidate_ids & {"rtx4090", "l4"}, (
            "Expected RTX4090 and/or L4 to be the candidates keeping this test "
            f"green, got {candidate_ids} instead"
        )
        assert "a100_sxm_80gb" not in candidate_ids, (
            "a100_sxm_80gb does not support fp8 (accuracy_tier='99') and must "
            "never appear as a real candidate — see the precision-support pre-filter"
        )

    def test_cheap_budget_top_pick_flagged_as_unmeasured(self, recommender):
        # a100_sxm_80gb, l4, and rtx4090 are in_model_scope but have zero rows
        # in mlperf_features.parquet — they're also the three cheapest GPUs in
        # pricing.yaml, so a tight budget query structurally tends to surface
        # them as the top (sometimes only) recommendation. Their predictions
        # are pure spec extrapolation, never validated against a real
        # measurement — has_training_data must say so.
        # A light batch/context override keeps the KV cache small enough that
        # RTX4090's 24 GB still fits gptj (the default batch=32/2048-token
        # assumption alone needs ~25 GB and would exclude it).
        result = recommender.recommend(
            model_name="gptj",
            accuracy_tier="99",
            batch_size=1,
            input_tokens=128,
            output_tokens=64,
            budget_per_gpu_hr=1.0,
        )
        assert len(result["frontier"]) > 0, (
            "Expected at least one candidate under $1.00/hr — RTX4090 ($0.39) "
            "is currently below this threshold"
        )
        top = result["frontier"][0]
        assert top["gpu_id"] == "rtx4090"
        assert top["has_training_data"] is False
        assert top["training_data_tier"] == "none"

        # Every candidate and filtered entry must carry both fields regardless
        # of which branch built it (predict_batch-derived vs the manually
        # built VRAM-fail reject entries) — a missing key here would fail
        # Pydantic validation silently at the API layer (extra field dropped,
        # not error), not surface as a crash.
        for r in result["frontier"] + result["dominated"] + result["filtered"]:
            assert "has_training_data" in r
            assert "training_data_tier" in r

    def test_mi300x_candidate_flagged_below_floor(self, recommender):
        # mi300x has real MLPerf + calibration rows (80) but sits under
        # this project's 100-row-per-GPU Must-have floor — has_training_data alone
        # reports this identically to a well-covered GPU like h100_sxm/h200_sxm
        # (178/283 rows).
        result = recommender.recommend(model_name="gptj", accuracy_tier="99")
        by_gpu = {
            r["gpu_id"]: r
            for r in result["frontier"] + result["dominated"] + result["filtered"]
        }
        assert "mi300x" in by_gpu, "mi300x should appear as a gptj candidate"
        assert by_gpu["mi300x"]["has_training_data"] is True
        assert by_gpu["mi300x"]["training_data_tier"] == "below_floor"

    def test_recommend_calls_training_data_tier_once_per_gpu(self, predictor, recommender, monkeypatch):
        # recommender.py's two manually-built reject-entry loops (precision-fail,
        # VRAM-fail) each originally called training_data_tier() twice per GPU —
        # once directly, once indirectly via has_training_data(). Found via
        # a performance review (wraps-mock instrumentation confirmed 16 calls
        # for 8 candidates on a VRAM-fail-only query; fixed to compute the tier
        # once per GPU and derive both fields from it). llama3.1-405b guarantees every candidate
        # takes the VRAM-fail reject path (nothing fits 405B).
        original = predictor.training_data_tier
        calls: list[str] = []

        def _counting(gpu_id):
            calls.append(gpu_id)
            return original(gpu_id)

        monkeypatch.setattr(predictor, "training_data_tier", _counting)
        result = recommender.recommend(model_name="llama3.1-405b", accuracy_tier="99.9")
        n_entries = len(result["frontier"]) + len(result["dominated"]) + len(result["filtered"])
        assert n_entries > 0, "expected candidates to check"
        assert len(calls) == n_entries, (
            f"training_data_tier() called {len(calls)}x for {n_entries} result "
            "entries, expected exactly 1 call per entry"
        )

    @staticmethod
    def _fr047_vector(c: dict) -> tuple[float, float, float]:
        """(throughput, price, watts) normalized to higher-is-better, matching
        the real dominance check in recommender._pareto_frontier."""
        price, watts = c["price_per_gpu_hr"], c["watts"]
        return (
            c["throughput"],
            -price if price is not None else float("-inf"),
            -watts if watts is not None else float("-inf"),
        )

    def test_no_dominated_option_strictly_worse_on_all(self, recommender):
        result = recommender.recommend(model_name="llama3.1-8b", accuracy_tier="base")
        frontier = result["frontier"]
        dominated = result["dominated"]

        # Guard: without this assertion the loop below never fires when dominated
        # is empty, giving zero assertions and false confidence.
        # llama3.1-8b base tier fits all 8 in-scope GPUs; under the
        # objective vector (throughput, price ↓, watts ↓) h200_sxm is beaten on
        # all three by h100_sxm (same watts, lower price, comparable throughput).
        assert len(dominated) >= 1, (
            "Expected at least one dominated candidate for llama3.1-8b base tier — "
            "if every in-scope GPU is now Pareto-optimal under (throughput, price, "
            "watts), verify this is intentional (pricing/spec change) before updating."
        )

        for dom in dominated:
            dv = self._fr047_vector(dom)
            # At least one frontier member must dominate this candidate
            is_dominated_by_frontier = any(
                all(fo >= do for fo, do in zip(self._fr047_vector(f), dv))
                and any(fo > do for fo, do in zip(self._fr047_vector(f), dv))
                for f in frontier
            )
            assert is_dominated_by_frontier, (
                f"{dom['gpu_id']} is in dominated list but is not dominated by any frontier member"
            )

    def test_min_throughput_filter(self, recommender):
        # High min_throughput should remove low-performing GPUs
        result = recommender.recommend(
            model_name="llama2-70b",
            accuracy_tier="99",
            min_throughput_tok_per_sec=10_000_000,  # absurdly high
        )
        # Nothing should pass
        assert len(result["frontier"]) == 0
        assert len(result["dominated"]) == 0

    def test_amd_99_9_vram_headroom_uses_fp8_model_size(self, recommender):
        # AMD at 99.9 tier uses FP8 (70 GB), not FP16 (140 GB).
        # vram_headroom for MI300X (192 GB) must be ~60% (70 GB model + 10%
        # overhead, negligible KV cache at this tiny batch/context override),
        # not ~27.1% (140 GB FP16 model with no overhead).  This would catch
        # the recommender computing vram_headroom from the workload-level
        # FP16 model_size_gb instead of the per-GPU FP8 total.  Uses a
        # minimal batch/context override so the KV-cache term doesn't
        # obscure the FP8-vs-FP16 comparison this test targets.
        result = recommender.recommend(
            model_name="llama2-70b", accuracy_tier="99.9",
            batch_size=1, input_tokens=64, output_tokens=1,
        )
        candidates = result["frontier"] + result["dominated"]
        mi300x = next((r for r in candidates if r["gpu_id"] == "mi300x"), None)
        assert mi300x is not None, "MI300X should be a candidate for llama2-70b tier 99.9"
        expected_headroom = (192 - 70 * 1.10) / 192   # weights + 10% overhead, ~zero KV cache
        got_headroom = mi300x["vram_headroom"]
        assert got_headroom == pytest.approx(expected_headroom, abs=0.01), (
            f"MI300X vram_headroom={got_headroom:.3f} — expected ~{expected_headroom:.3f}"
            " (FP8 70 GB model); got FP16 headroom instead?"
        )

    def test_frontier_is_pareto_optimal_gptj(self, recommender):
        # gptj (6.7B, ~6.7 GB FP8) fits all 8 in-scope GPUs — full-field Pareto test
        # with no VRAM pre-filter. Under the objective vector (throughput ↑,
        # price ↓, watts ↓), h100_sxm beats h200_sxm on all three (same watts, lower
        # price, higher throughput), so h200_sxm is dominated; the other 7 GPUs trade
        # off throughput against price/watts and are mutually Pareto-optimal. The test
        # verifies: (1) frontier is non-empty, (2) every dominated GPU is correctly
        # classified, (3) no frontier member dominates another.
        result = recommender.recommend(model_name="gptj", accuracy_tier="99")
        frontier = result["frontier"]
        dominated = result["dominated"]

        assert len(frontier) >= 1, "Expected ≥ 1 Pareto-optimal GPU for gptj"
        assert len(dominated) >= 1, (
            "Expected ≥ 1 dominated candidate for gptj — "
            "8 in-scope GPUs with diverse specs; not all can be Pareto-optimal"
        )

        # Every GPU in dominated must be dominated by at least one frontier member.
        for dom in dominated:
            dv = self._fr047_vector(dom)
            is_dominated_by_frontier = any(
                all(fo >= do for fo, do in zip(self._fr047_vector(f), dv))
                and any(fo > do for fo, do in zip(self._fr047_vector(f), dv))
                for f in frontier
            )
            assert is_dominated_by_frontier, (
                f"{dom['gpu_id']} is in dominated list but not dominated by any frontier member"
            )

        # No frontier member may dominate another.
        for i, a in enumerate(frontier):
            av = self._fr047_vector(a)
            for j, b in enumerate(frontier):
                if i == j:
                    continue
                bv = self._fr047_vector(b)
                dominates = (
                    all(bo >= ao for bo, ao in zip(bv, av))
                    and any(bo > ao for bo, ao in zip(bv, av))
                )
                assert not dominates, (
                    f"Frontier member {b['gpu_id']} dominates {a['gpu_id']} — "
                    "gptj frontier is not Pareto-optimal"
                )


# ---------------------------------------------------------------------------
# recommend() precision-support pre-filter
# ---------------------------------------------------------------------------

class TestRecommendPrecisionSupport:
    """a100_sxm_80gb (Ampere) has no native FP8 — data/gpu_specs.yaml has
    peak_tflops.fp8: ~ (null). accuracy_tier="99" selects fp8. Before this
    filter existed, recommend() sent a100 straight to predict_batch(), which
    silently substituted fp16's peak TFLOPS and returned a physically
    inconsistent but normal-looking candidate (fp8 memory footprint, fp16
    compute ceiling) — confirmed directly: it surfaced as a real "dominated"
    entry with a fabricated ~831 tok/s prediction for gptj.
    """

    def test_unsupported_precision_gpu_is_filtered_not_crashed(self, recommender):
        # gptj is small enough that a100 would otherwise pass the VRAM check
        # and reach predict_batch() — this isolates the precision filter from
        # the memory-fit filter (unlike the llama2-70b case, which happens to
        # also fail on VRAM and would mask this).
        result = recommender.recommend(model_name="gptj", accuracy_tier="99")
        candidate_ids = {r["gpu_id"] for r in result["frontier"] + result["dominated"]}
        assert "a100_sxm_80gb" not in candidate_ids

        entry = next(f for f in result["filtered"] if f["gpu_id"] == "a100_sxm_80gb")
        assert entry["memory_fit_verdict"] == "does_not_fit"
        assert entry["pred_throughput_tok_per_sec"] == 0.0
        assert "fp8" in entry["reject_reason"]
        assert "not supported" in entry["reject_reason"]

    def test_unsupported_precision_gpu_never_reaches_predict_batch(self, recommender, monkeypatch):
        # Assert the mechanism, not just the outcome: predict_batch() must
        # never even be called with a100 in the request list, since it would
        # raise and crash the whole recommend() call for every other
        # candidate GPU too.
        original = recommender._predictor.predict_batch

        def _spy(requests):
            gpu_ids = [r["gpu_id"] for r in requests]
            assert "a100_sxm_80gb" not in gpu_ids, (
                "a100_sxm_80gb must be filtered out before predict_batch() is called"
            )
            return original(requests)

        monkeypatch.setattr(recommender._predictor, "predict_batch", _spy)
        recommender.recommend(model_name="gptj", accuracy_tier="99")

    def test_supported_precision_gpus_unaffected(self, recommender):
        # The filter must not over-exclude: GPUs that DO support fp8 (every
        # other in-scope GPU) must still reach candidates/filtered-for-other-
        # reasons as before, not get swept up by this new filter.
        result = recommender.recommend(model_name="gptj", accuracy_tier="99")
        all_ids = {
            r["gpu_id"]
            for r in result["frontier"] + result["dominated"] + result["filtered"]
        }
        assert all_ids == set(recommender._in_scope_ids)
        precision_reasons = {
            f["gpu_id"]: f["reject_reason"]
            for f in result["filtered"]
            if "not supported" in f["reject_reason"]
        }
        assert set(precision_reasons) == {"a100_sxm_80gb"}


# ---------------------------------------------------------------------------
# recommend() serving-shape validation
# ---------------------------------------------------------------------------

class TestRecommendServingShapeValidation:
    """recommend()'s memory-fit pre-filter uses batch_size/input_tokens/
    output_tokens directly, before any predict_batch() call (which is where
    GpuPredictor does its own range validation) happens. Without an explicit
    check in recommend() itself, an out-of-range value could silently reach
    every in-scope GPU as "does_not_fit" and return a normal-looking response
    instead of raising — exactly what a garbage batch_size should NOT do
    silently, and exactly what predict() has always done correctly.
    """

    @pytest.mark.parametrize("kwargs", [
        {"batch_size": 0},
        {"batch_size": 257},
        {"batch_size": 999_999_999},   # the value that previously slipped
                                        # through: every GPU "does_not_fit",
                                        # so predict_batch() was never called
                                        # and its validation never ran.
        {"batch_size": -1000},
        {"input_tokens": 63},
        {"input_tokens": 100_000},
        {"output_tokens": 0},
        {"output_tokens": 4097},
    ])
    def test_out_of_range_serving_shape_raises(self, recommender, kwargs):
        # match= pins this to the parametrized field actually being rejected,
        # not just "some ValueError happened" — otherwise a future reordering
        # of recommend()'s validation could make this pass for the wrong reason.
        param_name = next(iter(kwargs))
        with pytest.raises(ValueError, match=f"Invalid {param_name}"):
            recommender.recommend(model_name="gptj", accuracy_tier="99", **kwargs)

    def test_raises_before_any_gpu_is_touched(self, recommender):
        # A batch_size this large makes every in-scope GPU "does_not_fit" —
        # the exact condition that let bad input slip past predict_batch()'s
        # validation before this test's namesake fix. Assert it still raises
        # rather than returning a "successfully filtered everything" result.
        with pytest.raises(ValueError, match="Invalid batch_size"):
            recommender.recommend(model_name="gptj", batch_size=999_999_999)


# ---------------------------------------------------------------------------
# recommend() accuracy_tier/scenario/framework validation
# ---------------------------------------------------------------------------

class TestRecommendCategoricalValidation:
    """GpuRecommender.recommend() is a public method (its own module docstring
    documents it as the library entry point Streamlit calls in-process) and
    must be safe to call directly with untrusted input — the same contract
    GpuPredictor.predict() already upholds via _validate(). Before this check
    existed, an invalid accuracy_tier reached the bare
    `TIER_TO_PRECISION[accuracy_tier]` subscript and raised an uncaught
    KeyError (not the ValueError every other invalid-input path in this
    codebase raises, so main.py's `except ValueError` handler would not have
    caught it), and an invalid scenario/framework would have silently passed
    through into the response — never validated — whenever every candidate
    GPU was excluded before reaching predict_batch()'s own checks. The
    FastAPI layer's Pydantic Literal types and the Streamlit UI's fixed
    selectboxes both happen to constrain these today, so this was not
    reachable through the shipped app, but recommend() itself had no
    independent guarantee of that.
    """

    def test_invalid_accuracy_tier_raises_value_error_not_key_error(self, recommender):
        with pytest.raises(ValueError, match="Invalid accuracy_tier"):
            recommender.recommend(model_name="gptj", accuracy_tier="fp99000; DROP TABLE")

    def test_invalid_scenario_raises(self, recommender):
        # llama3.1-405b (810 GB at FP16) fails memory-fit on every in-scope
        # GPU (see test_405b_fp16_all_filtered) — no GPU ever reaches
        # predict_batch(), so this isolates recommend()'s own scenario check
        # from predict_batch()'s. Using gptj here (as this test originally
        # did) passes for the wrong reason: gptj fits most GPUs, which reach
        # predict_batch() and its own _validate() catches the bad scenario
        # anyway — confirmed by mutation-testing this test against gptj with
        # recommend()'s own check removed: it still passed, proving nothing
        # about the code this test is named for.
        with pytest.raises(ValueError, match="Invalid scenario"):
            recommender.recommend(
                model_name="llama3.1-405b", accuracy_tier="99.9", scenario="Interactive",
            )

    def test_invalid_framework_raises(self, recommender):
        # Same reasoning as test_invalid_scenario_raises above.
        with pytest.raises(ValueError, match="Invalid framework"):
            recommender.recommend(
                model_name="llama3.1-405b", accuracy_tier="99.9", framework="pytorch",
            )


# ---------------------------------------------------------------------------
# ranking_objective — user-selectable Pareto-set ranking scalar
# ---------------------------------------------------------------------------

class TestRankingObjective:
    """recommend()'s Pareto-optimal (rank-1) set is sorted by a
    user-selectable scalar — tokens_per_dollar (default), tokens_per_second,
    tokens_per_watt, or lowest_cost_per_million_tokens.
    """

    def test_valid_ranking_objectives_contains_all_four_fr048_scalars(self):
        assert VALID_RANKING_OBJECTIVES == {
            "tokens_per_dollar", "tokens_per_second",
            "tokens_per_watt", "lowest_cost_per_million_tokens",
        }

    def test_invalid_ranking_objective_raises(self, recommender):
        with pytest.raises(ValueError, match="Invalid ranking_objective"):
            recommender.recommend(model_name="gptj", ranking_objective="bogus")

    def test_default_matches_explicit_tokens_per_dollar(self, recommender):
        default = recommender.recommend(model_name="gptj", accuracy_tier="99")
        explicit = recommender.recommend(
            model_name="gptj", accuracy_tier="99", ranking_objective="tokens_per_dollar",
        )
        assert [c["gpu_id"] for c in default["frontier"]] == [
            c["gpu_id"] for c in explicit["frontier"]
        ]

    def test_workload_echoes_ranking_objective(self, recommender):
        result = recommender.recommend(model_name="gptj", ranking_objective="tokens_per_watt")
        assert result["workload"]["ranking_objective"] == "tokens_per_watt"

    @pytest.mark.parametrize("objective,field,higher_is_better", [
        ("tokens_per_dollar", "cost_efficiency", True),
        ("tokens_per_second", "throughput", True),
        ("tokens_per_watt", "tokens_per_watt", True),
        ("lowest_cost_per_million_tokens", "cost_per_million_tokens", False),
    ])
    def test_frontier_sorted_by_each_objective(
        self, recommender, objective, field, higher_is_better,
    ):
        # Direct wiring check first: catches _RANKING_FIELDS mapping the wrong
        # field to `objective` immediately, regardless of what any dataset's
        # values happen to look like.
        assert _RANKING_FIELDS[objective] == (field, higher_is_better)

        # gptj/tier-99's frontier (mi355x > mi325x > mi300x > h100_sxm) is a
        # false-confidence trap for this test: cost_efficiency, throughput,
        # and tokens_per_watt all happen to agree on that exact order, so a
        # mutation mapping "tokens_per_watt" to the wrong field (e.g.
        # "cost_efficiency") still produces a monotonic tokens_per_watt column
        # by coincidence — confirmed by mutation-testing this test directly,
        # it passed unchanged. llama2-70b/tier-99's frontier genuinely
        # diverges across all three ratio fields (confirmed empirically:
        # ce=[mi355x,mi325x,mi300x,h200_sxm], tput=[mi355x,mi325x,h200_sxm,mi300x],
        # tpw=[mi355x,h200_sxm,mi325x,mi300x]), so sorting by the wrong field
        # produces a visibly non-monotonic column here instead of an
        # accidentally-correct one.
        result = recommender.recommend(
            model_name="llama2-70b", accuracy_tier="99", ranking_objective=objective,
        )
        values = [c[field] for c in result["frontier"]]
        assert len(values) >= 3, "llama2-70b/99 should have >=3 frontier GPUs to make sorting meaningful"
        assert values == sorted(values, reverse=higher_is_better)

    def test_watts_tokens_per_watt_cost_per_million_hand_verified(self, predictor, recommender):
        # mi300x, gptj, tier 99 — hand-verify the three new fields against
        # gpu_specs.yaml's tdp_w=750, a standalone predictor.predict() call
        # (not mi300x["throughput"] itself), and recommender._pricing (not
        # mi300x["price_per_gpu_hr"] itself), so a bug that corrupts
        # "throughput"/"price_per_gpu_hr" and "tokens_per_watt"/
        # "cost_per_million_tokens" together (e.g. both wired to the wrong
        # source) can't also corrupt the expected value the same way and
        # still pass. Found by mutation-testing this test directly:
        # the original version derived "expected" from mi300x["throughput"]/
        # mi300x["price_per_gpu_hr"] — reusing pred["roofline_tput_tok_per_sec"]
        # for throughput, tokens_per_watt, AND cost_per_million_tokens all
        # three at once still passed, since all three moved together.
        result = recommender.recommend(model_name="gptj", accuracy_tier="99")
        candidates = result["frontier"] + result["dominated"]
        mi300x = next(c for c in candidates if c["gpu_id"] == "mi300x")
        assert mi300x["watts"] == 750

        standalone = predictor.predict(gpu_id="mi300x", model_name="gptj", accuracy_tier="99")
        real_tput = standalone["pred_throughput_tok_per_sec"]
        real_price = recommender._pricing["mi300x"]
        assert mi300x["throughput"] == pytest.approx(real_tput)
        assert mi300x["price_per_gpu_hr"] == pytest.approx(real_price)

        assert mi300x["tokens_per_watt"] == pytest.approx(real_tput / 750)
        expected_cpm = (real_price / 3600) / (real_tput / 1_000_000)
        assert mi300x["cost_per_million_tokens"] == pytest.approx(expected_cpm)

    def test_filtered_entry_carries_watts_but_not_derived_ratios(self, recommender):
        # llama3.1-405b fp16/fp8 (810 GB) fails memory-fit on every in-scope
        # GPU (see test_405b_fp16_all_filtered) — filtered entries still carry
        # the GPU's static watts (a spec fact, independent of whether a
        # prediction ran) but not tokens_per_watt/cost_per_million_tokens,
        # which need a real throughput prediction to be meaningful.
        result = recommender.recommend(model_name="llama3.1-405b", accuracy_tier="99.9")
        assert result["frontier"] == []
        assert len(result["filtered"]) > 0
        for f in result["filtered"]:
            assert f["watts"] is not None
            assert f["tokens_per_watt"] is None
            assert f["cost_per_million_tokens"] is None
