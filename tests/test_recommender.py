"""
Unit tests for src/recommend/recommender.py.
"""

from __future__ import annotations

import pytest

from src.models.predictor import GpuPredictor
from src.recommend.recommender import GpuRecommender, _pareto_frontier


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def recommender() -> GpuRecommender:
    pred = GpuPredictor()
    return GpuRecommender(pred)


# ---------------------------------------------------------------------------
# _pareto_frontier (pure function)
# ---------------------------------------------------------------------------

class TestParetoFrontier:
    def _c(self, throughput, cost_efficiency, vram_headroom) -> dict:
        return {
            "throughput": throughput,
            "cost_efficiency": cost_efficiency,
            "vram_headroom": vram_headroom,
            "gpu_id": "test",
        }

    def test_all_on_frontier_when_no_domination(self):
        # A: best throughput; B: best cost; C: best vram — no one dominates
        candidates = [
            self._c(1000, 100, 0.3),   # A
            self._c(500,  200, 0.2),   # B
            self._c(600,  150, 0.8),   # C
        ]
        frontier, dominated = _pareto_frontier(candidates)
        assert len(frontier) == 3
        assert len(dominated) == 0

    def test_dominated_candidate_excluded_from_frontier(self):
        # D is strictly worse than A on all objectives → dominated
        candidates = [
            self._c(1000, 200, 0.8),   # A — best on all
            self._c(500,  100, 0.3),   # D — dominated by A
        ]
        frontier, dominated = _pareto_frontier(candidates)
        assert len(frontier) == 1
        assert frontier[0]["throughput"] == 1000
        assert len(dominated) == 1

    def test_frontier_sorted_by_cost_efficiency_desc(self):
        candidates = [
            self._c(800, 100, 0.5),
            self._c(900, 300, 0.4),
            self._c(700, 200, 0.6),
        ]
        frontier, _ = _pareto_frontier(candidates)
        efficiencies = [c["cost_efficiency"] for c in frontier]
        assert efficiencies == sorted(efficiencies, reverse=True)

    def test_empty_input(self):
        frontier, dominated = _pareto_frontier([])
        assert frontier == []
        assert dominated == []

    def test_single_candidate_on_frontier(self):
        frontier, dominated = _pareto_frontier([self._c(500, 100, 0.5)])
        assert len(frontier) == 1
        assert len(dominated) == 0

    def test_none_cost_efficiency_sorts_last(self):
        # A GPU with cost_efficiency=None (unpriced) must sort last on the
        # frontier — and must not crash the Pareto comparison with TypeError.
        # Choose values where neither candidate dominates the other:
        # first has higher cost_efficiency, second has higher throughput & vram.
        candidates = [
            self._c(500, 200, 0.4),   # better cost_efficiency
            self._c(800, None, 0.8),  # better throughput & vram; unpriced
        ]
        frontier, dominated = _pareto_frontier(candidates)
        assert len(frontier) == 2
        assert len(dominated) == 0
        assert frontier[-1]["cost_efficiency"] is None  # None sorts last

    def test_none_cost_efficiency_dominated_by_priced(self):
        # An unpriced GPU (cost_efficiency=None, treated as -inf) is dominated
        # by any GPU that beats it on the remaining two objectives.
        candidates = [
            self._c(1000, 200, 0.8),  # beats unpriced on all objectives
            self._c(500,  None, 0.3),  # unpriced; worse on throughput & vram too
        ]
        frontier, dominated = _pareto_frontier(candidates)
        assert len(frontier) == 1
        assert frontier[0]["cost_efficiency"] == 200
        assert len(dominated) == 1
        assert dominated[0]["cost_efficiency"] is None


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
        # llama2-70b at tier 99.9: FP16 for NVIDIA (140 GB), FP8 for AMD (70 GB).
        # The VRAM filter is vendor-aware, so AMD GPUs use 70 GB for the check.
        # In-scope GPUs that fit:
        #   MI355X(288), MI325X(256), MI300X(192) — AMD, 70 GB check
        #   H200 SXM(141)                          — NVIDIA, 140 GB check (barely fits)
        # In-scope GPUs that don't fit:
        #   H100 SXM(80), A100 SXM 80GB(80), L4(24), RTX4090(24) — NVIDIA, 140 GB
        result = recommender.recommend(model_name="llama2-70b", accuracy_tier="99.9")
        candidate_ids = {r["gpu_id"] for r in result["frontier"] + result["dominated"]}
        vram_filtered_ids = {
            f["gpu_id"] for f in result["filtered"]
            if "too large" in f.get("reject_reason", "")
        }
        # GPUs that must appear in candidates (they fit the 140 GB model)
        for gid in ("mi355x", "mi325x", "mi300x", "h200_sxm"):
            assert gid in candidate_ids, f"{gid} should fit 140 GB but is absent from candidates"
        # GPUs that must be VRAM-filtered (they don't fit 140 GB FP16 model)
        for gid in ("h100_sxm", "a100_sxm_80gb", "l4", "rtx4090"):
            assert gid in vram_filtered_ids, f"{gid} should be VRAM-filtered but is missing"
        # The two sets must be disjoint — no GPU can be both candidate and filtered
        assert candidate_ids.isdisjoint(vram_filtered_ids)

    def test_405b_fp16_all_filtered(self, recommender):
        # 405B × 2 bytes = 810 GB; largest GPU is MI355X at 288 GB — none fit
        result = recommender.recommend(
            model_name="llama3.1-405b",
            accuracy_tier="99.9",
        )
        assert len(result["frontier"]) == 0
        assert len(result["dominated"]) == 0
        assert len(result["filtered"]) > 0
        for f in result["filtered"]:
            assert "too large" in f["reject_reason"]

    def test_budget_filter_respected(self, recommender):
        # Budget $1.50/hr — should filter out expensive GPUs.
        # All in-scope GPUs have pricing (validated at startup), so
        # price_per_gpu_hr is never None — the is-None arm was dead code.
        result = recommender.recommend(
            model_name="gptj",
            accuracy_tier="99",
            budget_per_gpu_hr=1.50,
        )
        candidates = result["frontier"] + result["dominated"]
        # RTX4090 ($0.39) and L4 ($0.44) are both priced under $1.50 and fit
        # gptj (6 GB). Without this guard the loop below fires zero assertions
        # if pricing changes so all GPUs exceed the budget.
        assert len(candidates) > 0, (
            "Expected at least one GPU within $1.50/hr — RTX4090 ($0.39) and "
            "L4 ($0.44) are currently below this threshold"
        )
        for r in candidates:
            assert r["price_per_gpu_hr"] <= 1.50

    def test_no_dominated_option_strictly_worse_on_all(self, recommender):
        result = recommender.recommend(model_name="llama2-70b", accuracy_tier="99")
        frontier = result["frontier"]
        dominated = result["dominated"]
        objectives = ["throughput", "cost_efficiency", "vram_headroom"]

        # Guard: without this assertion the loop below never fires when dominated
        # is empty, giving zero assertions and false confidence.
        # llama2-70b FP8 (70 GB) fits 6 in-scope GPUs; with 3 objectives and
        # diverse pricing ($0.39–$3.99/hr) at least one must be dominated.
        assert len(dominated) >= 1, (
            "Expected at least one dominated candidate for llama2-70b fp8 — "
            "if all 6 VRAM-fitting GPUs are now Pareto-optimal, verify this "
            "is intentional (pricing or model change) before updating."
        )

        for dom in dominated:
            # At least one frontier member must dominate this candidate
            is_dominated_by_frontier = any(
                all(f[obj] >= dom[obj] for obj in objectives)
                and any(f[obj] > dom[obj] for obj in objectives)
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
        # vram_headroom for MI300X (192 GB) must be ~63.5% (70 GB model),
        # not ~27.1% (140 GB model).  This would catch the recommender
        # computing vram_headroom from the workload-level FP16 model_size_gb
        # instead of pred["model_size_gb"] which applies the AMD override.
        result = recommender.recommend(model_name="llama2-70b", accuracy_tier="99.9")
        candidates = result["frontier"] + result["dominated"]
        mi300x = next((r for r in candidates if r["gpu_id"] == "mi300x"), None)
        assert mi300x is not None, "MI300X should be a candidate for llama2-70b tier 99.9"
        expected_headroom = (192 - 70) / 192
        got_headroom = mi300x["vram_headroom"]
        assert got_headroom == pytest.approx(expected_headroom, abs=0.01), (
            f"MI300X vram_headroom={got_headroom:.3f} — expected ~{expected_headroom:.3f}"
            " (FP8 70 GB model); got FP16 headroom instead?"
        )

    def test_frontier_is_pareto_optimal_gptj(self, recommender):
        # gptj (6.7B, ~6.7 GB FP8) fits all 8 in-scope GPUs — full-field Pareto test
        # with no VRAM pre-filter. After the MI355X CDNA4 spec update (FP8 5033 TFLOPS,
        # confirmed from AMD official spec sheet 2026-06-17), MI355X dominates all other
        # in-scope GPUs on throughput, cost_efficiency, and vram_headroom at current
        # pricing ($3.50/hr). The test verifies: (1) frontier is non-empty, (2) every
        # dominated GPU is correctly classified, (3) no frontier member dominates another.
        result = recommender.recommend(model_name="gptj", accuracy_tier="99")
        frontier = result["frontier"]
        dominated = result["dominated"]
        objectives = ["throughput", "cost_efficiency", "vram_headroom"]

        assert len(frontier) >= 1, "Expected ≥ 1 Pareto-optimal GPU for gptj"
        assert len(dominated) >= 1, (
            "Expected ≥ 1 dominated candidate for gptj — "
            "8 in-scope GPUs with diverse specs; not all can be Pareto-optimal"
        )

        # Every GPU in dominated must be dominated by at least one frontier member.
        for dom in dominated:
            is_dominated_by_frontier = any(
                all(f[obj] >= dom[obj] for obj in objectives)
                and any(f[obj] > dom[obj] for obj in objectives)
                for f in frontier
            )
            assert is_dominated_by_frontier, (
                f"{dom['gpu_id']} is in dominated list but not dominated by any frontier member"
            )

        # No frontier member may dominate another.
        for i, a in enumerate(frontier):
            for j, b in enumerate(frontier):
                if i == j:
                    continue
                dominates = (
                    all(b[obj] >= a[obj] for obj in objectives)
                    and any(b[obj] > a[obj] for obj in objectives)
                )
                assert not dominates, (
                    f"Frontier member {b['gpu_id']} dominates {a['gpu_id']} — "
                    "gptj frontier is not Pareto-optimal"
                )
