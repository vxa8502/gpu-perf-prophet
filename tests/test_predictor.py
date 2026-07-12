"""
Unit and integration tests for src/models/predictor.py.

All tests load the real trained model (data/models/prophet_v1.json) so they
verify the full inference path, not just the feature-construction logic.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from src.models.predictor import (
    FEATURE_COLS,
    GpuPredictor,
    _build_feature_vector,
    _SERVING_ROUND,
)
from src.data.gpu_spec_db import load_specs


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def predictor() -> GpuPredictor:
    return GpuPredictor()


@pytest.fixture(scope="module")
def spec_map() -> dict:
    return {s["id"]: s for s in load_specs()}


# ---------------------------------------------------------------------------
# _build_feature_vector — pure function tests (no model needed)
# ---------------------------------------------------------------------------

class TestBuildFeatureVector:
    def test_amd_nvidia_arch_ordinals_mutually_exclusive(self, spec_map):
        feat_amd, _, _ = _build_feature_vector(
            gpu_spec=spec_map["mi300x"],
            model_name="gptj",
            scenario="Server",
            accuracy_tier="base",
            framework="rocm_other",
        )
        feat_nv, _, _ = _build_feature_vector(
            gpu_spec=spec_map["h100_sxm"],
            model_name="gptj",
            scenario="Server",
            accuracy_tier="base",
            framework="tensorrt",
        )
        nvidia_idx = FEATURE_COLS.index("nvidia_arch_gen")
        amd_idx = FEATURE_COLS.index("amd_arch_gen")

        # AMD GPU: nvidia_arch_gen NaN, amd_arch_gen not NaN
        assert math.isnan(feat_amd[nvidia_idx])
        assert not math.isnan(feat_amd[amd_idx])

        # NVIDIA GPU: amd_arch_gen NaN, nvidia_arch_gen not NaN
        assert math.isnan(feat_nv[amd_idx])
        assert not math.isnan(feat_nv[nvidia_idx])

    def test_scenario_offline_flag(self, spec_map):
        feat_off, _, _ = _build_feature_vector(
            gpu_spec=spec_map["h200_sxm"],
            model_name="llama2-70b",
            scenario="Offline",
            accuracy_tier="99",
            framework="vllm",
        )
        feat_srv, _, _ = _build_feature_vector(
            gpu_spec=spec_map["h200_sxm"],
            model_name="llama2-70b",
            scenario="Server",
            accuracy_tier="99",
            framework="vllm",
        )
        off_idx = FEATURE_COLS.index("scenario_offline")
        assert feat_off[off_idx] == 1
        assert feat_srv[off_idx] == 0

    def test_is_cdna4_flag(self, spec_map):
        feat_cdna4, _, _ = _build_feature_vector(
            gpu_spec=spec_map["mi355x"],
            model_name="llama2-70b",
            scenario="Offline",
            accuracy_tier="99.9",
            framework="rocm_other",
        )
        feat_cdna3, _, _ = _build_feature_vector(
            gpu_spec=spec_map["mi300x"],
            model_name="llama2-70b",
            scenario="Offline",
            accuracy_tier="99.9",
            framework="rocm_other",
        )
        cdna4_idx = FEATURE_COLS.index("is_cdna4")
        assert feat_cdna4[cdna4_idx] == 1
        assert feat_cdna3[cdna4_idx] == 0

    def test_bytes_per_param_precision_selection(self, spec_map):
        bpp_idx = FEATURE_COLS.index("bytes_per_param")
        # AMD tier 99   → fp8  → 1.0 byte/param
        feat_amd_99, _, _ = _build_feature_vector(
            gpu_spec=spec_map["mi300x"],
            model_name="llama2-70b",
            scenario="Offline",
            accuracy_tier="99",
            framework="vllm",
        )
        assert feat_amd_99[bpp_idx] == pytest.approx(1.0)
        # AMD tier 99.9 → fp8 override → 1.0 byte/param (not 2.0)
        feat_amd_99_9, _, _ = _build_feature_vector(
            gpu_spec=spec_map["mi300x"],
            model_name="llama2-70b",
            scenario="Offline",
            accuracy_tier="99.9",
            framework="vllm",
        )
        assert feat_amd_99_9[bpp_idx] == pytest.approx(1.0)
        # NVIDIA tier 99.9 → fp16 → 2.0 bytes/param (no override)
        feat_nv_99_9, _, _ = _build_feature_vector(
            gpu_spec=spec_map["h100_sxm"],
            model_name="llama2-70b",
            scenario="Offline",
            accuracy_tier="99.9",
            framework="tensorrt",
        )
        assert feat_nv_99_9[bpp_idx] == pytest.approx(2.0)

    def test_fw_other_all_dummies_zero(self, spec_map):
        # framework="other" must set all three fw_* flags to 0.
        # Previously untested — a bug that accidentally set one flag to 1 would
        # be invisible to the model (wrong feature for unknown frameworks).
        features, _, _ = _build_feature_vector(
            gpu_spec=spec_map["mi300x"],
            model_name="gptj",
            scenario="Offline",
            accuracy_tier="99",
            framework="other",
        )
        for fw_col in ("fw_tensorrt", "fw_vllm", "fw_rocm_other"):
            assert features[FEATURE_COLS.index(fw_col)] == 0, (
                f"{fw_col} should be 0 for framework='other'"
            )

    def test_is_base_tier_flag(self, spec_map):
        base_idx = FEATURE_COLS.index("is_base_tier")
        feat_base, _, _ = _build_feature_vector(
            gpu_spec=spec_map["mi300x"],
            model_name="gptj",
            scenario="Offline",
            accuracy_tier="base",
            framework="rocm_other",
        )
        feat_99, _, _ = _build_feature_vector(
            gpu_spec=spec_map["mi300x"],
            model_name="gptj",
            scenario="Offline",
            accuracy_tier="99",
            framework="rocm_other",
        )
        feat_99_9, _, _ = _build_feature_vector(
            gpu_spec=spec_map["h100_sxm"],
            model_name="gptj",
            scenario="Offline",
            accuracy_tier="99.9",
            framework="tensorrt",
        )
        assert feat_base[base_idx] == 1
        assert feat_99[base_idx] == 0
        assert feat_99_9[base_idx] == 0

    def test_mlperf_round_num_serving_uses_latest(self, spec_map):
        from src.features.build_features import ROUND_ORDINAL
        # _SERVING_ROUND must be the maximum defined ordinal — any future round
        # added to ROUND_ORDINAL automatically raises this floor.
        assert _SERVING_ROUND == float(max(ROUND_ORDINAL.values()))
        # The feature vector must include it at the correct position.
        features, _, _ = _build_feature_vector(
            gpu_spec=spec_map["mi300x"],
            model_name="llama2-70b",
            scenario="Offline",
            accuracy_tier="99",
            framework="vllm",
        )
        assert features[FEATURE_COLS.index("mlperf_round_num")] == _SERVING_ROUND


# ---------------------------------------------------------------------------
# GpuPredictor.predict — full inference path
# ---------------------------------------------------------------------------

class TestGpuPredictorPredict:
    def test_predict_returns_expected_keys(self, predictor):
        result = predictor.predict(
            gpu_id="mi300x",
            model_name="llama2-70b",
        )
        expected = {
            "gpu_id", "model_name", "scenario", "accuracy_tier", "framework",
            "pred_throughput_tok_per_sec", "roofline_tput_tok_per_sec",
            "efficiency_ratio", "vram_fits", "model_size_gb", "has_training_data",
            "training_data_tier",
            "memory_fit_verdict", "kv_cache_gb", "memory_total_gb", "vram_utilization",
        }
        assert set(result.keys()) == expected

    def test_has_training_data_true_for_trained_gpu(self, predictor):
        # mi300x has real MLPerf + calibration rows in the training set.
        assert predictor.has_training_data("mi300x") is True
        assert predictor.predict(gpu_id="mi300x", model_name="llama2-70b")["has_training_data"] is True

    def test_has_training_data_false_for_zero_row_gpus(self, predictor):
        # These three are in_model_scope but have zero rows in mlperf_features.parquet —
        # any prediction for them is pure spec extrapolation, never validated against
        # a real measurement.
        # accuracy_tier="99.9" (fp16), not the default "99" (fp8): a100_sxm_80gb
        # has no native FP8 (Ampere), which now correctly raises rather
        # than silently substituting fp16 — fp16 is supported by all three GPUs
        # here, and this test's actual subject is has_training_data, not precision.
        for gpu_id in ("a100_sxm_80gb", "l4", "rtx4090"):
            assert predictor.has_training_data(gpu_id) is False
            result = predictor.predict(gpu_id=gpu_id, model_name="gptj", accuracy_tier="99.9")
            assert result["has_training_data"] is False

    def test_predict_batch_includes_has_training_data(self, predictor):
        results = predictor.predict_batch([
            {"gpu_id": "mi300x", "model_name": "gptj"},
            {"gpu_id": "rtx4090", "model_name": "gptj"},
        ])
        by_gpu = {r["gpu_id"]: r for r in results}
        assert by_gpu["mi300x"]["has_training_data"] is True
        assert by_gpu["rtx4090"]["has_training_data"] is False

    def test_training_data_tier_none_for_zero_row_gpu(self, predictor):
        # rtx4090 has zero rows — pure spec extrapolation (see the
        # has_training_data test above for the full set of zero-row GPUs).
        assert predictor.training_data_tier("rtx4090") == "none"
        result = predictor.predict(gpu_id="rtx4090", model_name="gptj", accuracy_tier="99.9")
        assert result["training_data_tier"] == "none"

    def test_training_data_tier_below_floor_for_thin_amd_gpus(self, predictor):
        # mi300x (80), mi325x (82), mi355x (50) all have real rows but sit
        # under this project's 100-row-per-GPU Must-have floor — the exact gap a
        # boolean has_training_data can't distinguish from "plenty of data".
        for gpu_id in ("mi300x", "mi325x", "mi355x"):
            assert predictor.has_training_data(gpu_id) is True
            assert predictor.training_data_tier(gpu_id) == "below_floor"

    def test_training_data_tier_sufficient_for_well_covered_gpus(self, predictor):
        # h100_sxm (178) and h200_sxm (283) both clear the 100-row floor.
        for gpu_id in ("h100_sxm", "h200_sxm"):
            assert predictor.training_data_tier(gpu_id) == "sufficient"
        result = predictor.predict(gpu_id="h100_sxm", model_name="llama2-70b")
        assert result["training_data_tier"] == "sufficient"

    def test_predict_batch_includes_training_data_tier(self, predictor):
        results = predictor.predict_batch([
            {"gpu_id": "mi300x", "model_name": "gptj"},
            {"gpu_id": "h100_sxm", "model_name": "gptj"},
            {"gpu_id": "rtx4090", "model_name": "gptj"},
        ])
        by_gpu = {r["gpu_id"]: r for r in results}
        assert by_gpu["mi300x"]["training_data_tier"] == "below_floor"
        assert by_gpu["h100_sxm"]["training_data_tier"] == "sufficient"
        assert by_gpu["rtx4090"]["training_data_tier"] == "none"

    def test_predict_calls_training_data_tier_exactly_once(self, predictor, monkeypatch):
        # predict()'s dict-construction originally called training_data_tier()
        # twice per request — once directly for the training_data_tier field,
        # once indirectly via has_training_data() calling it internally.
        # Found 2026-07-11 via a performance review, instrumented with a
        # wraps mock (2x confirmed, then fixed to compute the tier once and
        # derive both fields from it — same shape as the existing
        # verdict/vram_fits/memory_fit_verdict dedup).
        original = predictor.training_data_tier
        calls: list[str] = []

        def _counting(gpu_id):
            calls.append(gpu_id)
            return original(gpu_id)

        monkeypatch.setattr(predictor, "training_data_tier", _counting)
        predictor.predict(gpu_id="mi300x", model_name="llama2-70b")
        assert calls == ["mi300x"], (
            f"training_data_tier() called {len(calls)}x for one predict() call, expected 1"
        )

    def test_predict_batch_calls_training_data_tier_once_per_gpu(self, predictor, monkeypatch):
        original = predictor.training_data_tier
        calls: list[str] = []

        def _counting(gpu_id):
            calls.append(gpu_id)
            return original(gpu_id)

        monkeypatch.setattr(predictor, "training_data_tier", _counting)
        predictor.predict_batch([
            {"gpu_id": "mi300x", "model_name": "gptj"},
            {"gpu_id": "h100_sxm", "model_name": "gptj"},
        ])
        assert calls == ["mi300x", "h100_sxm"], (
            f"training_data_tier() called {len(calls)}x for a 2-GPU predict_batch() call, expected 2"
        )

    def test_predict_throughput_positive(self, predictor):
        result = predictor.predict(
            gpu_id="h100_sxm",
            model_name="llama2-70b",
            scenario="Offline",
            accuracy_tier="99",
            framework="tensorrt",
        )
        # Must reach at least 1% of the roofline — a degenerate model predicting
        # ~0.001 tok/s would pass a bare > 0 check but not this threshold.
        assert result["pred_throughput_tok_per_sec"] >= 0.01 * result["roofline_tput_tok_per_sec"]

    def test_predict_never_exceeds_roofline(self, predictor):
        for gpu_id in ["mi300x", "h100_sxm", "h200_sxm", "mi325x", "mi355x"]:
            result = predictor.predict(
                gpu_id=gpu_id,
                model_name="llama2-70b",
                accuracy_tier="99",
            )
            assert (
                result["pred_throughput_tok_per_sec"]
                <= result["roofline_tput_tok_per_sec"] + 1e-3
            )

    def test_predict_vram_fits_llama70b_on_mi300x(self, predictor):
        result = predictor.predict(
            gpu_id="mi300x",
            model_name="llama2-70b",
            accuracy_tier="99",
        )
        # llama2-70b at FP8 = 70 GB, MI300X has 192 GB → fits
        assert result["vram_fits"] is True

    def test_predict_vram_not_fit_405b_on_l4(self, predictor):
        result = predictor.predict(
            gpu_id="l4",
            model_name="llama3.1-405b",
            accuracy_tier="99",
        )
        # 405B FP8 = 405 GB >> 24 GB L4 VRAM
        assert result["vram_fits"] is False

    def test_unknown_gpu_raises(self, predictor):
        with pytest.raises(ValueError, match="Unknown gpu_id"):
            predictor.predict(gpu_id="rtx9090", model_name="gptj")

    def test_unknown_model_raises(self, predictor):
        with pytest.raises(ValueError, match="Unknown model_name"):
            predictor.predict(gpu_id="mi300x", model_name="gpt4")

    def test_invalid_scenario_raises(self, predictor):
        with pytest.raises(ValueError, match="Invalid scenario"):
            predictor.predict(gpu_id="mi300x", model_name="gptj", scenario="Interactive")

    def test_invalid_tier_raises(self, predictor):
        with pytest.raises(ValueError, match="Invalid accuracy_tier"):
            predictor.predict(gpu_id="mi300x", model_name="gptj", accuracy_tier="99.5")

    def test_invalid_framework_raises(self, predictor):
        with pytest.raises(ValueError, match="Invalid framework"):
            predictor.predict(gpu_id="mi300x", model_name="gptj", framework="pytorch")

    def test_unsupported_precision_raises(self, predictor):
        # A100 (Ampere) has no native FP8 Tensor Core path —
        # data/gpu_specs.yaml has a100_sxm_80gb.peak_tflops.fp8: ~ (null).
        # accuracy_tier="99" selects fp8; before this check existed, predict()
        # silently substituted fp16's peak TFLOPS instead of raising, mixing
        # fp8's bytes-per-param (half the memory) with fp16's compute ceiling.
        with pytest.raises(ValueError, match="does not support precision 'fp8'"):
            predictor.predict(gpu_id="a100_sxm_80gb", model_name="gptj", accuracy_tier="99")

    @pytest.mark.parametrize("accuracy_tier", ["99.9", "base"])
    def test_a100_supports_fp16_and_bf16(self, predictor, accuracy_tier):
        # Sanity check that the precision-support guard only rejects the genuinely
        # unsupported precision (fp8), not every tier for this GPU.
        result = predictor.predict(
            gpu_id="a100_sxm_80gb", model_name="gptj", accuracy_tier=accuracy_tier,
        )
        assert result["pred_throughput_tok_per_sec"] >= 0.0

    def test_model_size_gb_correct_fp8(self, predictor):
        result = predictor.predict(
            gpu_id="mi300x",
            model_name="llama2-70b",
            accuracy_tier="99",
        )
        # 70B params × 1 byte/param (FP8) = 70 GB
        assert result["model_size_gb"] == pytest.approx(70.0)

    def test_memory_total_gb_matches_independent_calculation(self, predictor):
        # Recompute weights/kv/total from first principles — a fresh
        # kv_cache_gb() call fed by MODEL_ARCH/BYTES_PER_PARAM looked up here,
        # not by reusing predict()'s own model_size_gb/kv_cache_gb outputs as
        # "expected". This verifies predict() *wires* model_name/accuracy_tier
        # into the right MODEL_ARCH entry and the right precision correctly
        # (e.g. would catch predict() using the wrong bpp, the wrong model's
        # architecture tuple, or passing kv_cache_gb's args out of order) —
        # confirmed by temporarily forcing predict() to use FP16's bpp instead
        # of the correct FP8 one and watching this test fail (24.16 vs
        # expected 12.08 GB). It can NOT catch a wrong *value* inside
        # MODEL_ARCH/BYTES_PER_PARAM itself (both this test and predict() read
        # the same table) — that's a data-accuracy concern the source
        # citations in build_features.py address, not something a unit test
        # can verify without an external oracle.
        from src.features.build_features import MODEL_ARCH, BYTES_PER_PARAM, kv_cache_gb

        batch_size, input_tokens, output_tokens = 32, 2048, 256
        result = predictor.predict(
            gpu_id="mi300x", model_name="llama2-70b", accuracy_tier="99",
            batch_size=batch_size, input_tokens=input_tokens, output_tokens=output_tokens,
        )

        # llama2-70b @ tier 99 = FP8 (1 byte/param) for every vendor.
        bpp = BYTES_PER_PARAM["fp8"]
        expected_weights_gb = 70.0 * bpp
        n_layers, n_kv_heads, head_dim = MODEL_ARCH["llama2-70b"]
        expected_kv_gb = kv_cache_gb(
            n_layers, n_kv_heads, head_dim, batch_size, input_tokens, output_tokens, bpp
        )
        expected_total = (expected_weights_gb + expected_kv_gb) * 1.10
        expected_util = expected_total / 192.0  # MI300X VRAM

        # memory_total_gb and vram_utilization are independently rounded in
        # the response, so compare with an absolute tolerance rather than a
        # tight relative one to avoid double-rounding false negatives.
        assert result["model_size_gb"] == pytest.approx(expected_weights_gb, abs=1e-2)
        assert result["kv_cache_gb"] == pytest.approx(expected_kv_gb, abs=1e-2)
        assert result["memory_total_gb"] == pytest.approx(expected_total, abs=1e-2)
        assert result["vram_utilization"] == pytest.approx(expected_util, abs=1e-3)

    def test_memory_fit_verdict_tight_llama2_70b_base_tier_mi300x(self, predictor):
        # Golden "tight" case (default batch=32/in=2048/out=256): base tier is
        # FP16 on AMD (no FP8 override — only the 99.9 tier gets that), so
        # weights=140 GB + kv≈24 GB + 10% overhead ≈ 180.6 GB on MI300X's
        # 192 GB → util≈0.94, inside the (0.90, 0.98] tight band.
        result = predictor.predict(
            gpu_id="mi300x", model_name="llama2-70b", accuracy_tier="base",
        )
        assert result["memory_fit_verdict"] == "tight"
        assert result["vram_fits"] is True  # tight is not a hard exclusion

    def test_memory_fit_verdict_does_not_fit_excludes_vram_fits(self, predictor):
        result = predictor.predict(
            gpu_id="l4", model_name="llama3.1-405b", accuracy_tier="99",
        )
        assert result["memory_fit_verdict"] == "does_not_fit"
        assert result["vram_fits"] is False

    def test_larger_batch_can_flip_fits_to_does_not_fit(self, predictor):
        # Same GPU/model/tier — only batch size changes. Demonstrates the
        # KV-cache term (not just weights) drives the verdict end-to-end.
        small = predictor.predict(
            gpu_id="h200_sxm", model_name="llama2-70b", accuracy_tier="99",
            batch_size=1, input_tokens=64, output_tokens=1,
        )
        large = predictor.predict(
            gpu_id="h200_sxm", model_name="llama2-70b", accuracy_tier="99",
            batch_size=256, input_tokens=8192, output_tokens=4096,
        )
        assert small["memory_fit_verdict"] == "fits"
        assert large["memory_fit_verdict"] == "does_not_fit"
        assert large["kv_cache_gb"] > small["kv_cache_gb"]

    @pytest.mark.parametrize("kwargs", [
        {"batch_size": 0}, {"batch_size": 257},
        {"input_tokens": 63}, {"input_tokens": 8193},
        {"output_tokens": 0}, {"output_tokens": 4097},
    ])
    def test_invalid_memory_fit_params_raise(self, predictor, kwargs):
        # match= pins this to the parametrized field actually being rejected,
        # not just "some ValueError happened".
        param_name = next(iter(kwargs))
        with pytest.raises(ValueError, match=f"Invalid {param_name}"):
            predictor.predict(gpu_id="mi300x", model_name="gptj", **kwargs)


# ---------------------------------------------------------------------------
# GpuPredictor.predict_batch
# ---------------------------------------------------------------------------

class TestGpuPredictorBatch:
    def test_batch_preserves_order_and_identity(self, predictor):
        reqs = [
            {"gpu_id": "mi300x",   "model_name": "llama2-70b"},
            {"gpu_id": "h100_sxm", "model_name": "gptj"},
            {"gpu_id": "h200_sxm", "model_name": "mixtral-8x7b"},
        ]
        results = predictor.predict_batch(reqs)
        assert len(results) == 3
        # Count alone doesn't catch a bug that returns the mi300x result 3 times.
        for req, res in zip(reqs, results):
            assert res["gpu_id"] == req["gpu_id"]
            assert res["model_name"] == req["model_name"]

    def test_batch_empty_returns_empty(self, predictor):
        assert predictor.predict_batch([]) == []

    def test_batch_unsupported_precision_raises(self, predictor):
        # The precision-support guard must hold for every request in the batch independently, not
        # just single predict() calls — a100_sxm_80gb has no native fp8.
        with pytest.raises(ValueError, match="does not support precision 'fp8'"):
            predictor.predict_batch([
                {"gpu_id": "mi300x", "model_name": "gptj", "accuracy_tier": "99"},
                {"gpu_id": "a100_sxm_80gb", "model_name": "gptj", "accuracy_tier": "99"},
            ])

    @pytest.mark.parametrize("req", [
        # tier 99 (fp8) — baseline case
        {"gpu_id": "mi325x", "model_name": "llama2-70b",
         "scenario": "Server", "accuracy_tier": "99", "framework": "rocm_other"},
        # AMD tier 99.9 — fp8 override path; a batch/single divergence here would
        # produce the wrong model_size_gb (70 vs 140 GB) and thus wrong vram_fits.
        {"gpu_id": "mi300x", "model_name": "llama2-70b",
         "scenario": "Offline", "accuracy_tier": "99.9", "framework": "vllm"},
    ])
    def test_batch_results_match_single(self, predictor, req):
        single = predictor.predict(**{k: v for k, v in req.items()})
        batch = predictor.predict_batch([req])
        # Compare every numeric output — the batch path accumulates metadata in a
        # separate dict; a divergence (e.g. vram_fits always True) was invisible
        # when only pred_throughput_tok_per_sec was checked.
        for key in ("pred_throughput_tok_per_sec", "roofline_tput_tok_per_sec",
                    "efficiency_ratio", "model_size_gb",
                    "kv_cache_gb", "memory_total_gb", "vram_utilization"):
            assert batch[0][key] == pytest.approx(single[key], rel=1e-4), (
                f"{key}: batch={batch[0][key]!r}  single={single[key]!r}"
            )
        assert batch[0]["vram_fits"] == single["vram_fits"]
        assert batch[0]["memory_fit_verdict"] == single["memory_fit_verdict"]

    def test_precomputed_memory_fit_matches_recomputed(self, predictor):
        # GpuRecommender passes an already-computed (verdict, kv_cache_gb,
        # memory_total_gb, vram_utilization) tuple via the optional
        # "memory_fit" request key so predict_batch() doesn't redo that work
        # per candidate GPU (see predict_batch()'s docstring). Supplying it
        # must produce byte-identical output to leaving it out and letting
        # predict_batch() compute it fresh — otherwise the two code paths
        # could silently drift apart.
        base_req = {
            "gpu_id": "mi300x", "model_name": "llama2-70b",
            "accuracy_tier": "99.9", "batch_size": 16,
            "input_tokens": 1024, "output_tokens": 256,
        }
        recomputed = predictor.predict_batch([dict(base_req)])[0]

        precomputed_req = {
            **base_req,
            "memory_fit": (
                recomputed["memory_fit_verdict"],
                recomputed["kv_cache_gb"],
                recomputed["memory_total_gb"],
                recomputed["vram_utilization"],
            ),
        }
        with_hint = predictor.predict_batch([precomputed_req])[0]

        for key in ("memory_fit_verdict", "kv_cache_gb", "memory_total_gb",
                    "vram_utilization", "vram_fits", "pred_throughput_tok_per_sec"):
            assert with_hint[key] == recomputed[key], (
                f"{key}: precomputed={with_hint[key]!r} recomputed={recomputed[key]!r}"
            )

    def test_batch_all_in_scope_gpus(self, predictor):
        from src.data.gpu_spec_db import load_specs
        in_scope = [s["id"] for s in load_specs() if s.get("in_model_scope")]
        # accuracy_tier="99.9" (fp16): fp16 is supported by every in-scope GPU.
        # The default "99" (fp8) would raise for a100_sxm_80gb (no native FP8
        # on Ampere) — correct behavior, but not what this test is
        # checking (batch prediction across every GPU respects the roofline).
        reqs = [
            {"gpu_id": g, "model_name": "llama2-70b", "accuracy_tier": "99.9"}
            for g in in_scope
        ]
        results = predictor.predict_batch(reqs)
        assert len(results) == len(in_scope)
        for r in results:
            assert r["pred_throughput_tok_per_sec"] >= 0.01 * r["roofline_tput_tok_per_sec"]


# ---------------------------------------------------------------------------
# Cross-check: _encode() (training) vs _build_feature_vector() (serving)
# ---------------------------------------------------------------------------

class TestEncodeVsPredictor:
    """Verify that train_final._encode() and predictor._build_feature_vector()
    produce identical categorical encodings for all shared columns.

    If these two implementations drift the model is trained on different features
    than serving produces — a silent, test-invisible bug without this class.
    """

    @pytest.mark.parametrize("scenario,tier,framework,gpu_id", [
        ("Offline", "99",   "tensorrt",  "h100_sxm"),
        ("Server",  "99.9", "vllm",      "mi300x"),
        ("Offline", "base", "rocm_other", "mi355x"),
        ("Server",  "99",   "other",     "h200_sxm"),
        ("Offline", "99",   "vllm",      "mi325x"),
    ])
    def test_categorical_encodings_match(self, spec_map, scenario, tier, framework, gpu_id):
        import math
        import pandas as pd
        from src.models.train_final import _encode

        # Get ground-truth feature values from the serving path.
        feat, _, _ = _build_feature_vector(
            gpu_spec=spec_map[gpu_id],
            model_name="gptj",
            scenario=scenario,
            accuracy_tier=tier,
            framework=framework,
        )
        amd_arch_gen_raw = feat[FEATURE_COLS.index("amd_arch_gen")]

        # Build a row that mimics what build_training_df produces before _encode().
        # is_base_tier is computed in build_training_df (not _encode), so inject it directly.
        row = pd.DataFrame([{
            "scenario": scenario,
            "benchmark_accuracy_tier": tier,
            "framework_family": framework,
            "amd_arch_gen": None if math.isnan(amd_arch_gen_raw) else amd_arch_gen_raw,
            "is_base_tier": int(tier == "base"),
        }])
        enc = _encode(row).iloc[0]

        # Compare every column that _encode() or build_training_df writes and
        # _build_feature_vector() computes.
        for col in ("scenario_offline", "is_base_tier",
                    "fw_tensorrt", "fw_vllm", "fw_rocm_other", "is_cdna4"):
            feat_val = feat[FEATURE_COLS.index(col)]
            enc_val = enc[col]
            assert feat_val == enc_val, (
                f"{col} mismatch for gpu={gpu_id} scenario={scenario!r} "
                f"tier={tier!r} fw={framework!r}: "
                f"serving={feat_val}, training={enc_val}"
            )


# ---------------------------------------------------------------------------
# File-guard security tests — GpuPredictor.__init__
# ---------------------------------------------------------------------------

class TestGpuPredictorFileGuards:
    """Security tests for the file guards in GpuPredictor.__init__.

    These tests verify three properties:
    1. Feature-cols mismatch raises ValueError — not AssertionError — so the
       check is never compiled away under `python -O` / PYTHONOPTIMIZE=1.
    2. Symlinked artifact files are refused (path-traversal defence).
    3. Oversized metadata is refused (memory-exhaustion defence).
    """

    def test_feature_cols_mismatch_raises_valueerror(self, tmp_path):
        # Both files must pass the file guards (non-symlink, non-oversized)
        # before the feature-cols check fires.  prophet_v1.json is never
        # loaded in this test path — the check raises before xgb.load_model.
        (tmp_path / "feature_metadata.json").write_text(
            json.dumps({"feature_cols": ["wrong", "cols"], "target": "efficiency_ratio"}),
            encoding="utf-8",
        )
        (tmp_path / "prophet_v1.json").write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="feature_cols mismatch"):
            GpuPredictor(model_dir=tmp_path)

    def test_missing_trained_gpu_ids_raises_valueerror(self, tmp_path):
        # An old feature_metadata.json predating the has_training_data feature
        # must fail loudly at load time, not silently report every GPU as
        # having real training data.
        (tmp_path / "feature_metadata.json").write_text(
            json.dumps({"feature_cols": FEATURE_COLS, "target": "efficiency_ratio"}),
            encoding="utf-8",
        )
        real_model_dir = Path(__file__).parent.parent / "data" / "models"
        (tmp_path / "prophet_v1.json").write_bytes(
            (real_model_dir / "prophet_v1.json").read_bytes()
        )
        with pytest.raises(ValueError, match="trained_gpu_ids"):
            GpuPredictor(model_dir=tmp_path)

    def test_missing_trained_gpu_row_counts_raises_valueerror(self, tmp_path):
        # An old feature_metadata.json predating training_data_tier() must
        # fail loudly at load time, not silently report every trained GPU
        # as meeting the 100-row floor (same rationale as the trained_gpu_ids
        # guard above).
        (tmp_path / "feature_metadata.json").write_text(
            json.dumps({
                "feature_cols": FEATURE_COLS,
                "target": "efficiency_ratio",
                "trained_gpu_ids": ["mi300x"],
            }),
            encoding="utf-8",
        )
        real_model_dir = Path(__file__).parent.parent / "data" / "models"
        (tmp_path / "prophet_v1.json").write_bytes(
            (real_model_dir / "prophet_v1.json").read_bytes()
        )
        with pytest.raises(ValueError, match="trained_gpu_row_counts"):
            GpuPredictor(model_dir=tmp_path)

    def test_symlink_meta_raises_valueerror(self, tmp_path):
        link = tmp_path / "feature_metadata.json"
        link.symlink_to(tmp_path / "nonexistent_target")
        with pytest.raises(ValueError, match="symlink"):
            GpuPredictor(model_dir=tmp_path)

    def test_symlink_model_raises_valueerror(self, tmp_path):
        # meta must pass guards so the loop reaches model_path.
        (tmp_path / "feature_metadata.json").write_text(
            json.dumps({"feature_cols": [], "target": ""}), encoding="utf-8"
        )
        (tmp_path / "prophet_v1.json").symlink_to(tmp_path / "nonexistent_target")
        with pytest.raises(ValueError, match="symlink"):
            GpuPredictor(model_dir=tmp_path)

    def test_oversized_meta_raises_valueerror(self, tmp_path):
        from src.models.predictor import _MAX_META_BYTES
        (tmp_path / "feature_metadata.json").write_bytes(b"x" * (_MAX_META_BYTES + 1))
        (tmp_path / "prophet_v1.json").write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="too large"):
            GpuPredictor(model_dir=tmp_path)
