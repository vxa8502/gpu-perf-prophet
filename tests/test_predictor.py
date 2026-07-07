"""
Unit and integration tests for src/models/predictor.py.

All tests load the real trained model (data/models/prophet_v1.json) so they
verify the full inference path, not just the feature-construction logic.
"""

from __future__ import annotations

import json
import math

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
            "efficiency_ratio", "vram_fits", "model_size_gb",
        }
        assert set(result.keys()) == expected

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

    def test_model_size_gb_correct_fp8(self, predictor):
        result = predictor.predict(
            gpu_id="mi300x",
            model_name="llama2-70b",
            accuracy_tier="99",
        )
        # 70B params × 1 byte/param (FP8) = 70 GB
        assert result["model_size_gb"] == pytest.approx(70.0)


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
                    "efficiency_ratio", "model_size_gb"):
            assert batch[0][key] == pytest.approx(single[key], rel=1e-4), (
                f"{key}: batch={batch[0][key]!r}  single={single[key]!r}"
            )
        assert batch[0]["vram_fits"] == single["vram_fits"]

    def test_batch_all_in_scope_gpus(self, predictor):
        from src.data.gpu_spec_db import load_specs
        in_scope = [s["id"] for s in load_specs() if s.get("in_model_scope")]
        reqs = [{"gpu_id": g, "model_name": "llama2-70b"} for g in in_scope]
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
