"""
Tests for src/features/build_features.py.

Strategy: unit-test the pure functions with hand-computed expected values,
then run a smoke-test of the full pipeline on a synthetic fixture DataFrame
that mimics the enriched MLPerf schema (avoiding a real parquet dependency).
"""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import yaml

from src.features.build_features import (
    BYTES_PER_PARAM,
    MODEL_PARAMS,
    TIER_TO_PRECISION,
    _normalize_framework,
    build_training_df,
    roofline_ceilings,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def spec_path(tmp_path: Path) -> Path:
    """Minimal gpu_specs.yaml with two GPUs for pipeline tests."""
    spec = {
        "schema_version": "1.0",
        "gpus": [
            {
                "id": "h200_sxm",
                "name": "NVIDIA H200 SXM",
                "vendor": "nvidia",
                "architecture": "hopper",
                "memory_type": "hbm3e",
                "vram_gb": 141,
                "hbm_bandwidth_tbps": 4.8,
                "peak_tflops": {
                    "fp32": 66.9,
                    "bf16": 989.4,
                    "fp16": 989.4,
                    "fp8": 1978.9,
                    "fp6": None,
                    "fp4": None,
                    "int8": 3957.8,
                },
                "streaming_multiprocessors": 132,
                "l2_cache_mb": 50,
                "tdp_w": 700,
                "spec_confidence": "verified",
                "in_model_scope": True,
                "aliases": ["NVIDIA H200-SXM-141GB"],
            },
            {
                "id": "mi300x",
                "name": "AMD Instinct MI300X",
                "vendor": "amd",
                "architecture": "cdna3",
                "memory_type": "hbm3",
                "vram_gb": 192,
                "hbm_bandwidth_tbps": 5.3,
                "peak_tflops": {
                    "fp32": 163.4,
                    "bf16": 1307.4,
                    "fp16": 1307.4,
                    "fp8": 2614.9,
                    "fp6": None,
                    "fp4": None,
                    "int8": 2614.9,
                },
                "compute_units": 304,
                "l2_cache_mb": 32,
                "tdp_w": 750,
                "spec_confidence": "verified",
                "in_model_scope": True,
                "aliases": ["AMD Instinct MI300X 192GB HBM3"],
            },
        ],
    }
    p = tmp_path / "gpu_specs.yaml"
    p.write_text(yaml.dump(spec))
    return p


def _make_raw_df(spec_path: Path) -> pd.DataFrame:
    """Synthetic raw MLPerf DataFrame with the minimal schema expected by build_training_df."""
    rows = [
        # H200 SXM, llama2-70b, 99-tier (FP8 selected), Offline, 1 node × 8 GPUs
        {
            "round": "v5.0",
            "division": "closed",
            "submitter": "TestOrg",
            "system_name": "8xH200_test",
            "gpu_name": "NVIDIA H200-SXM-141GB",
            "num_gpus": 8,
            "vram_gb": 1128.0,
            "framework": "TensorRT 10.2.0, CUDA 12.4",
            "system_type": "datacenter",
            "hw_status": "available",
            "benchmark": "llama2-70b-99",
            "benchmark_base": "llama2-70b",
            "benchmark_accuracy_tier": "99",
            "scenario": "Offline",
            "precision": None,
            "tokens_per_sample": 294,
            "throughput_tokens_per_sec": 50000.0,
            "throughput_tok_per_sec_per_gpu": 6250.0,
            "result_valid": True,
            "throughput_samples_per_sec": 170.0,
            "latency_mean_ms": None,
            "latency_p99_ms": None,
            "ttft_mean_ms": None,
            "ttft_p99_ms": None,
            "tpot_mean_ms": None,
            "tpot_p99_ms": None,
            "log_path": "fake/path",
        },
        # MI300X, llama2-70b, 99.9-tier (FP16 selected), Server, 1 node × 8 GPUs
        {
            "round": "v5.0",
            "division": "closed",
            "submitter": "TestOrg",
            "system_name": "8xMI300X_test",
            "gpu_name": "AMD Instinct MI300X 192GB HBM3",
            "num_gpus": 8,
            "vram_gb": 1536.0,
            "framework": "vLLM 0.4.3+rocm614, PyTorch 2.3.0, ROCm 6.1.2",
            "system_type": "datacenter",
            "hw_status": "available",
            "benchmark": "llama2-70b-99.9",
            "benchmark_base": "llama2-70b",
            "benchmark_accuracy_tier": "99.9",
            "scenario": "Server",
            "precision": None,
            "tokens_per_sample": 294,
            "throughput_tokens_per_sec": 24000.0,
            "throughput_tok_per_sec_per_gpu": 3000.0,
            "result_valid": True,
            "throughput_samples_per_sec": 81.6,
            "latency_mean_ms": None,
            "latency_p99_ms": None,
            "ttft_mean_ms": None,
            "ttft_p99_ms": None,
            "tpot_mean_ms": None,
            "tpot_p99_ms": None,
            "log_path": "fake/path2",
        },
        # H200 SXM, llama2-70b, result_valid=False — must be dropped
        {
            "round": "v5.0",
            "division": "closed",
            "submitter": "TestOrg",
            "system_name": "8xH200_invalid",
            "gpu_name": "NVIDIA H200-SXM-141GB",
            "num_gpus": 8,
            "vram_gb": 1128.0,
            "framework": "TensorRT 10.2.0, CUDA 12.4",
            "system_type": "datacenter",
            "hw_status": "available",
            "benchmark": "llama2-70b-99",
            "benchmark_base": "llama2-70b",
            "benchmark_accuracy_tier": "99",
            "scenario": "Offline",
            "precision": None,
            "tokens_per_sample": 294,
            "throughput_tokens_per_sec": 99999.0,
            "throughput_tok_per_sec_per_gpu": 12499.8,
            "result_valid": False,
            "throughput_samples_per_sec": 340.0,
            "latency_mean_ms": None,
            "latency_p99_ms": None,
            "ttft_mean_ms": None,
            "ttft_p99_ms": None,
            "tpot_mean_ms": None,
            "tpot_p99_ms": None,
            "log_path": "fake/path3",
        },
        # Unknown GPU — should be dropped (no spec DB entry → gpu_hbm_bandwidth_tbps is NaN
        # → caught by the missing-specs guard in build_training_df, not by gpu_in_model_scope)
        {
            "round": "v5.0",
            "division": "closed",
            "submitter": "TestOrg",
            "system_name": "8xUnknownGPU",
            "gpu_name": "SomeFutureGPU 9000",
            "num_gpus": 8,
            "vram_gb": 512.0,
            "framework": "TensorRT 99.0, CUDA 99.0",
            "system_type": "datacenter",
            "hw_status": "available",
            "benchmark": "llama2-70b-99",
            "benchmark_base": "llama2-70b",
            "benchmark_accuracy_tier": "99",
            "scenario": "Offline",
            "precision": None,
            "tokens_per_sample": 294,
            "throughput_tokens_per_sec": 100.0,
            "throughput_tok_per_sec_per_gpu": 12.5,
            "result_valid": True,
            "throughput_samples_per_sec": 0.34,
            "latency_mean_ms": None,
            "latency_p99_ms": None,
            "ttft_mean_ms": None,
            "ttft_p99_ms": None,
            "tpot_mean_ms": None,
            "tpot_p99_ms": None,
            "log_path": "fake/path4",
        },
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# roofline_ceilings
# ---------------------------------------------------------------------------

class TestRooflineCeilings:
    """Hand-verify the bandwidth and compute ceilings for known GPU–model pairs."""

    def test_h200_llama2_70b_fp16(self):
        # H200 SXM: HBM BW = 4.8 TB/s, FP16 peak = 989.4 TFLOPS
        # Llama2-70B: 70B params (dense, total == compute)
        # BW ceiling  = 4.8e12 / (70e9 × 2)        = 34.29 tok/s
        # Compute ceil = 989.4e12 / (2 × 70e9)      = 7067.14 tok/s
        # roofline_tput = compute_ceil (not BW limit)
        bw, compute, roofline = roofline_ceilings(
            total_params_b=70.0,
            compute_params_b=70.0,
            bytes_per_param=2.0,
            hbm_bw_tbps=4.8,
            peak_tflops=989.4,
        )
        assert math.isclose(bw, 4.8e12 / (70e9 * 2), rel_tol=1e-6)
        assert math.isclose(compute, 989.4e12 / (2 * 70e9), rel_tol=1e-6)
        assert roofline == compute

    def test_mi300x_llama2_70b_fp8(self):
        # MI300X: HBM BW = 5.3 TB/s, FP8 peak = 2614.9 TFLOPS
        # BW ceiling  = 5.3e12 / (70e9 × 1)        = 75.71 tok/s
        # Compute ceil = 2614.9e12 / (2 × 70e9)     = 18677.86 tok/s
        bw, compute, roofline = roofline_ceilings(
            total_params_b=70.0,
            compute_params_b=70.0,
            bytes_per_param=1.0,
            hbm_bw_tbps=5.3,
            peak_tflops=2614.9,
        )
        assert math.isclose(bw, 5.3e12 / (70e9 * 1), rel_tol=1e-6)
        assert math.isclose(compute, 2614.9e12 / (2 * 70e9), rel_tol=1e-6)
        assert roofline == compute

    def test_moe_uses_active_params_for_compute(self):
        # Mixtral 8×7B: total=46.7B, active=14.1B
        # BW ceiling uses total (all weights in HBM): 5.3e12 / (46.7e9 × 2)
        # Compute ceiling uses active: 1307.4e12 / (2 × 14.1e9)
        bw, compute, roofline = roofline_ceilings(
            total_params_b=46.7,
            compute_params_b=14.1,
            bytes_per_param=2.0,
            hbm_bw_tbps=5.3,
            peak_tflops=1307.4,
        )
        assert math.isclose(bw, 5.3e12 / (46.7e9 * 2), rel_tol=1e-6)
        assert math.isclose(compute, 1307.4e12 / (2 * 14.1e9), rel_tol=1e-6)
        assert roofline == compute

    def test_roofline_equals_compute_ceiling(self):
        # roofline_tput is defined as the compute ceiling (a project-level choice:
        # the model learns the hardware efficiency ratio relative to compute, not BW).
        # All three cases are compute-bound (compute >> bw) by design — the invariant
        # is that roofline == compute regardless of what bw_tbps is.
        for bw_tbps, tflops in [(0.3, 121.6), (4.8, 989.4), (8.0, 2620.0)]:
            bw, compute, roofline = roofline_ceilings(
                total_params_b=70.0,
                compute_params_b=70.0,
                bytes_per_param=2.0,
                hbm_bw_tbps=bw_tbps,
                peak_tflops=tflops,
            )
            assert roofline == compute, (
                f"roofline should be compute ceiling, got roofline={roofline} compute={compute}"
            )

    def test_bandwidth_ceiling_scales_with_bytes_per_param(self):
        # Halving bytes_per_param (FP16→FP8) doubles the BW ceiling.
        bw_fp16, _, _ = roofline_ceilings(70.0, 70.0, 2.0, 4.8, 989.4)
        bw_fp8, _, _ = roofline_ceilings(70.0, 70.0, 1.0, 4.8, 989.4)
        assert math.isclose(bw_fp8, 2 * bw_fp16, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Reference tables
# ---------------------------------------------------------------------------

class TestReferenceTables:
    def test_all_model_params_have_positive_values(self):
        for name, (total, compute) in MODEL_PARAMS.items():
            assert total > 0, f"{name}: total_params_b must be > 0"
            assert compute > 0, f"{name}: compute_params_b must be > 0"
            assert compute <= total, f"{name}: compute_params_b must be ≤ total_params_b"

    def test_mixtral_moe_compute_lt_total(self):
        total, compute = MODEL_PARAMS["mixtral-8x7b"]
        assert compute < total

    def test_dense_models_compute_equals_total(self):
        for name in ("llama2-70b", "llama3.1-405b", "gptj"):
            total, compute = MODEL_PARAMS[name]
            assert total == compute, f"{name}: dense model should have compute == total"

    def test_bytes_per_param_ordering(self):
        # FP4 < FP8 < FP16 == BF16 < FP32
        assert BYTES_PER_PARAM["fp4"] < BYTES_PER_PARAM["fp8"]
        assert BYTES_PER_PARAM["fp8"] < BYTES_PER_PARAM["fp16"]
        assert BYTES_PER_PARAM["fp16"] == BYTES_PER_PARAM["bf16"]
        assert BYTES_PER_PARAM["bf16"] < BYTES_PER_PARAM["fp32"]

    def test_tier_to_precision_coverage(self):
        for tier in ("99.9", "99", "base"):
            assert tier in TIER_TO_PRECISION
            assert TIER_TO_PRECISION[tier] in BYTES_PER_PARAM


# ---------------------------------------------------------------------------
# _normalize_framework
# ---------------------------------------------------------------------------

class TestNormalizeFramework:
    @pytest.mark.parametrize("raw,expected", [
        ("TensorRT 10.2.0, CUDA 12.4",          "tensorrt"),
        ("TensorRT 9.3.0, CUDA 12.2",           "tensorrt"),
        ("vLLM 0.4.3+rocm614, PyTorch 2.3.0",   "vllm"),
        ("vLLM 0.9.0, Pytorch 2.7, ROCm 6.4",   "vllm"),
        ("PyTorch 2.9.1+git, ROCm 7.0.0",       "rocm_other"),
        ("ROCm 6.3.1",                           "rocm_other"),
        ("Mango LLMBoost, ROCm 6.12",            "rocm_other"),
        ("SomeUnknownFramework 1.0",             "other"),
        ("",                                     "other"),
        (None,                                   "unknown"),
        (123,                                    "unknown"),
    ])
    def test_classification(self, raw, expected):
        assert _normalize_framework(raw) == expected


# ---------------------------------------------------------------------------
# build_training_df — integration smoke test
# ---------------------------------------------------------------------------

class TestBuildTrainingDf:
    def test_filters_invalid_and_missing_spec_rows(self, spec_path, tmp_path):
        raw = _make_raw_df(spec_path)
        feat = build_training_df(raw, spec_path=spec_path)
        # invalid row (result_valid=False) and unknown-GPU row (no spec → NaN HBM BW) are dropped
        assert len(feat) == 2
        # Verify the correct 2 rows survived — a filter bug that kept the wrong pair
        # (e.g. result_valid=False row + unknown-GPU row) would still give len == 2.
        assert set(feat["canonical_gpu_id"]) == {"h200_sxm", "mi300x"}

    def test_new_columns_present(self, spec_path):
        raw = _make_raw_df(spec_path)
        feat = build_training_df(raw, spec_path=spec_path)
        expected_cols = [
            "model_total_params_b", "model_compute_params_b",
            "bytes_per_param", "selected_precision", "peak_tflops_selected",
            "bandwidth_ceiling_tok_per_sec", "compute_ceiling_tok_per_sec",
            "roofline_tput", "efficiency_ratio", "model_size_gb",
            "model_to_vram_ratio", "framework_family",
            "nvidia_arch_gen", "amd_arch_gen", "vendor_is_amd",
            "mlperf_round_num", "is_base_tier",
        ]
        for col in expected_cols:
            assert col in feat.columns, f"Missing column: {col}"

    def test_mlperf_round_num_mapping(self, spec_path):
        raw = _make_raw_df(spec_path)
        # _make_raw_df uses "round": "v5.0" → ROUND_ORDINAL["v5.0"] == 2
        feat = build_training_df(raw, spec_path=spec_path)
        assert (feat["mlperf_round_num"] == 2).all()

    def test_is_base_tier_encoding(self, spec_path):
        raw = _make_raw_df(spec_path).copy()
        # Inject a base-tier row for MI300X.
        base_row = raw[raw["gpu_name"] == "AMD Instinct MI300X 192GB HBM3"].iloc[0].copy()
        base_row["benchmark_accuracy_tier"] = "base"
        base_row["benchmark"] = "llama2-70b-base"
        raw = pd.concat([raw, base_row.to_frame().T], ignore_index=True)
        feat = build_training_df(raw, spec_path=spec_path)
        base_rows = feat[feat["benchmark_accuracy_tier"] == "base"]
        non_base_rows = feat[feat["benchmark_accuracy_tier"] != "base"]
        assert (base_rows["is_base_tier"] == 1).all()
        assert (non_base_rows["is_base_tier"] == 0).all()

    def test_mlperf_round_num_unknown_round_is_nan(self, spec_path):
        raw = _make_raw_df(spec_path).copy()
        raw["round"] = "v99.0"
        feat = build_training_df(raw, spec_path=spec_path)
        assert feat["mlperf_round_num"].isna().all()

    def test_roofline_equals_compute_ceiling(self, spec_path):
        raw = _make_raw_df(spec_path)
        feat = build_training_df(raw, spec_path=spec_path)
        pd.testing.assert_series_equal(
            feat["roofline_tput"],
            feat["compute_ceiling_tok_per_sec"],
            check_names=False,
        )

    def test_h200_fp8_precision_selection(self, spec_path):
        raw = _make_raw_df(spec_path)
        feat = build_training_df(raw, spec_path=spec_path)
        h200_row = feat[feat["canonical_gpu_id"] == "h200_sxm"].iloc[0]
        assert h200_row["selected_precision"] == "fp8"
        assert h200_row["bytes_per_param"] == BYTES_PER_PARAM["fp8"]
        # Pin the actual TFLOPS value — catches bugs in the vectorised peak-TFLOPS
        # selection (e.g. accidentally returning fp16=989.4 instead of fp8=1978.9).
        assert h200_row["peak_tflops_selected"] == pytest.approx(1978.9)

    def test_amd_tier_99_9_uses_fp8(self, spec_path):
        raw = _make_raw_df(spec_path)
        feat = build_training_df(raw, spec_path=spec_path)
        mi_row = feat[feat["canonical_gpu_id"] == "mi300x"].iloc[0]
        # AMD 99.9 tier → fp8 override (not fp16).  MI300X fp8 = 2614.9 TFLOPS.
        assert mi_row["selected_precision"] == "fp8"
        assert mi_row["bytes_per_param"] == BYTES_PER_PARAM["fp8"]
        assert mi_row["peak_tflops_selected"] == pytest.approx(2614.9)

    def test_nvidia_tier_99_9_still_uses_fp16(self, spec_path):
        raw = _make_raw_df(spec_path).copy()
        # Swap the H200 row to tier 99.9 — NVIDIA must NOT be overridden to fp8.
        mask = raw["gpu_name"] == "NVIDIA H200-SXM-141GB"
        raw.loc[mask, "benchmark_accuracy_tier"] = "99.9"
        raw.loc[mask, "benchmark"] = "llama2-70b-99.9"
        feat = build_training_df(raw, spec_path=spec_path)
        h200 = feat[feat["canonical_gpu_id"] == "h200_sxm"].iloc[0]
        assert h200["selected_precision"] == "fp16"
        assert h200["bytes_per_param"] == BYTES_PER_PARAM["fp16"]
        assert h200["peak_tflops_selected"] == pytest.approx(989.4)

    def test_vendor_is_amd_flag(self, spec_path):
        raw = _make_raw_df(spec_path)
        feat = build_training_df(raw, spec_path=spec_path)
        assert feat.loc[feat["canonical_gpu_id"] == "mi300x",  "vendor_is_amd"].iloc[0] == 1
        assert feat.loc[feat["canonical_gpu_id"] == "h200_sxm", "vendor_is_amd"].iloc[0] == 0

    def test_framework_family_assigned(self, spec_path):
        raw = _make_raw_df(spec_path)
        feat = build_training_df(raw, spec_path=spec_path)
        h200 = feat.loc[feat["canonical_gpu_id"] == "h200_sxm", "framework_family"].iloc[0]
        mi = feat.loc[feat["canonical_gpu_id"] == "mi300x", "framework_family"].iloc[0]
        assert h200 == "tensorrt"
        assert mi == "vllm"

    def test_no_nulls_in_feature_columns(self, spec_path):
        raw = _make_raw_df(spec_path)
        feat = build_training_df(raw, spec_path=spec_path)
        feature_cols = [
            "model_total_params_b", "model_compute_params_b", "bytes_per_param",
            "bandwidth_ceiling_tok_per_sec", "compute_ceiling_tok_per_sec",
            "roofline_tput", "efficiency_ratio", "model_size_gb",
            "model_to_vram_ratio", "vendor_is_amd",
        ]
        for col in feature_cols:
            assert feat[col].notna().all(), f"Nulls found in {col}"

    def test_vendor_arch_gen_ordinals(self, spec_path):
        raw = _make_raw_df(spec_path)
        feat = build_training_df(raw, spec_path=spec_path)
        h200 = feat[feat["canonical_gpu_id"] == "h200_sxm"].iloc[0]
        mi300 = feat[feat["canonical_gpu_id"] == "mi300x"].iloc[0]
        # H200 is hopper (nvidia_arch_gen=3); amd_arch_gen must be NaN
        assert h200["nvidia_arch_gen"] == 3
        assert pd.isna(h200["amd_arch_gen"])
        # MI300X is cdna3 (amd_arch_gen=1); nvidia_arch_gen must be NaN
        assert mi300["amd_arch_gen"] == 1
        assert pd.isna(mi300["nvidia_arch_gen"])

    def test_unknown_benchmark_dropped(self, spec_path):
        raw = _make_raw_df(spec_path).copy()
        # Inject a row with a benchmark_base not in MODEL_PARAMS
        extra = raw.iloc[0].copy()
        extra["benchmark_base"] = "future-model-1t"
        extra["result_valid"] = True
        raw = pd.concat([raw, extra.to_frame().T], ignore_index=True)
        feat = build_training_df(raw, spec_path=spec_path)
        assert "future-model-1t" not in feat["benchmark_base"].values
        # Guard against a catastrophic bug that drops ALL rows — an empty
        # DataFrame would also satisfy the assertion above.
        assert len(feat) == 2

    def test_rows_with_missing_gpu_specs_are_dropped(self, spec_path):
        # Patch enrich_df in the build_features namespace (where it was imported).
        import src.features.build_features as bf_mod
        from src.data.gpu_spec_db import enrich_df as real_enrich

        def patched_enrich(df, **kwargs):
            enriched = real_enrich(df, **kwargs)
            mask = enriched["canonical_gpu_id"] == "h200_sxm"
            enriched.loc[mask, "gpu_hbm_bandwidth_tbps"] = float("nan")
            return enriched

        raw = _make_raw_df(spec_path)
        with patch.object(bf_mod, "enrich_df", patched_enrich):
            feat = build_training_df(raw, spec_path=spec_path)

        # H200 row dropped; only MI300X survives
        assert len(feat) == 1
        assert feat["canonical_gpu_id"].iloc[0] == "mi300x"

    def test_violation_rows_logged_not_dropped(self, spec_path, caplog):
        import logging
        # When throughput > roofline_tput, rows are kept (not dropped) AND a warning
        # is emitted.  The test name claims both; both are now asserted.
        raw = _make_raw_df(spec_path).copy()
        # Force throughput_tok_per_sec_per_gpu to a huge value to trigger the path.
        h200_mask = raw["gpu_name"] == "NVIDIA H200-SXM-141GB"
        raw.loc[h200_mask, "throughput_tok_per_sec_per_gpu"] = 1_000_000.0
        with caplog.at_level(logging.WARNING, logger="src.features.build_features"):
            feat = build_training_df(raw, spec_path=spec_path)
        # Rows are kept, not filtered.
        assert len(feat) == 2
        assert (feat.loc[feat["canonical_gpu_id"] == "h200_sxm", "efficiency_ratio"] > 1).all()
        # Violations must also be logged (the "logged" half of the test name).
        assert any(
            "compute ceiling" in r.getMessage().lower()
            for r in caplog.records
            if r.levelno == logging.WARNING
        ), "Expected a WARNING about throughput > compute ceiling but none was emitted"
