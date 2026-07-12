"""Reliability-target gate tests — machine-executable counterpart of the design review checklist (`pytest tests/test_reliability_gates.py -v` or `pytest -m gate -v`); each test class name states the guarantee it protects and fills gaps left by the per-module test files (MoE param split, feature vector parity, schema/predictor agreement, model disk size, notebook FEATURE_COLS parity, training-data disclosure completeness, precision-support enforcement — see each class's own docstring for specifics)."""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path
from typing import get_args

import pandas as pd
import pytest
import yaml

from src.api.schemas import AccuracyTier, Framework, MemoryFitVerdict, RankingObjective, Scenario
from src.data import manifest
from src.data.gpu_spec_db import load_specs
from src.features.build_features import (
    BYTES_PER_PARAM,
    MEMORY_OVERHEAD_FACTOR,
    MODEL_PARAMS,
    TIER_TO_PRECISION,
    VALID_MEMORY_FIT_VERDICTS,
    _FRAMEWORK_PATTERNS,
    _normalize_framework,
    build_training_df,
    cost_per_million_tokens,
    gpu_supports_precision,
    memory_fit_verdict,
)
from src.models.predictor import (
    FEATURE_COLS,
    VALID_FRAMEWORKS,
    VALID_SCENARIOS,
    VALID_TIERS,
    GpuPredictor,
    _build_feature_vector,
    _selected_precision,
)
from src.recommend.recommender import (
    GpuRecommender,
    VALID_RANKING_OBJECTIVES,
    _pareto_frontier,
    _RANKING_FIELDS,
)


# Helpers

def _nan_match(a: float, b: float) -> bool:
    """True when both are NaN, or both are finite and within 1e-3 of each other."""
    a_nan = math.isnan(a)
    b_nan = math.isnan(b)
    if a_nan and b_nan:
        return True
    if a_nan or b_nan:
        return False
    return abs(a - b) < 1e-3


def _make_single_row(
    gpu_alias: str,
    model_name: str,
    scenario: str,
    tier: str,
    raw_framework: str,
    throughput: float = 1000.0,
) -> pd.DataFrame:
    """One synthetic MLPerf row with every field build_training_df expects."""
    return pd.DataFrame([{
        "round":                       "v5.0",
        "division":                    "closed",
        "submitter":                   "test",
        "system_name":                 "test_system",
        "gpu_name":                    gpu_alias,
        "num_gpus":                    1,
        "vram_gb":                     None,
        "framework":                   raw_framework,
        "system_type":                 "datacenter",
        "hw_status":                   "available",
        "benchmark":                   f"{model_name}-{tier}",
        "benchmark_base":              model_name,
        "benchmark_accuracy_tier":     tier,
        "scenario":                    scenario,
        "precision":                   None,
        "tokens_per_sample":           294,
        "throughput_tokens_per_sec":   throughput,
        "throughput_tok_per_sec_per_gpu": throughput,
        "result_valid":                True,
        "throughput_samples_per_sec":  throughput / 294,
        "latency_mean_ms":             None,
        "latency_p99_ms":              None,
        "ttft_mean_ms":                None,
        "ttft_p99_ms":                 None,
        "tpot_mean_ms":                None,
        "tpot_p99_ms":                 None,
        "log_path":                    "fake/path",
    }])


# MoE param split — bandwidth ceiling uses total, compute uses active

@pytest.mark.gate
class TestMoEParamSplitGate:
    """GATE: MoE bandwidth ceiling must use total_params_b, compute ceiling must use compute_params_b — swapping these args in roofline_ceilings() would overestimate mixtral-8x7b compute throughput 3.31x (total=46.7B vs active=14.1B) and underestimate bandwidth by the same factor."""

    @pytest.fixture(scope="class")
    def mi300x_spec(self) -> dict:
        return {s["id"]: s for s in load_specs()}["mi300x"]

    def test_mixtral_total_ne_active_params(self) -> None:
        total_b, active_b = MODEL_PARAMS["mixtral-8x7b"]
        assert total_b != active_b, "mixtral-8x7b must have distinct total and active params"
        assert total_b == pytest.approx(46.7, abs=0.1)
        assert active_b == pytest.approx(14.1, abs=0.1)

    def test_bw_ceiling_uses_total_params(self, mi300x_spec: dict) -> None:
        """Back-calculate which param count produced the BW ceiling in the serving path."""
        total_b, active_b = MODEL_PARAMS["mixtral-8x7b"]
        precision = TIER_TO_PRECISION["99"]   # fp8
        bpp = BYTES_PER_PARAM[precision]
        hbm_bw = mi300x_spec["hbm_bandwidth_tbps"]

        feats, _, _ = _build_feature_vector(
            gpu_spec=mi300x_spec,
            model_name="mixtral-8x7b",
            scenario="Offline",
            accuracy_tier="99",
            framework="vllm",
        )
        bw_ceil = feats[FEATURE_COLS.index("bandwidth_ceiling_tok_per_sec")]

        # bw_ceil = (hbm_bw * 1e12) / (params * 1e9 * bpp)  ⟹  params = …
        implied_params = (hbm_bw * 1e12) / (bw_ceil * 1e9 * bpp)
        assert implied_params == pytest.approx(total_b, rel=1e-3), (
            f"BW ceiling implies {implied_params:.2f}B params in model footprint; "
            f"expected total_params_b={total_b}B.  "
            f"Compute ceiling likely received total params instead of active params."
        )

    def test_compute_ceiling_uses_active_params(self, mi300x_spec: dict) -> None:
        """Back-calculate which param count produced the compute ceiling in the serving path."""
        total_b, active_b = MODEL_PARAMS["mixtral-8x7b"]
        precision = TIER_TO_PRECISION["99"]
        pt = mi300x_spec.get("peak_tflops") or {}
        peak_tflops = pt.get(precision) or pt.get("fp16")

        feats, _, _ = _build_feature_vector(
            gpu_spec=mi300x_spec,
            model_name="mixtral-8x7b",
            scenario="Offline",
            accuracy_tier="99",
            framework="vllm",
        )
        compute_ceil = feats[FEATURE_COLS.index("compute_ceiling_tok_per_sec")]

        # compute_ceil = (peak_tflops * 1e12) / (2 * params * 1e9)  ⟹  params = …
        implied_params = (peak_tflops * 1e12) / (2.0 * compute_ceil * 1e9)
        assert implied_params == pytest.approx(active_b, rel=1e-3), (
            f"Compute ceiling implies {implied_params:.2f}B active params; "
            f"expected compute_params_b={active_b}B.  "
            f"Bandwidth ceiling likely received active params instead of total params."
        )


# Memory-fit verdict enforcement — does_not_fit excluded, tight kept

@pytest.mark.gate
class TestMemoryFitVerdictEnforcementGate:
    """GATE: recommend()'s hard memory-fit exclusion must track the tri-state verdict exactly — every 'does_not_fit' lands in `filtered` (never frontier/dominated) and 'tight'/'fits' are never silently dropped; drift (e.g. reusing the old boolean vram_fits check) would recommend an unrunnable config or wrongly exclude a runnable one."""

    @pytest.fixture(scope="class")
    def recommender(self) -> GpuRecommender:
        return GpuRecommender(GpuPredictor())

    def test_sweep_all_models_verdict_matches_bucket(self, recommender) -> None:
        checked_any = False
        for model_name in MODEL_PARAMS:
            for tier in get_args(AccuracyTier):
                # No budget/throughput constraint is passed anywhere in this sweep, so every filtered entry can only be a memory-fit rejection; budget/throughput filtering is covered separately in test_recommender.py.
                result = recommender.recommend(model_name=model_name, accuracy_tier=tier)
                for f in result["filtered"]:
                    checked_any = True
                    assert f["memory_fit_verdict"] == "does_not_fit", (
                        f"{model_name}/{tier}/{f['gpu_id']} is filtered with verdict "
                        f"{f['memory_fit_verdict']!r}, but this sweep sets no budget/throughput "
                        "constraint, so the only valid filter reason is does_not_fit"
                    )
                for c in result["frontier"] + result["dominated"]:
                    checked_any = True
                    assert c["memory_fit_verdict"] != "does_not_fit", (
                        f"{model_name}/{tier}/{c['gpu_id']} is a candidate but "
                        f"verdict is does_not_fit — should have been excluded"
                    )
        assert checked_any, "no result entries were produced to check"


# Precision-support enforcement — never silently substituted

@pytest.mark.gate
class TestPrecisionSupportEnforcementGate:
    """GATE: predict()/predict_batch()/recommend() must never return a normal-looking result for a (GPU, tier) whose selected precision has no native peak_tflops entry — before this existed, a100_sxm_80gb@99 (fp8, unsupported on Ampere) silently substituted fp16 TFLOPS for the roofline ceiling while still using fp8 bytes-per-param, surfacing as a real "dominated" recommend() candidate with no error or flag."""

    @pytest.fixture(scope="class")
    def predictor(self) -> GpuPredictor:
        return GpuPredictor()

    @pytest.fixture(scope="class")
    def recommender(self, predictor) -> GpuRecommender:
        return GpuRecommender(predictor)

    def test_predict_raises_iff_precision_unsupported(self, predictor) -> None:
        specs = {s["id"]: s for s in load_specs() if s.get("in_model_scope")}
        checked_any = False
        checked_unsupported = False
        for gpu_id, spec in specs.items():
            for tier in get_args(AccuracyTier):
                selected = _selected_precision(spec, tier)
                supported = gpu_supports_precision(spec, selected)
                checked_any = True
                if supported:
                    result = predictor.predict(
                        gpu_id=gpu_id, model_name="gptj", accuracy_tier=tier
                    )
                    assert result["pred_throughput_tok_per_sec"] >= 0.0, (
                        f"{gpu_id}/{tier} (selected={selected}, supported) "
                        "should predict normally"
                    )
                else:
                    checked_unsupported = True
                    with pytest.raises(ValueError, match="does not support precision"):
                        predictor.predict(
                            gpu_id=gpu_id, model_name="gptj", accuracy_tier=tier
                        )
        assert checked_any, "no (gpu, tier) combos were checked"
        assert checked_unsupported, (
            "no unsupported (gpu, tier) combo was exercised — the gate "
            "would pass vacuously if gpu_specs.yaml ever dropped its one "
            "unsupported-precision case (a100_sxm_80gb.peak_tflops.fp8)"
        )

    def test_recommend_never_returns_unsupported_precision_as_candidate(
        self, recommender
    ) -> None:
        specs = {s["id"]: s for s in load_specs() if s.get("in_model_scope")}
        checked_any = False
        checked_unsupported = False
        for model_name in MODEL_PARAMS:
            for tier in get_args(AccuracyTier):
                result = recommender.recommend(model_name=model_name, accuracy_tier=tier)
                for c in result["frontier"] + result["dominated"]:
                    checked_any = True
                    spec = specs[c["gpu_id"]]
                    selected = _selected_precision(spec, tier)
                    assert gpu_supports_precision(spec, selected), (
                        f"{model_name}/{tier}/{c['gpu_id']} is a candidate "
                        f"but does not support its selected precision {selected!r}"
                    )
                for f in result["filtered"]:
                    checked_any = True
                    spec = specs[f["gpu_id"]]
                    selected = _selected_precision(spec, tier)
                    if not gpu_supports_precision(spec, selected):
                        checked_unsupported = True
                        assert selected in f["reject_reason"], (
                            f"{model_name}/{tier}/{f['gpu_id']} filtered for "
                            f"an unsupported precision, but reject_reason "
                            f"{f['reject_reason']!r} doesn't name it"
                        )
        assert checked_any, "no result entries were produced to check"
        assert checked_unsupported, (
            "no unsupported-precision filtered entry was exercised across "
            "the full model x tier sweep"
        )


# Schema/predictor agreement — Literal types match VALID_* frozensets

@pytest.mark.gate
class TestSchemaPredictorAgreementGate:
    """GATE: Pydantic Literal types in src/api/schemas.py must exactly match the independently-defined VALID_* frozensets in src/models/predictor.py — schemas accepting more causes a 500 at the predictor, schemas accepting less makes a valid predictor call unreachable via the API."""

    def test_scenario_literals_match_valid_scenarios(self) -> None:
        assert set(get_args(Scenario)) == VALID_SCENARIOS, (
            f"schemas.Scenario={set(get_args(Scenario))} "
            f"!= predictor.VALID_SCENARIOS={VALID_SCENARIOS}"
        )

    def test_accuracy_tier_literals_match_valid_tiers(self) -> None:
        assert set(get_args(AccuracyTier)) == VALID_TIERS, (
            f"schemas.AccuracyTier={set(get_args(AccuracyTier))} "
            f"!= predictor.VALID_TIERS={VALID_TIERS}"
        )

    def test_framework_literals_match_valid_frameworks(self) -> None:
        assert set(get_args(Framework)) == VALID_FRAMEWORKS, (
            f"schemas.Framework={set(get_args(Framework))} "
            f"!= predictor.VALID_FRAMEWORKS={VALID_FRAMEWORKS}"
        )


# has_training_data completeness across every recommend() code path

@pytest.mark.gate
class TestHasTrainingDataCompletenessGate:
    """GATE: every dict recommend() returns (frontier/dominated/filtered) must carry 'has_training_data' and 'training_data_tier' regardless of which of the three independent code paths built it (predict_batch-derived, VRAM-fail reject, precision-fail reject) — a future path that forgets either field would pass unit tests then 500 at request time on real traffic; both fields are checked together since they're added/read at the same call sites."""

    @pytest.fixture(scope="class")
    def predictor(self) -> GpuPredictor:
        return GpuPredictor()

    @pytest.fixture(scope="class")
    def recommender(self, predictor: GpuPredictor) -> GpuRecommender:
        return GpuRecommender(predictor)

    def test_zero_row_gpus_flagged_false(self, predictor: GpuPredictor) -> None:
        # a100_sxm_80gb/l4/rtx4090 are in_model_scope with zero training rows — a regression here means an unvalidated prediction is silently presented with the same confidence as a real one.
        for gpu_id in ("a100_sxm_80gb", "l4", "rtx4090"):
            assert predictor.has_training_data(gpu_id) is False, (
                f"{gpu_id} unexpectedly has training data — "
                "if this is intentional (new calibration data added), "
                "update this gate and the disclosure copy in the UI/README."
            )
            assert predictor.training_data_tier(gpu_id) == "none", (
                f"{gpu_id} unexpectedly has a non-'none' training_data_tier."
            )

    def test_amd_gpus_below_the_reliability_floor_flagged(self, predictor: GpuPredictor) -> None:
        # mi300x/mi325x/mi355x have real rows but fewer than the 100-row-per-GPU floor — the exact case a plain boolean silently conflated with "sufficient" until this gate/field existed.
        for gpu_id in ("mi300x", "mi325x", "mi355x"):
            assert predictor.has_training_data(gpu_id) is True, (
                f"{gpu_id} unexpectedly has zero training rows — "
                "if the corpus shrank, update this gate."
            )
            assert predictor.training_data_tier(gpu_id) == "below_floor", (
                f"{gpu_id}'s training_data_tier is no longer 'below_floor' — "
                "if this GPU's row count crossed 100 (e.g. new calibration data), "
                "that's a real improvement — update this gate and the disclosure "
                "copy in the UI/README rather than silently letting it drift."
            )

    def test_every_result_dict_carries_both_fields(self, recommender: GpuRecommender) -> None:
        # Exercises all three branches in one pass: llama3.1-405b guarantees VRAM-fail entries (nothing fits 405B); a $1/hr budget guarantees budget-reject entries plus a populated frontier/dominated.
        combos = [
            dict(model_name="llama3.1-405b", accuracy_tier="99.9"),
            dict(model_name="gptj", accuracy_tier="99", budget_per_gpu_hr=1.0),
        ]
        checked_any = False
        for kwargs in combos:
            result = recommender.recommend(**kwargs)
            for bucket in ("frontier", "dominated", "filtered"):
                for entry in result[bucket]:
                    checked_any = True
                    assert "has_training_data" in entry, (
                        f"{bucket} entry for {entry.get('gpu_id')} "
                        f"(query {kwargs}) is missing has_training_data — "
                        "this would fail Pydantic response validation at "
                        "request time, not at test time."
                    )
                    assert "training_data_tier" in entry, (
                        f"{bucket} entry for {entry.get('gpu_id')} "
                        f"(query {kwargs}) is missing training_data_tier — "
                        "this would fail Pydantic response validation at "
                        "request time, not at test time."
                    )
                    assert entry["training_data_tier"] in ("none", "below_floor", "sufficient"), (
                        f"{bucket} entry for {entry.get('gpu_id')} has an "
                        f"invalid training_data_tier: {entry['training_data_tier']!r}"
                    )
        assert checked_any, (
            "no result entries were produced to check — queries need adjusting"
        )


# Feature vector parity — training path vs serving path

@pytest.mark.gate
class TestFeatureVectorParityGate:
    """GATE: all 20 FEATURE_COLS values match between build_training_df (training path) and _build_feature_vector (serving path) — 13 continuous features here, 6 categorical in test_predictor.py::TestBuildFeatureVector, and mlperf_round_num intentionally excluded (training uses the real round ordinal, serving always uses the latest) and verified separately there."""

    # (gpu_alias, gpu_id, model_name, scenario, tier, raw_framework); gpu_alias must match a known alias in data/gpu_specs.yaml, raw_framework is a realistic MLPerf string that _normalize_framework maps to the family label _build_feature_vector expects.
    CASES = [
        # NVIDIA Hopper, FP8 tier (99 -> fp8, falls back for GPUs without fp8)
        (
            "NVIDIA H200-SXM-141GB",          "h200_sxm",  "llama2-70b",
            "Offline", "99",   "TensorRT 10.2.0",
        ),
        # AMD CDNA3, FP8 tier (99.9 -> fp8 for AMD) — verifies both paths apply the vendor override consistently.
        (
            "AMD Instinct MI300X 192GB HBM3",  "mi300x",    "gptj",
            "Server",  "99.9", "vLLM 0.4.3+rocm614",
        ),
        # AMD CDNA3, BF16 tier (base -> bf16); MoE model (active != total params)
        (
            "AMD Instinct MI325X",             "mi325x",    "mixtral-8x7b",
            "Offline", "99",   "ROCm 7.0",
        ),
        # AMD CDNA4 — only GPU where amd_arch_gen=2 and is_cdna4=1; a CDNA4 arch-ordinal encoding divergence can't be detected by the 3 CDNA3/NVIDIA cases above.
        (
            "AMD Instinct MI355X 288GB HBM3e", "mi355x",    "llama2-70b",
            "Offline", "99",   "ROCm 7.0",
        ),
        # NVIDIA at tier 99.9 selects fp16 TFLOPS (not fp8, no AMD override) — without this case a training/serving divergence in NVIDIA fp16 TFLOPS selection would pass undetected (all other cases are tier 99 or AMD 99.9, both fp8).
        (
            "NVIDIA H100-SXM-80GB",            "h100_sxm", "gptj",
            "Server",  "99.9", "TensorRT 10.4.0",
        ),
    ]

    @pytest.fixture(scope="class")
    def spec_map(self) -> dict:
        return {s["id"]: s for s in load_specs()}

    @pytest.mark.parametrize(
        "gpu_alias, gpu_id, model_name, scenario, tier, raw_fw",
        CASES,
    )
    def test_continuous_features_match(
        self, spec_map, gpu_alias, gpu_id, model_name, scenario, tier, raw_fw,
    ):
        """GATE: every continuous FEATURE_COL has the same value in both paths — a mismatch means the model evaluates features at serving time that it was never trained on."""
        # ── Training path ─────────────────────────────────────────────────
        raw_df = _make_single_row(gpu_alias, model_name, scenario, tier, raw_fw)
        feat_df = build_training_df(raw_df)
        assert len(feat_df) == 1, (
            f"build_training_df dropped the row — check gpu_alias={gpu_alias!r} "
            f"and benchmark_base={model_name!r}"
        )
        train_row = feat_df.iloc[0]

        # ── Serving path ──────────────────────────────────────────────────
        fw_family = _normalize_framework(raw_fw)
        serving_feats, _, _ = _build_feature_vector(
            gpu_spec=spec_map[gpu_id],
            model_name=model_name,
            scenario=scenario,
            accuracy_tier=tier,
            framework=fw_family,
        )

        # ── Compare continuous columns — mlperf_round_num excluded, its training/serving divergence is intentional, see class docstring.
        continuous_cols = [
            c for c in FEATURE_COLS
            if c not in {
                "scenario_offline", "is_base_tier",
                "fw_tensorrt", "fw_vllm", "fw_rocm_other", "is_cdna4",
                "mlperf_round_num",
            }
        ]
        mismatches = []
        for col in continuous_cols:
            idx = FEATURE_COLS.index(col)
            sv = float(serving_feats[idx])
            tv_raw = train_row[col]
            tv = float(tv_raw) if not pd.isna(tv_raw) else float("nan")

            if not _nan_match(sv, tv):
                mismatches.append(
                    f"  {col:<40} serving={sv:.6g}  training={tv:.6g}"
                )

        assert not mismatches, (
            f"feature mismatch for gpu={gpu_id!r} model={model_name!r} "
            f"scenario={scenario!r} tier={tier!r}:\n"
            + "\n".join(mismatches)
        )


# Model artifact total disk size < 50 MB

_MODEL_DIR = Path("data/models")
_DISK_LIMIT_MB = 50


@pytest.mark.gate
class TestModelDiskSizeGate:
    """GATE: data/models/ total size is < 50 MB — the deployment budget for free-tier Hugging Face Spaces (shared storage, slow cold-start I/O)."""

    def test_model_dir_exists(self) -> None:
        assert _MODEL_DIR.is_dir(), (
            f"{_MODEL_DIR} not found — run train_final.py to generate artifacts"
        )

    def test_total_size_under_limit(self) -> None:
        total_bytes = sum(f.stat().st_size for f in _MODEL_DIR.rglob("*") if f.is_file())
        total_mb = total_bytes / (1024 ** 2)
        assert total_mb < _DISK_LIMIT_MB, (
            f"data/models/ is {total_mb:.1f} MB — exceeds {_DISK_LIMIT_MB} MB limit. "
            "Trim model artifacts before deploying to HF Spaces."
        )

    def test_required_artifacts_present(self) -> None:
        required = ["prophet_v1.json", "feature_metadata.json"]
        for name in required:
            assert (_MODEL_DIR / name).exists(), (
                f"required artifact {name} is missing from {_MODEL_DIR}"
            )


# Pricing coverage — all in-scope GPUs have pricing entries

_PRICING_PATH = Path("data/pricing.yaml")


@pytest.mark.gate
class TestPricingCoverageGate:
    """GATE: every in_model_scope=True GPU has a data/pricing.yaml entry — GpuRecommender.__init__()'s ValueError for missing pricing is only visible at startup, and a gap would otherwise cause cost_efficiency=None, raising TypeError in _pareto_frontier's sort and silently breaking all recommendation responses."""

    def test_pricing_file_exists(self) -> None:
        assert _PRICING_PATH.is_file(), (
            f"{_PRICING_PATH} not found — create data/pricing.yaml before deployment"
        )

    def test_all_in_scope_gpus_have_pricing(self) -> None:
        with _PRICING_PATH.open() as f:
            pricing: dict = yaml.safe_load(f).get("pricing", {})
        in_scope_ids = [s["id"] for s in load_specs() if s.get("in_model_scope")]
        assert in_scope_ids, (
            "no in-scope GPUs found — check gpu_specs.yaml in_model_scope flags"
        )
        missing = [gid for gid in in_scope_ids if gid not in pricing]
        assert not missing, (
            f"pricing.yaml missing entries for in-scope GPUs: {missing}. "
            "GpuRecommender.__init__() will raise ValueError at startup."
        )


# AMD FP8 override — three-way consistency gate

@pytest.mark.gate
class TestAmdFp8OverrideGate:
    """GATE: the AMD 99.9-tier FP8 override must apply consistently across build_training_df (training), _build_feature_vector (serving), and _gpu_model_size_gb (VRAM pre-filter) — a regression skews MI355X training efficiency_ratio or wrongly rejects MI300X for llama2-70b@99.9 (140 GB FP16 vs 192 GB VRAM) despite fitting as 70 GB FP8; verified end-to-end via vram_headroom ≈0.635 (FP8) vs ≈0.271 (FP16, drifted)."""

    @pytest.fixture(scope="class")
    def recommender(self) -> GpuRecommender:
        return GpuRecommender(GpuPredictor())

    def test_mi300x_vram_headroom_uses_fp8_not_fp16(self, recommender) -> None:
        # Minimal batch/context override isolates the FP8-vs-FP16 weight comparison this gate targets from the KV-cache term (covered separately by test_recommender.py).
        result = recommender.recommend(
            model_name="llama2-70b", accuracy_tier="99.9",
            batch_size=1, input_tokens=64, output_tokens=1,
        )
        candidates = result["frontier"] + result["dominated"]
        mi300x = next((r for r in candidates if r["gpu_id"] == "mi300x"), None)
        assert mi300x is not None, (
            "MI300X absent from llama2-70b tier-99.9 candidates — "
            "VRAM filter may be applying FP16 model size (140 GB) instead of FP8 (70 GB), "
            "causing MI300X (192 GB) to be incorrectly rejected."
        )
        fp8_model_gb = 70.0  # 70B params × 1 byte/param (FP8)
        expected_headroom = (192.0 - fp8_model_gb * 1.10) / 192.0   # + 10% overhead, ≈ 0.599
        assert mi300x["vram_headroom"] == pytest.approx(expected_headroom, abs=0.01), (
            f"MI300X vram_headroom={mi300x['vram_headroom']:.3f}, "
            f"expected {expected_headroom:.3f} (FP8 70 GB model). "
            "A value near 0.271 indicates FP16 (140 GB) is being used instead."
        )


# Framework normalization — pattern outputs subset of VALID_FRAMEWORKS

@pytest.mark.gate
class TestFrameworkNormalizationGate:
    """GATE: every _FRAMEWORK_PATTERNS label (and the "other" fallback) is in VALID_FRAMEWORKS — a pattern added with a label outside VALID_FRAMEWORKS would enter training data but never be requestable via the API, causing silent 500s (predictor._validate() rejecting an apparently valid input)."""

    def test_all_pattern_outputs_in_valid_frameworks(self) -> None:
        labels_from_patterns = {label for _, label in _FRAMEWORK_PATTERNS}
        unknown_labels = labels_from_patterns - VALID_FRAMEWORKS
        assert not unknown_labels, (
            f"_FRAMEWORK_PATTERNS returns labels not in VALID_FRAMEWORKS: "
            f"{unknown_labels}. Add to predictor.VALID_FRAMEWORKS or remove from patterns."
        )

    def test_fallback_other_in_valid_frameworks(self) -> None:
        fallback = _normalize_framework("some_completely_unknown_framework_xyz")
        assert fallback == "other", (
            f"no-match fallback should be 'other', got {fallback!r}. "
            "The fallback must be in VALID_FRAMEWORKS so unrecognized frameworks "
            "can be requested via the API without a 422 validation error."
        )
        assert "other" in VALID_FRAMEWORKS, (
            "'other' is _normalize_framework's no-match return value "
            "but is not in VALID_FRAMEWORKS — unrecognized frameworks fail predictor._validate()."
        )


# Notebook FEATURE_COLS parity — training/serving vs. analysis notebooks

_NOTEBOOKS_DIR = Path("notebooks")
_FEATURE_COLS_NOTEBOOKS: tuple[str, ...] = (
    "03_model_training.ipynb",
    "04_top1_benchmark.ipynb",
    "05_mr008_ablation.ipynb",
)

# In-place mutations of FEATURE_COLS (or a list it's built from) can't be verified statically, so the value this module resolves may not match what the notebook computes at runtime.
_MUTATING_LIST_METHODS = frozenset({
    "append", "extend", "insert", "remove", "pop", "clear", "sort", "reverse",
})


def _parse_notebook_feature_cols(notebook_path: Path) -> list[str]:
    """Extract the notebook's actual FEATURE_COLS = X + Y + ... expression via AST (without executing the notebook) — order matters, not just the set, since XGBoost's colsample_bytree does positional random feature subsampling."""
    nb = json.loads(notebook_path.read_text())
    code = "\n".join(
        "".join(cell["source"])
        for cell in nb["cells"]
        if cell["cell_type"] == "code"
    )
    tree = ast.parse(code)

    list_literals: dict[str, list[str]] = {}
    feature_cols_expr: ast.expr | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if not isinstance(target, ast.Name):
                    continue
                if isinstance(node.value, ast.List):
                    list_literals[target.id] = [
                        elt.value for elt in node.value.elts
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                    ]
                elif target.id == "FEATURE_COLS":
                    feature_cols_expr = node.value

    if feature_cols_expr is None:
        raise RuntimeError(f"FEATURE_COLS assignment not found in {notebook_path}")

    # Names this resolution depends on, populated by _resolve below; FEATURE_COLS is always included so a bare `FEATURE_COLS += [...]` after the found assignment is still caught.
    referenced_names: set[str] = {"FEATURE_COLS"}

    def _resolve(expr: ast.expr) -> list[str]:
        if isinstance(expr, ast.Name):
            referenced_names.add(expr.id)
            if expr.id not in list_literals:
                raise RuntimeError(f"{expr.id} list literal not found in {notebook_path}")
            return list_literals[expr.id]
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
            return _resolve(expr.left) + _resolve(expr.right)
        raise RuntimeError(
            f"Unsupported FEATURE_COLS expression in {notebook_path}: {ast.dump(expr)}"
        )

    result = _resolve(feature_cols_expr)

    # The scan above only sees each name's initial literal assignment, not later in-place mutation (`+=`, `.append(...)`, etc) — since that's exactly the drift this gate exists to catch, fail loudly rather than silently trust a possibly-stale value.
    for node in ast.walk(tree):
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            if node.target.id in referenced_names:
                raise RuntimeError(
                    f"{notebook_path} mutates {node.target.id!r} with '+=' — "
                    "static AST parsing can't verify the resulting FEATURE_COLS. "
                    "Rewrite as a single 'FEATURE_COLS = A + B + ...' assignment."
                )
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _MUTATING_LIST_METHODS
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in referenced_names
        ):
            raise RuntimeError(
                f"{notebook_path} calls "
                f"{node.func.value.id}.{node.func.attr}(...) — static AST parsing "
                "can't verify the resulting FEATURE_COLS. Rewrite as a single "
                "'FEATURE_COLS = A + B + ...' assignment."
            )

    return result


@pytest.mark.gate
class TestNotebookFeatureColsParityGate:
    """GATE: every notebook manually reconstructing the feature list must match predictor.FEATURE_COLS exactly, in order — TestFeatureVectorParityGate protects the two source-code copies, but only this gate catches the two notebooks silently drifting out of sync after a FEATURE_COLS change (otherwise only caught by manually re-running them)."""

    @pytest.mark.parametrize("notebook_name", _FEATURE_COLS_NOTEBOOKS)
    def test_notebook_feature_cols_match_predictor(self, notebook_name: str) -> None:
        notebook_path = _NOTEBOOKS_DIR / notebook_name
        assert notebook_path.is_file(), f"{notebook_path} not found"
        notebook_cols = _parse_notebook_feature_cols(notebook_path)
        assert notebook_cols == list(FEATURE_COLS), (
            f"{notebook_name} FEATURE_COLS diverges from predictor.FEATURE_COLS.\n"
            f"  notebook:  {notebook_cols}\n"
            f"  predictor: {list(FEATURE_COLS)}"
        )


# memory_fit_verdict() output subset of VALID_MEMORY_FIT_VERDICTS

@pytest.mark.gate
class TestMemoryFitVerdictLabelsGate:
    """GATE: memory_fit_verdict() only ever returns a value from VALID_MEMORY_FIT_VERDICTS, and schemas.MemoryFitVerdict matches that set exactly — mirrors the framework-label closed-set gate; an unlisted fourth verdict from a future threshold-logic change would otherwise silently 500 at request time via Pydantic validation instead of failing a test."""

    @pytest.mark.parametrize("utilization_target", [0.0, 0.50, 0.90, 0.9001, 0.98, 0.9801, 5.0])
    def test_verdict_always_in_valid_set(self, utilization_target: float) -> None:
        vram_gb = 100.0
        weights_gb = utilization_target * vram_gb / MEMORY_OVERHEAD_FACTOR
        verdict, _, _ = memory_fit_verdict(weights_gb, 0.0, vram_gb)
        assert verdict in VALID_MEMORY_FIT_VERDICTS, (
            f"memory_fit_verdict() returned {verdict!r}, "
            f"not in VALID_MEMORY_FIT_VERDICTS={VALID_MEMORY_FIT_VERDICTS}"
        )

    def test_schema_literal_matches_valid_set(self) -> None:
        assert set(get_args(MemoryFitVerdict)) == VALID_MEMORY_FIT_VERDICTS, (
            f"schemas.MemoryFitVerdict={set(get_args(MemoryFitVerdict))} "
            f"!= build_features.VALID_MEMORY_FIT_VERDICTS={VALID_MEMORY_FIT_VERDICTS}"
        )


# RecommendRequest.ranking_objective subset of VALID_RANKING_OBJECTIVES

@pytest.mark.gate
class TestRankingObjectiveLabelsGate:
    """GATE: schemas.RankingObjective matches recommender.VALID_RANKING_OBJECTIVES exactly, and _RANKING_FIELDS covers every member — mirrors the memory-fit-verdict/framework-label closed-set gates; a future typo in either set would silently diverge (accepted by one, KeyError/422 from the other) with no test failing."""

    def test_schema_literal_matches_valid_set(self) -> None:
        assert set(get_args(RankingObjective)) == VALID_RANKING_OBJECTIVES, (
            f"schemas.RankingObjective={set(get_args(RankingObjective))} "
            f"!= recommender.VALID_RANKING_OBJECTIVES={VALID_RANKING_OBJECTIVES}"
        )

    def test_every_objective_is_a_real_recommender_ranking_field(self) -> None:
        assert set(_RANKING_FIELDS) == VALID_RANKING_OBJECTIVES, (
            f"recommender._RANKING_FIELDS keys={set(_RANKING_FIELDS)} "
            f"!= VALID_RANKING_OBJECTIVES={VALID_RANKING_OBJECTIVES}"
        )

    @pytest.mark.parametrize("objective", sorted(VALID_RANKING_OBJECTIVES))
    def test_every_objective_accepted_by_recommend(self, objective: str) -> None:
        # Full round-trip through the public entry point, not just the internal frozenset — catches a rejection at the recommend() validation layer even if the frozenset/Literal agree.
        predictor = GpuPredictor()
        recommender = GpuRecommender(predictor)
        result = recommender.recommend(model_name="gptj", ranking_objective=objective)
        assert result["workload"]["ranking_objective"] == objective


# Pareto objective vector uses (throughput, price, watts), not the earlier (throughput, cost_efficiency, vram_headroom)

@pytest.mark.gate
class TestParetoObjectiveVectorGate:
    """GATE: _pareto_frontier()'s dominance check must read watts/price_per_gpu_hr directly — a pair differing only in watts must change the outcome, and one differing only in vram_headroom must NOT, proving the spec's (throughput, usd_per_hour, watts) vector, guarding against silently backsliding to the old (throughput, cost_efficiency, vram_headroom) vector."""

    @staticmethod
    def _cand(
        gpu_id: str, throughput: float, price: float, watts: float, vram_headroom: float,
    ) -> dict:
        return {
            "gpu_id": gpu_id,
            "throughput": throughput,
            "price_per_gpu_hr": price,
            "watts": watts,
            "vram_headroom": vram_headroom,
            "cost_efficiency": throughput / price,
            "tokens_per_watt": throughput / watts,
            "cost_per_million_tokens": cost_per_million_tokens(price, throughput),
        }

    def test_watts_affects_dominance(self) -> None:
        # Identical throughput/price/vram_headroom; B draws less power than A.
        a = self._cand("a", 1000.0, 2.0, 1000.0, 0.5)
        b = self._cand("b", 1000.0, 2.0, 100.0, 0.5)
        frontier, dominated = _pareto_frontier([a, b])
        assert [c["gpu_id"] for c in frontier] == ["b"], (
            "lower watts at identical throughput/price must dominate — "
            "watts is not affecting Pareto dominance"
        )
        assert [c["gpu_id"] for c in dominated] == ["a"]

    def test_vram_headroom_does_not_affect_dominance(self) -> None:
        # Identical throughput/price/watts, only vram_headroom differs — the objective vector has no room for a fourth soft axis, so neither may dominate on vram_headroom alone.
        a = self._cand("a", 1000.0, 2.0, 500.0, 0.9)
        b = self._cand("b", 1000.0, 2.0, 500.0, 0.1)
        frontier, dominated = _pareto_frontier([a, b])
        assert {c["gpu_id"] for c in frontier} == {"a", "b"}, (
            "vram_headroom must not affect Pareto dominance — "
            f"got frontier={[c['gpu_id'] for c in frontier]}, "
            f"dominated={[c['gpu_id'] for c in dominated]}"
        )
        assert dominated == []


# Calibration runner/merge script CSV compatibility

_RUNNER_PATH = Path("benchmarks/run_mi300x_calibration.py")

# Fields merge_calibration_rows.py accesses directly from CSV rows (r["field"] or r.get(...)); update whenever the merge script adds a new field access.
_MERGE_REQUIRED_FIELDS: frozenset[str] = frozenset({
    "round_tag",
    "gpu_name",
    "benchmark_base",
    "benchmark_accuracy_tier",
    "scenario",
    "throughput_tok_per_sec",
    "vllm_version",
    # DataFrame subscript access (merge_calibration_rows.py L64) is not a .get() — raises KeyError if the column is absent from the CSV.
    "precision_used",
})


def _parse_runner_csv_fields() -> frozenset[str]:
    """Extract CSV_FIELDS list from runner script via AST — avoids importing the script directly, which would fail without vllm installed."""
    tree = ast.parse(_RUNNER_PATH.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "CSV_FIELDS":
                    if isinstance(node.value, ast.List):
                        return frozenset(
                            elt.value
                            for elt in node.value.elts
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                        )
    raise RuntimeError("CSV_FIELDS constant not found in benchmarks/run_mi300x_calibration.py")


@pytest.mark.gate
class TestRunnerMergeCompatibilityGate:
    """GATE: run_mi300x_calibration.py's CSV_FIELDS is a superset of the columns merge_calibration_rows.py reads — drift causes KeyError (or silent None) when processing real AMD Dev Cloud results after the 25-hr credit clock is already spent, so this gate runs locally before the timed session starts."""

    def test_runner_script_exists(self) -> None:
        assert _RUNNER_PATH.is_file(), (
            f"{_RUNNER_PATH} not found — calibration runner script is missing"
        )

    def test_runner_csv_fields_cover_merge_required(self) -> None:
        runner_fields = _parse_runner_csv_fields()
        missing = _MERGE_REQUIRED_FIELDS - runner_fields
        assert not missing, (
            f"merge_calibration_rows.py accesses fields not in runner CSV_FIELDS: "
            f"{missing}. Add these to CSV_FIELDS in run_mi300x_calibration.py, or "
            "remove the access from merge_calibration_rows.py."
        )


@pytest.mark.gate
class TestDataManifestLockFreshnessGate:
    """GATE: data_manifest.lock's committed calibration-CSV entry matches the CSV's real SHA-256 — the only manifest entry checkable in CI this way, since the gitignored MLPerf mirrors need a live local checkout (checked instead by train_final.py's warn-only verify_manifest()); this gate catches the one drift class reproducible from a fresh clone: the lock going stale against its own tracked sibling."""

    def test_manifest_lock_file_exists(self) -> None:
        assert manifest.MANIFEST_PATH.is_file(), (
            f"{manifest.MANIFEST_PATH} not found — run "
            "`python -m src.data.manifest` to generate it."
        )

    def test_calibration_entry_matches_tracked_csv(self) -> None:
        locked = manifest.load_manifest()
        assert locked is not None, "data_manifest.lock failed to load"
        entry = locked["sources"].get("mi300x_calibration")
        assert entry is not None, "data_manifest.lock missing 'mi300x_calibration' entry"
        current_sha256 = manifest._file_sha256(manifest.CALIBRATION_CSV)
        assert entry["sha256"] == current_sha256, (
            "data_manifest.lock's mi300x_calibration sha256 is stale — "
            "the tracked CSV changed since the lock file was last regenerated. "
            "Run `python -m src.data.manifest` and commit the refreshed lock file."
        )


# feature_metadata.json carries a well-formed corpus_sha256

_FEATURE_METADATA_PATH = Path("data/models/feature_metadata.json")
_TRAINING_CORPUS_PATH = Path("data/processed/mlperf_features.parquet")
_SHA256_HEX_CHARS = frozenset("0123456789abcdef")


@pytest.mark.gate
class TestCorpusShaSidecarGate:
    """GATE: feature_metadata.json records a well-formed corpus_sha256, completing the lineage chain served prediction -> feature_metadata.json -> data_manifest.lock -> pinned MLPerf commits; the field-presence/shape check always runs (metadata is committed), but the byte-for-byte match against the live corpus is best-effort since data/processed/ is gitignored and usually absent in CI."""

    def test_feature_metadata_exists(self) -> None:
        assert _FEATURE_METADATA_PATH.is_file(), (
            f"{_FEATURE_METADATA_PATH} not found — run train_final.py "
            "to generate artifacts"
        )

    def test_corpus_sha256_present_and_well_formed(self) -> None:
        meta = json.loads(_FEATURE_METADATA_PATH.read_text())
        corpus_hash = meta.get("corpus_sha256")
        assert corpus_hash, (
            "feature_metadata.json missing 'corpus_sha256' — retrain "
            "with `python -m src.models.train_final` to populate it."
        )
        assert len(corpus_hash) == 64 and set(corpus_hash) <= _SHA256_HEX_CHARS, (
            f"corpus_sha256 {corpus_hash!r} is not a well-formed SHA-256 hex digest"
        )

    def test_corpus_sha256_matches_live_corpus_when_present(self) -> None:
        if not _TRAINING_CORPUS_PATH.is_file():
            pytest.skip(
                f"{_TRAINING_CORPUS_PATH} not present locally (gitignored, "
                "not fetched/built) — nothing to compare against."
            )
        meta = json.loads(_FEATURE_METADATA_PATH.read_text())
        current_hash = manifest.corpus_sha256(_TRAINING_CORPUS_PATH)
        assert meta.get("corpus_sha256") == current_hash, (
            "feature_metadata.json's corpus_sha256 is stale — the training "
            "corpus changed since the model was last trained. Run "
            "`python -m src.models.train_final` and commit the refreshed artifacts."
        )

    def test_manifest_verified_present_and_boolean(self) -> None:
        """corpus_sha256 alone proves only *which* corpus trained the model, not whether its raw sources matched data_manifest.lock at build time — manifest_verified records that check's outcome (train_final.py's verify_manifest(), which only logs a warning and would otherwise leave no trace)."""
        meta = json.loads(_FEATURE_METADATA_PATH.read_text())
        assert "manifest_verified" in meta, (
            "feature_metadata.json missing 'manifest_verified' — retrain "
            "with `python -m src.models.train_final` to populate it."
        )
        assert isinstance(meta["manifest_verified"], bool), (
            f"manifest_verified should be a bool, got {type(meta['manifest_verified'])!r}"
        )

    def test_manifest_sources_coverage_present_and_sane(self) -> None:
        """manifest_verified alone can't distinguish "every source present and matched" from "every source simply absent, trivially not mismatched" — manifest_sources_present/total give the denominator needed to judge what manifest_verified: true is actually worth for this build."""
        meta = json.loads(_FEATURE_METADATA_PATH.read_text())
        assert "manifest_sources_present" in meta and "manifest_sources_total" in meta, (
            "feature_metadata.json missing manifest_sources_present/total — "
            "retrain with `python -m src.models.train_final` to populate it."
        )
        present, total = meta["manifest_sources_present"], meta["manifest_sources_total"]
        assert isinstance(present, int) and isinstance(total, int), (
            "manifest_sources_present/total should be ints"
        )
        assert 0 <= present <= total, (
            f"manifest_sources_present ({present}) must be between 0 and "
            f"manifest_sources_total ({total})"
        )

    def test_validation_metrics_present_and_well_formed(self) -> None:
        """The key must always exist (possibly null, if notebooks/03 hasn't been re-run since the last corpus build); when non-null, its primary summary fields must be sane finite fractions, not NaN/inf/negative placeholders that would silently pass a looser "key exists" check."""
        meta = json.loads(_FEATURE_METADATA_PATH.read_text())
        assert "validation_metrics" in meta, (
            "feature_metadata.json missing 'validation_metrics' key "
            "(should be present, possibly null) — retrain with "
            "`python -m src.models.train_final` to populate it."
        )
        vm = meta["validation_metrics"]
        if vm is None:
            pytest.skip("validation_metrics is null (notebooks/03 not re-run since last build)")

        primary = vm["primary"]
        for key in ("mape", "smape", "median_ape"):
            val = primary[key]
            assert isinstance(val, (int, float)) and math.isfinite(val) and val >= 0.0, (
                f"validation_metrics.primary.{key} = {val!r} is not a sane fraction"
            )
        rho = primary["spearman_rho"]
        assert isinstance(rho, (int, float)) and math.isfinite(rho) and -1.0 <= rho <= 1.0, (
            f"validation_metrics.primary.spearman_rho = {rho!r} out of [-1, 1]"
        )
        assert isinstance(primary["n_folds"], int) and primary["n_folds"] > 0, (
            f"validation_metrics.primary.n_folds = {primary.get('n_folds')!r} "
            "should be a positive int"
        )

    def test_trained_gpu_row_counts_present_and_consistent_with_trained_gpu_ids(self) -> None:
        """training_data_tier() needs a per-GPU row count to distinguish below-the-floor from ample data — a gap the older 'trained_gpu_ids' boolean set couldn't express; the two fields must describe the same set, so every id in trained_gpu_ids has a positive count here and vice versa."""
        meta = json.loads(_FEATURE_METADATA_PATH.read_text())
        assert "trained_gpu_row_counts" in meta, (
            "feature_metadata.json missing 'trained_gpu_row_counts' — "
            "retrain with `python -m src.models.train_final` to populate it."
        )
        counts = meta["trained_gpu_row_counts"]
        trained_ids = set(meta["trained_gpu_ids"])
        assert set(counts.keys()) == trained_ids, (
            f"trained_gpu_row_counts keys {sorted(counts.keys())} != "
            f"trained_gpu_ids {sorted(trained_ids)}"
        )
        for gpu_id, n in counts.items():
            assert isinstance(n, int) and n > 0, (
                f"trained_gpu_row_counts[{gpu_id!r}] = {n!r} should be a positive int"
            )
