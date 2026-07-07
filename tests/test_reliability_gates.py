"""
Reliability-target gate tests  (RT-X.Y IDs from JOURNAL.md §"Reliability Targets").

These are the machine-executable counterpart of the Design Review Checklist.
Run all gates before merging any PR:

    pytest tests/test_reliability_gates.py -v

Or run only GATE-marked tests across the entire suite:

    pytest -m gate -v

Each test name encodes the RT it protects so a CI failure immediately names
the violated requirement.

Coverage scope
--------------
This file fills the gaps left by the existing per-module test files:

  RT-2.2  MoE models (mixtral-8x7b) use total_params_b for the bandwidth
          ceiling and compute_params_b (active expert params) for the compute
          ceiling.  An arg-swap would overestimate mixtral throughput 3.31×.
          Not covered by any per-module test.

  RT-3.6  Full 20-feature parity (training path vs serving path).
          test_predictor.py::TestEncodeVsPredictor already covers the 6
          categorical columns.  This file covers all 13 continuous features
          (roofline ceilings, model sizes, VRAM ratio, precision TFLOPS)
          plus vendor_is_amd.
          A formula divergence there is a silent bug: predictions appear
          correct but the model is evaluated on features it was never trained on.

  RT-4.1  Pydantic Literal types in schemas.py match VALID_* frozensets in
          predictor.py.  Both are defined independently; silent divergence
          means valid API inputs are rejected by the predictor (500) or
          invalid inputs accepted by Pydantic then fail in the predictor.

  RT-5.3  Model artifact total disk size < 50 MB.
          No other test verifies this; the limit matters for free-tier HF Spaces
          deployment (2 vCPU / 16 GB RAM, shared bandwidth).
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import get_args

import pandas as pd
import pytest
import yaml

from src.api.schemas import AccuracyTier, Framework, Scenario
from src.data.gpu_spec_db import load_specs
from src.features.build_features import (
    BYTES_PER_PARAM,
    MODEL_PARAMS,
    TIER_TO_PRECISION,
    _FRAMEWORK_PATTERNS,
    _normalize_framework,
    build_training_df,
)
from src.models.predictor import (
    FEATURE_COLS,
    VALID_FRAMEWORKS,
    VALID_SCENARIOS,
    VALID_TIERS,
    GpuPredictor,
    _build_feature_vector,
)
from src.recommend.recommender import GpuRecommender


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# RT-2.2  MoE param split — bandwidth ceiling uses total, compute uses active
# ---------------------------------------------------------------------------

@pytest.mark.gate
class TestRT22MoEParamSplit:
    """RT-2.2 GATE: For MoE models, the bandwidth ceiling must use total_params_b
    (full model footprint in HBM) and the compute ceiling must use compute_params_b
    (active-expert params per token).

    mixtral-8x7b: total=46.7B, active=14.1B (ratio 3.31×).  Swapping these args
    in roofline_ceilings() would overestimate mixtral compute throughput 3.31×
    and underestimate the bandwidth ceiling by the same factor — corrupting both
    the feature values and the hard roofline cap in predict().
    """

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
            f"RT-2.2: BW ceiling implies {implied_params:.2f}B params in model footprint; "
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
            f"RT-2.2: Compute ceiling implies {implied_params:.2f}B active params; "
            f"expected compute_params_b={active_b}B.  "
            f"Bandwidth ceiling likely received active params instead of total params."
        )


# ---------------------------------------------------------------------------
# RT-4.1  Schema/predictor agreement — Literal types match VALID_* frozensets
# ---------------------------------------------------------------------------

@pytest.mark.gate
class TestRT41SchemaPredictorAgreement:
    """RT-4.1 GATE: Pydantic Literal types in src/api/schemas.py must exactly
    match the VALID_* frozensets in src/models/predictor.py.

    Both are defined independently.  If schemas.py accepts a value that
    VALID_* rejects, a well-formed API request causes a 500 (predictor raises
    ValueError).  If VALID_* accepts a value schemas.py rejects, a valid
    predictor call cannot be reached via the API at all.
    """

    def test_scenario_literals_match_valid_scenarios(self) -> None:
        assert set(get_args(Scenario)) == VALID_SCENARIOS, (
            f"RT-4.1: schemas.Scenario={set(get_args(Scenario))} "
            f"!= predictor.VALID_SCENARIOS={VALID_SCENARIOS}"
        )

    def test_accuracy_tier_literals_match_valid_tiers(self) -> None:
        assert set(get_args(AccuracyTier)) == VALID_TIERS, (
            f"RT-4.1: schemas.AccuracyTier={set(get_args(AccuracyTier))} "
            f"!= predictor.VALID_TIERS={VALID_TIERS}"
        )

    def test_framework_literals_match_valid_frameworks(self) -> None:
        assert set(get_args(Framework)) == VALID_FRAMEWORKS, (
            f"RT-4.1: schemas.Framework={set(get_args(Framework))} "
            f"!= predictor.VALID_FRAMEWORKS={VALID_FRAMEWORKS}"
        )


# ---------------------------------------------------------------------------
# RT-3.6  Feature vector parity — training path vs serving path
# ---------------------------------------------------------------------------

@pytest.mark.gate
class TestRT36FeatureVectorParity:
    """RT-3.6 GATE: all 20 FEATURE_COLS values match between build_training_df
    (training path) and _build_feature_vector (serving path), with one
    intentional exception noted below.

    Column groups verified
    ----------------------
    Continuous (this file):
        gpu_hbm_bandwidth_tbps, gpu_vram_gb, peak_tflops_selected,
        compute_ceiling_tok_per_sec, bandwidth_ceiling_tok_per_sec,
        model_total_params_b, model_compute_params_b, model_size_gb,
        model_to_vram_ratio, bytes_per_param, vendor_is_amd,
        nvidia_arch_gen, amd_arch_gen

    Categorical (test_predictor.py::TestBuildFeatureVector):
        scenario_offline, is_base_tier, fw_tensorrt, fw_vllm,
        fw_rocm_other, is_cdna4

    Intentionally excluded from parity check:
        mlperf_round_num — training uses the actual submission round ordinal;
        serving always uses _SERVING_ROUND (max(ROUND_ORDINAL.values())) to
        predict for the most mature software stack.  Verified separately in
        test_predictor.py::TestBuildFeatureVector::test_mlperf_round_num_serving_uses_latest.
    """

    # (gpu_alias, gpu_id, model_name, scenario, tier, raw_framework)
    # gpu_alias must match a known alias in data/gpu_specs.yaml.
    # raw_framework is a realistic MLPerf string; _normalize_framework maps it
    # to the family label that _build_feature_vector expects.
    CASES = [
        # NVIDIA Hopper, FP8 tier (99 → fp8, falls back for GPUs without fp8)
        (
            "NVIDIA H200-SXM-141GB",          "h200_sxm",  "llama2-70b",
            "Offline", "99",   "TensorRT 10.2.0",
        ),
        # AMD CDNA3, FP8 tier (99.9 → fp8 for AMD; parity check verifies both
        # paths apply the vendor override consistently)
        (
            "AMD Instinct MI300X 192GB HBM3",  "mi300x",    "gptj",
            "Server",  "99.9", "vLLM 0.4.3+rocm614",
        ),
        # AMD CDNA3, BF16 tier (base → bf16); MoE model (active ≠ total params)
        (
            "AMD Instinct MI325X",             "mi325x",    "mixtral-8x7b",
            "Offline", "99",   "ROCm 7.0",
        ),
        # AMD CDNA4 — only GPU where amd_arch_gen=2 and is_cdna4=1.
        # A divergence in CDNA4 arch-ordinal encoding (training vs serving)
        # cannot be detected by the 3 cases above (all use CDNA3 or NVIDIA).
        (
            "AMD Instinct MI355X 288GB HBM3e", "mi355x",    "llama2-70b",
            "Offline", "99",   "ROCm 7.0",
        ),
        # NVIDIA at tier 99.9 — selects fp16 TFLOPS (NOT fp8, no AMD override).
        # The four cases above are all tier 99 (fp8) or AMD 99.9 (fp8 override);
        # without this case a divergence in NVIDIA fp16 TFLOPS selection between
        # the training and serving paths would pass the gate undetected.
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
        """RT-3.6 GATE: every continuous FEATURE_COL has the same value in
        both paths.  A mismatch means the model evaluates features at serving
        time that it was never trained on.
        """
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

        # ── Compare continuous columns ────────────────────────────────────
        # mlperf_round_num is excluded: serving always uses the latest round
        # ordinal (_SERVING_ROUND); training uses the actual submission round.
        # This divergence is intentional — see class docstring.
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
            f"RT-3.6: feature mismatch for gpu={gpu_id!r} model={model_name!r} "
            f"scenario={scenario!r} tier={tier!r}:\n"
            + "\n".join(mismatches)
        )


# ---------------------------------------------------------------------------
# RT-5.3  Model artifact total disk size < 50 MB
# ---------------------------------------------------------------------------

_MODEL_DIR = Path("data/models")
_DISK_LIMIT_MB = 50


@pytest.mark.gate
class TestRT53ModelDiskSize:
    """RT-5.3 GATE: data/models/ total size is < 50 MB.

    Free-tier Hugging Face Spaces (the deploy target) has shared storage
    and slow cold-start I/O.  50 MB is the deployment budget.
    """

    def test_model_dir_exists(self) -> None:
        assert _MODEL_DIR.is_dir(), (
            f"RT-5.3: {_MODEL_DIR} not found — run train_final.py to generate artifacts"
        )

    def test_total_size_under_limit(self) -> None:
        total_bytes = sum(f.stat().st_size for f in _MODEL_DIR.rglob("*") if f.is_file())
        total_mb = total_bytes / (1024 ** 2)
        assert total_mb < _DISK_LIMIT_MB, (
            f"RT-5.3: data/models/ is {total_mb:.1f} MB — exceeds {_DISK_LIMIT_MB} MB limit. "
            "Trim model artifacts before deploying to HF Spaces."
        )

    def test_required_artifacts_present(self) -> None:
        required = ["prophet_v1.json", "feature_metadata.json"]
        for name in required:
            assert (_MODEL_DIR / name).exists(), (
                f"RT-5.3: required artifact {name} is missing from {_MODEL_DIR}"
            )


# ---------------------------------------------------------------------------
# RT-1.3  Pricing coverage — all in-scope GPUs have pricing entries
# ---------------------------------------------------------------------------

_PRICING_PATH = Path("data/pricing.yaml")


@pytest.mark.gate
class TestRT13PricingCoverage:
    """RT-1.3 GATE: every GPU with in_model_scope=True in gpu_specs.yaml has a
    corresponding entry in data/pricing.yaml.

    GpuRecommender.__init__() raises ValueError for missing pricing, but that
    failure is only visible at runtime (startup).  This gate catches the gap
    before the Docker image is built or before AMD Dev Cloud session starts.

    Adding a new in-scope GPU without a pricing entry causes cost_efficiency=None,
    which raises TypeError in _pareto_frontier's sort — silently breaking all
    recommendation responses rather than failing cleanly at startup.
    """

    def test_pricing_file_exists(self) -> None:
        assert _PRICING_PATH.is_file(), (
            f"RT-1.3: {_PRICING_PATH} not found — create data/pricing.yaml before deployment"
        )

    def test_all_in_scope_gpus_have_pricing(self) -> None:
        with _PRICING_PATH.open() as f:
            pricing: dict = yaml.safe_load(f).get("pricing", {})
        in_scope_ids = [s["id"] for s in load_specs() if s.get("in_model_scope")]
        assert in_scope_ids, (
            "RT-1.3: no in-scope GPUs found — check gpu_specs.yaml in_model_scope flags"
        )
        missing = [gid for gid in in_scope_ids if gid not in pricing]
        assert not missing, (
            f"RT-1.3: pricing.yaml missing entries for in-scope GPUs: {missing}. "
            "GpuRecommender.__init__() will raise ValueError at startup."
        )


# ---------------------------------------------------------------------------
# RT-3.7  AMD FP8 override — three-way consistency gate
# ---------------------------------------------------------------------------

@pytest.mark.gate
class TestRT37AmdFp8Override:
    """RT-3.7 GATE: the AMD 99.9-tier FP8 override is applied consistently in
    build_training_df (training), _build_feature_vector (serving), and
    GpuRecommender._gpu_model_size_gb (VRAM pre-filter).

    A regression in any one of the three paths causes:
      - Training: efficiency_ratio > 1.0 for MI355X 99.9-tier rows
        (FP16 ceiling used instead of FP8, ceiling violations in 22/50 rows,
        reducing MI355X LOGO ρ from ~0.60 to ~0.40)
      - Serving/recommender: wrong VRAM fit decision for llama2-70b at 99.9 tier
        on MI300X (140 GB FP16 model exceeds 192 GB VRAM by ~27%, so MI300X
        would be incorrectly rejected despite fitting comfortably as 70 GB FP8)

    Verified end-to-end via recommender: if vram_headroom ≈ 0.635 (FP8 70 GB),
    all three paths agree.  If headroom ≈ 0.271 (FP16 140 GB), at least one
    path has drifted.
    """

    @pytest.fixture(scope="class")
    def recommender(self) -> GpuRecommender:
        return GpuRecommender(GpuPredictor())

    def test_mi300x_vram_headroom_uses_fp8_not_fp16(self, recommender) -> None:
        result = recommender.recommend(model_name="llama2-70b", accuracy_tier="99.9")
        candidates = result["frontier"] + result["dominated"]
        mi300x = next((r for r in candidates if r["gpu_id"] == "mi300x"), None)
        assert mi300x is not None, (
            "RT-3.7: MI300X absent from llama2-70b tier-99.9 candidates — "
            "VRAM filter may be applying FP16 model size (140 GB) instead of FP8 (70 GB), "
            "causing MI300X (192 GB) to be incorrectly rejected."
        )
        fp8_model_gb = 70.0  # 70B params × 1 byte/param (FP8)
        expected_headroom = (192.0 - fp8_model_gb) / 192.0   # ≈ 0.635
        assert mi300x["vram_headroom"] == pytest.approx(expected_headroom, abs=0.01), (
            f"RT-3.7: MI300X vram_headroom={mi300x['vram_headroom']:.3f}, "
            f"expected {expected_headroom:.3f} (FP8 70 GB model). "
            "A value near 0.271 indicates FP16 (140 GB) is being used instead."
        )


# ---------------------------------------------------------------------------
# RT-3.8  Framework normalization — pattern outputs ⊆ VALID_FRAMEWORKS
# ---------------------------------------------------------------------------

@pytest.mark.gate
class TestRT38FrameworkNormalization:
    """RT-3.8 GATE: every label returned by a _FRAMEWORK_PATTERNS entry is in
    VALID_FRAMEWORKS, and the no-match fallback ("other") is also in VALID_FRAMEWORKS.

    If a new framework pattern is added to _FRAMEWORK_PATTERNS with a label that
    is NOT in VALID_FRAMEWORKS, that framework family enters the training data but
    can never be requested via the API — predictor._validate() raises ValueError
    on what appears to be a perfectly valid input, causing silent 500 errors for
    any workload that uses that framework.
    """

    def test_all_pattern_outputs_in_valid_frameworks(self) -> None:
        labels_from_patterns = {label for _, label in _FRAMEWORK_PATTERNS}
        unknown_labels = labels_from_patterns - VALID_FRAMEWORKS
        assert not unknown_labels, (
            f"RT-3.8: _FRAMEWORK_PATTERNS returns labels not in VALID_FRAMEWORKS: "
            f"{unknown_labels}. Add to predictor.VALID_FRAMEWORKS or remove from patterns."
        )

    def test_fallback_other_in_valid_frameworks(self) -> None:
        fallback = _normalize_framework("some_completely_unknown_framework_xyz")
        assert fallback == "other", (
            f"RT-3.8: no-match fallback should be 'other', got {fallback!r}. "
            "The fallback must be in VALID_FRAMEWORKS so unrecognized frameworks "
            "can be requested via the API without a 422 validation error."
        )
        assert "other" in VALID_FRAMEWORKS, (
            "RT-3.8: 'other' is _normalize_framework's no-match return value "
            "but is not in VALID_FRAMEWORKS — unrecognized frameworks fail predictor._validate()."
        )


# ---------------------------------------------------------------------------
# RT-6.1  Calibration runner/merge script CSV compatibility
# ---------------------------------------------------------------------------

_RUNNER_PATH = Path("benchmarks/run_mi300x_calibration.py")

# Fields that merge_calibration_rows.py accesses directly from CSV rows
# (r["field"] or r.get("field", ...)).  Update whenever merge script adds
# a new field access.
_MERGE_REQUIRED_FIELDS: frozenset[str] = frozenset({
    "round_tag",
    "gpu_name",
    "benchmark_base",
    "benchmark_accuracy_tier",
    "scenario",
    "throughput_tok_per_sec",
    "vllm_version",
    # DataFrame subscript access on line 64 of merge_calibration_rows.py:
    # failed[["benchmark_base", "precision_used", "scenario"]].to_dict("records")
    # Not a .get() — raises KeyError if the column is absent from the CSV.
    "precision_used",
})


def _parse_runner_csv_fields() -> frozenset[str]:
    """Extract CSV_FIELDS list from runner script via AST — avoids importing
    the script directly, which would fail without vllm installed."""
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
class TestRT61RunnerMergeCompatibility:
    """RT-6.1 GATE: benchmarks/run_mi300x_calibration.py CSV_FIELDS is a superset
    of the columns that benchmarks/merge_calibration_rows.py reads from each row.

    Drift between the runner's CSV schema and the merge script's column accesses
    causes KeyError (or silent None via .get()) when processing real AMD Dev Cloud
    results — after the 25-hr credit clock has already been consumed.  This gate
    runs locally before starting the timed session.
    """

    def test_runner_script_exists(self) -> None:
        assert _RUNNER_PATH.is_file(), (
            f"RT-6.1: {_RUNNER_PATH} not found — calibration runner script is missing"
        )

    def test_runner_csv_fields_cover_merge_required(self) -> None:
        runner_fields = _parse_runner_csv_fields()
        missing = _MERGE_REQUIRED_FIELDS - runner_fields
        assert not missing, (
            f"RT-6.1: merge_calibration_rows.py accesses fields not in runner CSV_FIELDS: "
            f"{missing}. Add these to CSV_FIELDS in run_mi300x_calibration.py, or "
            "remove the access from merge_calibration_rows.py."
        )
