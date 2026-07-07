"""
Feature engineering for GPU Perf Prophet.

Public API
----------
build_training_df(raw_df)
    Filter to in-scope rows, join GPU specs, add roofline features,
    and return a model-ready DataFrame.

roofline_ceilings(total_params_b, compute_params_b, bytes_per_param, hbm_bw_tbps, peak_tflops)
    Pure-function roofline computation used by the pipeline and notebooks.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from src.data.gpu_spec_db import enrich_df

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reference tables
# ---------------------------------------------------------------------------

# benchmark_base → (total_params_b, compute_params_b)
#
# total_params_b   : all weights loaded into VRAM — used for the
#                    memory-bandwidth ceiling and the VRAM-fit check.
# compute_params_b : active params per forward pass — equals total_params_b
#                    for dense models; equals active-expert params for MoE.
#
# Mixtral 8×7B: 8 experts of ~7B params each, but only 2 are activated per
# token. Shared layers (attention, embeddings) add ~0.7B. We approximate
# total = 46.7B and active = 2 × 6.7B + 0.7B ≈ 14.1B. Memory-bandwidth
# ceiling uses total (all weights reside in HBM); compute ceiling uses active.
#
# Sources:
#   llama2-70b    : Meta "Llama 2" paper, Table 1 (2023) — 70B parameters
#   llama3.1-405b : Meta "Llama 3" blog (2024) — 405B parameters
#   llama3.1-8b   : Meta "Llama 3" blog (2024) — 8B parameters
#   gptj          : EleutherAI GPT-J-6B model card (2021) — 6.05B parameters
#   mixtral-8x7b  : Mistral AI blog (2023) — 46.7B total / ~14.1B active
MODEL_PARAMS: dict[str, tuple[float, float]] = {
    "llama2-70b":    (70.0,  70.0),
    "llama3.1-405b": (405.0, 405.0),
    "llama3.1-8b":   (8.03,  8.03),   # anticipated for MLPerf v7.0+; no rows yet
    "gptj":          (6.05,  6.05),
    "mixtral-8x7b":  (46.7,  14.1),   # (total, active)
}

# benchmark_accuracy_tier → precision label used to select peak TFLOPS
# and compute bytes-per-param.
#
# "99.9" requires near-lossless accuracy → FP16 (no quantization risk).
# "99"   allows modest accuracy drop     → FP8  (halves memory footprint).
# "base" has the loosest constraint      → BF16 (widely supported baseline).
#
# If the selected precision is not supported on a GPU (peak column is NaN or
# None), _select_peak_tflops falls back to FP16.
TIER_TO_PRECISION: dict[str, str] = {
    "99.9": "fp16",
    "99":   "fp8",
    "base": "bf16",
}

# Bytes occupied per stored parameter at each precision.
BYTES_PER_PARAM: dict[str, float] = {
    "fp32": 4.0,
    "bf16": 2.0,
    "fp16": 2.0,
    "fp8":  1.0,
    "fp6":  0.75,
    "fp4":  0.5,
    "int8": 1.0,
}

# GPU peak TFLOPS column name for each precision label.
_PRECISION_TO_COL: dict[str, str] = {
    "fp32": "gpu_peak_fp32_tflops",
    "bf16": "gpu_peak_bf16_tflops",
    "fp16": "gpu_peak_fp16_tflops",
    "fp8":  "gpu_peak_fp8_tflops",
    "fp6":  "gpu_peak_fp6_tflops",
    "fp4":  "gpu_peak_fp4_tflops",
    "int8": "gpu_peak_int8_tops",
}

# Framework string → normalized family label.
# Matched in order; first hit wins.
_FRAMEWORK_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"TensorRT", re.IGNORECASE),  "tensorrt"),
    (re.compile(r"vLLM",     re.IGNORECASE),  "vllm"),
    (re.compile(r"ROCm|Mango", re.IGNORECASE), "rocm_other"),
]

# Per-vendor architecture generation ordinals (higher = newer).
# Split by vendor so the model cannot learn spurious cross-vendor "newer = better"
# patterns from a single linear scale (e.g. CDNA3 > Hopper is meaningless).
# NaN for the other vendor's GPUs is intentional and correct.
_NVIDIA_ARCH_ORDINAL: dict[str, int] = {
    "ampere":      1,
    "ada_lovelace": 2,
    "hopper":      3,
    "blackwell":   4,
}
_AMD_ARCH_ORDINAL: dict[str, int] = {
    "cdna3": 1,
    "cdna4": 2,
}

# MLPerf round tag → chronological ordinal (higher = more recent submission round).
# Prices in framework/driver maturity (e.g. ROCm version) as a feature: early
# rounds for a given GPU tend to run on less-tuned software stacks, which the
# model can otherwise mistake for a hardware effect. Unrecognized round tags
# map to NaN rather than raising, the same convention as the arch ordinals
# above — a future round just needs a new entry here.
ROUND_ORDINAL: dict[str, int] = {
    "v4.1": 1,
    "v5.0": 2,
    "v5.1": 3,
    "v6.0": 4,
}


# ---------------------------------------------------------------------------
# Core roofline computation (pure function — also used by notebooks directly)
# ---------------------------------------------------------------------------

def roofline_ceilings(
    total_params_b: float,
    compute_params_b: float,
    bytes_per_param: float,
    hbm_bw_tbps: float,
    peak_tflops: float,
) -> tuple[float, float, float]:
    """Return (bandwidth_ceiling, compute_ceiling, roofline_tput) in tokens/sec.

    bandwidth_ceiling
        A bandwidth-density proxy: how many tokens/sec HBM bandwidth could
        sustain if reading the full model weight footprint per token.  Uses
        *total* params (VRAM footprint), not active params, so that for MoE
        models the feature reflects how much HBM capacity the model occupies
        rather than per-token access volume.  Not a hard ceiling for batched
        inference, but a useful feature capturing GPU memory-bandwidth richness
        relative to model size.

    compute_ceiling
        The absolute maximum tokens/sec at any batch size, limited by peak
        FLOP/s.  Uses *compute* (active) params because only the activated
        weights participate in each forward pass.  This IS the hard physical
        ceiling for throughput; MLPerf Offline/Server run large batches and
        approach this bound in practice.

    roofline_tput = compute_ceiling.
        The compute ceiling is the correct upper bound for batched LLM
        inference.  We keep bandwidth_ceiling as a separate feature rather
        than folding it into the roofline.
    """
    model_bytes = total_params_b * 1e9 * bytes_per_param          # bytes
    bw_bytes_per_sec = hbm_bw_tbps * 1e12                         # bytes/s
    bw_ceil = bw_bytes_per_sec / model_bytes                       # tokens/s

    flops_per_token = 2.0 * compute_params_b * 1e9                # FLOPs
    peak_flops_per_sec = peak_tflops * 1e12                        # FLOPs/s
    compute_ceil = peak_flops_per_sec / flops_per_token            # tokens/s

    return bw_ceil, compute_ceil, compute_ceil


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_framework(raw: Optional[str]) -> str:
    if not isinstance(raw, str):
        return "unknown"
    for pattern, label in _FRAMEWORK_PATTERNS:
        if pattern.search(raw):
            return label
    return "other"


def _select_peak_tflops(row: pd.Series, precision: str) -> Optional[float]:
    """Return the peak TFLOPS for `precision`, falling back to fp16 if absent."""
    col = _PRECISION_TO_COL.get(precision)
    val = row.get(col) if col else None
    if val is None or (isinstance(val, float) and pd.isna(val)):
        # GPU doesn't support this precision natively — fall back to fp16.
        val = row.get("gpu_peak_fp16_tflops")
    return val


# ---------------------------------------------------------------------------
# Public pipeline
# ---------------------------------------------------------------------------

def build_training_df(
    raw_df: pd.DataFrame,
    spec_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Filter, enrich, and featurise raw MLPerf data for model training.

    Steps
    -----
    1. Enrich with GPU spec columns via ``enrich_df``.
    2. Filter: ``result_valid == True`` only.  ``gpu_in_model_scope`` is a
       serving-layer gate, not a training filter.
    3. Attach model-parameter reference values.
    4. Compute roofline ceilings (bandwidth, compute ceiling only).
    5. Derive secondary features (efficiency ratio, VRAM fit, framework family,
       architecture ordinal, vendor indicator).

    Returns
    -------
    pd.DataFrame with all original columns plus the features listed below.
    Rows that cannot be featurised (unknown benchmark_base or missing GPU
    specs) are dropped with a warning rather than silently propagating NaN
    into training.
    """
    kwargs = {"spec_path": spec_path} if spec_path is not None else {}
    df = enrich_df(raw_df, **kwargs).copy()

    # --- filter ---
    # gpu_in_model_scope gates recommendation exposure, not training inclusion.
    # Out-of-scope GPUs (e.g. B200, H200 NVL) are valid training signal; they
    # inform the model about architectural trends even if not served to users v1.
    df = df[df["result_valid"]]
    log.info("After result_valid filter: %d rows", len(df))

    # --- model params ---
    df["model_total_params_b"] = df["benchmark_base"].map(
        {k: v[0] for k, v in MODEL_PARAMS.items()}
    )
    df["model_compute_params_b"] = df["benchmark_base"].map(
        {k: v[1] for k, v in MODEL_PARAMS.items()}
    )

    unknown_benchmarks = df[df["model_total_params_b"].isna()]["benchmark_base"].unique()
    if len(unknown_benchmarks):
        log.warning("Dropping %d rows with unknown benchmark_base: %s",
                    df["model_total_params_b"].isna().sum(), unknown_benchmarks)
        df = df[df["model_total_params_b"].notna()]

    # --- precision selection ---
    df["selected_precision"] = df["benchmark_accuracy_tier"].map(TIER_TO_PRECISION)
    # AMD CDNA hardware achieves 99.9 accuracy with FP8, not FP16.  The
    # TIER_TO_PRECISION default (99.9→fp16) is correct for NVIDIA; override
    # it for AMD so efficiency_ratio targets are computed against the right
    # ceiling (2× the TFLOPS of the FP16 ceiling, halving the efficiency
    # ratio for those rows and eliminating the ceiling violations in training).
    amd_tier_99_9 = (df["gpu_vendor"] == "amd") & (df["benchmark_accuracy_tier"] == "99.9")
    df.loc[amd_tier_99_9, "selected_precision"] = "fp8"
    df["bytes_per_param"] = df["selected_precision"].map(BYTES_PER_PARAM)

    # Vectorised equivalent of _select_peak_tflops over all rows.
    # Covers the three precision values that TIER_TO_PRECISION produces;
    # fp16 is the fallback for any unknown precision or for GPUs whose
    # selected-precision column is NaN (same as the scalar helper).
    _fp16 = df["gpu_peak_fp16_tflops"]
    _fp8 = df["gpu_peak_fp8_tflops"].where(df["gpu_peak_fp8_tflops"].notna(), _fp16)
    _bf16 = df["gpu_peak_bf16_tflops"].where(df["gpu_peak_bf16_tflops"].notna(), _fp16)
    _prec = df["selected_precision"]
    df["peak_tflops_selected"] = _fp8.where(
        _prec == "fp8", _bf16.where(_prec == "bf16", _fp16)
    )

    # --- roofline ---
    missing_specs = df[
        df["gpu_hbm_bandwidth_tbps"].isna() | df["peak_tflops_selected"].isna()
    ]
    if len(missing_specs):
        log.warning("Dropping %d rows with missing GPU specs (cannot compute roofline)",
                    len(missing_specs))
        df = df[
            df["gpu_hbm_bandwidth_tbps"].notna() & df["peak_tflops_selected"].notna()
        ]

    _model_bytes = df["model_total_params_b"] * 1e9 * df["bytes_per_param"]
    df["bandwidth_ceiling_tok_per_sec"] = (df["gpu_hbm_bandwidth_tbps"] * 1e12) / _model_bytes
    df["compute_ceiling_tok_per_sec"] = (df["peak_tflops_selected"] * 1e12) / (
        2.0 * df["model_compute_params_b"] * 1e9
    )
    df["roofline_tput"] = df["compute_ceiling_tok_per_sec"]

    # --- derived features ---
    df["efficiency_ratio"] = (
        df["throughput_tok_per_sec_per_gpu"] / df["roofline_tput"]
    )
    df["model_size_gb"] = (
        df["model_total_params_b"] * df["bytes_per_param"]
    )
    df["model_to_vram_ratio"] = df["model_size_gb"] / df["gpu_vram_gb"]
    # Deduplicate framework strings before normalizing — ~10–20 unique values
    # across ~1112 rows, so .apply() would call _normalize_framework (3 regex
    # searches) ~1112 times.  Map unique values once, then broadcast.
    # Parallels the per-unique-gpu-name optimization in enrich_df ("Fix 2").
    _fw_unique_map = {fw: _normalize_framework(fw) for fw in df["framework"].dropna().unique()}
    df["framework_family"] = df["framework"].map(_fw_unique_map).fillna("unknown")
    df["nvidia_arch_gen"] = df["gpu_architecture"].map(_NVIDIA_ARCH_ORDINAL)
    df["amd_arch_gen"] = df["gpu_architecture"].map(_AMD_ARCH_ORDINAL)
    df["vendor_is_amd"] = (df["gpu_vendor"] == "amd").astype(int)
    df["mlperf_round_num"] = df["round"].map(ROUND_ORDINAL)
    # Binary flag separating the "base" accuracy tier (BF16, loose accuracy
    # constraint) from the "99" and "99.9" tiers (FP8 / FP16 or the AMD FP8
    # override).  Replaces the three-level accuracy_tier_ord: after the AMD
    # FP8 override, tiers "99" and "99.9" are both FP8 for AMD, making
    # accuracy_tier_ord a spurious discriminator within AMD LOGO folds.
    # bytes_per_param already captures the FP8 vs FP16 precision split for
    # NVIDIA; is_base_tier captures the remaining base-vs-non-base distinction
    # that bytes_per_param cannot express (base BF16 = 2.0, same as NVIDIA FP16).
    df["is_base_tier"] = (df["benchmark_accuracy_tier"] == "base").astype(int)

    # Efficiency ratio > 1 means actual throughput exceeds the compute ceiling
    # at the precision we inferred from benchmark_accuracy_tier.  This is
    # expected for AMD CDNA4 rows at the "99.9" tier: AMD's vLLM/ROCm stack
    # achieves 99.9 accuracy with FP8 (higher TFLOPS ceiling), but our proxy
    # maps 99.9 → FP16.  These rows are valid training data — the model learns
    # that AMD systematically outperforms the FP16 ceiling for this tier.
    n_violations = (df["throughput_tok_per_sec_per_gpu"] > df["roofline_tput"]).sum()
    if n_violations:
        log.warning(
            "%d rows (%.1f%%) have throughput > compute ceiling at selected "
            "precision — likely precision-proxy mismatch (AMD FP8 at 99.9 tier).",
            n_violations,
            100 * n_violations / len(df),
        )

    log.info("build_training_df complete: %d rows, %d columns", len(df), df.shape[1])
    return df.reset_index(drop=True)
