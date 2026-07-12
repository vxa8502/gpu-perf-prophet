"""Feature engineering for GPU Perf Prophet: build_training_df(raw_df) filters/enriches/featurises raw MLPerf rows into a model-ready DataFrame; roofline_ceilings(...) is the pure-function roofline computation reused by the pipeline and notebooks."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from src.data.gpu_spec_db import enrich_df

log = logging.getLogger(__name__)

# --- Reference tables ---

# benchmark_base → (total_params_b, compute_params_b): total = all weights in VRAM (bandwidth ceiling / VRAM-fit check); compute = active params per forward pass (dense: same as total; MoE: active-expert only) — e.g. Mixtral 8x7B approximated as total=46.7B (8 x ~7B experts), active=14.1B (2 active experts x 6.7B + ~0.7B shared layers); sources: Meta Llama 2/3 papers/blog, EleutherAI GPT-J-6B model card, Mistral AI blog (2023).
MODEL_PARAMS: dict[str, tuple[float, float]] = {
    "llama2-70b":    (70.0,  70.0),
    "llama3.1-405b": (405.0, 405.0),
    "llama3.1-8b":   (8.03,  8.03),   # anticipated for MLPerf v7.0+; no rows yet
    "gptj":          (6.05,  6.05),
    "mixtral-8x7b":  (46.7,  14.1),   # (total, active)
}

# benchmark_base → (n_layers, n_kv_heads, head_dim) for KV-cache sizing, not derivable from MLPerf rows — sourced from each model's published config.json/paper (GQA models have a proportionally smaller KV cache than MHA); Mixtral 8x7B's MoE only changes the FFN, so its KV-cache math is treated as a dense 32-layer/8-kv-head/128-head-dim model with no sliding window (unlike base Mistral 7B v0.1).
MODEL_ARCH: dict[str, tuple[int, int, int]] = {
    "llama2-70b":    (80,  8,  128),
    "llama3.1-405b": (126, 8,  128),
    "llama3.1-8b":   (32,  8,  128),
    "gptj":          (28,  16, 256),
    "mixtral-8x7b":  (32,  8,  128),
}

# benchmark_accuracy_tier → precision label used to select peak TFLOPS and bytes-per-param: "99.9"→FP16 (near-lossless), "99"→FP8 (modest accuracy drop, halves memory), "base"→BF16 (loosest constraint); if a GPU lacks the selected precision, _select_peak_tflops falls back to FP16 for TRAINING-DATA ingestion only — the live serving path must raise an "unsupported precision" error instead, see gpu_supports_precision() below.
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

# Framework string → normalized family label, matched in order (first hit wins).
_FRAMEWORK_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"TensorRT", re.IGNORECASE),  "tensorrt"),
    (re.compile(r"vLLM",     re.IGNORECASE),  "vllm"),
    (re.compile(r"ROCm|Mango", re.IGNORECASE), "rocm_other"),
]

# Per-vendor architecture generation ordinals (higher = newer), split by vendor so the model can't learn spurious cross-vendor "newer = better" patterns (e.g. CDNA3 > Hopper is meaningless) — NaN for the other vendor's GPUs is intentional.
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

# MLPerf round tag → chronological ordinal (higher = more recent), prices in framework/driver maturity (e.g. ROCm version) as a feature since early rounds run on less-tuned software stacks the model could otherwise mistake for a hardware effect; unrecognized round tags map to NaN rather than raising, same convention as the arch ordinals above.
ROUND_ORDINAL: dict[str, int] = {
    "v4.1": 1,
    "v5.0": 2,
    "v5.1": 3,
    "v6.0": 4,
}


# --- Core roofline computation (pure function — also used by notebooks directly) ---

def roofline_ceilings(
    total_params_b: float,
    compute_params_b: float,
    bytes_per_param: float,
    hbm_bw_tbps: float,
    peak_tflops: float,
) -> tuple[float, float, float]:
    """Return (bandwidth_ceiling, compute_ceiling, roofline_tput) in tokens/sec: bandwidth_ceiling is a memory-bandwidth-richness proxy using *total* params (VRAM footprint, so MoE reflects HBM occupancy not per-token access), compute_ceiling is the hard physical throughput ceiling using *compute* (active) params, and roofline_tput = compute_ceiling (the correct upper bound for batched LLM inference; bandwidth_ceiling is kept as a separate feature rather than folded in)."""
    model_bytes = total_params_b * 1e9 * bytes_per_param          # bytes
    bw_bytes_per_sec = hbm_bw_tbps * 1e12                         # bytes/s
    bw_ceil = bw_bytes_per_sec / model_bytes                       # tokens/s

    flops_per_token = 2.0 * compute_params_b * 1e9                # FLOPs
    peak_flops_per_sec = peak_tflops * 1e12                        # FLOPs/s
    compute_ceil = peak_flops_per_sec / flops_per_token            # tokens/s

    return bw_ceil, compute_ceil, compute_ceil


# Default batch/context-length assumption for KV-cache sizing — MLPerf rows carry no per-row batch/context length so this can't be learned like efficiency_ratio; it's a stated, overridable-per-request assumption (same spirit as the static pricing snapshot) representing a moderately loaded Offline-scenario batched-serving workload.
DEFAULT_BATCH_SIZE: int = 32
DEFAULT_INPUT_TOKENS: int = 2048
DEFAULT_OUTPUT_TOKENS: int = 256

MIN_BATCH_SIZE, MAX_BATCH_SIZE = 1, 256
MIN_INPUT_TOKENS, MAX_INPUT_TOKENS = 64, 8192
MIN_OUTPUT_TOKENS, MAX_OUTPUT_TOKENS = 1, 4096

# Outlier-rejection bound on efficiency_ratio: a row outside (0, MAX_EFFICIENCY_RATIO] signals a spec-DB or parse error (e.g. the precision-proxy mismatch that drove pre-FP8-override AMD 99.9-tier rows to ~1.35) and is dropped; values in (1.0, 1.2] are still kept as expected precision-proxy noise, see the diagnostic-only warning in build_training_df.
MAX_EFFICIENCY_RATIO: float = 1.2

# 10% activation/framework overhead on top of weights + KV cache, aligned with vLLM's default --gpu-memory-utilization 0.90.
MEMORY_OVERHEAD_FACTOR: float = 1.10

# Verdict thresholds on VRAM utilization: does_not_fit is a hard exclusion, tight is a disclosure-only flag (expected to run but with little headroom for allocator fragmentation).
_FITS_MAX_UTIL: float = 0.90
_TIGHT_MAX_UTIL: float = 0.98

# The only values memory_fit_verdict() ever returns — single source of truth for closed-set membership checks (mirrors VALID_FRAMEWORKS/_normalize_framework).
VALID_MEMORY_FIT_VERDICTS: frozenset[str] = frozenset({"fits", "tight", "does_not_fit"})


def validate_serving_shape(batch_size: int, input_tokens: int, output_tokens: int) -> None:
    """Raise ValueError if batch_size/input_tokens/output_tokens are out of range; single source of truth for this check shared by GpuPredictor.predict()/predict_batch() and GpuRecommender.recommend() so both entry points enforce the same input contract (recommend() previously had no check at all)."""
    if not (MIN_BATCH_SIZE <= batch_size <= MAX_BATCH_SIZE):
        raise ValueError(
            f"Invalid batch_size {batch_size!r}. "
            f"Valid range: [{MIN_BATCH_SIZE}, {MAX_BATCH_SIZE}]"
        )
    if not (MIN_INPUT_TOKENS <= input_tokens <= MAX_INPUT_TOKENS):
        raise ValueError(
            f"Invalid input_tokens {input_tokens!r}. "
            f"Valid range: [{MIN_INPUT_TOKENS}, {MAX_INPUT_TOKENS}]"
        )
    if not (MIN_OUTPUT_TOKENS <= output_tokens <= MAX_OUTPUT_TOKENS):
        raise ValueError(
            f"Invalid output_tokens {output_tokens!r}. "
            f"Valid range: [{MIN_OUTPUT_TOKENS}, {MAX_OUTPUT_TOKENS}]"
        )


def kv_cache_gb(
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    batch_size: int,
    input_tokens: int,
    output_tokens: int,
    bytes_per_value: float,
) -> float:
    """KV-cache size in GB for one batch at the given context length: 2 (K and V) x batch x seq_len x n_layers x n_kv_heads x head_dim x bytes; GQA models (n_kv_heads < n_heads) shrink this proportionally, the reduction that makes GQA cheap to serve."""
    seq_len = input_tokens + output_tokens
    kv_bytes = (
        2 * batch_size * seq_len * n_layers * n_kv_heads * head_dim * bytes_per_value
    )
    return kv_bytes / 1e9


def memory_fit_verdict(
    weights_gb: float,
    kv_gb: float,
    vram_gb: float,
) -> tuple[str, float, float]:
    """Return (verdict, total_gb, utilization), where verdict is one of "fits" (util <= 0.90), "tight" (<= 0.98), or "does_not_fit" (> 0.98)."""
    total_gb = (weights_gb + kv_gb) * MEMORY_OVERHEAD_FACTOR
    utilization = total_gb / vram_gb
    if utilization <= _FITS_MAX_UTIL:
        verdict = "fits"
    elif utilization <= _TIGHT_MAX_UTIL:
        verdict = "tight"
    else:
        verdict = "does_not_fit"
    return verdict, total_gb, utilization


def cost_per_million_tokens(
    price_per_gpu_hr: Optional[float],
    tokens_per_sec: float,
) -> Optional[float]:
    """USD per 1M tokens served: (usd_per_hour / 3600) / (tok/s / 1e6); returns None when price is unknown or throughput is non-positive (undefined, not zero or an error, matching cost_efficiency's None-for-unpriced convention)."""
    if price_per_gpu_hr is None or tokens_per_sec <= 0:
        return None
    return (price_per_gpu_hr / 3600.0) / (tokens_per_sec / 1_000_000.0)


# --- Internal helpers ---

def _normalize_framework(raw: Optional[str]) -> str:
    if not isinstance(raw, str):
        return "unknown"
    for pattern, label in _FRAMEWORK_PATTERNS:
        if pattern.search(raw):
            return label
    return "other"


def _select_peak_tflops(row: pd.Series, precision: str) -> Optional[float]:
    """Return the peak TFLOPS for `precision`, falling back to fp16 if absent — training-data ingestion only, see the note on gpu_supports_precision()."""
    col = _PRECISION_TO_COL.get(precision)
    val = row.get(col) if col else None
    if val is None or (isinstance(val, float) and pd.isna(val)):
        # GPU doesn't support this precision natively — fall back to fp16.
        val = row.get("gpu_peak_fp16_tflops")
    return val


def gpu_supports_precision(gpu_spec: dict, precision: str) -> bool:
    """Whether gpu_spec's peak_tflops table has a real (non-null) entry for `precision` — single source of truth for the rule that unsupported precision must raise rather than silently substitute; `gpu_specs.yaml` encodes non-support as `~` (e.g. `a100_sxm_80gb.peak_tflops.fp8: ~`, Ampere has no native FP8 path), and both GpuPredictor and GpuRecommender call this before building a prediction."""
    peak = (gpu_spec.get("peak_tflops") or {}).get(precision)
    if peak is None:
        return False
    if isinstance(peak, float) and pd.isna(peak):
        return False
    return True


# --- Public pipeline ---

def build_training_df(
    raw_df: pd.DataFrame,
    spec_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Filter to result_valid rows, enrich with GPU specs, attach model-param references, compute roofline ceilings, and derive secondary features (efficiency ratio, VRAM fit, framework family, architecture ordinal, vendor indicator); rows that can't be featurised (unknown benchmark_base or missing GPU specs) are dropped with a warning rather than propagating NaN into training."""
    kwargs = {"spec_path": spec_path} if spec_path is not None else {}
    df = enrich_df(raw_df, **kwargs).copy()

    # --- filter --- gpu_in_model_scope gates recommendation exposure, not training inclusion; out-of-scope GPUs (e.g. B200, H200 NVL) remain valid training signal even though not served to users v1.
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
    # AMD CDNA hardware achieves 99.9 accuracy with FP8, not FP16 (the TIER_TO_PRECISION default, correct for NVIDIA) — override for AMD so efficiency_ratio is computed against the right (2x FP16 TFLOPS) ceiling, eliminating ceiling violations in training.
    amd_tier_99_9 = (df["gpu_vendor"] == "amd") & (df["benchmark_accuracy_tier"] == "99.9")
    df.loc[amd_tier_99_9, "selected_precision"] = "fp8"
    df["bytes_per_param"] = df["selected_precision"].map(BYTES_PER_PARAM)

    # Vectorised equivalent of _select_peak_tflops over all rows: covers the three TIER_TO_PRECISION values, falling back to fp16 for unknown precision or NaN selected-precision (same as the scalar helper).
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
    # Deduplicate framework strings before normalizing (~10-20 unique values across ~1112 rows) to avoid ~1112 redundant _normalize_framework calls — parallels the per-unique-gpu-name optimization in enrich_df.
    _fw_unique_map = {fw: _normalize_framework(fw) for fw in df["framework"].dropna().unique()}
    df["framework_family"] = df["framework"].map(_fw_unique_map).fillna("unknown")
    df["nvidia_arch_gen"] = df["gpu_architecture"].map(_NVIDIA_ARCH_ORDINAL)
    df["amd_arch_gen"] = df["gpu_architecture"].map(_AMD_ARCH_ORDINAL)
    df["vendor_is_amd"] = (df["gpu_vendor"] == "amd").astype(int)
    df["mlperf_round_num"] = df["round"].map(ROUND_ORDINAL)
    # Binary flag separating "base" tier (BF16) from "99"/"99.9" (FP8/FP16 or AMD FP8 override); replaces three-level accuracy_tier_ord, which became a spurious discriminator within AMD LOGO folds once the AMD FP8 override made "99" and "99.9" both FP8 — bytes_per_param already covers the FP8-vs-FP16 split, is_base_tier covers the remaining base-vs-non-base distinction it can't express (base BF16 = 2.0, same value as NVIDIA FP16).
    df["is_base_tier"] = (df["benchmark_accuracy_tier"] == "base").astype(int)

    # Efficiency ratio > 1 (actual throughput exceeds the inferred-precision compute ceiling) is expected for AMD CDNA4 "99.9"-tier rows, whose vLLM/ROCm stack really achieves 99.9 accuracy with FP8 while our proxy maps 99.9→FP16 — these rows are valid training data showing AMD outperforming the FP16 ceiling for this tier.
    n_violations = (df["throughput_tok_per_sec_per_gpu"] > df["roofline_tput"]).sum()
    if n_violations:
        log.warning(
            "%d rows (%.1f%%) have throughput > compute ceiling at selected "
            "precision — likely precision-proxy mismatch (AMD FP8 at 99.9 tier).",
            n_violations,
            100 * n_violations / len(df),
        )

    # Outlier-rejection rule (hard bound, unlike the >1.0 warning above): drop rows whose efficiency_ratio falls outside (0, MAX_EFFICIENCY_RATIO], including NaN/inf from a zero or missing roofline_tput; .notna() is technically redundant with `> 0` (NaN comparisons are always False under IEEE 754, confirmed by mutation testing 2026-07-12) but kept explicit for readability.
    valid_ratio = (
        df["efficiency_ratio"].notna()
        & (df["efficiency_ratio"] > 0)
        & (df["efficiency_ratio"] <= MAX_EFFICIENCY_RATIO)
    )
    if (~valid_ratio).any():
        dropped = df.loc[~valid_ratio, "efficiency_ratio"]
        log.warning(
            "Dropping %d rows with efficiency_ratio outside (0, %.1f] "
            "(sample values: %s) — outlier-rejection rule; likely a "
            "spec-DB or parse error, not valid training signal.",
            len(dropped),
            MAX_EFFICIENCY_RATIO,
            sorted(dropped.round(3).tolist())[:10],
        )
        df = df[valid_ratio]

    log.info("build_training_df complete: %d rows, %d columns", len(df), df.shape[1])
    return df.reset_index(drop=True)
