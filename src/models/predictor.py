"""GPU Perf Prophet inference module: GpuPredictor loads the trained XGBoost model and predicts/predict_batch serve tokens/sec, with feature construction mirroring notebooks/03_model_training.ipynb exactly so the serving path cannot diverge."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import stat as _stat
from pathlib import Path

import numpy as np
import xgboost as xgb

from src.data.gpu_spec_db import load_specs
from src.features.build_features import (
    MODEL_PARAMS,
    MODEL_ARCH,
    TIER_TO_PRECISION,
    ROUND_ORDINAL,
    BYTES_PER_PARAM,
    DEFAULT_BATCH_SIZE,
    DEFAULT_INPUT_TOKENS,
    DEFAULT_OUTPUT_TOKENS,
    _NVIDIA_ARCH_ORDINAL,
    _AMD_ARCH_ORDINAL,
    roofline_ceilings,
    kv_cache_gb,
    memory_fit_verdict,
    validate_serving_shape,
    gpu_supports_precision,
)

log = logging.getLogger(__name__)

_DEFAULT_MODEL_DIR = Path(__file__).parent.parent.parent / "data" / "models"

# File-guard caps for model artifacts — mirrors load_specs() and _load_pricing().
_MAX_META_BYTES: int = 1 * 1024 * 1024   # 1 MB; real metadata JSON is ~1 KB
_MAX_MODEL_BYTES: int = 50 * 1024 * 1024  # 50 MB; matches the model-disk-size gate's budget

# Serving always predicts for the most mature software stack (most recent MLPerf round), correcting the ROCm-maturity confound without exposing round_tag as an API parameter.
_SERVING_ROUND: float = float(max(ROUND_ORDINAL.values()))

# This project's per-GPU Must-have minimum (>=100 rows/GPU); v1 ships all 8 GPUs regardless of whether they clear it — a deliberate, disclosed departure from the stated fallback (defer or gate off under-floor GPUs) — so training_data_tier() makes that gap visible instead of collapsing it into a single has-any-data bool.
MIN_TRAINING_ROWS_PER_GPU: int = 100

# Feature column order — must match FEATURE_COLS in the training notebook exactly.
FEATURE_COLS: list[str] = [
    # GPU hardware
    "gpu_hbm_bandwidth_tbps",
    "gpu_vram_gb",
    "peak_tflops_selected",
    "compute_ceiling_tok_per_sec",
    "bandwidth_ceiling_tok_per_sec",
    # Model
    "model_total_params_b",
    "model_compute_params_b",
    "model_size_gb",
    "model_to_vram_ratio",
    "bytes_per_param",
    # Architecture
    "nvidia_arch_gen",
    "amd_arch_gen",
    "vendor_is_amd",
    # Encoded categoricals
    "scenario_offline",
    "is_base_tier",
    "fw_tensorrt",
    "fw_vllm",
    "fw_rocm_other",
    "is_cdna4",
    # Context
    "mlperf_round_num",
]

VALID_SCENARIOS: frozenset[str] = frozenset({"Offline", "Server"})
VALID_TIERS: frozenset[str] = frozenset({"base", "99", "99.9"})
VALID_FRAMEWORKS: frozenset[str] = frozenset({"vllm", "tensorrt", "rocm_other", "other"})
VALID_MODELS: frozenset[str] = frozenset(MODEL_PARAMS.keys())


def _selected_precision(gpu_spec: dict, accuracy_tier: str) -> str:
    """Precision label this (GPU, tier) pair actually runs at; single source of truth for the AMD 99.9-tier FP8 override, shared by the ML feature vector and KV-cache memory-fit calc so the two can't diverge."""
    selected_precision = TIER_TO_PRECISION[accuracy_tier]
    if gpu_spec.get("vendor") == "amd" and accuracy_tier == "99.9":
        selected_precision = "fp8"
    return selected_precision


def _check_precision_supported(
    gpu_id: str, gpu_spec: dict, accuracy_tier: str, selected_precision: str
) -> None:
    """Raise ValueError if gpu_id has no native peak TFLOPS for selected_precision — without this guard, _build_feature_vector used to silently substitute the fp16 ceiling (e.g. FP8 on Ampere's a100_sxm_80gb), mixing FP8's bytes-per-param with FP16's compute ceiling with no signal to the caller."""
    if not gpu_supports_precision(gpu_spec, selected_precision):
        peak_tflops = gpu_spec.get("peak_tflops") or {}
        valid = sorted(p for p in peak_tflops if gpu_supports_precision(gpu_spec, p))
        raise ValueError(
            f"GPU {gpu_id!r} does not support precision {selected_precision!r} "
            f"(accuracy_tier={accuracy_tier!r}). "
            f"Valid precisions for this GPU: {valid}"
        )


def _memory_fit(
    *,
    gpu_spec: dict,
    model_name: str,
    bpp: float,
    weights_gb: float,
    batch_size: int,
    input_tokens: int,
    output_tokens: int,
) -> tuple[str, float, float, float]:
    """Return (verdict, kv_cache_gb, total_gb, utilization) for the memory-fit check; takes bpp directly since the caller already derived it via _selected_precision() when building the feature vector, avoiding a redundant re-derivation."""
    n_layers, n_kv_heads, head_dim = MODEL_ARCH[model_name]
    kv_gb = kv_cache_gb(
        n_layers, n_kv_heads, head_dim, batch_size, input_tokens, output_tokens, bpp
    )
    verdict, total_gb, utilization = memory_fit_verdict(
        weights_gb, kv_gb, gpu_spec["vram_gb"]
    )
    return verdict, kv_gb, total_gb, utilization


def _build_feature_vector(
    *,
    gpu_spec: dict,
    model_name: str,
    scenario: str,
    accuracy_tier: str,
    framework: str,
    selected_precision: str | None = None,
) -> tuple[list[float], float, float]:
    """Return (feature_vector, roofline_tput, model_size_gb) for one (GPU, workload) pair; pass selected_precision when the caller already derived it (e.g. for the KV-cache memory-fit calc too) to avoid deriving it twice, and roofline_tput/model_size_gb are returned so callers don't re-derive them with a second copy of the AMD precision override logic."""
    total_params_b, compute_params_b = MODEL_PARAMS[model_name]
    if selected_precision is None:
        selected_precision = _selected_precision(gpu_spec, accuracy_tier)
    bpp = BYTES_PER_PARAM[selected_precision]

    # Peak TFLOPS at the selected precision; never None/NaN via GpuPredictor.predict()/predict_batch() (already passed _check_precision_supported) — the fp16 fallback below only guards direct callers (e.g. tests) that skip that check.
    pt = gpu_spec.get("peak_tflops") or {}
    peak_tflops = pt.get(selected_precision)
    if peak_tflops is None or (isinstance(peak_tflops, float) and np.isnan(peak_tflops)):
        peak_tflops = pt.get("fp16")

    hbm_bw = gpu_spec["hbm_bandwidth_tbps"]
    vram_gb = gpu_spec["vram_gb"]

    bw_ceil, compute_ceil, roofline_tput = roofline_ceilings(
        total_params_b, compute_params_b, bpp, hbm_bw, peak_tflops
    )

    model_size_gb = total_params_b * bpp
    model_to_vram_ratio = model_size_gb / vram_gb

    arch = gpu_spec.get("architecture", "")
    nvidia_arch_gen = _NVIDIA_ARCH_ORDINAL.get(arch)
    amd_arch_gen = _AMD_ARCH_ORDINAL.get(arch)
    vendor_is_amd = int(gpu_spec.get("vendor", "") == "amd")

    is_cdna4 = int(amd_arch_gen == 2) if amd_arch_gen is not None else 0

    scenario_offline = int(scenario == "Offline")
    is_base_tier = int(accuracy_tier == "base")
    fw_tensorrt = int(framework == "tensorrt")
    fw_vllm = int(framework == "vllm")
    fw_rocm_other = int(framework == "rocm_other")

    # NaN for the other vendor's arch ordinal — XGBoost handles missing natively.
    features: list[float] = [
        hbm_bw,
        vram_gb,
        peak_tflops,
        compute_ceil,
        bw_ceil,
        total_params_b,
        compute_params_b,
        model_size_gb,
        model_to_vram_ratio,
        bpp,
        float(nvidia_arch_gen) if nvidia_arch_gen is not None else float("nan"),
        float(amd_arch_gen) if amd_arch_gen is not None else float("nan"),
        vendor_is_amd,
        scenario_offline,
        is_base_tier,
        fw_tensorrt,
        fw_vllm,
        fw_rocm_other,
        is_cdna4,
        _SERVING_ROUND,
    ]
    return features, roofline_tput, model_size_gb


class GpuPredictor:
    """Load-once, predict-many XGBoost inference wrapper."""

    def __init__(self, model_dir: Path | str = _DEFAULT_MODEL_DIR) -> None:
        model_dir = Path(model_dir)
        meta_path = model_dir / "feature_metadata.json"
        model_path = model_dir / "prophet_v1.json"

        # File guards (mirrors load_specs()/_load_pricing()): symlink checks block path-traversal redirects, size caps block unbounded memory use; applied before open() so they're never compiled away like assert under python -O.
        for _path, _cap in ((meta_path, _MAX_META_BYTES), (model_path, _MAX_MODEL_BYTES)):
            try:
                _st = _path.lstat()
            except OSError as exc:
                raise FileNotFoundError(f"Model artifact not found: {_path}") from exc
            if _stat.S_ISLNK(_st.st_mode):
                raise ValueError(f"Model artifact path is a symlink (refused): {_path}")
            if _st.st_size > _cap:
                raise ValueError(
                    f"Model artifact too large ({_st.st_size} bytes > {_cap}): {_path}"
                )

        with meta_path.open() as f:
            self._meta = json.load(f)

        # Explicit raise (not assert) so this check is never a no-op under `python -O` / PYTHONOPTIMIZE=1.
        if self._meta["feature_cols"] != FEATURE_COLS:
            raise ValueError(
                "feature_metadata.json feature_cols mismatch — retrain the model. "
                f"Expected {len(FEATURE_COLS)} cols, got "
                f"{len(self._meta.get('feature_cols', []))}."
            )

        self._model = xgb.XGBRegressor()
        self._model.load_model(str(model_path))

        # meta.model_artifact_sha256: hash of the actual model file bytes, distinct from feature_metadata.json's corpus_sha256 (hash of the *training data*) — this one changes iff the artifact itself changes.
        self._model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
        self._model_version: str = self._meta.get("model_version", "unknown")

        # GPUs with zero training rows extrapolate purely from specs; required key (not .get()) so an old feature_metadata.json predating this field fails loudly rather than silently reporting every GPU as having real training data.
        if "trained_gpu_ids" not in self._meta:
            raise ValueError(
                "feature_metadata.json missing 'trained_gpu_ids' — retrain the model."
            )
        self._trained_gpu_ids: frozenset[str] = frozenset(self._meta["trained_gpu_ids"])

        # Per-GPU row counts behind training_data_tier() — same required-key, fail-loudly convention as trained_gpu_ids above, so an old artifact doesn't silently report every trained GPU as meeting the 100-row floor.
        if "trained_gpu_row_counts" not in self._meta:
            raise ValueError(
                "feature_metadata.json missing 'trained_gpu_row_counts' — retrain the model."
            )
        self._trained_gpu_row_counts: dict[str, int] = self._meta["trained_gpu_row_counts"]

        specs = load_specs()
        # Deep-copy each spec dict: load_specs() is lru_cache'd and returns its live list, so storing direct references would let any write silently corrupt the global cache for all callers.
        self._id_map: dict[str, dict] = {s["id"]: copy.deepcopy(s) for s in specs}

        log.info(
            "GpuPredictor loaded: model=%s  features=%d  gpus=%d",
            model_path.name, len(FEATURE_COLS), len(self._id_map),
        )

    # ---- Public API ----

    @property
    def model_artifact_sha256(self) -> str:
        return self._model_sha256

    @property
    def model_version(self) -> str:
        return self._model_version

    def training_data_tier(self, gpu_id: str) -> str:
        """Where gpu_id's training-row count sits relative to the reliability floor: "none" (zero real rows, pure spec extrapolation), "below_floor" (nonzero but under MIN_TRAINING_ROWS_PER_GPU=100; see the module-level comment on that constant for why v1 ships these anyway), or "sufficient" (meets or exceeds the 100-row floor)."""
        n = self._trained_gpu_row_counts.get(gpu_id, 0)
        if n == 0:
            return "none"
        if n < MIN_TRAINING_ROWS_PER_GPU:
            return "below_floor"
        return "sufficient"

    def has_training_data(self, gpu_id: str) -> bool:
        """Whether gpu_id had at least one real measured row in training; True for both "below_floor" and "sufficient" tiers, so check training_data_tier() for whether it actually clears the reliability floor."""
        return self.training_data_tier(gpu_id) != "none"

    def predict(
        self,
        *,
        gpu_id: str,
        model_name: str,
        scenario: str = "Offline",
        accuracy_tier: str = "99",
        framework: str = "vllm",
        batch_size: int = DEFAULT_BATCH_SIZE,
        input_tokens: int = DEFAULT_INPUT_TOKENS,
        output_tokens: int = DEFAULT_OUTPUT_TOKENS,
    ) -> dict:
        """Predict inference throughput for one (GPU, workload) pair; batch_size/input_tokens/output_tokens drive the KV-cache memory-fit calc only (not ML features, since MLPerf rows carry no per-row batch/context-length info); returns a dict with gpu_id, model_name, scenario, accuracy_tier, framework, pred_throughput_tok_per_sec, roofline_tput_tok_per_sec, efficiency_ratio, vram_fits, memory_fit_verdict, kv_cache_gb, memory_total_gb, vram_utilization, has_training_data."""
        selected_precision = self._validate(
            gpu_id, model_name, scenario, accuracy_tier, framework,
            batch_size, input_tokens, output_tokens,
        )
        gpu_spec = self._id_map[gpu_id]

        features, roofline_tput, model_size_gb = _build_feature_vector(
            gpu_spec=gpu_spec,
            model_name=model_name,
            scenario=scenario,
            accuracy_tier=accuracy_tier,
            framework=framework,
            selected_precision=selected_precision,
        )

        X = np.array([features], dtype=np.float32)
        pred_eff = float(self._model.predict(X)[0])
        pred_tput = pred_eff * roofline_tput

        # Enforce roofline ceiling on the output (< 2% violation rate in CV; clamp rather than raise so API remains responsive).
        pred_tput = min(pred_tput, roofline_tput)

        verdict, kv_gb, total_gb, utilization = _memory_fit(
            gpu_spec=gpu_spec,
            model_name=model_name,
            bpp=BYTES_PER_PARAM[selected_precision],
            weights_gb=model_size_gb,
            batch_size=batch_size,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        tier = self.training_data_tier(gpu_id)

        return {
            "gpu_id": gpu_id,
            "model_name": model_name,
            "scenario": scenario,
            "accuracy_tier": accuracy_tier,
            "framework": framework,
            "pred_throughput_tok_per_sec": round(pred_tput, 2),
            "roofline_tput_tok_per_sec": round(roofline_tput, 2),
            "efficiency_ratio": round(pred_eff, 4),
            # True for "fits" AND "tight" — only "does_not_fit" is False; check memory_fit_verdict for the three-tier detail.
            "vram_fits": verdict != "does_not_fit",
            "memory_fit_verdict": verdict,
            "kv_cache_gb": round(kv_gb, 2),
            "memory_total_gb": round(total_gb, 2),
            "vram_utilization": round(utilization, 4),
            "model_size_gb": round(model_size_gb, 2),
            # True for both "below_floor" and "sufficient" — only "none" is False; check training_data_tier for the three-tier detail.
            "has_training_data": tier != "none",
            "training_data_tier": tier,
        }

    def predict_batch(self, requests: list[dict]) -> list[dict]:
        """Vectorised prediction over a list of request dicts (same keys as predict()'s kwargs); an optional "memory_fit" tuple is used verbatim instead of recomputed, since GpuRecommender already computes it per candidate GPU for its VRAM pre-filter before calling predict_batch() — without this, the same KV-cache/threshold math would run twice per GPU per recommend() call (other callers, e.g. /predict/batch, are unaffected since the key is simply absent)."""
        if not requests:
            return []

        feature_matrix: list[list[float]] = []
        roofline_tputs: list[float] = []
        meta: list[dict] = []

        for req in requests:
            gpu_id = req["gpu_id"]
            model_name = req["model_name"]
            scenario = req.get("scenario", "Offline")
            accuracy_tier = req.get("accuracy_tier", "99")
            framework = req.get("framework", "vllm")
            batch_size = req.get("batch_size", DEFAULT_BATCH_SIZE)
            input_tokens = req.get("input_tokens", DEFAULT_INPUT_TOKENS)
            output_tokens = req.get("output_tokens", DEFAULT_OUTPUT_TOKENS)

            selected_precision = self._validate(
                gpu_id, model_name, scenario, accuracy_tier, framework,
                batch_size, input_tokens, output_tokens,
            )
            gpu_spec = self._id_map[gpu_id]
            features, roofline_tput, model_size_gb = _build_feature_vector(
                gpu_spec=gpu_spec,
                model_name=model_name,
                scenario=scenario,
                accuracy_tier=accuracy_tier,
                framework=framework,
                selected_precision=selected_precision,
            )
            feature_matrix.append(features)
            roofline_tputs.append(roofline_tput)

            precomputed_fit = req.get("memory_fit")
            if precomputed_fit is not None:
                verdict, kv_gb, total_gb, utilization = precomputed_fit
            else:
                verdict, kv_gb, total_gb, utilization = _memory_fit(
                    gpu_spec=gpu_spec,
                    model_name=model_name,
                    bpp=BYTES_PER_PARAM[selected_precision],
                    weights_gb=model_size_gb,
                    batch_size=batch_size,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )

            tier = self.training_data_tier(gpu_id)

            meta.append({
                "gpu_id": gpu_id,
                "model_name": model_name,
                "scenario": scenario,
                "accuracy_tier": accuracy_tier,
                "framework": framework,
                "model_size_gb": round(model_size_gb, 2),
                "vram_fits": verdict != "does_not_fit",
                "memory_fit_verdict": verdict,
                "kv_cache_gb": round(kv_gb, 2),
                "memory_total_gb": round(total_gb, 2),
                "vram_utilization": round(utilization, 4),
                # True for both "below_floor" and "sufficient" — only "none" is False; check training_data_tier for the three-tier detail.
                "has_training_data": tier != "none",
                "training_data_tier": tier,
            })

        X = np.array(feature_matrix, dtype=np.float32)
        pred_effs = self._model.predict(X)

        results = []
        for req_meta, pred_eff, roofline_tput in zip(meta, pred_effs, roofline_tputs):
            pred_tput = min(float(pred_eff) * roofline_tput, roofline_tput)
            results.append({
                **req_meta,
                "pred_throughput_tok_per_sec": round(pred_tput, 2),
                "roofline_tput_tok_per_sec": round(roofline_tput, 2),
                "efficiency_ratio": round(float(pred_eff), 4),
            })
        return results

    # ---- Internal ----

    def _validate(
        self,
        gpu_id: str,
        model_name: str,
        scenario: str,
        accuracy_tier: str,
        framework: str,
        batch_size: int = DEFAULT_BATCH_SIZE,
        input_tokens: int = DEFAULT_INPUT_TOKENS,
        output_tokens: int = DEFAULT_OUTPUT_TOKENS,
    ) -> str:
        """Validate every field; return the selected_precision derived along the way so callers don't need a second _selected_precision() call (same reason _build_feature_vector()/_memory_fit() take selected_precision/bpp as parameters instead of re-deriving them)."""
        if gpu_id not in self._id_map:
            raise ValueError(
                f"Unknown gpu_id {gpu_id!r}. "
                f"Valid: {sorted(self._id_map)}"
            )
        if model_name not in VALID_MODELS:
            raise ValueError(
                f"Unknown model_name {model_name!r}. "
                f"Valid: {sorted(VALID_MODELS)}"
            )
        if scenario not in VALID_SCENARIOS:
            raise ValueError(
                f"Invalid scenario {scenario!r}. Valid: {sorted(VALID_SCENARIOS)}"
            )
        if accuracy_tier not in VALID_TIERS:
            raise ValueError(
                f"Invalid accuracy_tier {accuracy_tier!r}. Valid: {sorted(VALID_TIERS)}"
            )
        # gpu_id/accuracy_tier are known-valid here (checked above), so this can never KeyError; folded into _validate() rather than a separate call, the same "one shared gate" principle validate_serving_shape already established.
        gpu_spec = self._id_map[gpu_id]
        selected_precision = _selected_precision(gpu_spec, accuracy_tier)
        _check_precision_supported(gpu_id, gpu_spec, accuracy_tier, selected_precision)
        validate_serving_shape(batch_size, input_tokens, output_tokens)
        if framework not in VALID_FRAMEWORKS:
            raise ValueError(
                f"Invalid framework {framework!r}. Valid: {sorted(VALID_FRAMEWORKS)}"
            )
        return selected_precision
