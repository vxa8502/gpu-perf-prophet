"""
GPU Perf Prophet — inference module.

Public API
----------
GpuPredictor(model_dir)
    Load the trained XGBoost model and metadata from disk.

predictor.predict(gpu_id, model_name, scenario, accuracy_tier, framework)
    Predict tokens/sec for one (GPU, workload) pair.

predictor.predict_batch(requests)
    Vectorised prediction for a list of dicts.

Feature construction mirrors the exact encoding used in
notebooks/03_model_training.ipynb so the serving path cannot diverge.
"""

from __future__ import annotations

import copy
import json
import logging
import stat as _stat
from pathlib import Path

import numpy as np
import xgboost as xgb

from src.data.gpu_spec_db import load_specs
from src.features.build_features import (
    MODEL_PARAMS,
    TIER_TO_PRECISION,
    ROUND_ORDINAL,
    BYTES_PER_PARAM,
    _NVIDIA_ARCH_ORDINAL,
    _AMD_ARCH_ORDINAL,
    roofline_ceilings,
)

log = logging.getLogger(__name__)

_DEFAULT_MODEL_DIR = Path(__file__).parent.parent.parent / "data" / "models"

# File-guard caps for model artifacts — mirrors load_specs() and _load_pricing().
_MAX_META_BYTES: int = 1 * 1024 * 1024   # 1 MB; real metadata JSON is ~1 KB
_MAX_MODEL_BYTES: int = 50 * 1024 * 1024  # 50 MB; matches RT-5.3 disk budget

# At serving time we always predict for the most mature software stack —
# i.e. the most recent MLPerf round.  This corrects the ROCm-maturity
# confound without exposing round_tag as an API parameter.
_SERVING_ROUND: float = float(max(ROUND_ORDINAL.values()))

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


def _build_feature_vector(
    *,
    gpu_spec: dict,
    model_name: str,
    scenario: str,
    accuracy_tier: str,
    framework: str,
) -> tuple[list[float], float, float]:
    """Return (feature_vector, roofline_tput, model_size_gb) for one (GPU, workload) pair.

    roofline_tput and model_size_gb are returned so callers don't re-derive
    them with a second copy of the AMD precision override logic.
    """
    total_params_b, compute_params_b = MODEL_PARAMS[model_name]

    selected_precision = TIER_TO_PRECISION[accuracy_tier]
    # Mirror the AMD 99.9-tier override from build_features.build_training_df.
    if gpu_spec.get("vendor") == "amd" and accuracy_tier == "99.9":
        selected_precision = "fp8"
    bpp = BYTES_PER_PARAM[selected_precision]

    # Peak TFLOPS: use selected precision, fall back to fp16 if None/NaN.
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

        # File guards — mirrors the policy in load_specs() and _load_pricing().
        # Symlink checks prevent path-traversal redirects; size caps prevent
        # unbounded memory consumption.  Applied before open() so the checks
        # are not compiled away (unlike assert statements under python -O).
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

        # assert is compiled away under `python -O` / PYTHONOPTIMIZE=1 — use
        # explicit raise so this check is never a no-op in any deployment mode.
        if self._meta["feature_cols"] != FEATURE_COLS:
            raise ValueError(
                "feature_metadata.json feature_cols mismatch — retrain the model. "
                f"Expected {len(FEATURE_COLS)} cols, got "
                f"{len(self._meta.get('feature_cols', []))}."
            )

        self._model = xgb.XGBRegressor()
        self._model.load_model(str(model_path))

        specs = load_specs()
        # Deep-copy each spec dict so _id_map holds independent objects.
        # load_specs() is lru_cache'd and returns its live list; storing direct
        # references means any write to a spec value (e.g. caching a derived
        # field) would silently corrupt the global cache for all callers.
        self._id_map: dict[str, dict] = {s["id"]: copy.deepcopy(s) for s in specs}

        log.info(
            "GpuPredictor loaded: model=%s  features=%d  gpus=%d",
            model_path.name, len(FEATURE_COLS), len(self._id_map),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        *,
        gpu_id: str,
        model_name: str,
        scenario: str = "Offline",
        accuracy_tier: str = "99",
        framework: str = "vllm",
    ) -> dict:
        """Predict inference throughput for one (GPU, workload) pair.

        Returns
        -------
        dict with keys:
            gpu_id, model_name, scenario, accuracy_tier, framework,
            pred_throughput_tok_per_sec, roofline_tput_tok_per_sec,
            efficiency_ratio, vram_fits
        """
        self._validate(gpu_id, model_name, scenario, accuracy_tier, framework)
        gpu_spec = self._id_map[gpu_id]

        features, roofline_tput, model_size_gb = _build_feature_vector(
            gpu_spec=gpu_spec,
            model_name=model_name,
            scenario=scenario,
            accuracy_tier=accuracy_tier,
            framework=framework,
        )

        X = np.array([features], dtype=np.float32)
        pred_eff = float(self._model.predict(X)[0])
        pred_tput = pred_eff * roofline_tput

        # Enforce roofline ceiling on the output (< 2% violation rate in CV;
        # clamp rather than raise so API remains responsive).
        pred_tput = min(pred_tput, roofline_tput)

        vram_fits = model_size_gb <= gpu_spec["vram_gb"]

        return {
            "gpu_id": gpu_id,
            "model_name": model_name,
            "scenario": scenario,
            "accuracy_tier": accuracy_tier,
            "framework": framework,
            "pred_throughput_tok_per_sec": round(pred_tput, 2),
            "roofline_tput_tok_per_sec": round(roofline_tput, 2),
            "efficiency_ratio": round(pred_eff, 4),
            "vram_fits": vram_fits,
            "model_size_gb": round(model_size_gb, 2),
        }

    def predict_batch(self, requests: list[dict]) -> list[dict]:
        """Vectorised prediction over a list of request dicts.

        Each dict must contain the same keys as predict() keyword args.
        """
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

            self._validate(gpu_id, model_name, scenario, accuracy_tier, framework)
            gpu_spec = self._id_map[gpu_id]
            features, roofline_tput, model_size_gb = _build_feature_vector(
                gpu_spec=gpu_spec,
                model_name=model_name,
                scenario=scenario,
                accuracy_tier=accuracy_tier,
                framework=framework,
            )
            feature_matrix.append(features)
            roofline_tputs.append(roofline_tput)

            meta.append({
                "gpu_id": gpu_id,
                "model_name": model_name,
                "scenario": scenario,
                "accuracy_tier": accuracy_tier,
                "framework": framework,
                "model_size_gb": round(model_size_gb, 2),
                "vram_fits": model_size_gb <= gpu_spec["vram_gb"],
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

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _validate(
        self,
        gpu_id: str,
        model_name: str,
        scenario: str,
        accuracy_tier: str,
        framework: str,
    ) -> None:
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
        if framework not in VALID_FRAMEWORKS:
            raise ValueError(
                f"Invalid framework {framework!r}. Valid: {sorted(VALID_FRAMEWORKS)}"
            )
