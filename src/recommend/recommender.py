"""
GPU Perf Prophet — Pareto recommendation engine.

Public API
----------
GpuRecommender(predictor, pricing_path)
    Wraps GpuPredictor and adds Pareto multi-objective ranking.

recommender.recommend(model_name, scenario, accuracy_tier, framework,
                      budget_per_gpu_hr, min_throughput_tok_per_sec)
    Return a ranked list of GPU recommendations for the given workload.

Objectives (all higher-is-better after normalisation):
    1. throughput — predicted tokens/sec
    2. cost_efficiency — tokens per dollar (throughput / price_per_hr)
    3. vram_headroom — fraction of VRAM unused (1 - model_to_vram_ratio)

Constraints (hard filters applied before ranking):
    • VRAM fit:  model must fit on a single GPU (model_size_gb ≤ gpu_vram_gb)
    • Budget:    price_per_gpu_hr ≤ budget_per_gpu_hr  (if provided)
    • Min tput:  pred_throughput ≥ min_throughput       (if provided)

Pareto frontier:
    A GPU is Pareto-dominated if another GPU is at least as good on all
    three objectives and strictly better on at least one.  The frontier
    contains all non-dominated GPUs, sorted by cost_efficiency descending
    (best tokens-per-dollar first).  Dominated GPUs are returned in a
    separate list so the UI can show them as alternatives.
"""

from __future__ import annotations

import logging
import stat as _stat
from pathlib import Path
from typing import Optional

import yaml

from src.data.gpu_spec_db import load_specs
from src.models.predictor import GpuPredictor, MODEL_PARAMS, TIER_TO_PRECISION, BYTES_PER_PARAM

log = logging.getLogger(__name__)

_DEFAULT_PRICING_PATH = Path(__file__).parent.parent.parent / "data" / "pricing.yaml"

# Real pricing files are < 1 KB.  1 MB cap matches gpu_spec_db.py's policy.
_MAX_PRICING_BYTES: int = 1 * 1024 * 1024  # 1 MB


def _load_pricing(path: Path) -> dict[str, float]:
    # Mirror the symlink and size guards from gpu_spec_db.load_specs so the
    # pricing file cannot be swapped out via a filesystem symlink.
    try:
        st = path.lstat()
    except OSError as exc:
        raise FileNotFoundError(f"Pricing DB not found: {path}") from exc
    if _stat.S_ISLNK(st.st_mode):
        raise ValueError(f"Pricing DB path is a symlink (refused): {path}")
    if st.st_size > _MAX_PRICING_BYTES:
        raise ValueError(
            f"Pricing DB too large ({st.st_size} bytes > {_MAX_PRICING_BYTES}): {path}"
        )
    with path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or "pricing" not in data:
        raise ValueError(f"pricing.yaml at {path} is missing required 'pricing' key")
    result: dict[str, float] = {}
    for gpu_id, entry in data["pricing"].items():
        if not isinstance(entry, dict) or "price_per_gpu_hr" not in entry:
            raise ValueError(
                f"pricing.yaml entry {gpu_id!r} is missing 'price_per_gpu_hr' key"
            )
        result[gpu_id] = entry["price_per_gpu_hr"]
    return result


def _pareto_frontier(candidates: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split candidates into (frontier, dominated).

    Each candidate dict must have numeric keys:
        throughput, cost_efficiency, vram_headroom  (all higher-is-better)

    None is treated as -inf for dominance comparisons and sorts last.
    """
    objectives = ["throughput", "cost_efficiency", "vram_headroom"]
    frontier: list[dict] = []
    dominated: list[dict] = []

    # None objective value → treat as -inf so unpriced GPUs are always
    # dominated by any priced GPU and don't crash comparisons.
    def _obj(v) -> float:
        return v if v is not None else float("-inf")

    def _dominates(a: dict, b: dict) -> bool:
        """Return True if a dominates b: a >= b on all objectives, a > b on ≥1.

        Single pass over objectives: returns False immediately on the first
        objective where a < b, avoiding the O(2k) double-evaluation of the
        separate all()/any() generators.
        """
        has_strict = False
        for obj in objectives:
            ao, bo = _obj(a[obj]), _obj(b[obj])
            if ao < bo:
                return False
            if ao > bo:
                has_strict = True
        return has_strict

    for i, cand in enumerate(candidates):
        is_dominated = any(
            _dominates(other, cand)
            for j, other in enumerate(candidates)
            if j != i
        )
        if is_dominated:
            dominated.append(cand)
        else:
            frontier.append(cand)

    # Sort frontier by cost_efficiency descending (best tokens-per-dollar first).
    # None sorts last so unpriced GPUs appear after all priced GPUs.
    def _ce(x: dict) -> float:
        v = x["cost_efficiency"]
        return v if v is not None else float("-inf")

    frontier.sort(key=_ce, reverse=True)
    dominated.sort(key=_ce, reverse=True)
    return frontier, dominated


class GpuRecommender:
    """Multi-objective GPU recommender wrapping GpuPredictor."""

    def __init__(
        self,
        predictor: GpuPredictor,
        pricing_path: Path | str = _DEFAULT_PRICING_PATH,
    ) -> None:
        self._predictor = predictor
        self._pricing = _load_pricing(Path(pricing_path))

        specs = load_specs()
        self._in_scope_ids: list[str] = [
            s["id"] for s in specs if s.get("in_model_scope")
        ]
        # Re-use the predictor's already-deep-copied spec map — this class never
        # writes to spec dicts, so sharing is safe and avoids a second full
        # deepcopy of every spec at init time.
        self._spec_map: dict[str, dict] = predictor._id_map

        # Fail fast: a missing pricing entry produces cost_efficiency=None,
        # which causes a TypeError in _pareto_frontier's sort and comparisons.
        missing = [gid for gid in self._in_scope_ids if gid not in self._pricing]
        if missing:
            raise ValueError(
                f"pricing.yaml is missing entries for in-scope GPUs: {missing}. "
                "Add a price_per_gpu_hr entry before enabling these GPUs."
            )

        log.info(
            "GpuRecommender ready: %d in-scope GPUs, %d pricing entries",
            len(self._in_scope_ids), len(self._pricing),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recommend(
        self,
        *,
        model_name: str,
        scenario: str = "Offline",
        accuracy_tier: str = "99",
        framework: str = "vllm",
        budget_per_gpu_hr: Optional[float] = None,
        min_throughput_tok_per_sec: Optional[float] = None,
    ) -> dict:
        """Return a recommendation result dict.

        Keys
        ----
        frontier : list[dict]   — Pareto-optimal GPUs, best tokens/$ first
        dominated: list[dict]   — remaining GPUs that passed hard constraints
        filtered : list[dict]   — GPUs removed by hard constraints (vram / budget)
        workload : dict         — echoed inputs + model_size_gb
        """
        if model_name not in MODEL_PARAMS:
            raise ValueError(
                f"Unknown model_name {model_name!r}. Valid: {sorted(MODEL_PARAMS)}"
            )
        total_params_b, _ = MODEL_PARAMS[model_name]
        bpp = BYTES_PER_PARAM[TIER_TO_PRECISION[accuracy_tier]]
        model_size_gb = total_params_b * bpp  # workload summary (canonical, FP16 for tier 99.9)

        # Per-GPU effective model size — AMD uses FP8 at 99.9 tier, halving the
        # footprint vs the FP16 default.  The VRAM pre-filter and reject messages
        # must use this per-GPU value; candidates use pred["model_size_gb"] which
        # comes from predict_batch() and already applies the same override.
        def _gpu_model_size_gb(gpu_id: str) -> float:
            spec = self._spec_map[gpu_id]
            eff_bpp = bpp
            if spec.get("vendor") == "amd" and accuracy_tier == "99.9":
                eff_bpp = BYTES_PER_PARAM["fp8"]
            return total_params_b * eff_bpp

        # Pre-filter by VRAM before calling predict_batch: skip XGBoost inference
        # for GPUs where the model provably cannot fit.  Matters most for large
        # models (llama3.1-405b at fp8 = 405 GB; no in-scope GPU reaches that).
        # Compute effective model size once per GPU (avoid 4× calls per failing GPU
        # across the dual list comprehensions and the reject entry builder below).
        gpu_sizes: dict[str, float] = {
            gid: _gpu_model_size_gb(gid) for gid in self._in_scope_ids
        }
        vram_ok_ids: list[str] = []
        vram_fail_ids: list[str] = []
        for gid in self._in_scope_ids:
            fits = self._spec_map[gid]["vram_gb"] >= gpu_sizes[gid]
            (vram_ok_ids if fits else vram_fail_ids).append(gid)

        requests = [
            {
                "gpu_id": gpu_id,
                "model_name": model_name,
                "scenario": scenario,
                "accuracy_tier": accuracy_tier,
                "framework": framework,
            }
            for gpu_id in vram_ok_ids
        ]
        predictions = self._predictor.predict_batch(requests)

        candidates: list[dict] = []
        filtered: list[dict] = []

        # Build reject entries for VRAM-failing GPUs without running inference.
        for gpu_id in vram_fail_ids:
            spec = self._spec_map[gpu_id]
            price = self._pricing.get(gpu_id)
            _sz = gpu_sizes[gpu_id]
            filtered.append({
                "gpu_id": gpu_id,
                "gpu_name": spec.get("name", gpu_id),
                "vendor": spec.get("vendor", ""),
                "model_name": model_name,
                "scenario": scenario,
                "accuracy_tier": accuracy_tier,
                "framework": framework,
                "pred_throughput_tok_per_sec": 0.0,
                "roofline_tput_tok_per_sec": 0.0,
                "efficiency_ratio": 0.0,
                "vram_fits": False,
                "model_size_gb": round(_sz, 2),
                "vram_gb": spec.get("vram_gb"),
                "price_per_gpu_hr": price,
                "vram_headroom": 0.0,
                "cost_efficiency": None,
                "throughput": 0.0,
                "reject_reason": (
                    f"model too large ({_sz:.1f} GB"
                    f" > {spec['vram_gb']} GB VRAM)"
                ),
            })

        for pred in predictions:
            gpu_id = pred["gpu_id"]
            spec = self._spec_map[gpu_id]
            price = self._pricing.get(gpu_id)
            pred_tput = pred["pred_throughput_tok_per_sec"]

            reject_reason = None
            if budget_per_gpu_hr is not None and price is not None and price > budget_per_gpu_hr:
                reject_reason = f"price ${price:.2f}/hr > budget ${budget_per_gpu_hr:.2f}/hr"
            elif min_throughput_tok_per_sec is not None and pred_tput < min_throughput_tok_per_sec:
                reject_reason = (
                    f"predicted {pred_tput:.0f} tok/s"
                    f" < minimum {min_throughput_tok_per_sec:.0f} tok/s"
                )

            entry = {
                **pred,
                "gpu_name":        spec.get("name", gpu_id),
                "vendor":          spec.get("vendor", ""),
                "vram_gb":         spec.get("vram_gb"),
                "price_per_gpu_hr": price,
                "vram_headroom":   max(0.0, 1.0 - pred["model_size_gb"] / spec["vram_gb"]),
                "cost_efficiency": (pred_tput / price) if price else None,
                "throughput":      pred_tput,
            }

            if reject_reason:
                entry["reject_reason"] = reject_reason
                filtered.append(entry)
            else:
                candidates.append(entry)

        frontier, dominated = _pareto_frontier(candidates)

        return {
            "frontier":  frontier,
            "dominated": dominated,
            "filtered":  filtered,
            "workload": {
                "model_name":     model_name,
                "scenario":       scenario,
                "accuracy_tier":  accuracy_tier,
                "framework":      framework,
                "model_size_gb":  round(model_size_gb, 2),
                "budget_per_gpu_hr": budget_per_gpu_hr,
                "min_throughput_tok_per_sec": min_throughput_tok_per_sec,
            },
        }
