"""GPU Perf Prophet recommendation engine: GpuRecommender.recommend() ranks GPUs on a 3-objective (throughput maximize, price_per_gpu_hr minimize, watts minimize) Pareto frontier under hard VRAM-fit/budget/min-throughput constraints; vram_headroom/cost_efficiency/tokens_per_watt/cost_per_million_tokens are computed per candidate but are not part of the dominance check itself, and dominated GPUs are returned separately (sorted the same way) as alternatives."""

from __future__ import annotations

import logging
import stat as _stat
from pathlib import Path
from typing import Optional

import yaml

from src.data.gpu_spec_db import load_specs
from src.features.build_features import cost_per_million_tokens
from src.models.predictor import (
    GpuPredictor,
    MODEL_PARAMS,
    MODEL_ARCH,
    TIER_TO_PRECISION,
    BYTES_PER_PARAM,
    DEFAULT_BATCH_SIZE,
    DEFAULT_INPUT_TOKENS,
    DEFAULT_OUTPUT_TOKENS,
    VALID_SCENARIOS,
    VALID_TIERS,
    VALID_FRAMEWORKS,
    kv_cache_gb,
    memory_fit_verdict,
    validate_serving_shape,
    gpu_supports_precision,
    _selected_precision,
)

log = logging.getLogger(__name__)

_DEFAULT_PRICING_PATH = Path(__file__).parent.parent.parent / "data" / "pricing.yaml"

# Real pricing files are < 1 KB.  1 MB cap matches gpu_spec_db.py's policy.
_MAX_PRICING_BYTES: int = 1 * 1024 * 1024  # 1 MB


def _load_pricing(path: Path) -> tuple[dict[str, float], Optional[str]]:
    """Return (gpu_id -> price_per_gpu_hr, source_date). source_date is the pricing snapshot date (meta.pricing_snapshot_date) — None for a pricing.yaml predating that key rather than a hard failure, since pricing itself still loads fine without it."""
    # Mirror gpu_spec_db.load_specs' symlink/size guards so the pricing file can't be swapped out via a filesystem symlink.
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
    return result, data.get("source_date")


# ranking_objective name -> (candidate dict field, higher_is_better); lowest_cost_per_million_tokens is the one ascending (lower-is-better) case.
_RANKING_FIELDS: dict[str, tuple[str, bool]] = {
    "tokens_per_dollar":              ("cost_efficiency", True),
    "tokens_per_second":              ("throughput", True),
    "tokens_per_watt":                ("tokens_per_watt", True),
    "lowest_cost_per_million_tokens": ("cost_per_million_tokens", False),
}

# The only values recommend()'s ranking_objective accepts; declared independently of _RANKING_FIELDS' keys (not derived) so the gate cross-check test has something real to catch, and lives here rather than build_features.py since (unlike VALID_MEMORY_FIT_VERDICTS) nothing outside recommender.py reads it.
VALID_RANKING_OBJECTIVES: frozenset[str] = frozenset({
    "tokens_per_dollar",
    "tokens_per_second",
    "tokens_per_watt",
    "lowest_cost_per_million_tokens",
})


def _ranking_key(ranking_objective: str):
    """Sort key for a candidate dict, best-first: negates higher-is-better fields so ascending sort = best-first; None (unpriced/no TDP) maps to +inf so it always sorts last."""
    field, higher_is_better = _RANKING_FIELDS[ranking_objective]

    def key(cand: dict) -> float:
        v = cand[field]
        if v is None:
            return float("inf")
        return -v if higher_is_better else v

    return key


def _pareto_frontier(
    candidates: list[dict],
    ranking_objective: str = "tokens_per_dollar",
) -> tuple[list[dict], list[dict]]:
    """Split candidates into (frontier, dominated) using the (throughput maximize, price_per_gpu_hr minimize, watts minimize) objective vector, sorted post-split by ranking_objective; None is treated as worst-possible so it never wins a dominance comparison."""
    frontier: list[dict] = []
    dominated: list[dict] = []

    def _obj_vector(cand: dict) -> tuple[float, float, float]:
        tput = cand["throughput"]
        price = cand["price_per_gpu_hr"]
        watts = cand["watts"]
        return (
            tput if tput is not None else float("-inf"),
            -price if price is not None else float("-inf"),
            -watts if watts is not None else float("-inf"),
        )

    def _dominates(a_vec: tuple[float, float, float], b_vec: tuple[float, float, float]) -> bool:
        """Return True if a_vec dominates b_vec (>= on all objectives, > on ≥1) in a single pass, avoiding the O(2k) double-evaluation of separate all()/any() generators."""
        has_strict = False
        for ao, bo in zip(a_vec, b_vec):
            if ao < bo:
                return False
            if ao > bo:
                has_strict = True
        return has_strict

    # Precompute each candidate's objective vector once instead of recomputing it per pairwise comparison in _dominates() (was O(n^2) total; measured 336 field accesses for 8 candidates vs. a 24 theoretical minimum, a 14x redundancy factor).
    vectors = [_obj_vector(cand) for cand in candidates]

    for i, cand in enumerate(candidates):
        is_dominated = any(
            _dominates(vectors[j], vectors[i])
            for j in range(len(candidates))
            if j != i
        )
        if is_dominated:
            dominated.append(cand)
        else:
            frontier.append(cand)

    key = _ranking_key(ranking_objective)
    frontier.sort(key=key)
    dominated.sort(key=key)
    return frontier, dominated


class GpuRecommender:
    """Multi-objective GPU recommender wrapping GpuPredictor."""

    def __init__(
        self,
        predictor: GpuPredictor,
        pricing_path: Path | str = _DEFAULT_PRICING_PATH,
    ) -> None:
        self._predictor = predictor
        self._pricing, self._pricing_source_date = _load_pricing(Path(pricing_path))

        specs = load_specs()
        self._in_scope_ids: list[str] = [
            s["id"] for s in specs if s.get("in_model_scope")
        ]
        # Re-use the predictor's already-deep-copied spec map — this class never writes to spec dicts, so sharing is safe and avoids a second full deepcopy at init.
        self._spec_map: dict[str, dict] = predictor._id_map

        # Fail fast: a missing pricing entry produces cost_efficiency=None, which would TypeError in _pareto_frontier's sort/comparisons.
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

    @property
    def pricing_source_date(self) -> Optional[str]:
        return self._pricing_source_date

    # Public API

    def recommend(
        self,
        *,
        model_name: str,
        scenario: str = "Offline",
        accuracy_tier: str = "99",
        framework: str = "vllm",
        batch_size: int = DEFAULT_BATCH_SIZE,
        input_tokens: int = DEFAULT_INPUT_TOKENS,
        output_tokens: int = DEFAULT_OUTPUT_TOKENS,
        budget_per_gpu_hr: Optional[float] = None,
        min_throughput_tok_per_sec: Optional[float] = None,
        ranking_objective: str = "tokens_per_dollar",
    ) -> dict:
        """Return a recommendation result dict with frontier/dominated/filtered candidate lists plus the echoed workload; batch_size/input_tokens/output_tokens only drive the KV-cache memory-fit check (not the throughput model), and ranking_objective only orders the frontier (default "tokens_per_dollar"), never changes which GPUs make it."""
        if ranking_objective not in VALID_RANKING_OBJECTIVES:
            raise ValueError(
                f"Invalid ranking_objective {ranking_objective!r}. "
                f"Valid: {sorted(VALID_RANKING_OBJECTIVES)}"
            )
        # Validated up front (not just implicitly via predict_batch()) since the memory-fit pre-filter below uses these values directly; an out-of-range batch_size that excludes every GPU would otherwise return a normal-looking response instead of raising, unlike predict() — two entry points silently disagreeing on the input contract.
        validate_serving_shape(batch_size, input_tokens, output_tokens)

        if model_name not in MODEL_PARAMS:
            raise ValueError(
                f"Unknown model_name {model_name!r}. Valid: {sorted(MODEL_PARAMS)}"
            )
        # accuracy_tier/scenario/framework are validated here too (not just implicitly by FastAPI/Streamlit) since recommend() is a public method that must be safe for untrusted input; before this check, an invalid accuracy_tier raised an uncaught KeyError (not the usual ValueError) and a garbage scenario/framework could silently pass through unvalidated whenever every candidate GPU was excluded before reaching predict_batch()'s own checks.
        if accuracy_tier not in VALID_TIERS:
            raise ValueError(
                f"Invalid accuracy_tier {accuracy_tier!r}. Valid: {sorted(VALID_TIERS)}"
            )
        if scenario not in VALID_SCENARIOS:
            raise ValueError(
                f"Invalid scenario {scenario!r}. Valid: {sorted(VALID_SCENARIOS)}"
            )
        if framework not in VALID_FRAMEWORKS:
            raise ValueError(
                f"Invalid framework {framework!r}. Valid: {sorted(VALID_FRAMEWORKS)}"
            )
        total_params_b, _ = MODEL_PARAMS[model_name]
        bpp = BYTES_PER_PARAM[TIER_TO_PRECISION[accuracy_tier]]
        model_size_gb = total_params_b * bpp  # workload summary (canonical, FP16 for tier 99.9)
        n_layers, n_kv_heads, head_dim = MODEL_ARCH[model_name]

        # Per-GPU memory fit: AMD uses FP8 at the 99.9 tier, halving weights and KV cache vs. the FP16 default (KV is stored at the same precision as weights); the VRAM pre-filter and reject messages must use this per-GPU value, matching predict_batch()'s own override.
        def _gpu_memory_fit(gpu_id: str, selected_precision: str) -> tuple[str, float, float, float, float]:
            """Return (verdict, weights_gb, kv_gb, total_gb, utilization); takes selected_precision as a parameter instead of re-deriving it, since the caller already computed it once (was 16 calls for 8 GPUs, now 8)."""
            spec = self._spec_map[gpu_id]
            eff_bpp = BYTES_PER_PARAM[selected_precision]
            weights_gb = total_params_b * eff_bpp
            kv_gb = kv_cache_gb(
                n_layers, n_kv_heads, head_dim,
                batch_size, input_tokens, output_tokens, eff_bpp,
            )
            verdict, total_gb, utilization = memory_fit_verdict(
                weights_gb, kv_gb, spec["vram_gb"]
            )
            return verdict, weights_gb, kv_gb, total_gb, utilization

        # Precision-support pre-filter (before memory-fit): a GPU whose peak_tflops table has no native entry for the tier's selected precision must never reach predict_batch() (which now raises for this case), so it's excluded here with a reason instead of crashing the whole recommend() call, reusing "does_not_fit" since there's no "unsupported_precision" verdict in the closed MemoryFitVerdict schema; selected_precision is derived once per GPU and threaded through everything below instead of re-derived at each use site (was 16 calls for 8 in-scope GPUs, now 8).
        precisions: dict[str, str] = {
            gid: _selected_precision(self._spec_map[gid], accuracy_tier)
            for gid in self._in_scope_ids
        }
        precision_ok_ids: list[str] = []
        precision_fail_ids: list[str] = []
        for gid in self._in_scope_ids:
            if gpu_supports_precision(self._spec_map[gid], precisions[gid]):
                precision_ok_ids.append(gid)
            else:
                precision_fail_ids.append(gid)

        # Pre-filter by memory fit before predict_batch to skip XGBoost inference for GPUs that provably can't fit (e.g. llama3.1-405b at fp8 = 405 GB, no in-scope GPU reaches that); fit tuple computed once per GPU and reused below.
        gpu_mem: dict[str, tuple[str, float, float, float, float]] = {
            gid: _gpu_memory_fit(gid, precisions[gid]) for gid in precision_ok_ids
        }
        vram_ok_ids: list[str] = []
        vram_fail_ids: list[str] = []
        for gid in precision_ok_ids:
            verdict, *_ = gpu_mem[gid]
            (vram_fail_ids if verdict == "does_not_fit" else vram_ok_ids).append(gid)

        # Pass the memory fit already computed above straight through — predict_batch() would otherwise redo the same KV-cache + threshold math per GPU.
        requests = [
            {
                "gpu_id": gpu_id,
                "model_name": model_name,
                "scenario": scenario,
                "accuracy_tier": accuracy_tier,
                "framework": framework,
                "batch_size": batch_size,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "memory_fit": (
                    gpu_mem[gpu_id][0],  # verdict
                    gpu_mem[gpu_id][2],  # kv_gb
                    gpu_mem[gpu_id][3],  # total_gb
                    gpu_mem[gpu_id][4],  # utilization
                ),
            }
            for gpu_id in vram_ok_ids
        ]
        predictions = self._predictor.predict_batch(requests)

        candidates: list[dict] = []
        filtered: list[dict] = []

        # Build reject entries for precision-unsupported GPUs, which never touched memory-fit or predict_batch() (see the precision pre-filter above).
        for gpu_id in precision_fail_ids:
            spec = self._spec_map[gpu_id]
            price = self._pricing.get(gpu_id)
            sel_prec = precisions[gpu_id]
            tier = self._predictor.training_data_tier(gpu_id)
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
                "memory_fit_verdict": "does_not_fit",
                "kv_cache_gb": 0.0,
                "memory_total_gb": 0.0,
                "vram_utilization": 0.0,
                "has_training_data": tier != "none",
                "training_data_tier": tier,
                "model_size_gb": 0.0,
                "vram_gb": spec.get("vram_gb"),
                "price_per_gpu_hr": price,
                "vram_headroom": 0.0,
                "cost_efficiency": None,
                "throughput": 0.0,
                "watts": spec.get("tdp_w"),
                "tokens_per_watt": None,
                "cost_per_million_tokens": None,
                "reject_reason": (
                    f"{sel_prec} not supported on {spec.get('name', gpu_id)}"
                ),
            })

        # Build reject entries for memory-failing GPUs without running inference.
        for gpu_id in vram_fail_ids:
            spec = self._spec_map[gpu_id]
            price = self._pricing.get(gpu_id)
            verdict, weights_gb, kv_gb, total_gb, utilization = gpu_mem[gpu_id]
            tier = self._predictor.training_data_tier(gpu_id)
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
                "memory_fit_verdict": verdict,
                "kv_cache_gb": round(kv_gb, 2),
                "memory_total_gb": round(total_gb, 2),
                "vram_utilization": round(utilization, 4),
                "has_training_data": tier != "none",
                "training_data_tier": tier,
                "model_size_gb": round(weights_gb, 2),
                "vram_gb": spec.get("vram_gb"),
                "price_per_gpu_hr": price,
                "vram_headroom": 0.0,
                "cost_efficiency": None,
                "throughput": 0.0,
                "watts": spec.get("tdp_w"),
                "tokens_per_watt": None,
                "cost_per_million_tokens": None,
                "reject_reason": (
                    f"model needs {total_gb:.1f} GB (weights + KV cache + overhead)"
                    f" > {spec['vram_gb']} GB VRAM"
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

            watts = spec.get("tdp_w")
            entry = {
                **pred,
                "gpu_name":        spec.get("name", gpu_id),
                "vendor":          spec.get("vendor", ""),
                "vram_gb":         spec.get("vram_gb"),
                "price_per_gpu_hr": price,
                "vram_headroom":   max(0.0, 1.0 - pred["memory_total_gb"] / spec["vram_gb"]),
                "cost_efficiency": (pred_tput / price) if price else None,
                "throughput":      pred_tput,
                "watts":           watts,
                "tokens_per_watt": (pred_tput / watts) if watts else None,
                "cost_per_million_tokens": cost_per_million_tokens(price, pred_tput),
            }

            if reject_reason:
                entry["reject_reason"] = reject_reason
                filtered.append(entry)
            else:
                candidates.append(entry)

        frontier, dominated = _pareto_frontier(candidates, ranking_objective)

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
                "batch_size":     batch_size,
                "input_tokens":   input_tokens,
                "output_tokens":  output_tokens,
                "budget_per_gpu_hr": budget_per_gpu_hr,
                "min_throughput_tok_per_sec": min_throughput_tok_per_sec,
                "ranking_objective": ranking_objective,
            },
        }
