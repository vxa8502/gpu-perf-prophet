"""
GPU Perf Prophet — Pareto recommendation engine.

Public API
----------
GpuRecommender(predictor, pricing_path)
    Wraps GpuPredictor and adds Pareto multi-objective ranking.

recommender.recommend(model_name, scenario, accuracy_tier, framework,
                      budget_per_gpu_hr, min_throughput_tok_per_sec,
                      ranking_objective)
    Return a ranked list of GPU recommendations for the given workload.

Pareto objectives (all normalized to higher-is-better internally):
    1. throughput      — predicted tokens/sec                    (maximize)
    2. price_per_gpu_hr — static cloud-pricing snapshot           (minimize)
    3. watts           — GPU TDP, gpu_specs.yaml tdp_w            (minimize)

vram_headroom/cost_efficiency/tokens_per_watt/cost_per_million_tokens are
still computed and returned per candidate (UI/API-visible, and the last
three double as ranking scalars), but are not part of the dominance check
itself; memory-fit is already a hard constraint (below), so a soft
VRAM-headroom axis was never part of the objective vector.

Constraints (hard filters applied before ranking):
    • VRAM fit:  model must fit on a single GPU (model_size_gb ≤ gpu_vram_gb)
    • Budget:    price_per_gpu_hr ≤ budget_per_gpu_hr  (if provided)
    • Min tput:  pred_throughput ≥ min_throughput       (if provided)

Pareto frontier:
    A GPU is Pareto-dominated if another GPU is at least as good on all
    three objectives and strictly better on at least one.  The frontier
    contains all non-dominated GPUs, sorted by the caller's ranking_objective
    (default tokens_per_dollar).  Dominated GPUs are returned in a
    separate list, sorted the same way, so the UI can show them as
    alternatives.
"""

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


# ranking_objective name -> (candidate dict field, higher_is_better).
# lowest_cost_per_million_tokens is the one ascending (lower-is-better) case.
_RANKING_FIELDS: dict[str, tuple[str, bool]] = {
    "tokens_per_dollar":              ("cost_efficiency", True),
    "tokens_per_second":              ("throughput", True),
    "tokens_per_watt":                ("tokens_per_watt", True),
    "lowest_cost_per_million_tokens": ("cost_per_million_tokens", False),
}

# The only values recommend()'s ranking_objective parameter ever accepts.
# Declared independently of _RANKING_FIELDS' keys (not derived via
# frozenset(_RANKING_FIELDS)) so the gate cross-check test — the two must be
# edited together — actually has something to catch; a derived frozenset
# would make that assertion vacuously true. Lives here, not build_features.py
# (where it was originally placed): nothing outside recommender.py reads it —
# unlike VALID_MEMORY_FIT_VERDICTS, which pairs with memory_fit_verdict() and
# is genuinely shared by both GpuPredictor and GpuRecommender.
VALID_RANKING_OBJECTIVES: frozenset[str] = frozenset({
    "tokens_per_dollar",
    "tokens_per_second",
    "tokens_per_watt",
    "lowest_cost_per_million_tokens",
})


def _ranking_key(ranking_objective: str):
    """Sort key for a candidate dict, best-first, per ranking_objective.

    Transforms every objective into "ascending sort = best first": negate
    higher-is-better fields, pass lower-is-better fields through as-is.
    None (unknown/undefined for this candidate — e.g. unpriced, or no TDP
    on file) always maps to +inf so it sorts last regardless of direction.
    """
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
    """Split candidates into (frontier, dominated).

    Each candidate dict must have numeric keys:
        throughput, price_per_gpu_hr, watts
    (the objective vector — throughput maximized, the other two
    minimized) plus whichever field `ranking_objective` names (see
    _RANKING_FIELDS) for the post-split sort.

    None is treated as the worst possible value for dominance comparisons
    (never lets an unknown value make a candidate look artificially good)
    and always sorts last.
    """
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
        """Return True if a_vec dominates b_vec: >= on all objectives, > on ≥1.

        Single pass over objectives: returns False immediately on the first
        objective where a < b, avoiding the O(2k) double-evaluation of the
        separate all()/any() generators.
        """
        has_strict = False
        for ao, bo in zip(a_vec, b_vec):
            if ao < bo:
                return False
            if ao > bo:
                has_strict = True
        return has_strict

    # Precompute each candidate's objective vector once — _dominates() used to
    # take raw candidate dicts and recompute _obj_vector() on every pairwise
    # comparison, an O(n) recomputation per candidate (O(n^2) total across the
    # full sweep below) for a value that depends only on that candidate's own
    # fields. Measured 336 field accesses for 8 mutually non-dominated
    # candidates (a real gptj/tier-99 shape) against a theoretical minimum of
    # 24 — a 14x redundancy factor.
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
        batch_size: int = DEFAULT_BATCH_SIZE,
        input_tokens: int = DEFAULT_INPUT_TOKENS,
        output_tokens: int = DEFAULT_OUTPUT_TOKENS,
        budget_per_gpu_hr: Optional[float] = None,
        min_throughput_tok_per_sec: Optional[float] = None,
        ranking_objective: str = "tokens_per_dollar",
    ) -> dict:
        """Return a recommendation result dict.

        batch_size/input_tokens/output_tokens drive the KV-cache memory-fit
        calculation only, not the throughput model itself —
        see GpuPredictor.predict() docstring.

        ranking_objective selects the scalar the Pareto-optimal
        (rank-1) set is sorted by — one of VALID_RANKING_OBJECTIVES, default
        "tokens_per_dollar". Does not affect which GPUs make the frontier
        (that's the fixed 3-objective dominance check), only their order.

        Keys
        ----
        frontier : list[dict]   — Pareto-optimal GPUs, ranked by ranking_objective
        dominated: list[dict]   — remaining GPUs that passed hard constraints
        filtered : list[dict]   — GPUs removed by hard constraints (vram / budget)
        workload : dict         — echoed inputs + model_size_gb
        """
        if ranking_objective not in VALID_RANKING_OBJECTIVES:
            raise ValueError(
                f"Invalid ranking_objective {ranking_objective!r}. "
                f"Valid: {sorted(VALID_RANKING_OBJECTIVES)}"
            )
        # Validated up front, not just implicitly via predict_batch(): the
        # memory-fit pre-filter below uses these three values directly, and
        # only GPUs that pass it ever reach predict_batch()'s own validation.
        # An out-of-range batch_size that makes every in-scope GPU look like
        # "does_not_fit" would otherwise return a normal-looking response
        # instead of raising, while the same value passed to predict() always
        # raises — two entry points into the same system silently disagreeing
        # on the input contract.
        validate_serving_shape(batch_size, input_tokens, output_tokens)

        if model_name not in MODEL_PARAMS:
            raise ValueError(
                f"Unknown model_name {model_name!r}. Valid: {sorted(MODEL_PARAMS)}"
            )
        # accuracy_tier/scenario/framework are validated here too — not just
        # implicitly downstream. The FastAPI layer's Pydantic Literal types
        # and the Streamlit UI's fixed selectboxes both happen to constrain
        # these today, but GpuRecommender.recommend() is a public method
        # (its own module docstring documents it as the library entry point)
        # and must be safe to call directly with untrusted input, the same
        # contract GpuPredictor.predict() already upholds via _validate().
        # Before this check, an invalid accuracy_tier reached the bare
        # `TIER_TO_PRECISION[accuracy_tier]` subscript below and raised an
        # uncaught KeyError — not the ValueError every other invalid-input
        # path in this codebase raises, so main.py's `except ValueError`
        # handler would not have caught it, and a garbage scenario/framework
        # would have silently passed through into the response (echoed
        # unchanged, never validated) whenever every candidate GPU was
        # excluded before reaching predict_batch()'s own checks.
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

        # Per-GPU memory fit — AMD uses FP8 at 99.9 tier, halving the weight
        # footprint vs the FP16 default, which also shrinks the KV cache since
        # KV values are stored at the same precision as the weights here.  The
        # VRAM pre-filter and reject messages must use this per-GPU value;
        # candidates use pred[...] fields which come from predict_batch() and
        # already apply the same override (GpuPredictor._selected_precision).
        def _gpu_memory_fit(gpu_id: str, selected_precision: str) -> tuple[str, float, float, float, float]:
            """Return (verdict, weights_gb, kv_gb, total_gb, utilization).

            Takes selected_precision as a parameter rather than re-deriving it
            via _selected_precision(spec, accuracy_tier) — the caller already
            derived it once for the precision pre-filter below, and every GPU
            here is looked up exactly once per recommend() call (was twice,
            measured 16 calls for 8 in-scope GPUs, now 8).
            """
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

        # Precision-support pre-filter (ahead of the memory-fit
        # filter, order-of-operations step 0): a GPU whose peak_tflops
        # table has no native entry for the tier's selected precision (e.g.
        # accuracy_tier="99" → fp8 on a100_sxm_80gb, which has no native FP8
        # Tensor Core path) must never reach predict_batch(), which now raises
        # for this case (GpuPredictor._check_precision_supported) — one
        # unsupported GPU must not crash the whole recommend() call for every
        # other candidate, so it is excluded here instead, with a reason.
        # There is no "unsupported_precision" memory_fit_verdict value (the
        # schema's MemoryFitVerdict Literal is closed to fits/tight/
        # does_not_fit, enforced by a reliability gate) — "does_not_fit" is reused here
        # since the GPU categorically cannot serve this request either way;
        # reject_reason carries the real, precision-specific explanation.
        #
        # selected_precision is derived once per GPU here and threaded through
        # everything below (_gpu_memory_fit, the filtered-entry builder) —
        # not re-derived at each use site, which is how this loop and
        # _gpu_memory_fit each independently called _selected_precision()
        # before (16 calls for 8 in-scope GPUs, now 8).
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

        # Pre-filter by memory fit before calling predict_batch: skip XGBoost
        # inference for GPUs where the model provably cannot fit (weights + KV
        # cache + 10% overhead > VRAM).  Matters most for large models
        # (llama3.1-405b at fp8 = 405 GB; no in-scope GPU reaches that).
        # Compute the fit tuple once per GPU (avoid repeat calls across the
        # dual list comprehensions and the reject entry builder below).
        gpu_mem: dict[str, tuple[str, float, float, float, float]] = {
            gid: _gpu_memory_fit(gid, precisions[gid]) for gid in precision_ok_ids
        }
        vram_ok_ids: list[str] = []
        vram_fail_ids: list[str] = []
        for gid in precision_ok_ids:
            verdict, *_ = gpu_mem[gid]
            (vram_fail_ids if verdict == "does_not_fit" else vram_ok_ids).append(gid)

        # Pass the memory fit we already computed above straight through —
        # predict_batch() would otherwise redo the exact same KV-cache +
        # threshold math for every one of these GPUs (see its docstring).
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

        # Build reject entries for precision-unsupported GPUs — never touched
        # memory-fit or predict_batch() at all (see the precision pre-filter
        # above).
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
