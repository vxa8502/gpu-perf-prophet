"""GPU Perf Prophet — Streamlit UI."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is importable when running from the project root.
# Use append (not insert) so the project root is searched LAST — if a file in
# the project root happened to share a name with a stdlib module, insert(0)
# would shadow the stdlib for the entire process.
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

from src.features.build_features import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_INPUT_TOKENS,
    DEFAULT_OUTPUT_TOKENS,
    MIN_BATCH_SIZE,
    MAX_BATCH_SIZE,
    MIN_INPUT_TOKENS,
    MAX_INPUT_TOKENS,
    MIN_OUTPUT_TOKENS,
    MAX_OUTPUT_TOKENS,
)
from src.models.predictor import GpuPredictor, VALID_MODELS
from src.recommend.recommender import GpuRecommender

_SORTED_MODELS: list[str] = sorted(VALID_MODELS)

# The four ranking scalars, human-readable label -> API value.
_RANKING_OBJECTIVE_LABELS: dict[str, str] = {
    "Tokens per dollar": "tokens_per_dollar",
    "Tokens per second": "tokens_per_second",
    "Tokens per watt": "tokens_per_watt",
    "Lowest cost per 1M tokens": "lowest_cost_per_million_tokens",
}

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="GPU Perf Prophet",
    layout="wide",
)

st.title("GPU Perf Prophet")
st.caption(
    "Cross-vendor LLM inference forecasting · AMD Instinct + NVIDIA · "
    "Powered by roofline physics + XGBoost"
)

# ---------------------------------------------------------------------------
# Load model (cached so it runs only once)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model …")
def _load() -> tuple[GpuPredictor, GpuRecommender]:
    pred = GpuPredictor()
    rec  = GpuRecommender(pred)
    return pred, rec


predictor, recommender = _load()

# ---------------------------------------------------------------------------
# Sidebar — workload inputs
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Workload")

    model_name = st.selectbox(
        "LLM model",
        options=_SORTED_MODELS,
        index=_SORTED_MODELS.index("llama2-70b"),
    )
    scenario = st.selectbox("Scenario", ["Offline", "Server"])
    accuracy_tier = st.selectbox(
        "Accuracy tier",
        ["99", "99.9", "base"],
        help="99 → FP8 | 99.9 → FP8 (AMD) / FP16 (NVIDIA) | base → BF16",
    )
    framework = st.selectbox(
        "Framework",
        ["vllm", "tensorrt", "rocm_other", "other"],
    )

    st.divider()
    st.header("Serving shape")
    st.caption(
        "Drives the KV-cache memory-fit check only — MLPerf submissions "
        "don't report per-row batch/context length, so these are stated "
        "assumptions, not learned features."
    )
    batch_size = st.number_input(
        "Batch size", min_value=MIN_BATCH_SIZE, max_value=MAX_BATCH_SIZE,
        value=DEFAULT_BATCH_SIZE, step=1,
    )
    input_tokens = st.number_input(
        "Input tokens", min_value=MIN_INPUT_TOKENS, max_value=MAX_INPUT_TOKENS,
        value=DEFAULT_INPUT_TOKENS, step=64,
    )
    output_tokens = st.number_input(
        "Output tokens", min_value=MIN_OUTPUT_TOKENS, max_value=MAX_OUTPUT_TOKENS,
        value=DEFAULT_OUTPUT_TOKENS, step=64,
    )

    st.divider()
    st.header("Constraints (optional)")
    budget = st.number_input(
        "Max $/GPU/hr", min_value=0.0, max_value=20.0,
        value=0.0, step=0.25,
        help="Set to 0 to disable budget filter",
    )
    min_tput = st.number_input(
        "Min throughput (tok/s)", min_value=0.0,
        value=0.0, step=100.0,
        help="Set to 0 to disable throughput filter",
    )

    st.divider()
    ranking_label = st.selectbox(
        "Rank by",
        options=list(_RANKING_OBJECTIVE_LABELS),
        help="The scalar the Pareto-optimal set below is sorted by. "
             "Does not change which GPUs make the frontier — only their order.",
    )
    ranking_objective = _RANKING_OBJECTIVE_LABELS[ranking_label]

    run_btn = st.button("Recommend", use_container_width=True, type="primary")

# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

if not run_btn:
    st.info("Configure your workload in the sidebar and click **Recommend**.")
    st.stop()

with st.spinner("Running predictions …"):
    result = recommender.recommend(
        model_name=model_name,
        scenario=scenario,
        accuracy_tier=accuracy_tier,
        framework=framework,
        batch_size=batch_size,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        budget_per_gpu_hr=budget if budget > 0 else None,
        min_throughput_tok_per_sec=min_tput if min_tput > 0 else None,
        ranking_objective=ranking_objective,
    )

workload = result["workload"]
frontier = result["frontier"]
dominated = result["dominated"]
filtered = result["filtered"]

# ---------------------------------------------------------------------------
# Workload summary
# ---------------------------------------------------------------------------

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Model", workload["model_name"])
col2.metric("Model size", f"{workload['model_size_gb']:.1f} GB")
col3.metric("Scenario", workload["scenario"])
col4.metric("Accuracy tier", workload["accuracy_tier"])
col5.metric(
    "Serving shape",
    f"batch {workload['batch_size']}",
    f"{workload['input_tokens']}→{workload['output_tokens']} tok",
)

st.divider()

# ---------------------------------------------------------------------------
# Pareto frontier
# ---------------------------------------------------------------------------

_ALL_CANDIDATES = frontier + dominated

if not _ALL_CANDIDATES:
    st.warning("No GPUs passed the hard constraints (VRAM fit / budget / throughput).")
    if filtered:
        st.subheader("Filtered GPUs")
        fdf = pd.DataFrame([
            {
                "GPU": r["gpu_name"],
                "Vendor": r["vendor"].upper(),
                "Pred. tput (tok/s)": f"{r['pred_throughput_tok_per_sec']:,.0f}",
                "Reason": r["reject_reason"],
            }
            for r in filtered
        ])
        st.dataframe(fdf, use_container_width=True, hide_index=True)
    st.stop()

st.subheader("Pareto-Optimal Recommendations")
st.caption(
    "GPUs on the Pareto frontier are not dominated by any other candidate "
    "across throughput, price ($/hr), and power draw (watts). "
    f"Ranked by **{ranking_label}**."
)

_MEMORY_FIT_LABELS = {
    "fits": "✓ fits",
    "tight": "⚠ tight",
    "does_not_fit": "✗ does not fit",
}

# "sufficient" reads as plain "✓ measured" (no extra noise for the common
# case); "below_floor" gets its own amber signal rather than being folded
# into "✓ measured" — real data, but short of this project's 100-row-per-GPU
# reliability floor, which a bare boolean previously hid.
_TRAINING_DATA_LABELS = {
    "sufficient": "✓ measured",
    "below_floor": "⚠ limited data",
    "none": "✗ unmeasured",
}


def _make_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "GPU":               r["gpu_name"],
            "Vendor":            r["vendor"].upper(),
            "Pred. tput (tok/s)": f"{r['pred_throughput_tok_per_sec']:,.0f}",
            "Roofline (tok/s)":  f"{r['roofline_tput_tok_per_sec']:,.0f}",
            "Efficiency":        f"{r['efficiency_ratio']:.2f}×",
            "VRAM (GB)":         r["vram_gb"],
            "Weights + KV (GB)": f"{r['memory_total_gb']:.1f}",
            "Memory fit":        _MEMORY_FIT_LABELS.get(r["memory_fit_verdict"], r["memory_fit_verdict"]),
            "VRAM headroom":     f"{r['vram_headroom']:.0%}",
            "$/GPU/hr":          f"${r['price_per_gpu_hr']:.2f}" if r["price_per_gpu_hr"] else "—",
            "Tok/$":             f"{r['cost_efficiency']:,.0f}" if r["cost_efficiency"] else "—",
            "Watts":             r["watts"] if r["watts"] else "—",
            "Tok/W":             f"{r['tokens_per_watt']:,.1f}" if r["tokens_per_watt"] else "—",
            "$/1M tok":          (
                f"${r['cost_per_million_tokens']:.2f}" if r["cost_per_million_tokens"] else "—"
            ),
            "Data":              _TRAINING_DATA_LABELS.get(
                r["training_data_tier"], r["training_data_tier"]
            ),
        }
        for r in rows
    ])

if frontier:
    st.dataframe(
        _make_table(frontier),
        use_container_width=True,
        hide_index=True,
    )

    # Highlight the top pick
    top = frontier[0]
    st.success(
        f"**Top pick: {top['gpu_name']}** (ranked by {ranking_label.lower()}) — "
        f"{top['pred_throughput_tok_per_sec']:,.0f} tok/s · "
        f"${top['price_per_gpu_hr']:.2f}/hr · "
        f"{top['cost_efficiency']:,.0f} tok/$ · "
        + (f"{top['tokens_per_watt']:,.1f} tok/W · " if top["tokens_per_watt"] else "")
        + f"{top['vram_headroom']:.0%} VRAM free"
    )
    if top["training_data_tier"] == "none":
        st.warning(
            f"⚠ **{top['gpu_name']} has no real measured data in this model's training set.** "
            "This prediction is extrapolated from other GPUs' specs, not validated against "
            "an actual benchmark for this SKU — treat it as a rough estimate, not a "
            "measured number."
        )
    elif top["training_data_tier"] == "below_floor":
        st.warning(
            f"⚠ **{top['gpu_name']} has limited measured data** — real benchmark rows went "
            "into training, but fewer than this project's own 100-row-per-GPU reliability "
            "target. Treat the prediction as directionally useful, not as confidently "
            "measured as GPUs with more training data."
        )
    if top["memory_fit_verdict"] == "tight":
        st.warning(
            f"⚠ **{top['gpu_name']} is a tight memory fit** — weights + KV cache + 10% "
            f"overhead use {top['vram_utilization']:.0%} of its {top['vram_gb']:.0f} GB VRAM "
            "at this batch size/context length. Expected to run, but with little headroom "
            "for allocator fragmentation; consider a smaller batch or a bigger GPU."
        )
else:
    st.info("No Pareto-optimal candidates after constraints.")

# ---------------------------------------------------------------------------
# All candidates (dominated)
# ---------------------------------------------------------------------------

if dominated:
    with st.expander(f"Other passing GPUs ({len(dominated)} dominated)", expanded=False):
        st.dataframe(
            _make_table(dominated),
            use_container_width=True,
            hide_index=True,
        )

# ---------------------------------------------------------------------------
# Filtered GPUs
# ---------------------------------------------------------------------------

if filtered:
    with st.expander(f"Filtered out ({len(filtered)} GPUs)", expanded=False):
        fdf = pd.DataFrame([
            {
                "GPU":    r["gpu_name"],
                "Vendor": r["vendor"].upper(),
                "Pred. tput (tok/s)": f"{r['pred_throughput_tok_per_sec']:,.0f}",
                "Reason": r["reject_reason"],
            }
            for r in filtered
        ])
        st.dataframe(fdf, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# AMD vs NVIDIA context
# ---------------------------------------------------------------------------

with st.expander("AMD vs NVIDIA breakdown", expanded=False):
    all_rows = _ALL_CANDIDATES
    amd_rows  = [r for r in all_rows if r["vendor"] == "amd"]
    nvidia_rows = [r for r in all_rows if r["vendor"] == "nvidia"]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**AMD Instinct**")
        if amd_rows:
            best_amd = max(amd_rows, key=lambda r: r["pred_throughput_tok_per_sec"])
            st.metric("Best throughput", f"{best_amd['pred_throughput_tok_per_sec']:,.0f} tok/s", best_amd["gpu_name"])
        else:
            st.info("No AMD GPUs passed filters.")
    with c2:
        st.markdown("**NVIDIA**")
        if nvidia_rows:
            best_nv = max(nvidia_rows, key=lambda r: r["pred_throughput_tok_per_sec"])
            st.metric("Best throughput", f"{best_nv['pred_throughput_tok_per_sec']:,.0f} tok/s", best_nv["gpu_name"])
        else:
            st.info("No NVIDIA GPUs passed filters.")

st.divider()
st.caption(
    "Predictions use a roofline physics model + XGBoost trained on MLPerf Inference v4.1–v6.0. "
    "Prices are static estimates (June 2026). AMD MAPE ≈ 25%, NVIDIA MAPE ≈ 21% — use for "
    "ranking, not precise capacity planning."
)
