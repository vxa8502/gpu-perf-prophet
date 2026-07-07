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

from src.models.predictor import GpuPredictor, VALID_MODELS
from src.recommend.recommender import GpuRecommender

_SORTED_MODELS: list[str] = sorted(VALID_MODELS)

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
        budget_per_gpu_hr=budget if budget > 0 else None,
        min_throughput_tok_per_sec=min_tput if min_tput > 0 else None,
    )

workload = result["workload"]
frontier = result["frontier"]
dominated = result["dominated"]
filtered = result["filtered"]

# ---------------------------------------------------------------------------
# Workload summary
# ---------------------------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)
col1.metric("Model", workload["model_name"])
col2.metric("Model size", f"{workload['model_size_gb']:.1f} GB")
col3.metric("Scenario", workload["scenario"])
col4.metric("Accuracy tier", workload["accuracy_tier"])

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
    "across throughput, cost-efficiency (tok/$), and VRAM headroom."
)

def _make_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "GPU":               r["gpu_name"],
            "Vendor":            r["vendor"].upper(),
            "Pred. tput (tok/s)": f"{r['pred_throughput_tok_per_sec']:,.0f}",
            "Roofline (tok/s)":  f"{r['roofline_tput_tok_per_sec']:,.0f}",
            "Efficiency":        f"{r['efficiency_ratio']:.2f}×",
            "VRAM (GB)":         r["vram_gb"],
            "Model (GB)":        f"{r['model_size_gb']:.1f}",
            "VRAM headroom":     f"{r['vram_headroom']:.0%}",
            "$/GPU/hr":          f"${r['price_per_gpu_hr']:.2f}" if r["price_per_gpu_hr"] else "—",
            "Tok/$":             f"{r['cost_efficiency']:,.0f}" if r["cost_efficiency"] else "—",
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
        f"**Top pick: {top['gpu_name']}** — "
        f"{top['pred_throughput_tok_per_sec']:,.0f} tok/s · "
        f"${top['price_per_gpu_hr']:.2f}/hr · "
        f"{top['cost_efficiency']:,.0f} tok/$ · "
        f"{top['vram_headroom']:.0%} VRAM free"
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
