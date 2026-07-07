---
title: GPU Perf Prophet
emoji: ⚡
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# GPU Perf Prophet

**Cross-vendor LLM inference performance forecasting and hardware recommendation engine.**

Predict throughput (tokens/sec) for major LLM workloads across AMD Instinct and NVIDIA GPU families — then get a Pareto-optimal recommendation ranked by throughput, cost-efficiency, and VRAM fit.

---

## What it does

1. **Forecasts** per-GPU inference throughput for 5 LLM models × 8 GPU SKUs using a roofline physics model corrected by XGBoost trained on MLPerf Inference v4.1–v6.0 plus self-run AMD Dev Cloud MI300X calibration (1,136 benchmark rows).
2. **Recommends** GPUs via multi-objective Pareto ranking across throughput, cost-efficiency (tok/$), and VRAM headroom.
3. **Covers AMD Instinct** MI300X, MI325X, MI355X alongside NVIDIA H100, H200, A100, L4, RTX 4090.

## Supported workloads

| LLM | Params |
|-----|--------|
| Llama 2 70B | 70B |
| Llama 3.1 8B | 8B |
| Llama 3.1 405B | 405B |
| Mixtral 8×7B | 46.7B total / 14.1B active |
| GPT-J 6B | 6B |

**Scenarios:** Offline · Server  
**Accuracy tiers:** base (BF16) · 99 (FP8) · 99.9 (FP8 on AMD, FP16 on NVIDIA)  
**Frameworks:** vLLM · TensorRT-LLM · ROCm/other

## How to use

1. Select an LLM model, scenario, accuracy tier, and framework in the sidebar.
2. Optionally set a budget cap ($/GPU/hr) or minimum throughput threshold.
3. Click **Recommend**.

The app returns:
- **Pareto-optimal GPUs** — no dominated option appears in this list.
- **Top pick** — the frontier member with the highest cost-efficiency.
- **Filtered GPUs** — candidates eliminated by VRAM, budget, or throughput constraints.
- **AMD vs NVIDIA breakdown** — best per-vendor throughput at a glance.

## Model accuracy (LOGO-CV on 5 in-scope GPU SKUs)

| Metric | NVIDIA (H100, H200) | AMD (MI300X, MI325X, MI355X) |
|--------|--------------------|-----------------------------|
| Mean MAPE | ~20% | ~24% |
| Spearman ρ | 0.902 | 0.713 |
| Roofline violations | 0 / 461 rows | 0 / 212 rows |

**Use for relative ranking and hardware shortlisting, not precise capacity planning.**  
AMD predictions carry higher uncertainty due to a smaller training corpus (212 vs 461 NVIDIA rows) and, for MI300X specifically, a mix of official MLPerf submissions and self-run calibration benchmarks run without serving-stack tuning — reported metrics score against official submissions only, calibration rows are used purely as extra training signal.

## Architecture

```
MLPerf Inference v4.1–v6.0  →  Roofline model (physics ceiling)
                             →  XGBoost efficiency-gap correction (20 features)
                             →  Multi-objective Pareto recommender
                             →  FastAPI backend + Streamlit UI
```

Key design principle borrowed from [NeuSight](https://arxiv.org/abs/2405.12031): physics-bounded ML generalizes to unseen GPUs; pure ML fails.

## Data sources

- **MLPerf Inference results** v4.1, v5.0, v5.1, v6.0 — [mlcommons/inference_results_*](https://github.com/mlcommons)
- **AMD Dev Cloud calibration** — 24 self-run vLLM benchmarks on MI300X (GPT-J, Llama 2 70B, Llama 3.1 8B, Mixtral 8×7B)
- **GPU specs** — AMD and NVIDIA product pages (HBM bandwidth, TFLOPS, VRAM)
- **Pricing** — static estimates as of June 2026 from cloud provider list prices

## Limitations

- Prices are static (June 2026). Cloud spot/reserved pricing varies significantly.
- MI355X predictions have higher variance (50 training rows, CDNA4 architecture with limited cross-GPU training signal).
- Multi-GPU scaling, training-time workloads, and Blackwell/MI400 families are out of scope.
- No live API calls — all inference is local to the Docker container.

## Tech stack

Python · XGBoost · FastAPI · Streamlit · Docker · MLflow · SHAP

---

*Built by [Victoria Alabi](https://github.com/vxa8502) · Trained on MLPerf Inference v4.1–v6.0*
