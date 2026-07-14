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

Predict throughput (tokens/sec) for major LLM workloads across AMD Instinct and NVIDIA GPU families — then get a Pareto-optimal recommendation ranked by throughput, price, and power draw, with a user-selectable ranking scalar (tokens/$, tokens/sec, tokens/watt, or lowest cost per million tokens).

---

## What it does

1. **Forecasts** per-GPU inference throughput for 5 LLM models × 8 GPU SKUs using a roofline physics model corrected by XGBoost trained on MLPerf Inference v4.1–v6.0 plus self-run AMD Dev Cloud MI300X calibration (1,136 benchmark rows).
2. **Recommends** GPUs via multi-objective Pareto ranking across throughput, price ($/hr), and power draw (watts) — with VRAM fit enforced as a hard constraint, not a Pareto axis — ranked by a user-selectable scalar (tokens/$ by default, or tokens/sec, tokens/watt, lowest cost per million tokens).
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
2. Optionally adjust the serving shape (batch size, input/output tokens) — this drives the KV-cache memory-fit check, not the throughput prediction itself (MLPerf submissions don't report per-row batch size, so these are stated, overridable defaults).
3. Optionally set a budget cap ($/GPU/hr) or minimum throughput threshold.
4. Optionally choose a ranking objective (tokens/$ by default, or tokens/sec, tokens/watt, lowest cost per million tokens) — reorders the frontier, doesn't change which GPUs are on it.
5. Click **Recommend**.

The app returns:
- **Pareto-optimal GPUs** — no dominated option appears in this list.
- **Top pick** — the frontier member ranked first by the selected ranking objective (tokens/$ by default).
- **Memory fit** — each candidate is flagged `fits`, `tight` (runs, but with little headroom for allocator fragmentation), or excluded entirely as `does not fit`, based on weights + KV cache + 10% overhead against VRAM capacity.
- **Filtered GPUs** — candidates eliminated by memory fit, budget, or throughput constraints.
- **AMD vs NVIDIA breakdown** — best per-vendor throughput at a glance.

## Model accuracy (LOGO-CV on 5 in-scope GPU SKUs)

| Metric | NVIDIA (H100, H200) | AMD (MI300X, MI325X, MI355X) |
|--------|--------------------|-----------------------------|
| Mean MAPE | ~21% | ~25% |
| Spearman ρ | 0.894 | 0.719 |
| Roofline violations | 0 / 461 rows | 0 / 212 rows |

**Use for relative ranking and hardware shortlisting, not precise capacity planning.**  
AMD predictions carry higher uncertainty due to a smaller training corpus (212 vs 461 NVIDIA rows) and, for MI300X specifically, a mix of official MLPerf submissions and self-run calibration benchmarks run without serving-stack tuning — reported metrics score against official submissions only, calibration rows are used purely as extra training signal.

## Recommendation accuracy

MAPE and Spearman ρ measure point-prediction accuracy, not what the recommender is actually for: does it point at the right GPU? `notebooks/04_top1_benchmark.ipynb` tests this directly — for every real workload where ≥2 in-scope GPUs have measured throughput, across a range of budget and minimum-throughput constraints, does the recommender's top pick (from out-of-fold predictions, so it's never trained on the GPU it's ranking) match the GPU that actually measured highest?

**Result: 24/42 scenarios (57.1%) — below the 70% target.**

Most of the misses (12 of 18) trace to one specific cause: MI300X and MI325X share the same CDNA3 compute die — MI325X is a memory-only upgrade (6.0 vs 5.3 TB/s HBM bandwidth, 256 vs 192 GB VRAM), so their physics-based roofline ceiling is identical. Real measurements show MI325X running ~30% faster than MI300X on `llama2-70b`, but the model under-weights that bandwidth-driven gap when generalizing to whichever of the two is held out, and repeatedly recommends the cheaper MI300X ($1.99/hr) over the faster MI325X ($2.50/hr) in budget-constrained scenarios — the wrong call. This is a specific, addressable limitation (the model needs better signal to distinguish two SKUs with identical compute but different memory bandwidth), not a general "AMD is unreliable" finding.

**Recommendation diversity is lower than the "it depends on the workload" framing above implies.** Swept every feasible model × accuracy-tier combination through the recommender under all four ranking objectives (tokens/dollar, tokens/sec, tokens/watt, lowest cost per million tokens): `mi355x` is the #1 Pareto pick in 73–80% of queries, regardless of which objective is selected. It's not a ranking bug — mi355x measures as genuinely dominant on throughput, price, and power at its listed $4.50/hr — but it means the tool's answer is largely single-valued today rather than swinging between vendors by workload.

## Architecture

```
MLPerf Inference v4.1–v6.0  →  Roofline model (physics ceiling)
                             →  XGBoost efficiency-gap correction (20 features)
                             →  Multi-objective Pareto recommender
                             →  FastAPI service  →  Streamlit UI
```

Both processes run in the deployed container under `supervisord` (`docker/supervisord.conf`,
restarts either process if it dies): FastAPI (`src/api/`) serves the roofline+XGBoost
predictions and the Pareto recommender on an internal port, plus a
`meta`/`request_id`/`GET /version`/rate-limiting/structured-access-log layer for provenance and
basic abuse protection; Streamlit is the UI HF Spaces exposes publicly, and calls the API over
HTTP (`app/api_client.py`) rather than importing the prediction/recommendation modules
in-process — so every UI interaction is real, observable API traffic, not a bypass of it.

Key design principle borrowed from [NeuSight](https://arxiv.org/abs/2405.12031): physics-bounded ML generalizes to unseen GPUs; pure ML fails.

## Data sources

- **MLPerf Inference results** v4.1, v5.0, v5.1, v6.0 — [mlcommons/inference_results_*](https://github.com/mlcommons)
- **AMD Dev Cloud calibration** — 24 self-run vLLM benchmarks on MI300X (GPT-J, Llama 2 70B, Llama 3.1 8B, Mixtral 8×7B)
- **GPU specs** — AMD and NVIDIA product pages (HBM bandwidth, TFLOPS, VRAM)
- **Pricing** — static estimates as of June 2026 from cloud provider list prices

## Limitations

- Prices are static (June 2026). Cloud spot/reserved pricing varies significantly.
- All three AMD GPUs (MI300X: 80 rows, MI325X: 82, MI355X: 50) ship below this project's own 100-row-per-GPU reliability floor; MI355X is additionally single-round (v6.0 only, one benchmark family) with the highest resulting variance. Every prediction discloses this per-request (`training_data_tier: below_floor`) rather than being presented with the same confidence as H100/H200.
- The self-run AMD calibration set (24 rows, MI300X, 2 precisions, 4 models) falls short of its own ≥50-row / ≥3-batch-size target — it has no batch-size variation.
- Multi-GPU scaling, training-time workloads, and Blackwell/MI400 families are out of scope.
- No live API calls — all inference is local to the Docker container.
- `/predict`/`/recommend` p95 latency measured at ~1.5ms (1,000-request synthetic load test), far under the 200ms budget — but measured locally in-process, not on the deployed Hugging Face Spaces CPU tier over a real network connection.

## Tech stack

Python · XGBoost · FastAPI · Streamlit · Docker

---

*Built by [Victoria Alabi](https://github.com/vxa8502) · Trained on MLPerf Inference v4.1–v6.0*
