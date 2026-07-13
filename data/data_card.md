# GPU Perf Prophet — Data Card v0.1

> **Status:** Hand-authored baseline. This file will be superseded once `build_data_card()`
> is implemented and `make data` is run.
> §1, §2's MLPerf figures, and §4 are verified against `data/processed/mlperf_raw.parquet`
> on 2026-06-12. §2 also covers the AMD Dev Cloud calibration rows added 2026-07-07, which
> are a separate provenance layered on top of the MLPerf corpus at the training-features
> stage (`data/processed/mlperf_features.parquet`) — they never enter `mlperf_raw.parquet`.

---

## 1 · Provenance

| Field | Value |
|-------|-------|
| Primary source | MLCommons MLPerf Inference results repos v4.1, v5.0, v5.1, v6.0 |
| Divisions | closed + open |
| Ingestor | `src/data/mlperf_parser.py` |
| GPU spec enrichment | `src/data/gpu_spec_db.py` → `data/gpu_specs.yaml` |
| Processed artifact | `data/processed/mlperf_raw.parquet` |
| Generated | 2026-06-12 |
| Total parsed rows | **1,223** |
| Unique submitters | 32 |
| Unique system configs | 167 |
| Result validity | 100% VALID (zero INVALID rows in corpus) |

Source repos are sparse-checked-out git clones. Fetch scripts in `scripts/fetch_mlperf.sh`.
Every parsed row carries `log_path` back to its originating `mlperf_log_summary.txt`.

---

## 2 · In-scope training corpus

Rows entering model training must pass the `gpu_in_model_scope == True` filter
(set in `data/gpu_specs.yaml`). CPU-only, heterogeneous, edge, and other
out-of-scope GPUs are excluded.

**Total MLPerf in-scope rows: 649** (2026-06-12). Plus 24 self-run AMD Dev Cloud MI300X
calibration rows added 2026-07-07 — **total current in-scope rows: 673.**

### Per-GPU breakdown

| Canonical ID | GPU | Vendor | Rounds present | MLPerf rows | Calibration rows | Total | Spec confidence |
|---|---|---|---|---:|---:|---:|---|
| `mi300x` | AMD Instinct MI300X | AMD | v4.1, v5.0, v5.1, v6.0 | 56 | 24 | **80** | verified |
| `mi325x` | AMD Instinct MI325X | AMD | v5.0, v5.1, v6.0 | 82 | 0 | 82 | verified |
| `mi355x` | AMD Instinct MI355X | AMD | **v6.0 only** | **50** | 0 | **50** | estimated |
| `h100_sxm` | NVIDIA H100 SXM5 | NVIDIA | v4.1, v5.0, v5.1 | 178 | 0 | 178 | verified |
| `h200_sxm` | NVIDIA H200 SXM | NVIDIA | v4.1, v5.0, v5.1 | 283 | 0 | 283 | verified |

AMD total: **212 rows** (31 %)  
NVIDIA total: **461 rows** (69 %)

The AMD/NVIDIA imbalance is the reason the AMD-specific MAPE target is relaxed to
< 20 % (vs. < 15 % overall).

**Calibration provenance:** the 24 MI300X rows were self-run on the AMD Developer Cloud
(vLLM 0.23.0, ROCm 7.2.4) covering `gptj`, `llama2-70b`, `llama3.1-8b`, and `mixtral-8x7b` —
`gptj` and `llama3.1-8b` had **zero** official MLPerf MI300X submissions before this.
These rows measure a genuinely different regime from official submissions (no
serving-stack tuning — see `README.md` § Recommendation accuracy); model-quality metrics
reported anywhere in this project score against official MLPerf rows only, with
calibration rows used purely as additional training signal, never as evaluation ground
truth.

### Per-round in-scope row counts (MLPerf only; excludes calibration rows)

| Round | mi300x | mi325x | mi355x | h100_sxm | h200_sxm | Round total |
|-------|-------:|-------:|-------:|---------:|---------:|------------:|
| v4.1 | 16 | — | — | 116 | 68 | 200 |
| v5.0 | 4 | 16 | — | 54 | 156 | 230 |
| v5.1 | 28 | 62 | — | 8 | 59 | 157 |
| v6.0 | 8 | 4 | **50** | — | — | 62 |
| **Total** | **56** | **82** | **50** | **178** | **283** | **649** |

---

## 3 · MI355X — v6.0-only GPU

**MI355X is the only in-scope GPU with data from a single round.**  
This is the primary known limitation of the training corpus.

### What "v6.0-only" means

| Property | Detail |
|---|---|
| First MLPerf round | v6.0 (no MI355X rows in v4.1, v5.0, or v5.1) |
| In-scope training rows | 50 |
| Benchmarks covered | `llama2-70b-99` (25 rows) and `llama2-70b-99.9` (25 rows) only |
| Benchmarks *not* covered | `gptj`, `mixtral-8x7b`, `llama3.1-405b` |
| Scenarios | Offline and Server |
| Spec confidence | **estimated** — CDNA4 dense FLOPS derived from with-sparsity ÷ 2 |

### Why this matters for the model

1. **No round-over-round variance.** The model cannot check MI355X consistency across
   rounds, making it harder to detect submission anomalies or performance regressions.

2. **Single benchmark family.** The model sees MI355X only on `llama2-70b`. Predictions
   for MI355X on `gptj`, `mixtral-8x7b`, or `llama3.1-405b` are extrapolations,
   not interpolations. SHAP explanations will not decompose MI355X benchmark-level
   effects from GPU-level effects for those benchmarks.

3. **50 rows < 100-row minimum.** If MI355X does not meet the 100-row minimum gate
   after further AMD Dev Cloud calibration, it will be marked `enabled: false` in
   `gpu_specs.yaml` and excluded from v1 recommendations. The current value is
   `in_model_scope: true` as a placeholder pending that gate. **Until that gate is
   built, the API/UI discloses the shortfall per-request instead:** MI300X, MI325X,
   and MI355X (all under the 100-row floor) get `training_data_tier: "below_floor"`
   rather than being presented with the same confidence as a well-covered GPU —
   an interim, per-request disclosure, not a substitute for the hard gate above.

4. **Estimated spec values.** `gpu_specs.yaml` MI355X entries for `peak_tflops` are
   `sparse/2` derivations (flagged `spec_confidence: estimated`). These **must be
   reconfirmed** against the AMD MI350-series whitepaper before the training data
   is finalized.

### Mitigation plan

| Step | Action | Status |
|------|--------|--------|
| 1 | Train with MI355X included; report MI355X MAPE separately; note in model card | Done |
| 2 | Run MI300X + MI355X benchmarks on AMD Dev Cloud; add calibration rows | MI300X run 2026-07-07, but under its own ≥50-row / ≥3-LLM / ≥3-batch-size / 2-precision target: delivered 24 rows, 4 LLMs, 2 precisions, **zero batch-size variation**. Meets 2 of 4 axes. MI355X not yet run — Dev Cloud access covers MI300X instances only |
| 3 | If MI355X row count ≥ 100 → keep `enabled: true`; else set `enabled: false` | Still blocked — MI355X remains at 50 rows |
| 4 | Public writeup explicitly discloses the single-round limitation | Done (`README.md`) |
| 5 | Until step 3's hard gate exists, disclose per-GPU data sufficiency in every API/UI response rather than presenting uniform confidence | Done — `training_data_tier` field (`none`/`below_floor`/`sufficient`), 2026-07-11. Interim measure; does not replace step 3. |

---

## 4 · Excluded rows

| Category | Count | Reason |
|---|---:|---|
| CPU-inference (`gpu_name = N/A`) | 26 | Intel EMR and similar; no GPU to predict |
| NVIDIA Blackwell (B200, B300, GB200, GB300) | 206 | Explicitly out of v1 scope |
| NVIDIA RTX PRO 6000 Blackwell | 52 | Not in spec DB; out of scope |
| Heterogeneous multi-GPU | 20 | Mixed SKUs; cannot normalize to a single canonical ID |
| Intel Arc | 16 | Not in spec DB |
| Edge GPUs (Jetson) | 4 | Edge, not datacenter |
| GH200, H100-NVL, H100-PCIe, H200-NVL, L40S | 250 | Insufficient LLM rows or out of form-factor scope |
| **Total excluded** | **574** | |

Excluded rows are present in `mlperf_raw.parquet` with `gpu_in_model_scope` null or false.
They are **not removed** — exclusion happens at training time via the scope filter.

---

## 5 · Feature encoding decisions

### `precision` field

`precision` is 0 % populated across the entire corpus. MLPerf system names are hardware
identifiers (e.g., `8xMI300X_2xEPYC-9374F`), not precision-tagged. Regex extraction
returns `None` on all real submissions.

**`benchmark_accuracy_tier` is the reliable precision proxy** (100 % populated):

| Tier | Benchmark suffix | Typical precision regime |
|------|-----------------|--------------------------|
| `99.9` | `-99.9` | BF16 / FP16 (high-fidelity) |
| `99` | `-99` | FP8 or better |
| `base` | none | Throughput-optimized; any precision |

Distribution: `99` → 550 rows, `99.9` → 488 rows, `base` → 185 rows.

### `tokens_per_sample` (verified 2026-06-12)

These values are the divisor in `throughput_tok_per_sec_per_gpu`.

| Benchmark base | Value used | Source-verified | Delta | Source |
|---|---:|---:|---:|---|
| `gptj` | 128 | 128 (fixed) | 0.0 % | MLPerf benchmark spec |
| `llama2-70b` | 294 | 294.45 | < 0.2 % | `mlcommons/inference` language/llama2-70b/README.md |
| `mixtral-8x7b` | 145 | 144.84 | < 0.1 % | `mlcommons/inference` language/mixtral-8x7b README |
| `llama3.1-405b` | 294 | 294.45 | < 0.2 % | Same Open ORCA dataset as llama2-70b per MLPerf rules |

All values verified within ± 5 % tolerance. No parquet re-export required.

### `vram_gb` vs `gpu_vram_gb`

`vram_gb` (from the system JSON `accelerator_memory_capacity`) can reflect **total
system VRAM** for multi-GPU submissions. `gpu_vram_gb` (from `gpu_specs.yaml`) is
always per-GPU capacity.

**Rule: use `gpu_vram_gb` for the memory-fit constraint. Treat `vram_gb` as unreliable
for any system with `num_gpus > 1`.**

---

## 6 · Deduplication policy

No deduplication is applied. Rows from different rounds for the same GPU + benchmark
combination are intentional — they represent genuinely independent performance measurements
submitted to different MLPerf rounds.

**Primary key:** `(submitter, system_name, benchmark, scenario, round, division)`  
**Duplicate rows in corpus:** 0

---

## 7 · Known limitations

| ID | Limitation | Impact | Mitigation |
|----|-----------|--------|-----------|
| KL-01 | **MI355X v6.0-only** — single round, one benchmark family, 50 rows | AMD MAPE for MI355X has higher variance; no multi-round consistency check | Further AMD Dev Cloud calibration; 100-row gate. Not yet actionable — MI355X instances aren't available on the same Dev Cloud access used for MI300X. Interim: `training_data_tier: "below_floor"` discloses this per-request (§2 mitigation step 5). |
| KL-02 | `precision` field is 0 % populated | Cannot distinguish FP8 vs FP16 directly | Use `benchmark_accuracy_tier` as proxy |
| KL-03 | `tokens_per_sample` are rounded estimates | < 0.2 % systematic bias in `throughput_tok_per_sec_per_gpu` | Verified; rounding error is negligible |
| KL-04 | AMD corpus is 29 % of in-scope rows | AMD predictions are lower-confidence than NVIDIA | Relaxed AMD MAPE gate (< 20 % vs < 15 %) |
| KL-05 | No INVALID rows in current corpus | INVALID filter in training code is not currently exercised | Filter remains; future rounds may include INVALID |
| KL-06 | MI355X CDNA4 spec values are estimated | Roofline ceilings for MI355X may be slightly off | Reconfirm against AMD MI350 whitepaper before finalizing training data |
| KL-07 | `TOKENS_PER_SAMPLE` for llama3.1-405b inherits llama2-70b value | If Open ORCA output lengths differ across models, minor bias | Verify if llama3.1-405b round adds dataset-specific stats |
| KL-08 | Recommendation diversity is low: `mi355x` is the #1 Pareto pick in 73–80% of feasible model×tier queries, under all 4 ranking objectives | Undercuts the "workload-dependent, don't always pick the same GPU" framing this data supports the recommender for | Not a data-quality defect — mi355x measures as genuinely dominant at its listed price. Re-verify the $4.50/hr price assumption and/or add further Pareto axes in a future pass |
