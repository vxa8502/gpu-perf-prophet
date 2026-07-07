#!/usr/bin/env python3
"""
Merge MI300X calibration results into the GPU Perf Prophet training set.

Run this on your local machine after SCP-ing mi300x_calibration_results.csv
back from AMD Developer Cloud.

Steps
-----
1. Reads mi300x_calibration_results.csv
2. Constructs a raw-MLPerf-format DataFrame (same schema as the parser output)
3. Calls build_training_df() to compute all features (roofline, precision, etc.)
4. Appends to data/processed/mlperf_features.parquet
5. Re-trains the production model via src/models/train_final.py

Usage
-----
    python benchmarks/merge_calibration_rows.py --csv benchmarks/mi300x_calibration_results.csv
    python benchmarks/merge_calibration_rows.py --csv benchmarks/mi300x_calibration_results.csv --dry-run
    python benchmarks/merge_calibration_rows.py --csv benchmarks/mi300x_calibration_results.csv --no-retrain
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd

# Ensure project root is on the path (script is in benchmarks/, project root is parent)
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(_PROJECT_ROOT))

from src.features.build_features import build_training_df  # noqa: E402
from src.data.mlperf_parser import TOKENS_PER_SAMPLE       # noqa: E402

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)

_FEATURES_PARQUET = _PROJECT_ROOT / "data" / "processed" / "mlperf_features.parquet"

# ---------------------------------------------------------------------------
# Construct raw-MLPerf-format rows from calibration CSV
# ---------------------------------------------------------------------------

def _build_raw_df(csv_path: Path) -> pd.DataFrame:
    """Read calibration CSV and return a DataFrame in raw-MLPerf parser format.

    The schema matches what mlperf_parser.py produces, so build_training_df()
    can process it without modification.
    """
    raw = pd.read_csv(csv_path)

    # Drop failed runs (throughput == 0 means the run errored out)
    failed = raw[raw["throughput_tok_per_sec"].astype(float) == 0.0]
    if len(failed):
        log.warning("Dropping %d failed runs: %s", len(failed),
                    failed[["benchmark_base", "precision_used", "scenario"]].to_dict("records"))
    raw = raw[raw["throughput_tok_per_sec"].astype(float) > 0.0].copy()

    if raw.empty:
        log.error("No successful runs in %s", csv_path)
        sys.exit(1)

    log.info("Loaded %d successful calibration rows from %s", len(raw), csv_path)

    rows = []
    for _, r in raw.iterrows():
        bench_base = r["benchmark_base"]
        tier       = r["benchmark_accuracy_tier"]
        scenario   = r["scenario"]
        tput       = float(r["throughput_tok_per_sec"])
        tps        = TOKENS_PER_SAMPLE.get(bench_base, TOKENS_PER_SAMPLE.get(f"{bench_base}-{tier}"))

        benchmark_name = bench_base if tier == "base" else f"{bench_base}-{tier}"

        rows.append({
            # MLPerf identity fields
            "round":            r["round_tag"],
            "division":         "open",
            "submitter":        "vxa8502",
            "system_name":      "1xMI300X_AMD_Dev_Cloud",
            "gpu_name":         r["gpu_name"],
            "num_gpus":         1,
            "vram_gb":          192.0,   # MI300X per-GPU VRAM (matches spec DB)
            "framework":        f"vLLM {r.get('vllm_version', '')}".strip(),
            "system_type":      "single_node",
            "hw_status":        "available",

            # Benchmark identity
            "benchmark":               benchmark_name,
            "benchmark_base":          bench_base,
            "benchmark_accuracy_tier": tier,
            "scenario":                scenario,
            "precision":               None,   # 0% populated in real MLPerf data — correct to leave null

            # Throughput (the training target)
            "throughput_tokens_per_sec":      tput,
            "throughput_tok_per_sec_per_gpu": tput,   # single GPU

            # Derived from throughput + tokens_per_sample
            "throughput_samples_per_sec": tput / tps if tps else None,
            "tokens_per_sample":          tps,

            # Latency fields — not measured in throughput benchmark
            "latency_mean_ms": None,
            "latency_p99_ms":  None,
            "ttft_mean_ms":    None,
            "ttft_p99_ms":     None,
            "tpot_mean_ms":    None,
            "tpot_p99_ms":     None,

            # Validation gate — True because we verified the run succeeded
            "result_valid": True,

            # Provenance
            "log_path": f"benchmarks/results/{r.get('round_tag', 'self-run')}/{bench_base}_{tier}_{scenario}",
        })

    df = pd.DataFrame(rows)
    log.info("Constructed %d raw rows ready for feature engineering", len(df))
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Merge MI300X calibration results into training set")
    parser.add_argument(
        "--csv", required=True,
        help="Path to mi300x_calibration_results.csv from AMD Dev Cloud",
    )
    parser.add_argument(
        "--features-parquet", default=str(_FEATURES_PARQUET),
        help="Path to existing mlperf_features.parquet (default: data/processed/mlperf_features.parquet)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Build features and print summary but do NOT write to disk or retrain",
    )
    parser.add_argument(
        "--no-retrain", action="store_true",
        help="Write updated parquet but skip model retraining",
    )
    args = parser.parse_args()

    csv_path      = Path(args.csv)
    parquet_path  = Path(args.features_parquet)

    if not csv_path.exists():
        log.error("CSV not found: %s", csv_path)
        sys.exit(1)
    if not parquet_path.exists():
        log.error("Features parquet not found: %s", parquet_path)
        sys.exit(1)

    # 1 — Build raw rows and run feature engineering
    raw_df = _build_raw_df(csv_path)
    cal_features = build_training_df(raw_df)

    if cal_features.empty:
        log.error("build_training_df returned 0 rows — check GPU name aliases and benchmark_base values")
        sys.exit(1)

    log.info(
        "Feature engineering complete: %d rows  |  "
        "efficiency_ratio range [%.3f, %.3f]  |  "
        "roofline violations: %d",
        len(cal_features),
        cal_features["efficiency_ratio"].min(),
        cal_features["efficiency_ratio"].max(),
        (cal_features["throughput_tok_per_sec_per_gpu"] > cal_features["roofline_tput"]).sum(),
    )

    # 2 — Load existing features and check for overlap
    existing = pd.read_parquet(parquet_path)
    log.info("Existing training set: %d rows", len(existing))

    # Check for duplicates on (submitter, system_name, benchmark, scenario, round, division)
    pk_cols = ["submitter", "system_name", "benchmark", "scenario", "round", "division"]
    overlap_mask = cal_features[pk_cols].apply(tuple, axis=1).isin(
        existing[pk_cols].apply(tuple, axis=1)
    )
    if overlap_mask.any():
        log.warning(
            "Dropping %d calibration rows that are already in the training set (PK collision)",
            overlap_mask.sum(),
        )
        cal_features = cal_features[~overlap_mask]

    if cal_features.empty:
        log.info("No new rows to add — all calibration configs already present.")
        return

    # 3 — Print summary
    print("\n=== Calibration rows to add ===")
    print(
        cal_features.groupby(["canonical_gpu_id", "benchmark_base", "benchmark_accuracy_tier", "scenario"])
        [["throughput_tok_per_sec_per_gpu", "efficiency_ratio"]]
        .agg({"throughput_tok_per_sec_per_gpu": "mean", "efficiency_ratio": "mean"})
        .round(3)
        .to_string()
    )
    print(f"\nTotal new rows: {len(cal_features)}")
    print(f"MI300X rows before: {(existing.canonical_gpu_id == 'mi300x').sum()}")
    print(f"MI300X rows after:  {(existing.canonical_gpu_id == 'mi300x').sum() + len(cal_features[cal_features.canonical_gpu_id == 'mi300x'])}")

    if args.dry_run:
        log.info("Dry run — no files written.")
        return

    # 4 — Concatenate and write
    combined = pd.concat([existing, cal_features], ignore_index=True)
    log.info("Combined training set: %d rows", len(combined))
    combined.to_parquet(parquet_path, index=False)
    log.info("Wrote updated parquet: %s", parquet_path)

    if args.no_retrain:
        log.info("Skipping retraining (--no-retrain).")
        return

    # 5 — Retrain
    log.info("Retraining production model via train_final.py ...")
    env = {**__import__("os").environ, "PYTHONPATH": str(_PROJECT_ROOT)}
    result = subprocess.run(
        [sys.executable, str(_PROJECT_ROOT / "src" / "models" / "train_final.py")],
        cwd=str(_PROJECT_ROOT),
        env=env,
        check=False,
    )
    if result.returncode != 0:
        log.error("train_final.py exited with code %d", result.returncode)
        sys.exit(result.returncode)

    log.info("Retraining complete. Run pytest to verify gate tests still pass:")
    log.info("  pytest -m gate -v")
    log.info("  pytest --cov=src --cov-fail-under=85")


if __name__ == "__main__":
    main()
