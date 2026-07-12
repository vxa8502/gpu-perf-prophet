#!/usr/bin/env python3
"""MI300X calibration benchmark runner: measures vLLM output throughput per (model, precision, scenario) on AMD Dev Cloud (ROCm) and appends results to mi300x_calibration_results.csv."""
from __future__ import annotations

import argparse
import csv
import gc
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)

# Sweep matrix

@dataclass
class ModelConfig:
    benchmark_base: str         # key used in GPU Perf Prophet MODEL_PARAMS
    hf_model_id: str            # HuggingFace model identifier
    input_len: int              # fixed prompt length in tokens
    output_len: int             # max output tokens (matches MLPerf standard)
    n_samples_offline: int      # number of prompts for Offline throughput run
    n_samples_server: int       # number of prompts for Server (lower-concurrency) run
    needs_hf_token: bool = False
    max_model_len: Optional[int] = None  # set if model's default context is too large
    # trust_remote_code must stay False unless the repo needs a custom architecture — True lets the repo run arbitrary code at load time (RCE risk).
    trust_remote_code: bool = False


# MLPerf standard output lengths: gptj=128, llama2-70b=294, mixtral-8x7b=145, llama3.1-8b=294 tok (OpenOCR dataset, MLPerf v4.1+ rules).
MODEL_CONFIGS: list[ModelConfig] = [
    ModelConfig(
        benchmark_base="gptj",
        hf_model_id="EleutherAI/gpt-j-6b",
        input_len=1919,
        output_len=128,
        n_samples_offline=256,
        n_samples_server=64,
        needs_hf_token=False,
    ),
    ModelConfig(
        benchmark_base="llama3.1-8b",
        hf_model_id="meta-llama/Meta-Llama-3.1-8B",
        input_len=1024,
        output_len=294,
        n_samples_offline=128,
        n_samples_server=32,
        needs_hf_token=True,
    ),
    ModelConfig(
        benchmark_base="llama2-70b",
        hf_model_id="meta-llama/Llama-2-70b-hf",
        input_len=1024,
        output_len=294,
        n_samples_offline=64,
        n_samples_server=16,
        needs_hf_token=True,
    ),
    ModelConfig(
        benchmark_base="mixtral-8x7b",
        hf_model_id="mistralai/Mixtral-8x7B-v0.1",
        input_len=1024,
        output_len=145,
        n_samples_offline=128,
        n_samples_server=32,
        needs_hf_token=False,
        max_model_len=4096,
    ),
]

@dataclass
class PrecisionConfig:
    name: str                          # "bf16" or "fp8"
    vllm_dtype: str                    # passed as dtype= to LLM()
    vllm_quantization: Optional[str]   # passed as quantization= to LLM()
    # MLPerf accuracy tiers this precision maps to; fp8 covers both "99" and "99.9" for AMD.
    tiers: list[str] = field(default_factory=list)


PRECISION_CONFIGS: list[PrecisionConfig] = [
    PrecisionConfig(
        name="bf16",
        vllm_dtype="bfloat16",
        vllm_quantization=None,
        tiers=["base"],
    ),
    PrecisionConfig(
        name="fp8",
        vllm_dtype="bfloat16",
        vllm_quantization="fp8",
        tiers=["99", "99.9"],   # AMD MI300X achieves both with FP8
    ),
]

# MLPerf scenario names
SCENARIOS = ["Offline", "Server"]

# GPU identifier that must match an alias in data/gpu_specs.yaml
GPU_NAME = "AMD Instinct MI300X"

# Use the most recent MLPerf round tag so mlperf_round_num maps to 4 (max maturity) — we're on a current ROCm/vLLM stack.
ROUND_TAG = "v6.0"

# CSV schema

CSV_FIELDS = [
    "gpu_name",
    "benchmark_base",
    "benchmark_accuracy_tier",
    "scenario",
    "precision_used",
    "hf_model_id",
    "input_len",
    "output_len",
    "n_samples",
    "total_output_tokens",
    "elapsed_s",
    "throughput_tok_per_sec",
    "round_tag",
    "vllm_version",
    "rocm_version",
    "notes",
]

# vLLM helpers

def _rocm_version() -> str:
    """Return ROCm version string, or 'unknown'."""
    try:
        import subprocess
        result = subprocess.run(
            ["rocm-smi", "--version"], capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.splitlines():
            if "ROCm" in line or "rocm" in line.lower():
                return line.strip()
        return result.stdout.strip()[:80] or "unknown"
    except Exception:
        return "unknown"


def _vllm_version() -> str:
    try:
        import vllm
        return getattr(vllm, "__version__", "unknown")
    except ImportError:
        return "not_installed"


def _load_model(config: ModelConfig, prec: PrecisionConfig) -> "vllm.LLM":
    """Load a vLLM LLM instance with the given precision."""
    from vllm import LLM

    kwargs: dict = {
        "model": config.hf_model_id,
        "dtype": prec.vllm_dtype,
        "gpu_memory_utilization": 0.93,
        "enforce_eager": False,   # allow graph capture for throughput
        "disable_log_stats": True,
    }
    if config.trust_remote_code:
        kwargs["trust_remote_code"] = True
    if prec.vllm_quantization:
        kwargs["quantization"] = prec.vllm_quantization
    if config.max_model_len:
        kwargs["max_model_len"] = config.max_model_len

    log.info(
        "Loading %s  dtype=%s  quant=%s",
        config.hf_model_id, prec.vllm_dtype, prec.vllm_quantization or "none",
    )
    t0 = time.perf_counter()
    llm = LLM(**kwargs)
    log.info("Model loaded in %.1f s", time.perf_counter() - t0)
    return llm


def _unload_model(llm) -> None:
    """Release GPU memory held by an LLM instance."""
    try:
        import torch
        del llm
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass


def _make_prompts(n: int, input_len: int) -> list[list[int]]:
    """Return n identical prompt_token_ids of length input_len, using token ID 1 for exact token-count control without running a tokenizer."""
    # Each prompt must be an independent list — vLLM may mutate lists during tokenization, and [prompt] * n would share one object.
    prompt = [1] * input_len
    return [list(prompt) for _ in range(n)]


def _run_throughput(
    llm,
    config: ModelConfig,
    n_samples: int,
    *,
    warmup: bool = True,
) -> tuple[float, int]:
    """Run a throughput pass and return (elapsed_seconds, total_output_tokens); if warmup=True, runs a discarded warm-up batch first."""
    from vllm import SamplingParams

    params = SamplingParams(
        temperature=0.0,
        max_tokens=config.output_len,
        ignore_eos=True,   # always generate exactly output_len tokens
    )

    if warmup:
        log.info("  Warmup pass (%d samples)...", min(n_samples, 8))
        warmup_prompts = _make_prompts(min(n_samples, 8), config.input_len)
        llm.generate(warmup_prompts, sampling_params=params, use_tqdm=False)

    prompts = _make_prompts(n_samples, config.input_len)
    log.info("  Timed pass (%d samples × %d output tokens)...", n_samples, config.output_len)

    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params=params, use_tqdm=False)
    elapsed = time.perf_counter() - t0

    total_out = sum(len(o.outputs[0].token_ids) for o in outputs)
    return elapsed, total_out


# CSV helpers

def _completed_keys(csv_path: Path) -> set[tuple]:
    """Return set of (benchmark_base, precision_used, scenario) already in CSV."""
    if not csv_path.exists():
        return set()
    keys = set()
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            keys.add((row["benchmark_base"], row["precision_used"], row["scenario"]))
    return keys


def _append_row(csv_path: Path, row: dict) -> None:
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# Main sweep

def run_sweep(
    model_filter: Optional[list[str]],
    resume: bool,
    dry_run: bool,
    output: Path,
) -> None:
    models = [m for m in MODEL_CONFIGS if model_filter is None or m.benchmark_base in model_filter]
    if not models:
        log.error("No models matched filter %s", model_filter)
        sys.exit(1)

    vllm_ver  = _vllm_version()
    rocm_ver  = _rocm_version()
    completed = _completed_keys(output) if resume else set()

    log.info("vLLM %s  |  ROCm: %s", vllm_ver, rocm_ver)
    log.info("Output: %s", output)

    # Count total configs
    total = sum(len(PRECISION_CONFIGS) * len(SCENARIOS) for _ in models)
    log.info("Sweep: %d model×precision×scenario configs", total)

    if dry_run:
        for m in models:
            for prec in PRECISION_CONFIGS:
                for scenario in SCENARIOS:
                    n = m.n_samples_offline if scenario == "Offline" else m.n_samples_server
                    key = (m.benchmark_base, prec.name, scenario)
                    status = "SKIP (completed)" if key in completed else "RUN"
                    log.info(
                        "  [%s] %s | %s | %s | n=%d | input=%d out=%d",
                        status, m.benchmark_base, prec.name, scenario,
                        n, m.input_len, m.output_len,
                    )
        return

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    for m in models:
        if m.needs_hf_token and not hf_token:
            log.warning(
                "Skipping %s — set HF_TOKEN env var to access gated model.",
                m.hf_model_id,
            )
            continue

        for prec in PRECISION_CONFIGS:
            # Check if all scenarios for this (model, prec) are already done
            needed_scenarios = [
                s for s in SCENARIOS
                if (m.benchmark_base, prec.name, s) not in completed
            ]
            if not needed_scenarios:
                log.info("Skipping %s/%s — all scenarios already completed.", m.benchmark_base, prec.name)
                continue

            # Load model once per (model, precision) pair
            llm = None
            try:
                llm = _load_model(m, prec)
            except Exception as exc:
                log.error("Failed to load %s/%s: %s", m.benchmark_base, prec.name, exc)
                if llm:
                    _unload_model(llm)
                continue

            for scenario in needed_scenarios:
                n_samples = m.n_samples_offline if scenario == "Offline" else m.n_samples_server
                log.info(
                    "Running %s | prec=%s | scenario=%s | n=%d",
                    m.benchmark_base, prec.name, scenario, n_samples,
                )
                try:
                    elapsed, total_out = _run_throughput(llm, m, n_samples, warmup=(scenario == "Offline"))
                    tput = total_out / elapsed
                    log.info(
                        "  => %.1f tok/s  (%d tokens in %.1f s)",
                        tput, total_out, elapsed,
                    )
                    # Write one row per (model, precision, scenario, tier) — fp8 covers both "99" and "99.9" tiers for AMD MI300X.
                    for tier in prec.tiers:
                        row = {
                            "gpu_name":                GPU_NAME,
                            "benchmark_base":          m.benchmark_base,
                            "benchmark_accuracy_tier": tier,
                            "scenario":                scenario,
                            "precision_used":          prec.name,
                            "hf_model_id":             m.hf_model_id,
                            "input_len":               m.input_len,
                            "output_len":              m.output_len,
                            "n_samples":               n_samples,
                            "total_output_tokens":     total_out,
                            "elapsed_s":               f"{elapsed:.3f}",
                            "throughput_tok_per_sec":  f"{tput:.2f}",
                            "round_tag":               ROUND_TAG,
                            "vllm_version":            vllm_ver,
                            "rocm_version":            rocm_ver,
                            "notes":                   "",
                        }
                        _append_row(output, row)
                    completed.add((m.benchmark_base, prec.name, scenario))

                except Exception as exc:
                    log.error(
                        "Run failed for %s/%s/%s: %s",
                        m.benchmark_base, prec.name, scenario, exc,
                    )
                    _append_row(output, {
                        "gpu_name":                GPU_NAME,
                        "benchmark_base":          m.benchmark_base,
                        "benchmark_accuracy_tier": prec.tiers[0] if prec.tiers else "?",
                        "scenario":                scenario,
                        "precision_used":          prec.name,
                        "hf_model_id":             m.hf_model_id,
                        "input_len":               m.input_len,
                        "output_len":              m.output_len,
                        "n_samples":               n_samples,
                        "total_output_tokens":     0,
                        "elapsed_s":               0,
                        "throughput_tok_per_sec":  0,
                        "round_tag":               ROUND_TAG,
                        "vllm_version":            vllm_ver,
                        "rocm_version":            rocm_ver,
                        "notes":                   f"FAILED: {exc}",
                    })

            _unload_model(llm)

    log.info("Sweep complete. Results: %s", output)


# CLI

def main() -> None:
    parser = argparse.ArgumentParser(description="MI300X calibration benchmark runner")
    parser.add_argument(
        "--models", nargs="+",
        choices=[m.benchmark_base for m in MODEL_CONFIGS],
        help="Run only these models (default: all)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print sweep plan without running",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip configs already present in the output CSV",
    )
    parser.add_argument(
        "--output", default="mi300x_calibration_results.csv",
        help="Output CSV path (default: mi300x_calibration_results.csv)",
    )
    args = parser.parse_args()

    run_sweep(
        model_filter=args.models,
        resume=args.resume,
        dry_run=args.dry_run,
        output=Path(args.output),
    )


if __name__ == "__main__":
    main()
