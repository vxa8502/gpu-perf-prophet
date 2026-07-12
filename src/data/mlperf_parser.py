"""MLPerf Inference results parser: walks local MLPerf Inference results repos (v4.1-v6.0, closed/open divisions) and produces a flat CSV/Parquet of (system, workload, scenario) performance rows; GPU/model specs are joined separately (see gpu_spec_db.py). CLI: `python -m src.data.mlperf_parser --repos-dir data/raw/mlperf --rounds v4.1 v5.0 v5.1 v6.0 --output data/processed/mlperf_raw.csv --parquet` (see scripts/fetch_mlperf.sh for fetching)."""

from __future__ import annotations

import argparse
import json
import logging
import re
import stat as _stat
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

# --- Reference tables ---

# All LLM benchmark names used across MLPerf v4.1-v6.0 (and anticipated v7.0+); accuracy-constrained variants (-99, -99.9) share the same dataset.
LLM_BENCHMARKS: frozenset[str] = frozenset({
    "gptj", "gptj-99", "gptj-99.9",
    "llama2-70b", "llama2-70b-99", "llama2-70b-99.9",
    "mixtral-8x7b", "mixtral-8x7b-99", "mixtral-8x7b-99.9",
    "llama3.1-405b", "llama3.1-405b-99", "llama3.1-405b-99.9",
    "llama3.1-8b",   "llama3.1-8b-99",   "llama3.1-8b-99.9",
})

# Mean output tokens/sample (MLPerf dataset docs) to convert samples/sec -> tok/sec: gptj fixed 128, llama2-70b/llama3.1 ~294 (Open ORCA), mixtral ~145 — these are estimates, re-verify if the dataset changes between rounds.
TOKENS_PER_SAMPLE: dict[str, int] = {
    "gptj":               128,
    "gptj-99":            128,
    "gptj-99.9":          128,
    "llama2-70b":         294,
    "llama2-70b-99":      294,
    "llama2-70b-99.9":    294,
    "mixtral-8x7b":       145,
    "mixtral-8x7b-99":    145,
    "mixtral-8x7b-99.9":  145,
    "llama3.1-405b":      294,
    "llama3.1-405b-99":   294,
    "llama3.1-405b-99.9": 294,
    "llama3.1-8b":        294,
    "llama3.1-8b-99":     294,
    "llama3.1-8b-99.9":   294,
}


def _base_benchmark(benchmark: str) -> str:
    """Strip accuracy-constraint suffix: 'llama2-70b-99' → 'llama2-70b'."""
    for suffix in ("-99.9", "-99"):
        if benchmark.endswith(suffix):
            return benchmark[: -len(suffix)]
    return benchmark


def _accuracy_tier(benchmark: str) -> str:
    """Extract the MLPerf accuracy-constraint tier ("99.9" | "99" | "base") from a benchmark name — used as a proxy for precision since real system names rarely tag it directly."""
    if benchmark.endswith("-99.9"):
        return "99.9"
    if benchmark.endswith("-99"):
        return "99"
    return "base"


# --- Safe file reading ---

# Real MLPerf files are < 50 KB; 512 KB cap (10x margin) refuses corrupted/adversarial files without loading them into memory.
_MAX_FILE_BYTES: int = 512 * 1024  # 512 KB


def _safe_read_text(path: Path) -> Optional[str]:
    """Read a text file, refusing symlinks and oversized files and catching only I/O errors; returns None (logged at WARNING) instead of raising on any guarded condition."""
    # Single lstat() call supplies both the symlink bit and the file size, avoiding a second stat() syscall.
    try:
        st = path.lstat()
    except OSError as exc:
        log.warning("Could not stat %s: %s", path, exc)
        return None
    if _stat.S_ISLNK(st.st_mode):
        log.warning("Refusing to read symlink: %s", path)
        return None
    if st.st_size > _MAX_FILE_BYTES:
        log.warning("Skipping oversized file (%d bytes > %d limit): %s",
                    st.st_size, _MAX_FILE_BYTES, path)
        return None
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeDecodeError) as exc:
        log.warning("Could not read %s: %s", path, exc)
        return None


# --- Field-parsing helpers ---

def _parse_vram_gb(s: object) -> Optional[float]:
    """'192 GB' → 192.0, '80 GiB' → 80.0, None → None."""
    if not s:
        return None
    # Require match to start with a digit so a bare '.' can't slip through.
    m = re.search(r"(\d[\d.]*)\s*(?:GB|GiB)", str(s), re.IGNORECASE)
    if m is None:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        log.warning("Could not parse VRAM value: %r", s)
        return None


def _parse_int(v: object) -> Optional[int]:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _ns_to_ms(ns: Optional[int]) -> Optional[float]:
    return ns / 1_000_000 if ns is not None else None


# --- system_desc.json parser ---

def _parse_system_json(path: Path) -> dict:
    """Parse one system description JSON file into a flat dict; missing fields return None rather than raising, so schema drift between rounds is handled gracefully."""
    text = _safe_read_text(path)
    if text is None:
        return {}
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        log.warning("Malformed JSON in %s: %s", path, exc)
        return {}

    # json.loads can return any JSON type; raw.get(...) would raise AttributeError on non-dict, aborting the whole round directory's parse.
    if not isinstance(raw, dict):
        log.warning(
            "Expected JSON object in %s, got %s — skipping",
            path, type(raw).__name__,
        )
        return {}

    # num_gpus = accelerators_per_node x number_of_nodes: multi-node submissions report cluster-wide throughput but only set the per-node count, so this gives the correct divisor for per-GPU normalisation.
    accel_per_node = _parse_int(raw.get("accelerators_per_node")) or 1
    num_nodes      = max(_parse_int(raw.get("number_of_nodes")) or 1, 1)

    return {
        "gpu_name":    raw.get("accelerator_model_name") or raw.get("accelerators"),
        "num_gpus":    accel_per_node * num_nodes,
        "vram_gb":     _parse_vram_gb(raw.get("accelerator_memory_capacity")),
        "framework":   raw.get("framework"),
        "system_type": raw.get("system_type"),   # "datacenter" or "edge"
        "hw_status":   raw.get("status"),         # "available" / "preview" / "rdi"
    }


# --- mlperf_log_summary.txt parser ---

# Regex patterns — cover formatting variations observed across v4.1–v6.0.
_RE_SCENARIO     = re.compile(r"Scenario\s*:\s*(\S+)",                              re.I)
_RE_VALID        = re.compile(r"Result is\s*:\s*(\w+)",                              re.I)
# Anchored to line-start so it doesn't match "Completed samples per second" in Server logs.
_RE_OFFLINE_TPUT = re.compile(r"^Samples per second\s*:\s*([\d.]+)",                re.I | re.MULTILINE)
_RE_SERVER_TPUT  = re.compile(r"Scheduled samples per second\s*:\s*([\d.]+)",       re.I)
_RE_LAT_MEAN     = re.compile(r"Mean latency\s*\(ns\)\s*:\s*(\d+)",              re.I)
_RE_LAT_P99      = re.compile(r"99\.00 percentile latency\s*\(ns\)\s*:\s*(\d+)", re.I)
# LLM-specific metrics (present in v4.1+ for LLM benchmarks)
_RE_TTFT_MEAN    = re.compile(r"Mean First Token Latency\s*\(ns\)\s*:\s*(\d+)",                      re.I)
_RE_TTFT_P99     = re.compile(r"99(?:th|\.00) Percentile First Token Latency\s*\(ns\)\s*:\s*(\d+)", re.I)
_RE_TPOT_MEAN    = re.compile(r"Mean Time Per Output Token\s*\(ns\)\s*:\s*(\d+)",                   re.I)
_RE_TPOT_P99     = re.compile(r"99(?:th|\.00) Percentile Time Per Output Token\s*\(ns\)\s*:\s*(\d+)", re.I)


def _parse_log_summary(path: Path) -> dict:
    """Parse one mlperf_log_summary.txt into a flat dict of performance metrics (latencies converted ns -> ms); returns an empty dict on read/parse failure (logged at WARNING)."""
    text = _safe_read_text(path)
    if text is None:
        return {}

    def _f(pat: re.Pattern) -> Optional[float]:
        m = pat.search(text)
        return float(m.group(1)) if m else None

    def _i(pat: re.Pattern) -> Optional[int]:
        m = pat.search(text)
        return int(m.group(1)) if m else None

    scenario_m = _RE_SCENARIO.search(text)
    valid_m    = _RE_VALID.search(text)

    tput = _f(_RE_OFFLINE_TPUT) or _f(_RE_SERVER_TPUT)

    return {
        "scenario_from_log": scenario_m.group(1).capitalize() if scenario_m else None,
        "result_valid":       (valid_m.group(1).upper() == "VALID") if valid_m else None,
        "throughput_samples_per_sec": tput,
        "latency_mean_ms": _ns_to_ms(_i(_RE_LAT_MEAN)),
        "latency_p99_ms":  _ns_to_ms(_i(_RE_LAT_P99)),
        "ttft_mean_ms":    _ns_to_ms(_i(_RE_TTFT_MEAN)),
        "ttft_p99_ms":     _ns_to_ms(_i(_RE_TTFT_P99)),
        "tpot_mean_ms":    _ns_to_ms(_i(_RE_TPOT_MEAN)),
        "tpot_p99_ms":     _ns_to_ms(_i(_RE_TPOT_P99)),
    }


# --- Precision extraction ---

# Ordered from most-specific to least-specific; first match wins.
_PRECISION_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("fp4",   re.compile(r"\bfp4\b|\bmxfp4\b",  re.I)),
    ("fp6",   re.compile(r"\bfp6\b|\bmxfp6\b",  re.I)),
    ("fp8",   re.compile(r"\bfp8\b|\be4m3\b|\be5m2\b", re.I)),
    ("int8",  re.compile(r"\bint8\b|\bw8a8\b",   re.I)),
    ("bf16",  re.compile(r"\bbf16\b|\bbfloat16\b", re.I)),
    ("fp16",  re.compile(r"\bfp16\b|\bfloat16\b", re.I)),
]


def _extract_precision(system_name: str, framework: Optional[str]) -> Optional[str]:
    """Best-effort precision extraction from system_name/framework fields (MLPerf has no dedicated precision field); returns None when not found — callers should treat None as 'unknown'."""
    # Replace underscores with spaces so \b word-boundaries match tokens like "AMD_MI300X_FP8_vLLM" -> "AMD MI300X FP8 vLLM".
    search_text = f"{system_name} {framework or ''}".replace("_", " ")
    for label, pat in _PRECISION_PATTERNS:
        if pat.search(search_text):
            return label
    return None


# --- Run-directory resolution ---

def _run_number(log_file: Path) -> int:
    """Extract the integer N from a 'run_N' directory name, -1 if not found."""
    m = re.search(r"run_(\d+)", log_file.parent.name)
    return int(m.group(1)) if m else -1


def _find_best_run(scenario_dir: Path) -> Optional[Path]:
    """Return the mlperf_log_summary.txt from the highest-numbered run_N dir under <scenario_dir>/performance/ (None if none exists); symlinked run dirs are skipped."""
    perf_dir = scenario_dir / "performance"
    if not perf_dir.is_dir() or perf_dir.is_symlink():
        return None

    # Single pass: filter symlinks at both dir/file level, extract run number, discard non-numeric names.
    runs = [
        (p, n)
        for p in perf_dir.glob("run_*/mlperf_log_summary.txt")
        if not p.parent.is_symlink()
        and not p.is_symlink()
        and (n := _run_number(p)) >= 0
    ]
    if not runs:
        return None

    return max(runs, key=lambda x: x[1])[0]


# --- Repo walker ---

def parse_repo(
    repo_root: Path,
    round_tag: str,
    divisions: tuple[str, ...] = ("closed", "open"),
    llm_only: bool = True,
) -> pd.DataFrame:
    """Walk one MLPerf Inference results repo (local clone of inference_results_vX.Y) and return a DataFrame of rows tagged with round_tag, filtered by divisions and (by default) to LLM-only benchmarks."""
    repo_root = Path(repo_root)
    rows: list[dict] = []

    for division in divisions:
        div_dir = repo_root / division
        if not div_dir.is_dir():
            continue

        for submitter_dir in sorted(div_dir.iterdir()):
            if not submitter_dir.is_dir() or submitter_dir.is_symlink():
                continue
            submitter = submitter_dir.name

            systems_dir = submitter_dir / "systems"
            results_dir = submitter_dir / "results"
            if not results_dir.is_dir() or results_dir.is_symlink():
                continue

            for system_dir in sorted(results_dir.iterdir()):
                if not system_dir.is_dir() or system_dir.is_symlink():
                    continue
                system_name = system_dir.name

                sys_json_path = systems_dir / f"{system_name}.json"
                # Single lstat() gives existence + symlink bit, avoiding the exists()+is_symlink()+is_symlink() three-call pattern.
                try:
                    _st = sys_json_path.lstat()
                except OSError:
                    hw = {}
                    log.debug("No system JSON for %r/%r", submitter, system_name)
                else:
                    if _stat.S_ISLNK(_st.st_mode):
                        hw = {}
                        log.warning("Skipping symlinked system JSON: %s", sys_json_path)
                    else:
                        hw = _parse_system_json(sys_json_path)

                # Precision is a system-level attribute — compute it once here rather than once per (benchmark, scenario) row.
                precision = _extract_precision(system_name, hw.get("framework"))
                # Clamp to >= 1: accelerators_per_node = "0" for CPU-only systems (e.g. Intel EMR); 0 as divisor would produce +-inf throughput_tok_per_sec_per_gpu.
                num_gpus = max(hw.get("num_gpus") or 1, 1)

                for benchmark_dir in sorted(system_dir.iterdir()):
                    if not benchmark_dir.is_dir() or benchmark_dir.is_symlink():
                        continue

                    # Lower-case once; reused for LLM_BENCHMARKS check, TOKENS_PER_SAMPLE lookup, and storage.
                    bench_lower = benchmark_dir.name.lower()

                    if llm_only and bench_lower not in LLM_BENCHMARKS:
                        continue

                    for scenario_dir in sorted(benchmark_dir.iterdir()):
                        if not scenario_dir.is_dir() or scenario_dir.is_symlink():
                            continue
                        scenario = scenario_dir.name  # "Offline", "Server", "SingleStream", …

                        log_path = _find_best_run(scenario_dir)
                        if log_path is None:
                            log.debug("No performance run found: %s", scenario_dir)
                            continue

                        metrics = _parse_log_summary(log_path)

                        # Cross-check: scenario from log should match directory name.
                        log_scenario = metrics.get("scenario_from_log")
                        if log_scenario and log_scenario.lower() != scenario.lower():
                            log.debug(
                                "Scenario mismatch: dir=%r log=%r (%s)",
                                scenario, log_scenario, log_path,
                            )

                        tput    = metrics.get("throughput_samples_per_sec")
                        tps     = TOKENS_PER_SAMPLE.get(bench_lower)
                        tok_sec = tput * tps if (tput and tps) else None
                        # Per-GPU throughput is the model target variable; roofline bounds are computed per-GPU, so we normalize here.
                        tok_sec_per_gpu = tok_sec / num_gpus if tok_sec else None

                        rows.append({
                            # Provenance
                            "round":       round_tag,
                            "division":    division,
                            "submitter":   submitter,
                            "system_name": system_name,
                            # Hardware (from system JSON)
                            **hw,
                            # num_gpus overrides **hw: the clamped value (>= 1) must match the divisor used for tok_sec_per_gpu.
                            "num_gpus": num_gpus,
                            # Workload
                            "benchmark":               bench_lower,
                            "benchmark_base":          _base_benchmark(bench_lower),
                            "benchmark_accuracy_tier": _accuracy_tier(bench_lower),
                            "scenario":                scenario,
                            # precision: best-effort regex extraction, typically None on real data — use benchmark_accuracy_tier as the reliable proxy feature.
                            "precision": precision,
                            # Token-rate (per-node and per-GPU)
                            "tokens_per_sample":           tps,
                            "throughput_tokens_per_sec":   tok_sec,
                            "throughput_tok_per_sec_per_gpu": tok_sec_per_gpu,
                            # Performance metrics
                            **{k: v for k, v in metrics.items() if k != "scenario_from_log"},
                            # Audit trail — relative path preferred; falls back to absolute if log_path resolves outside repo_root (e.g. via an unexpected symlink).
                            "log_path": (
                                str(log_path.relative_to(repo_root))
                                if log_path.is_relative_to(repo_root)
                                else str(log_path)
                            ),
                        })

    if not rows:
        log.warning("No rows parsed from %s — check repo layout and round tag.", repo_root)
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    log.info("Parsed %d rows from %s (%s)", len(df), repo_root.name, round_tag)
    return df


def parse_repos(
    repo_specs: list[tuple[Path | str, str]],
    llm_only: bool = True,
) -> pd.DataFrame:
    """Parse multiple repos (list of (repo_path, round_tag) pairs; nonexistent paths silently skipped) and return a single concatenated DataFrame."""
    frames = [
        parse_repo(Path(p), tag, llm_only=llm_only)
        for p, tag in repo_specs
        if Path(p).is_dir()
    ]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# --- CLI ---

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Parse MLPerf Inference results repos into a flat CSV/Parquet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--repos-dir",
        default="data/raw/mlperf",
        help="Parent directory containing version-named subdirs (v4.1/, v5.0/, …).",
    )
    p.add_argument(
        "--rounds", nargs="+",
        default=["v4.1", "v5.0", "v5.1", "v6.0"],
        help="Round tags to look for under --repos-dir.",
    )
    p.add_argument(
        "--output",
        default="data/processed/mlperf_raw.csv",
        help="Output CSV path.",
    )
    p.add_argument("--parquet", action="store_true",
                   help="Also write a .parquet file alongside the CSV.")
    p.add_argument("--all-benchmarks", action="store_true",
                   help="Include non-LLM benchmarks (ResNet, BERT, DLRM, …).")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s: %(message)s")

    repos_dir  = Path(args.repos_dir)
    repo_specs = [(repos_dir / tag, tag) for tag in args.rounds]
    found      = [(p, t) for p, t in repo_specs if p.is_dir()]

    if not found:
        log.error(
            "No round directories found in %s. "
            "Run scripts/fetch_mlperf.sh first, or pass --repos-dir.",
            repos_dir,
        )
        raise SystemExit(1)

    df = parse_repos(found, llm_only=not args.all_benchmarks)
    if df.empty:
        log.error("No rows collected — check repo layout and round tags.")
        raise SystemExit(1)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    log.info("Wrote %d rows → %s", len(df), out)

    if args.parquet:
        pq = out.with_suffix(".parquet")
        df.to_parquet(pq, index=False)
        log.info("Wrote Parquet → %s", pq)

    # Quick summary to stdout
    if "gpu_name" in df.columns:
        print("\nGPU coverage:")
        print(df.groupby(["gpu_name", "round"])["benchmark_base"].nunique()
                .rename("benchmarks")
                .to_string())
    print(f"\nTotal rows: {len(df)}")


if __name__ == "__main__":
    main()
