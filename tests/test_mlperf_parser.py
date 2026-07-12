"""Unit tests for src/data/mlperf_parser.py; fixtures use synthetic, in-memory data mirroring the exact directory layout and file format of MLPerf Inference v4.1-v6.0 closed-division submissions — no real MLPerf repos required."""

import json
import logging
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.data.mlperf_parser import (
    LLM_BENCHMARKS,
    TOKENS_PER_SAMPLE,
    _MAX_FILE_BYTES,
    _accuracy_tier,
    _base_benchmark,
    _extract_precision,
    _find_best_run,
    _parse_log_summary,
    _parse_system_json,
    _parse_vram_gb,
    _run_number,
    _safe_read_text,
    main,
    parse_repo,
    parse_repos,
)


# Fixture content (mirrors real MLPerf file content)

SYSTEM_JSON_NVIDIA = {
    "system_name": "H100-SXM5-80GBx8_TRT-LLM",
    # Must match an alias in data/gpu_specs.yaml so end-to-end parse_repo -> enrich_df tests produce a non-null canonical_gpu_id.
    "accelerator_model_name": "NVIDIA H100-SXM-80GB",
    "accelerators_per_node": "8",
    "accelerator_memory_capacity": "80 GB",
    "framework": "TensorRT-LLM v0.12.0, CUDA 12.6.3",
    "system_type": "datacenter",
    "status": "available",
    "division": "closed",
}

SYSTEM_JSON_AMD = {
    "system_name": "AMD_Instinct_MI300Xx8_vLLM",
    "accelerator_model_name": "AMD Instinct MI300X",
    "accelerators_per_node": "8",
    "accelerator_memory_capacity": "192 GB",
    "framework": "ROCm 6.2.4, vLLM 0.5.0",
    "system_type": "datacenter",
    "status": "available",
    "division": "closed",
}

LOG_SUMMARY_OFFLINE = """\
================================================
MLPerf Results Summary
================================================
SUT name : PySUT
Scenario : Offline
Mode     : PerformanceOnly
Samples per second: 25.34
Result is : VALID
  Min duration satisfied : Yes
  Min queries satisfied : Yes
  Early stopping satisfied: Yes
================================================
Additional Stats
================================================
Min latency (ns)                : 1000000000
Max latency (ns)                : 5000000000
Mean latency (ns)               : 2000000000
50.00 percentile latency (ns)   : 1800000000
90.00 percentile latency (ns)   : 3500000000
95.00 percentile latency (ns)   : 4000000000
97.00 percentile latency (ns)   : 4200000000
99.00 percentile latency (ns)   : 4800000000
99.90 percentile latency (ns)   : 4950000000
Mean First Token Latency (ns)   : 150000000
99th Percentile First Token Latency (ns) : 300000000
Mean Time Per Output Token (ns) : 50000
99th Percentile Time Per Output Token (ns) : 80000
"""

LOG_SUMMARY_SERVER = """\
================================================
MLPerf Results Summary
================================================
SUT name : PySUT
Scenario : Server
Mode     : PerformanceOnly
Scheduled samples per second : 12.56
Result is : VALID
  Min duration satisfied : Yes
  Min queries satisfied : Yes
  Early stopping satisfied: Yes
================================================
Additional Stats
================================================
Completed samples per second    : 11.80
Min latency (ns)                : 2000000000
Max latency (ns)                : 9000000000
Mean latency (ns)               : 4000000000
50.00 percentile latency (ns)   : 3800000000
90.00 percentile latency (ns)   : 7000000000
95.00 percentile latency (ns)   : 7800000000
97.00 percentile latency (ns)   : 8200000000
99.00 percentile latency (ns)   : 8900000000
99.90 percentile latency (ns)   : 8990000000
Mean First Token Latency (ns)   : 200000000
99th Percentile First Token Latency (ns) : 450000000
Mean Time Per Output Token (ns) : 60000
99th Percentile Time Per Output Token (ns) : 95000
"""

LOG_SUMMARY_INVALID = """\
================================================
MLPerf Results Summary
================================================
Scenario : Offline
Mode     : PerformanceOnly
Samples per second: 99.99
Result is : INVALID
"""


# Fixtures: build a minimal synthetic repo tree on disk

def _build_repo(
    tmp_path: Path,
    system_json: dict,
    benchmarks: list[str] | None = None,
    scenarios: list[str] | None = None,
) -> Path:
    """Create a minimal MLPerf-layout repo under tmp_path; returns the repo root path."""
    if benchmarks is None:
        benchmarks = ["llama2-70b"]
    if scenarios is None:
        scenarios = ["Offline", "Server"]

    repo = tmp_path / "inference_results_v6.0"
    submitter = "TestSubmitter"
    system_name = system_json["system_name"]

    # systems/<system>.json
    systems_dir = repo / "closed" / submitter / "systems"
    systems_dir.mkdir(parents=True)
    (systems_dir / f"{system_name}.json").write_text(
        json.dumps(system_json), encoding="utf-8"
    )

    # results/<system>/<benchmark>/<scenario>/performance/run_1/mlperf_log_summary.txt
    for bm in benchmarks:
        for sc in scenarios:
            log_content = LOG_SUMMARY_OFFLINE if sc == "Offline" else LOG_SUMMARY_SERVER
            run_dir = (
                repo / "closed" / submitter / "results"
                / system_name / bm / sc / "performance" / "run_1"
            )
            run_dir.mkdir(parents=True)
            (run_dir / "mlperf_log_summary.txt").write_text(log_content, encoding="utf-8")

    return repo


# Unit tests: helper functions

class TestParseVramGb:
    def test_gb_suffix(self):
        assert _parse_vram_gb("192 GB") == 192.0

    def test_gib_suffix(self):
        assert _parse_vram_gb("80 GiB") == 80.0

    def test_case_insensitive(self):
        assert _parse_vram_gb("80 gb") == 80.0

    def test_none_input(self):
        assert _parse_vram_gb(None) is None

    def test_empty_string(self):
        assert _parse_vram_gb("") is None

    def test_unrecognised_format(self):
        assert _parse_vram_gb("lots") is None

    def test_dot_only_match_returns_none(self):
        # Regression: [\d.]+ matched ". GB" -> float(".") raised ValueError and crashed the whole parse job; fixed by requiring the match to start with \d.
        assert _parse_vram_gb(". GB") is None

    def test_malformed_float_returns_none(self):
        # "1.2.3" is not a valid float; must return None, not crash.
        assert _parse_vram_gb("1.2.3 GB") is None


class TestBaseBenchmark:
    @pytest.mark.parametrize("name,expected", [
        ("llama2-70b",       "llama2-70b"),
        ("llama2-70b-99",    "llama2-70b"),
        ("llama2-70b-99.9",  "llama2-70b"),
        ("gptj-99",          "gptj"),
        ("mixtral-8x7b-99.9", "mixtral-8x7b"),
    ])
    def test_strip_suffix(self, name, expected):
        assert _base_benchmark(name) == expected


class TestAccuracyTier:
    @pytest.mark.parametrize("name,expected", [
        ("llama2-70b",        "base"),
        ("gptj",              "base"),
        ("llama2-70b-99",     "99"),
        ("gptj-99",           "99"),
        ("llama2-70b-99.9",   "99.9"),
        ("mixtral-8x7b-99.9", "99.9"),
    ])
    def test_tier_extraction(self, name, expected):
        assert _accuracy_tier(name) == expected


# Unit tests: _parse_system_json

class TestParseSystemJson:
    def test_nvidia_fields(self, tmp_path):
        p = tmp_path / "H100.json"
        p.write_text(json.dumps(SYSTEM_JSON_NVIDIA), encoding="utf-8")
        hw = _parse_system_json(p)
        assert hw["gpu_name"] == "NVIDIA H100-SXM-80GB"
        assert hw["num_gpus"] == 8
        assert hw["vram_gb"] == 80.0
        assert "TensorRT-LLM" in hw["framework"]
        assert hw["system_type"] == "datacenter"

    def test_amd_fields(self, tmp_path):
        p = tmp_path / "MI300X.json"
        p.write_text(json.dumps(SYSTEM_JSON_AMD), encoding="utf-8")
        hw = _parse_system_json(p)
        assert hw["gpu_name"] == "AMD Instinct MI300X"
        assert hw["num_gpus"] == 8
        assert hw["vram_gb"] == 192.0
        assert "ROCm" in hw["framework"]

    def test_missing_file_returns_empty(self, tmp_path):
        hw = _parse_system_json(tmp_path / "nonexistent.json")
        assert hw == {}

    def test_malformed_json_returns_empty(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("not json at all", encoding="utf-8")
        hw = _parse_system_json(p)
        assert hw == {}

    def test_json_array_returns_empty(self, tmp_path):
        # Bug fix: json.loads('[1,2,3]') returns a list; raw.get(...) on a list raised an uncaught AttributeError, aborting the entire round parse.
        p = tmp_path / "array.json"
        p.write_text("[1, 2, 3]", encoding="utf-8")
        hw = _parse_system_json(p)
        assert hw == {}

    def test_json_scalar_returns_empty(self, tmp_path):
        p = tmp_path / "scalar.json"
        p.write_text('"just a string"', encoding="utf-8")
        hw = _parse_system_json(p)
        assert hw == {}


# Unit tests: _parse_log_summary

class TestParseLogSummary:
    def test_offline_throughput(self, tmp_path):
        p = tmp_path / "summary.txt"
        p.write_text(LOG_SUMMARY_OFFLINE, encoding="utf-8")
        m = _parse_log_summary(p)
        assert m["throughput_samples_per_sec"] == pytest.approx(25.34)
        assert m["result_valid"] is True
        assert m["scenario_from_log"] == "Offline"

    def test_server_throughput(self, tmp_path):
        p = tmp_path / "summary.txt"
        p.write_text(LOG_SUMMARY_SERVER, encoding="utf-8")
        m = _parse_log_summary(p)
        assert m["throughput_samples_per_sec"] == pytest.approx(12.56)
        assert m["result_valid"] is True
        assert m["scenario_from_log"] == "Server"

    def test_invalid_result_flag(self, tmp_path):
        p = tmp_path / "summary.txt"
        p.write_text(LOG_SUMMARY_INVALID, encoding="utf-8")
        m = _parse_log_summary(p)
        assert m["result_valid"] is False

    def test_latency_ns_to_ms_conversion(self, tmp_path):
        p = tmp_path / "summary.txt"
        p.write_text(LOG_SUMMARY_OFFLINE, encoding="utf-8")
        m = _parse_log_summary(p)
        # 2_000_000_000 ns → 2000.0 ms
        assert m["latency_mean_ms"] == pytest.approx(2000.0)
        assert m["latency_p99_ms"] == pytest.approx(4800.0)

    def test_ttft_parsed(self, tmp_path):
        p = tmp_path / "summary.txt"
        p.write_text(LOG_SUMMARY_OFFLINE, encoding="utf-8")
        m = _parse_log_summary(p)
        # 150_000_000 ns → 150.0 ms
        assert m["ttft_mean_ms"] == pytest.approx(150.0)
        assert m["ttft_p99_ms"] == pytest.approx(300.0)

    def test_tpot_parsed(self, tmp_path):
        p = tmp_path / "summary.txt"
        p.write_text(LOG_SUMMARY_OFFLINE, encoding="utf-8")
        m = _parse_log_summary(p)
        # 50_000 ns → 0.05 ms
        assert m["tpot_mean_ms"] == pytest.approx(0.05)
        assert m["tpot_p99_ms"] == pytest.approx(0.08)

    def test_missing_file_returns_empty(self, tmp_path):
        m = _parse_log_summary(tmp_path / "nonexistent.txt")
        assert m == {}

    def test_no_llm_metrics_returns_none(self, tmp_path):
        p = tmp_path / "summary.txt"
        p.write_text(
            "Scenario : Offline\nSamples per second: 5.0\nResult is : VALID\n",
            encoding="utf-8",
        )
        m = _parse_log_summary(p)
        assert m["ttft_mean_ms"] is None
        assert m["tpot_mean_ms"] is None


# Integration tests: parse_repo

class TestParseRepo:
    def test_correct_row_count(self, tmp_path):
        # 2 scenarios x 1 benchmark = 2 rows — verifies both scenarios are present, not just the total count (two Offline rows would also give len==2).
        repo = _build_repo(
            tmp_path, SYSTEM_JSON_NVIDIA,
            benchmarks=["llama2-70b"],
            scenarios=["Offline", "Server"],
        )
        df = parse_repo(repo, "v6.0")
        assert len(df) == 2
        assert set(df["scenario"]) == {"Offline", "Server"}

    def test_round_tag_propagated(self, tmp_path):
        repo = _build_repo(tmp_path, SYSTEM_JSON_NVIDIA)
        df = parse_repo(repo, "v6.0")
        assert (df["round"] == "v6.0").all()

    def test_gpu_name_extracted(self, tmp_path):
        repo = _build_repo(tmp_path, SYSTEM_JSON_AMD)
        df = parse_repo(repo, "v6.0")
        assert (df["gpu_name"] == "AMD Instinct MI300X").all()

    def test_throughput_tokens_computed(self, tmp_path):
        repo = _build_repo(
            tmp_path, SYSTEM_JSON_NVIDIA,
            benchmarks=["llama2-70b"],
            scenarios=["Offline"],
        )
        df = parse_repo(repo, "v6.0")
        row = df.iloc[0]
        expected = 25.34 * TOKENS_PER_SAMPLE["llama2-70b"]
        assert row["throughput_tokens_per_sec"] == pytest.approx(expected)

    def test_gptj_tokens_per_sample(self, tmp_path):
        repo = _build_repo(
            tmp_path, SYSTEM_JSON_NVIDIA,
            benchmarks=["gptj"],
            scenarios=["Offline"],
        )
        df = parse_repo(repo, "v6.0")
        assert df.iloc[0]["tokens_per_sample"] == 128

    def test_non_llm_benchmark_excluded_by_default(self, tmp_path):
        repo = _build_repo(
            tmp_path, SYSTEM_JSON_NVIDIA,
            benchmarks=["resnet50", "llama2-70b"],
            scenarios=["Offline"],
        )
        df = parse_repo(repo, "v6.0", llm_only=True)
        assert set(df["benchmark"]) == {"llama2-70b"}

    def test_non_llm_included_when_flag_off(self, tmp_path):
        repo = _build_repo(
            tmp_path, SYSTEM_JSON_NVIDIA,
            benchmarks=["resnet50", "llama2-70b"],
            scenarios=["Offline"],
        )
        df = parse_repo(repo, "v6.0", llm_only=False)
        # Verify both sides: non-LLM included AND LLM not accidentally dropped
        assert set(df["benchmark"]) == {"resnet50", "llama2-70b"}

    def test_invalid_result_rows_present(self, tmp_path):
        # Parser includes INVALID rows — filtering is caller's responsibility
        system_name = SYSTEM_JSON_NVIDIA["system_name"]
        repo = tmp_path / "repo"
        run_dir = (
            repo / "closed" / "Sub" / "results"
            / system_name / "llama2-70b" / "Offline" / "performance" / "run_1"
        )
        run_dir.mkdir(parents=True)
        (run_dir / "mlperf_log_summary.txt").write_text(
            LOG_SUMMARY_INVALID, encoding="utf-8"
        )
        sys_dir = repo / "closed" / "Sub" / "systems"
        sys_dir.mkdir(parents=True)
        (sys_dir / f"{system_name}.json").write_text(
            json.dumps(SYSTEM_JSON_NVIDIA), encoding="utf-8"
        )
        df = parse_repo(repo, "v6.0")
        assert len(df) == 1
        assert df.iloc[0]["result_valid"] == False  # noqa: E712 — numpy.bool_ != Python bool

    def test_multiple_runs_uses_highest(self, tmp_path):
        # run_2 has higher throughput; parser should pick run_2
        system_name = SYSTEM_JSON_NVIDIA["system_name"]
        repo = tmp_path / "repo"
        base = (
            repo / "closed" / "Sub" / "results"
            / system_name / "llama2-70b" / "Offline" / "performance"
        )

        (base / "run_1").mkdir(parents=True)
        (base / "run_1" / "mlperf_log_summary.txt").write_text(
            LOG_SUMMARY_OFFLINE, encoding="utf-8"
        )
        run2_content = LOG_SUMMARY_OFFLINE.replace("Samples per second: 25.34",
                                                   "Samples per second: 99.00")
        (base / "run_2").mkdir(parents=True)
        (base / "run_2" / "mlperf_log_summary.txt").write_text(
            run2_content, encoding="utf-8"
        )
        sys_dir = repo / "closed" / "Sub" / "systems"
        sys_dir.mkdir(parents=True)
        (sys_dir / f"{system_name}.json").write_text(
            json.dumps(SYSTEM_JSON_NVIDIA), encoding="utf-8"
        )
        df = parse_repo(repo, "v6.0")
        assert df.iloc[0]["throughput_samples_per_sec"] == pytest.approx(99.00)

    def test_empty_repo_returns_empty_df(self, tmp_path):
        repo = tmp_path / "empty_repo"
        repo.mkdir()
        df = parse_repo(repo, "v6.0")
        assert df.empty

    def test_benchmark_base_column(self, tmp_path):
        repo = _build_repo(
            tmp_path, SYSTEM_JSON_NVIDIA,
            benchmarks=["llama2-70b-99"],
            scenarios=["Offline"],
        )
        df = parse_repo(repo, "v6.0")
        assert df.iloc[0]["benchmark_base"] == "llama2-70b"

    def test_submitter_field(self, tmp_path):
        repo = _build_repo(tmp_path, SYSTEM_JSON_AMD)
        df = parse_repo(repo, "v6.0")
        assert (df["submitter"] == "TestSubmitter").all()

    def test_benchmark_accuracy_tier_column(self, tmp_path):
        """benchmark_accuracy_tier must be present and correctly derived."""
        repo = _build_repo(
            tmp_path, SYSTEM_JSON_NVIDIA,
            benchmarks=["llama2-70b-99"],
            scenarios=["Offline"],
        )
        df = parse_repo(repo, "v6.0")
        assert "benchmark_accuracy_tier" in df.columns
        assert df.iloc[0]["benchmark_accuracy_tier"] == "99"

    def test_num_gpus_zero_clamped_to_one(self, tmp_path):
        """accelerators_per_node=0 (CPU-only systems, e.g. Intel EMR) must produce num_gpus=1, not 0 — storing 0 would make the column inconsistent with the throughput_tok_per_sec_per_gpu divisor."""
        zero_gpu_json = {**SYSTEM_JSON_NVIDIA, "accelerators_per_node": "0"}
        repo = _build_repo(
            tmp_path, zero_gpu_json,
            benchmarks=["llama2-70b"],
            scenarios=["Offline"],
        )
        df = parse_repo(repo, "v6.0")
        assert df.iloc[0]["num_gpus"] == 1
        # per-GPU throughput must be finite (not inf/nan)
        assert df.iloc[0]["throughput_tok_per_sec_per_gpu"] > 0


# Security tests: _safe_read_text, symlink guards, _run_number

class TestSafeReadText:
    def test_normal_file_read(self, tmp_path):
        p = tmp_path / "ok.txt"
        p.write_text("hello", encoding="utf-8")
        assert _safe_read_text(p) == "hello"

    def test_missing_file_returns_none(self, tmp_path):
        assert _safe_read_text(tmp_path / "missing.txt") is None

    def test_oversized_file_returns_none(self, tmp_path):
        p = tmp_path / "huge.txt"
        # Write a file just over the limit.
        p.write_bytes(b"x" * (_MAX_FILE_BYTES + 1))
        assert _safe_read_text(p) is None

    def test_symlink_returns_none(self, tmp_path):
        target = tmp_path / "real.txt"
        target.write_text("secret", encoding="utf-8")
        link = tmp_path / "link.txt"
        link.symlink_to(target)
        assert _safe_read_text(link) is None

    def test_symlink_to_outside_returns_none(self, tmp_path):
        # Simulates a git-repo symlink pointing outside the repo root.
        link = tmp_path / "escape.txt"
        link.symlink_to("/etc/hosts")
        assert _safe_read_text(link) is None

    def test_read_error_returns_none(self, tmp_path, monkeypatch):
        """OSError from read_text() after lstat() succeeds returns None (lines 140-142)."""
        p = tmp_path / "ok_to_stat.txt"
        p.write_text("data", encoding="utf-8")

        def _raise(*args, **kwargs):
            raise OSError("simulated read failure")

        monkeypatch.setattr(Path, "read_text", _raise)
        assert _safe_read_text(p) is None


class TestRunNumber:
    def test_extracts_integer(self, tmp_path):
        p = tmp_path / "run_3" / "mlperf_log_summary.txt"
        assert _run_number(p) == 3

    def test_double_digit(self, tmp_path):
        p = tmp_path / "run_12" / "mlperf_log_summary.txt"
        assert _run_number(p) == 12

    def test_no_match_returns_minus_one(self, tmp_path):
        p = tmp_path / "audit" / "mlperf_log_summary.txt"
        assert _run_number(p) == -1


class TestSymlinkGuards:
    def test_symlinked_submitter_dir_skipped(self, tmp_path):
        """A symlinked submitter directory must not be walked — without the guard the parser would follow it into a valid submission tree and return rows; with it, df must be empty."""
        repo = tmp_path / "repo"
        real_sub = tmp_path / "real_submitter"
        # Build a complete, parseable submission inside real_sub.
        run_dir = (
            real_sub / "results" / "SomeSystem" / "llama2-70b"
            / "Offline" / "performance" / "run_1"
        )
        run_dir.mkdir(parents=True)
        (run_dir / "mlperf_log_summary.txt").write_text(
            LOG_SUMMARY_OFFLINE, encoding="utf-8"
        )
        sys_dir = real_sub / "systems"
        sys_dir.mkdir()
        (sys_dir / "SomeSystem.json").write_text(
            json.dumps(SYSTEM_JSON_NVIDIA), encoding="utf-8"
        )
        # Symlink real_sub into the repo's closed/ directory.
        (repo / "closed").mkdir(parents=True)
        (repo / "closed" / "EvilSub").symlink_to(real_sub)
        df = parse_repo(repo, "v6.0")
        assert df.empty

    def test_symlinked_system_json_skipped(self, tmp_path):
        """A symlinked system JSON must not be read."""
        system_name = SYSTEM_JSON_NVIDIA["system_name"]
        repo = tmp_path / "repo"
        # Build the results tree normally.
        run_dir = (
            repo / "closed" / "Sub" / "results"
            / system_name / "llama2-70b" / "Offline" / "performance" / "run_1"
        )
        run_dir.mkdir(parents=True)
        (run_dir / "mlperf_log_summary.txt").write_text(
            LOG_SUMMARY_OFFLINE, encoding="utf-8"
        )
        # Create a symlinked system JSON instead of a real one.
        real_json = tmp_path / "real.json"
        real_json.write_text(json.dumps(SYSTEM_JSON_NVIDIA), encoding="utf-8")
        sys_dir = repo / "closed" / "Sub" / "systems"
        sys_dir.mkdir(parents=True)
        (sys_dir / f"{system_name}.json").symlink_to(real_json)
        # Parser should still produce a row from the log, but with no hw fields — the **hw spread adds no keys when hw == {}, so gpu_name is absent.
        df = parse_repo(repo, "v6.0")
        assert len(df) == 1
        assert "gpu_name" not in df.columns

    def test_parse_log_summary_rejects_symlink(self, tmp_path):
        """_parse_log_summary returns {} for a symlinked log file."""
        real = tmp_path / "real.txt"
        real.write_text(LOG_SUMMARY_OFFLINE, encoding="utf-8")
        link = tmp_path / "link.txt"
        link.symlink_to(real)
        assert _parse_log_summary(link) == {}


# Unit tests: _extract_precision

class TestExtractPrecision:
    @pytest.mark.parametrize("system_name,framework,expected", [
        # FP8 in system name
        ("AMD_MI300X_FP8_vLLM", "ROCm 6.2.4",            "fp8"),
        # FP16 in framework string
        ("AMD_MI300X_vLLM",     "ROCm 6.2.4 fp16",       "fp16"),
        # BF16
        ("H100_bf16_TRT",       "TRT-LLM",                "bf16"),
        # FP4 / MXFP4 — should win over FP8 when both present (priority order)
        ("MI355X_MXFP4",        "ROCm 7.2, fp8 fallback", "fp4"),
        # INT8
        ("H100_int8_TRT",       None,                      "int8"),
        # Nothing recognisable → None
        ("H100_SXM5_80GBx8",    "TensorRT-LLM v0.12.0",   None),
        # None framework arg handled gracefully
        ("MI300X_FP16",         None,                      "fp16"),
    ])
    def test_precision_extraction(self, system_name, framework, expected):
        assert _extract_precision(system_name, framework) == expected


# Additional parse_repo behavior tests (post-fix)

class TestParseRepoPostFix:
    def test_server_tput_uses_scheduled_not_completed(self, tmp_path):
        """_RE_OFFLINE_TPUT must not match 'Completed samples per second'."""
        log_with_diverged_completed = """\
================================================
MLPerf Results Summary
================================================
Scenario : Server
Mode     : PerformanceOnly
Scheduled samples per second : 12.56
Result is : VALID
================================================
Additional Stats
================================================
Completed samples per second    : 11.80
Mean latency (ns)               : 4000000000
99.00 percentile latency (ns)   : 8900000000
"""
        system_name = SYSTEM_JSON_NVIDIA["system_name"]
        repo = tmp_path / "repo"
        run_dir = (
            repo / "closed" / "Sub" / "results"
            / system_name / "llama2-70b" / "Server" / "performance" / "run_1"
        )
        run_dir.mkdir(parents=True)
        (run_dir / "mlperf_log_summary.txt").write_text(
            log_with_diverged_completed, encoding="utf-8"
        )
        sys_dir = repo / "closed" / "Sub" / "systems"
        sys_dir.mkdir(parents=True)
        (sys_dir / f"{system_name}.json").write_text(
            json.dumps(SYSTEM_JSON_NVIDIA), encoding="utf-8"
        )
        df = parse_repo(repo, "v6.0")
        assert df.iloc[0]["throughput_samples_per_sec"] == pytest.approx(12.56)

    def test_per_gpu_throughput_normalized(self, tmp_path):
        """Both throughput columns are independently pinned to fixture constants — deriving expected from row["throughput_tokens_per_sec"] / 8 would only test col_a/8 ≈ col_b, passing even if both columns were miscalculated by the same factor."""
        repo = _build_repo(
            tmp_path, SYSTEM_JSON_NVIDIA,   # 8 GPUs
            benchmarks=["llama2-70b"],
            scenarios=["Offline"],
        )
        df = parse_repo(repo, "v6.0")
        row = df.iloc[0]
        # 25.34 samples/sec (LOG_SUMMARY_OFFLINE) × 294 tokens/sample
        per_node = 25.34 * TOKENS_PER_SAMPLE["llama2-70b"]
        assert row["throughput_tokens_per_sec"] == pytest.approx(per_node)
        assert row["throughput_tok_per_sec_per_gpu"] == pytest.approx(per_node / 8)

    def test_precision_extracted_from_system_name(self, tmp_path):
        fp8_json = {**SYSTEM_JSON_NVIDIA, "system_name": "H100_SXM5_80GBx8_FP8_TRT"}
        repo = _build_repo(tmp_path, fp8_json, scenarios=["Offline"])
        df = parse_repo(repo, "v6.0")
        assert df.iloc[0]["precision"] == "fp8"

    def test_benchmark_column_is_lowercase(self, tmp_path):
        """benchmark must be lowercased even when the directory name is uppercase."""
        # _build_repo's default ("llama2-70b") is already lowercase and never exercises .lower(), so use an uppercase directory name here.
        repo = _build_repo(
            tmp_path, SYSTEM_JSON_NVIDIA,
            benchmarks=["LLAMA2-70B"],
            scenarios=["Offline"],
        )
        df = parse_repo(repo, "v6.0")
        assert df.iloc[0]["benchmark"] == "llama2-70b"


# Coverage gap A: open division

class TestOpenDivision:
    def test_open_division_rows_included(self, tmp_path):
        """Submissions under open/ must be walked, not only closed/."""
        repo = tmp_path / "repo"
        system_name = SYSTEM_JSON_AMD["system_name"]
        run_dir = (
            repo / "open" / "OpenSub" / "results"
            / system_name / "llama2-70b" / "Offline" / "performance" / "run_1"
        )
        run_dir.mkdir(parents=True)
        (run_dir / "mlperf_log_summary.txt").write_text(
            LOG_SUMMARY_OFFLINE, encoding="utf-8"
        )
        sys_dir = repo / "open" / "OpenSub" / "systems"
        sys_dir.mkdir(parents=True)
        (sys_dir / f"{system_name}.json").write_text(
            json.dumps(SYSTEM_JSON_AMD), encoding="utf-8"
        )
        df = parse_repo(repo, "v6.0", divisions=("open",))
        assert len(df) == 1
        assert df.iloc[0]["division"] == "open"
        assert df.iloc[0]["submitter"] == "OpenSub"

    def test_both_divisions_combined(self, tmp_path):
        """When both divisions are requested, rows from each are present."""
        # closed submission
        closed_repo = _build_repo(
            tmp_path / "closed_build", SYSTEM_JSON_NVIDIA,
            benchmarks=["llama2-70b"], scenarios=["Offline"],
        )
        # graft an open submission into the same repo root
        open_sys = SYSTEM_JSON_AMD["system_name"]
        run_dir = (
            closed_repo / "open" / "OpenSub" / "results"
            / open_sys / "llama2-70b" / "Offline" / "performance" / "run_1"
        )
        run_dir.mkdir(parents=True)
        (run_dir / "mlperf_log_summary.txt").write_text(
            LOG_SUMMARY_OFFLINE, encoding="utf-8"
        )
        sys_dir = closed_repo / "open" / "OpenSub" / "systems"
        sys_dir.mkdir(parents=True)
        (sys_dir / f"{open_sys}.json").write_text(
            json.dumps(SYSTEM_JSON_AMD), encoding="utf-8"
        )
        df = parse_repo(closed_repo, "v6.0", divisions=("closed", "open"))
        assert set(df["division"]) == {"closed", "open"}


# Coverage gap B: parse_repos (multi-round entry point)

class TestParseRepos:
    def test_combines_two_rounds(self, tmp_path):
        """Rows from two different rounds are concatenated."""
        repo_a = _build_repo(
            tmp_path / "a", SYSTEM_JSON_NVIDIA,
            benchmarks=["llama2-70b"], scenarios=["Offline"],
        )
        repo_b = _build_repo(
            tmp_path / "b", SYSTEM_JSON_AMD,
            benchmarks=["gptj"], scenarios=["Offline"],
        )
        df = parse_repos([(repo_a, "v5.1"), (repo_b, "v6.0")])
        assert set(df["round"]) == {"v5.1", "v6.0"}
        assert len(df) == 2

    def test_missing_round_silently_skipped(self, tmp_path):
        """A path that doesn't exist must be skipped, not raise."""
        repo = _build_repo(tmp_path, SYSTEM_JSON_NVIDIA, scenarios=["Offline"])
        df = parse_repos([
            (repo, "v6.0"),
            (tmp_path / "does_not_exist", "v5.1"),
        ])
        assert len(df) == 1
        assert df.iloc[0]["round"] == "v6.0"

    def test_all_missing_returns_empty_df(self, tmp_path):
        df = parse_repos([(tmp_path / "ghost_a", "v4.1"), (tmp_path / "ghost_b", "v5.0")])
        assert df.empty

    def test_deduplication_not_silently_applied(self, tmp_path):
        """parse_repos must NOT silently drop duplicate (GPU, benchmark) pairs that appear in multiple rounds — those are intentional cross-round rows."""
        repo_a = _build_repo(
            tmp_path / "a", SYSTEM_JSON_NVIDIA,
            benchmarks=["llama2-70b"], scenarios=["Offline"],
        )
        repo_b = _build_repo(
            tmp_path / "b", SYSTEM_JSON_NVIDIA,
            benchmarks=["llama2-70b"], scenarios=["Offline"],
        )
        df = parse_repos([(repo_a, "v5.1"), (repo_b, "v6.0")])
        # Same GPU + benchmark in two rounds → 2 rows (no silent dedup)
        assert len(df) == 2
        assert set(df["round"]) == {"v5.1", "v6.0"}


# Reference table sanity checks

class TestReferenceTables:
    def test_all_llm_benchmarks_have_token_count(self):
        for bm in LLM_BENCHMARKS:
            assert bm in TOKENS_PER_SAMPLE, f"Missing TOKENS_PER_SAMPLE entry for {bm!r}"

    def test_gptj_is_128(self):
        assert TOKENS_PER_SAMPLE["gptj"] == 128

    def test_accuracy_variants_match_base(self):
        for bm in LLM_BENCHMARKS:
            base = _base_benchmark(bm)
            if base != bm:
                assert TOKENS_PER_SAMPLE[bm] == TOKENS_PER_SAMPLE[base], (
                    f"{bm!r} and {base!r} should have same TOKENS_PER_SAMPLE"
                )


# Coverage gap C: _find_best_run early-return paths

class TestFindBestRun:
    def test_no_performance_dir_returns_none(self, tmp_path):
        """scenario_dir with no performance/ subdirectory returns None (line 304)."""
        scenario_dir = tmp_path / "Offline"
        scenario_dir.mkdir()
        assert _find_best_run(scenario_dir) is None

    def test_no_valid_runs_returns_none(self, tmp_path):
        """performance/ exists but is empty → no run_* files → returns None (line 316)."""
        scenario_dir = tmp_path / "Offline"
        (scenario_dir / "performance").mkdir(parents=True)
        assert _find_best_run(scenario_dir) is None


# Coverage gap D: parse_repo continue-guards and debug-log branches

class TestParseRepoEdgeCases:
    """Targets the five directory-guard continue statements and two debug branches."""

    def test_no_system_json_row_still_produced(self, tmp_path):
        """Missing systems/ dir triggers the OSError branch (hw = {}) — the row is still produced, but hw == {} contributes no hardware columns to the row dict spread."""
        system_name = SYSTEM_JSON_NVIDIA["system_name"]
        repo = tmp_path / "repo"
        run_dir = (
            repo / "closed" / "Sub" / "results"
            / system_name / "llama2-70b" / "Offline" / "performance" / "run_1"
        )
        run_dir.mkdir(parents=True)
        (run_dir / "mlperf_log_summary.txt").write_text(
            LOG_SUMMARY_OFFLINE, encoding="utf-8"
        )
        # Intentionally NO systems/ directory so lstat raises FileNotFoundError
        df = parse_repo(repo, "v6.0")
        assert len(df) == 1
        assert "gpu_name" not in df.columns  # hw spread was empty — no hardware columns
        # Performance data from the log must still be captured
        assert df.iloc[0]["benchmark"] == "llama2-70b"
        assert df.iloc[0]["throughput_tok_per_sec_per_gpu"] > 0

    def test_results_dir_is_file_skipped(self, tmp_path):
        """A file named 'results' (not a dir) triggers the is_dir() guard (line 357)."""
        repo = tmp_path / "repo"
        sub_dir = repo / "closed" / "Sub"
        sub_dir.mkdir(parents=True)
        (sub_dir / "results").write_text("not a directory", encoding="utf-8")
        assert parse_repo(repo, "v6.0").empty

    def test_system_dir_is_file_skipped(self, tmp_path):
        """A file inside results/ triggers the system-dir is_dir() guard (line 361)."""
        repo = tmp_path / "repo"
        results_dir = repo / "closed" / "Sub" / "results"
        results_dir.mkdir(parents=True)
        (results_dir / "not_a_system.txt").write_text("file", encoding="utf-8")
        assert parse_repo(repo, "v6.0").empty

    def test_benchmark_dir_is_file_skipped(self, tmp_path):
        """A file inside a system dir triggers the benchmark is_dir() guard: it must be selective, skipping the file while still processing a valid sibling benchmark dir — assert df.empty alone wouldn't catch an over-aggressive guard skipping the whole system."""
        system_name = SYSTEM_JSON_NVIDIA["system_name"]
        repo = tmp_path / "repo"
        system_dir = repo / "closed" / "Sub" / "results" / system_name
        system_dir.mkdir(parents=True)

        # Invalid entry — must be skipped
        (system_dir / "readme.txt").write_text("file", encoding="utf-8")

        # Valid sibling — must still be processed
        run_dir = system_dir / "llama2-70b" / "Offline" / "performance" / "run_1"
        run_dir.mkdir(parents=True)
        (run_dir / "mlperf_log_summary.txt").write_text(
            LOG_SUMMARY_OFFLINE, encoding="utf-8"
        )

        sys_dir = repo / "closed" / "Sub" / "systems"
        sys_dir.mkdir(parents=True)
        (sys_dir / f"{system_name}.json").write_text(
            json.dumps(SYSTEM_JSON_NVIDIA), encoding="utf-8"
        )
        df = parse_repo(repo, "v6.0")
        assert len(df) == 1                           # file skipped, valid dir kept
        assert df.iloc[0]["benchmark"] == "llama2-70b"

    def test_scenario_dir_is_file_skipped(self, tmp_path):
        """A file inside a benchmark dir triggers the scenario is_dir() guard (line 401)."""
        system_name = SYSTEM_JSON_NVIDIA["system_name"]
        repo = tmp_path / "repo"
        benchmark_dir = (
            repo / "closed" / "Sub" / "results" / system_name / "llama2-70b"
        )
        benchmark_dir.mkdir(parents=True)
        (benchmark_dir / "metadata.txt").write_text("file", encoding="utf-8")
        sys_dir = repo / "closed" / "Sub" / "systems"
        sys_dir.mkdir(parents=True)
        (sys_dir / f"{system_name}.json").write_text(
            json.dumps(SYSTEM_JSON_NVIDIA), encoding="utf-8"
        )
        assert parse_repo(repo, "v6.0").empty

    def test_no_performance_run_row_skipped(self, tmp_path):
        """Scenario dir with no performance/run_N tree produces no row (lines 406-407)."""
        system_name = SYSTEM_JSON_NVIDIA["system_name"]
        repo = tmp_path / "repo"
        # Scenario dir exists but NO performance/ inside it
        scenario_dir = (
            repo / "closed" / "Sub" / "results"
            / system_name / "llama2-70b" / "Offline"
        )
        scenario_dir.mkdir(parents=True)
        sys_dir = repo / "closed" / "Sub" / "systems"
        sys_dir.mkdir(parents=True)
        (sys_dir / f"{system_name}.json").write_text(
            json.dumps(SYSTEM_JSON_NVIDIA), encoding="utf-8"
        )
        assert parse_repo(repo, "v6.0").empty

    def test_scenario_mismatch_debug_logged(self, tmp_path, caplog):
        """Scenario in log != directory name emits a debug message (line 414)."""
        system_name = SYSTEM_JSON_NVIDIA["system_name"]
        repo = tmp_path / "repo"
        # Directory is "Offline" but the log file claims "Scenario : Server"
        mismatched = LOG_SUMMARY_OFFLINE.replace(
            "Scenario : Offline", "Scenario : Server"
        )
        run_dir = (
            repo / "closed" / "Sub" / "results"
            / system_name / "llama2-70b" / "Offline" / "performance" / "run_1"
        )
        run_dir.mkdir(parents=True)
        (run_dir / "mlperf_log_summary.txt").write_text(mismatched, encoding="utf-8")
        sys_dir = repo / "closed" / "Sub" / "systems"
        sys_dir.mkdir(parents=True)
        (sys_dir / f"{system_name}.json").write_text(
            json.dumps(SYSTEM_JSON_NVIDIA), encoding="utf-8"
        )
        with caplog.at_level(logging.DEBUG, logger="src.data.mlperf_parser"):
            df = parse_repo(repo, "v6.0")
        assert len(df) == 1  # row is still produced
        assert any("mismatch" in r.getMessage().lower() for r in caplog.records)


# Coverage gap E: CLI entry point — main() and _build_arg_parser()

class TestMainCLI:
    """Integration tests for main() via monkeypatched sys.argv; _build_repo(parent, ...) creates a repo at parent/inference_results_v6.0/, so --repos-dir parent --rounds inference_results_v6.0 is the correct shape."""

    def test_no_round_dirs_exits_1(self, tmp_path, monkeypatch):
        """--repos-dir with no matching subdirectories → SystemExit(1)."""
        repos_dir = tmp_path / "repos"
        repos_dir.mkdir()
        monkeypatch.setattr(sys, "argv", [
            "mlperf_parser",
            "--repos-dir", str(repos_dir),
            "--rounds", "v6.0",
        ])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1

    def test_empty_round_exits_1(self, tmp_path, monkeypatch):
        """Round dir exists but is empty → parse_repos returns empty df → SystemExit(1)."""
        repos_dir = tmp_path / "repos"
        (repos_dir / "v6.0").mkdir(parents=True)
        monkeypatch.setattr(sys, "argv", [
            "mlperf_parser",
            "--repos-dir", str(repos_dir),
            "--rounds", "v6.0",
        ])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1

    def test_normal_run_writes_csv(self, tmp_path, monkeypatch, capsys):
        """Normal invocation writes a CSV and prints a row-count summary."""
        repos_dir = tmp_path / "repos"
        _build_repo(repos_dir, SYSTEM_JSON_NVIDIA, scenarios=["Offline"])
        out_csv = tmp_path / "out.csv"
        monkeypatch.setattr(sys, "argv", [
            "mlperf_parser",
            "--repos-dir", str(repos_dir),
            "--rounds", "inference_results_v6.0",
            "--output", str(out_csv),
        ])
        main()
        assert out_csv.exists()
        result = pd.read_csv(out_csv)
        assert len(result) == 1
        assert result.iloc[0]["benchmark"] == "llama2-70b"   # not just any row
        assert result.iloc[0]["throughput_samples_per_sec"] == pytest.approx(25.34)
        assert "Total rows" in capsys.readouterr().out

    def test_parquet_flag_writes_parquet(self, tmp_path, monkeypatch):
        """--parquet produces a .parquet file alongside the CSV."""
        repos_dir = tmp_path / "repos"
        _build_repo(repos_dir, SYSTEM_JSON_NVIDIA, scenarios=["Offline"])
        out_csv = tmp_path / "out.csv"
        monkeypatch.setattr(sys, "argv", [
            "mlperf_parser",
            "--repos-dir", str(repos_dir),
            "--rounds", "inference_results_v6.0",
            "--output", str(out_csv),
            "--parquet",
        ])
        main()
        pq_path = out_csv.with_suffix(".parquet")
        assert pq_path.exists()
        assert len(pd.read_parquet(pq_path)) == 1   # file is readable, not just present

    def test_all_benchmarks_flag_includes_non_llm(self, tmp_path, monkeypatch):
        """--all-benchmarks passes llm_only=False, including non-LLM benchmarks."""
        repos_dir = tmp_path / "repos"
        _build_repo(
            repos_dir, SYSTEM_JSON_NVIDIA,
            benchmarks=["resnet50"],
            scenarios=["Offline"],
        )
        out_csv = tmp_path / "out.csv"
        monkeypatch.setattr(sys, "argv", [
            "mlperf_parser",
            "--repos-dir", str(repos_dir),
            "--rounds", "inference_results_v6.0",
            "--output", str(out_csv),
            "--all-benchmarks",
        ])
        main()
        assert "resnet50" in pd.read_csv(out_csv)["benchmark"].values
