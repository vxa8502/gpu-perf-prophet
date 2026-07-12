"""Tests for src/data/gpu_spec_db.py; tests reading data/gpu_specs.yaml use the real file since it's curated reference data, not generated output."""

import logging
from pathlib import Path
import pandas as pd
import pytest
import yaml

from src.data.gpu_spec_db import (
    _MAX_SPEC_BYTES,
    SPEC_COLUMNS,
    _build_alias_index,
    _is_heterogeneous,
    _strip_suffixes,
    enrich_df,
    load_specs,
    normalize_gpu_name,
)


# Fixtures

@pytest.fixture(scope="module")
def specs():
    # Shallow copy so test mutations (e.g. specs.append(...)) do not corrupt the load_specs() lru_cache, which is shared across the whole test process.
    return list(load_specs())


@pytest.fixture(scope="module")
def alias_index(specs):
    return _build_alias_index(specs)


# YAML integrity

class TestYamlIntegrity:
    REQUIRED_FIELDS = [
        "id", "name", "vendor", "architecture", "memory_type",
        "vram_gb", "hbm_bandwidth_tbps", "peak_tflops",
        "in_model_scope", "spec_confidence", "aliases",
    ]
    # Precisions every in-scope GPU supports (A100 Ampere lacks native FP8, intentionally excluded; covered by a separate test below).
    REQUIRED_TFLOPS = ["fp32", "bf16", "fp16"]

    def test_all_ids_unique(self, specs):
        ids = [s["id"] for s in specs]
        assert len(ids) == len(set(ids)), "Duplicate GPU ids in spec DB"

    def test_required_fields_present(self, specs):
        for spec in specs:
            for field in self.REQUIRED_FIELDS:
                assert field in spec, (
                    f"GPU {spec.get('id')!r} missing required field {field!r}"
                )

    def test_all_gpus_have_valid_spec_confidence(self, specs):
        # Renamed from "..._in_scope_gpus_..." — the loop had no in_model_scope guard, so it actually checked all GPUs; name now matches behavior.
        valid = {"verified", "estimated"}
        for spec in specs:
            assert spec["spec_confidence"] in valid, (
                f"{spec['id']}: spec_confidence must be 'verified' or 'estimated', "
                f"got {spec['spec_confidence']!r}"
            )

    def test_in_scope_gpus_have_bandwidth_and_bf16(self, specs):
        """In-scope GPUs must have positive bandwidth and BF16 TFLOPS — zero/None would silently corrupt the roofline model (div-by-zero or a 0-throughput ceiling)."""
        for spec in specs:
            if not spec["in_model_scope"]:
                continue
            bw = spec["hbm_bandwidth_tbps"]
            assert bw is not None and bw > 0, (
                f"{spec['id']}: hbm_bandwidth_tbps must be > 0, got {bw!r}"
            )
            bf16 = spec["peak_tflops"].get("bf16")
            assert bf16 is not None and bf16 > 0, (
                f"{spec['id']}: peak_tflops.bf16 must be > 0, got {bf16!r}"
            )

    def test_in_scope_gpus_have_positive_required_tflops(self, specs):
        """REQUIRED_TFLOPS must be positive for every in-scope GPU — catches accidentally-zero values that would silently make the roofline predict 0 throughput."""
        for spec in specs:
            if not spec["in_model_scope"]:
                continue
            pt = spec.get("peak_tflops") or {}
            for key in self.REQUIRED_TFLOPS:
                val = pt.get(key)
                assert val is not None and val > 0, (
                    f"{spec['id']}: peak_tflops.{key} must be > 0 for in-scope GPU, "
                    f"got {val!r}"
                )

    def test_non_null_tflops_are_positive(self, specs):
        """Any non-null peak_tflops value (including fp8, fp6, fp4) must be > 0 — null means unsupported (e.g. A100 fp8), but a non-null zero would silently corrupt roofline calculations."""
        for spec in specs:
            pt = spec.get("peak_tflops") or {}
            for key, val in pt.items():
                if val is not None:
                    assert val > 0, (
                        f"{spec['id']}: peak_tflops.{key} is non-null but not > 0: {val!r}"
                    )

    def test_no_conflicting_aliases_across_gpus(self, specs):
        """No alias should map to two different GPU ids after case-folding."""
        seen: dict[str, str] = {}
        for spec in specs:
            for alias in spec.get("aliases", []):
                key = alias.lower()
                if key in seen and seen[key] != spec["id"]:
                    pytest.fail(
                        f"Alias {alias!r} maps to both {seen[key]!r} and {spec['id']!r}"
                    )
                seen[key] = spec["id"]

    def test_amd_gpus_have_compute_units(self, specs):
        for spec in specs:
            if spec["vendor"] == "amd":
                assert "compute_units" in spec, (
                    f"{spec['id']}: AMD GPU missing compute_units"
                )

    def test_nvidia_gpus_have_streaming_multiprocessors(self, specs):
        for spec in specs:
            if spec["vendor"] == "nvidia":
                assert "streaming_multiprocessors" in spec, (
                    f"{spec['id']}: NVIDIA GPU missing streaming_multiprocessors"
                )

    def test_in_scope_gpu_ids_present(self, specs):
        """The eight v1-scope GPUs must all be present and in-scope."""
        required = {"mi300x", "mi325x", "mi355x", "h100_sxm", "h200_sxm",
                    "a100_sxm_80gb", "l4", "rtx4090"}
        in_scope_ids = {s["id"] for s in specs if s["in_model_scope"]}
        missing = required - in_scope_ids
        assert not missing, f"Required GPUs missing from spec DB: {missing}"


# Security tests: load_specs guards

class TestLoadSpecsSecurity:
    def _minimal_yaml(self, tmp_path, content: dict) -> Path:
        p = tmp_path / "specs.yaml"
        p.write_text(yaml.dump(content), encoding="utf-8")
        return p

    def test_rejects_symlink(self, tmp_path):
        real = self._minimal_yaml(tmp_path, {"gpus": []})
        link = tmp_path / "link.yaml"
        link.symlink_to(real)
        with pytest.raises(ValueError, match="symlink"):
            load_specs(link)

    def test_rejects_symlink_to_outside(self, tmp_path):
        link = tmp_path / "escape.yaml"
        link.symlink_to("/etc/hosts")
        with pytest.raises(ValueError, match="symlink"):
            load_specs(link)

    def test_rejects_oversized_file(self, tmp_path):
        p = tmp_path / "huge.yaml"
        p.write_bytes(b"# " + b"x" * _MAX_SPEC_BYTES)
        with pytest.raises(ValueError, match="too large"):
            load_specs(p)

    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_specs(tmp_path / "nonexistent.yaml")

    def test_missing_gpus_key_raises_value_error(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text(yaml.dump({"schema_version": "1.0"}), encoding="utf-8")
        with pytest.raises(ValueError, match="gpus"):
            load_specs(p)

    def test_empty_yaml_raises_value_error(self, tmp_path):
        p = tmp_path / "empty.yaml"
        p.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="gpus"):
            load_specs(p)

    def test_valid_minimal_yaml_loads(self, tmp_path):
        p = self._minimal_yaml(tmp_path, {"gpus": [{"id": "test"}]})
        result = load_specs(p)
        assert result == [{"id": "test"}]


class TestLogInjection:
    def test_newline_in_gpu_name_does_not_inject_log_line(self, caplog):
        """A GPU name containing \\n must not create a spurious second log line."""
        injected = "NVIDIA H100\n[CRITICAL] Forged log entry"
        df = pd.DataFrame([{"gpu_name": injected}])
        with caplog.at_level(logging.DEBUG, logger="src.data.gpu_spec_db"):
            enrich_df(df)
        # The injected newline must appear repr-quoted, not as a raw newline in a forged second log line.
        for record in caplog.records:
            assert "\n[CRITICAL]" not in record.getMessage(), (
                "Log injection: raw newline from GPU name appeared in log output"
            )

    def test_ansi_escape_in_gpu_name_repr_quoted(self, caplog):
        """ANSI escape sequences in GPU names must be repr-quoted in debug logs."""
        ansi_name = "GPU\x1b[31mRED\x1b[0m"
        df = pd.DataFrame([{"gpu_name": ansi_name}])
        with caplog.at_level(logging.DEBUG, logger="src.data.gpu_spec_db"):
            enrich_df(df)
        raw_ansi = "\x1b[31m"
        for record in caplog.records:
            assert raw_ansi not in record.getMessage(), (
                "ANSI escape from GPU name appeared unescaped in log output"
            )


# _strip_suffixes

class TestStripSuffixes:
    @pytest.mark.parametrize("raw,expected", [
        ("AMD Instinct MI355X 288GB HBM3e (x87)",         "AMD Instinct MI355X 288GB HBM3e"),
        ("AMD Instinct MI300X 192GB HBM3 (x8)",           "AMD Instinct MI300X 192GB HBM3"),
        ("AMD Instinct MI355X 288GB HBM3e (Power Cap 1000 W)", "AMD Instinct MI355X 288GB HBM3e"),
        ("NVIDIA H100-SXM-80GB",                           "NVIDIA H100-SXM-80GB"),
        ("AMD Instinct MI300X 192GB HBM3",                 "AMD Instinct MI300X 192GB HBM3"),
    ])
    def test_strips_known_patterns(self, raw, expected):
        assert _strip_suffixes(raw) == expected


# _is_heterogeneous

class TestIsHeterogeneous:
    def test_mixed_and(self):
        assert _is_heterogeneous(
            "AMD Instinct MI300X 192GB HBM3 (x8) and AMD Instinct MI325X 256GB HBM3e (x8)"
        )

    def test_mixed_comma_count(self):
        assert _is_heterogeneous(
            "AMD Instinct MI300X 192GB HBM3 (x32), AMD Instinct MI325X 256GB HBM3e (x16)"
        )

    def test_single_gpu_not_heterogeneous(self):
        assert not _is_heterogeneous("AMD Instinct MI300X 192GB HBM3")

    def test_comma_without_count_not_flagged(self):
        # A name with a comma but no "(xN)" is not a multi-SKU string.
        assert not _is_heterogeneous("NVIDIA H100-SXM-80GB, Rev 2")


# normalize_gpu_name

class TestNormalizeGpuName:
    # Known exact aliases from the real MLPerf corpus
    @pytest.mark.parametrize("raw,expected_id", [
        ("AMD Instinct MI300X 192GB HBM3",           "mi300x"),
        ("AMD Instinct MI300X-NPS1-SPX-192GB-750W",  "mi300x"),
        ("AMD MI300X-NPS1-SPX-192GB-750W",           "mi300x"),
        ("AMD Instinct MI325X 256GB HBM3E",          "mi325x"),  # uppercase E
        ("AMD Instinct MI325X 256GB HBM3e",          "mi325x"),  # lowercase e
        ("AMD Instinct MI355X 288GB HBM3e",          "mi355x"),
        ("AMD Instinct MI350X 288GB HBM3e",          "mi350x"),
        ("NVIDIA H100-SXM-80GB",                     "h100_sxm"),
        ("Virtualized NVIDIA H100-SXM-80GB",         "h100_sxm"),
        ("NVIDIA H200-SXM-141GB",                    "h200_sxm"),
        ("NVIDIA H200-SXM-141GB-CTS",                "h200_sxm"),
        ("Virtualized NVIDIA H200-SXM-141GB",        "h200_sxm"),
        ("NVIDIA H200-NVL-141GB",                    "h200_nvl"),
        ("NVIDIA H100-PCIe-80GB",                    "h100_pcie"),
        ("NVIDIA H100-NVL-94GB",                     "h100_nvl"),
        ("NVIDIA GH200 Grace Hopper Superchip 96GB",  "gh200_96gb"),
        ("NVIDIA GH200 Grace Hopper Superchip 144GB", "gh200_144gb"),
        ("NVIDIA L40S",                              "l40s"),
        ("NVIDIA B200-SXM-180GB",                    "b200_sxm"),
        ("NVIDIA B300-SXM-270GB",                    "b300_sxm"),
        ("NVIDIA GB200",                             "gb200"),
        ("NVIDIA GB300",                             "gb300"),
    ])
    def test_known_aliases(self, raw, expected_id, specs, alias_index):
        result = normalize_gpu_name(raw, specs, _index=alias_index)
        assert result == expected_id, f"Expected {expected_id!r}, got {result!r} for {raw!r}"

    def test_count_suffix_stripped(self, specs, alias_index):
        """'(x87)' suffix must be stripped before alias lookup."""
        result = normalize_gpu_name(
            "AMD Instinct MI355X 288GB HBM3e (x87)", specs, _index=alias_index
        )
        assert result == "mi355x"

    def test_power_suffix_stripped(self, specs, alias_index):
        result = normalize_gpu_name(
            "AMD Instinct MI355X 288GB HBM3e (Power Cap 1000 W)",
            specs, _index=alias_index,
        )
        assert result == "mi355x"

    def test_heterogeneous_returns_none(self, specs, alias_index):
        result = normalize_gpu_name(
            "AMD Instinct MI300X 192GB HBM3 (x8) and AMD Instinct MI325X 256GB HBM3e (x8)",
            specs, _index=alias_index,
        )
        assert result is None

    def test_na_returns_none(self, specs, alias_index):
        assert normalize_gpu_name("N/A", specs, _index=alias_index) is None

    def test_none_input_returns_none(self, specs, alias_index):
        assert normalize_gpu_name(None, specs, _index=alias_index) is None

    def test_unknown_gpu_returns_none(self, specs, alias_index):
        assert normalize_gpu_name("NVIDIA H9000-FAKE", specs, _index=alias_index) is None


# enrich_df

class TestEnrichDf:
    def _minimal_df(self, gpu_name: str) -> pd.DataFrame:
        return pd.DataFrame([{"gpu_name": gpu_name}])

    def test_adds_all_spec_columns(self):
        df = self._minimal_df("AMD Instinct MI300X 192GB HBM3")
        enriched = enrich_df(df)
        for col in SPEC_COLUMNS:
            assert col in enriched.columns, f"Missing column {col!r}"

    def test_known_gpu_values_populated(self):
        df = self._minimal_df("NVIDIA H100-SXM-80GB")
        enriched = enrich_df(df)
        row = enriched.iloc[0]
        assert row["canonical_gpu_id"] == "h100_sxm"
        assert row["gpu_vendor"] == "nvidia"
        assert row["gpu_architecture"] == "hopper"
        assert row["gpu_vram_gb"] == pytest.approx(80.0)
        assert row["gpu_hbm_bandwidth_tbps"] == pytest.approx(3.35)
        assert row["gpu_peak_bf16_tflops"] == pytest.approx(989.4)
        assert row["gpu_in_model_scope"] == True  # noqa: E712 — numpy.bool_

    def test_unknown_gpu_yields_null_spec_columns(self):
        df = self._minimal_df("NVIDIA FAKE-9000")
        enriched = enrich_df(df)
        assert enriched.iloc[0]["canonical_gpu_id"] is None
        assert pd.isna(enriched.iloc[0]["gpu_vram_gb"])

    def test_heterogeneous_gpu_yields_null(self):
        df = self._minimal_df(
            "AMD Instinct MI300X 192GB HBM3 (x8) "
            "and AMD Instinct MI325X 256GB HBM3e (x8)"
        )
        enriched = enrich_df(df)
        assert enriched.iloc[0]["canonical_gpu_id"] is None

    def test_original_df_not_mutated(self):
        df = self._minimal_df("NVIDIA H200-SXM-141GB")
        original_cols = set(df.columns)
        enrich_df(df)
        assert set(df.columns) == original_cols

    def test_multi_row_df(self):
        df = pd.DataFrame([
            {"gpu_name": "AMD Instinct MI300X 192GB HBM3"},
            {"gpu_name": "NVIDIA H200-SXM-141GB"},
            {"gpu_name": "N/A"},
        ])
        enriched = enrich_df(df)
        assert len(enriched) == 3
        assert enriched.iloc[0]["canonical_gpu_id"] == "mi300x"
        assert enriched.iloc[1]["canonical_gpu_id"] == "h200_sxm"
        assert pd.isna(enriched.iloc[2]["canonical_gpu_id"])

    def test_out_of_scope_gpu_flagged(self):
        df = self._minimal_df("NVIDIA B200-SXM-180GB")
        enriched = enrich_df(df)
        assert enriched.iloc[0]["gpu_in_model_scope"] == False  # noqa: E712 — numpy.bool_

    def test_cu_sm_count_unified(self):
        """AMD CUs and NVIDIA SMs both appear under gpu_cu_sm_count."""
        amd = enrich_df(self._minimal_df("AMD Instinct MI300X 192GB HBM3")).iloc[0]
        nv = enrich_df(self._minimal_df("NVIDIA H100-SXM-80GB")).iloc[0]
        assert amd["gpu_cu_sm_count"] == 304
        assert nv["gpu_cu_sm_count"] == 132

    def test_duplicate_gpu_name_rows_all_enriched(self):
        """All rows sharing the same gpu_name must get spec columns populated — the dedup must broadcast results to all rows, not just the first occurrence."""
        df = pd.DataFrame([{"gpu_name": "NVIDIA H200-SXM-141GB"}] * 50)
        enriched = enrich_df(df)
        assert len(enriched) == 50
        assert (enriched["canonical_gpu_id"] == "h200_sxm").all()
        assert enriched["gpu_hbm_bandwidth_tbps"].notna().all()

    def test_vram_conflict_warning_logged(self, caplog):
        """enrich_df warns when parser vram_gb differs from spec DB gpu_vram_gb; simulates a submitter reporting total system VRAM (8x80=640 GB) instead of per-GPU capacity (80 GB)."""
        df = pd.DataFrame([{
            "gpu_name": "NVIDIA H100-SXM-80GB",
            "vram_gb":  640.0,   # 8-GPU total — spec DB stores 80 GB per-GPU
            "num_gpus": 8,
        }])
        with caplog.at_level(logging.WARNING, logger="src.data.gpu_spec_db"):
            enriched = enrich_df(df)
        assert enriched.iloc[0]["gpu_vram_gb"] == pytest.approx(80.0)
        assert any(
            "vram" in r.getMessage().lower()
            for r in caplog.records
            if r.levelno == logging.WARNING
        )

    def test_no_vram_conflict_no_warning(self, caplog):
        """No warning when vram_gb already matches the spec DB value."""
        df = pd.DataFrame([{
            "gpu_name": "NVIDIA H100-SXM-80GB",
            "vram_gb":  80.0,   # matches spec DB
            "num_gpus": 1,
        }])
        with caplog.at_level(logging.WARNING, logger="src.data.gpu_spec_db"):
            enrich_df(df)
        assert not any(
            "vram" in r.getMessage().lower()
            for r in caplog.records
            if r.levelno == logging.WARNING
        )


# _build_alias_index: conflicting alias warning

class TestBuildAliasIndex:
    def test_conflicting_alias_warns(self, caplog):
        """Same lowercase alias in two different specs emits a warning (line 122)."""
        specs = [
            {"id": "gpu_a", "aliases": ["Shared Alias", "only_a"]},
            {"id": "gpu_b", "aliases": ["shared alias", "only_b"]},  # same after lower()
        ]
        with caplog.at_level(logging.WARNING, logger="src.data.gpu_spec_db"):
            index = _build_alias_index(specs)
        assert any("conflict" in r.getMessage().lower() for r in caplog.records)
        # Last writer wins — gpu_b is processed second
        assert index["shared alias"]["id"] == "gpu_b"

    def test_same_alias_same_gpu_no_warning(self, caplog):
        """Duplicate alias within one GPU (case variants) must not warn."""
        specs = [
            {"id": "gpu_a", "aliases": ["HBM3E", "hbm3e"]},  # same after lower()
        ]
        with caplog.at_level(logging.WARNING, logger="src.data.gpu_spec_db"):
            _build_alias_index(specs)
        assert not any(r.levelno == logging.WARNING for r in caplog.records)
