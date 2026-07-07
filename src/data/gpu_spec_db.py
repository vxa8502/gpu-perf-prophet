"""
GPU specification database loader for GPU Perf Prophet.

Loads data/gpu_specs.yaml and provides two public entry-points:

    normalize_gpu_name(raw_name, specs) → Optional[str]
        Maps a raw MLPerf accelerator_model_name to a canonical GPU id.
        Returns None for heterogeneous multi-GPU strings, unknown GPUs, or
        explicitly unrecognised names.

    enrich_df(df, spec_path) → pd.DataFrame
        Adds GPU hardware spec columns to a parsed MLPerf DataFrame.
        Rows whose gpu_name cannot be normalized get NaN in spec columns.
"""

from __future__ import annotations

import functools
import logging
import re
import stat as _stat
import types
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

# Real GPU spec files are ~10 KB.  1 MB cap blocks both accidental and
# adversarial oversized files; mirrors _safe_read_text in mlperf_parser.py.
_MAX_SPEC_BYTES: int = 1 * 1024 * 1024  # 1 MB

log = logging.getLogger(__name__)

_DEFAULT_SPEC_PATH = Path(__file__).parent.parent.parent / "data" / "gpu_specs.yaml"

# Columns added by enrich_df, in output order.
SPEC_COLUMNS = [
    "canonical_gpu_id",
    "gpu_vendor",
    "gpu_architecture",
    "gpu_memory_type",
    "gpu_vram_gb",
    "gpu_hbm_bandwidth_tbps",
    "gpu_peak_fp32_tflops",
    "gpu_peak_bf16_tflops",
    "gpu_peak_fp16_tflops",
    "gpu_peak_fp8_tflops",
    "gpu_peak_fp6_tflops",
    "gpu_peak_fp4_tflops",
    "gpu_peak_int8_tops",
    "gpu_cu_sm_count",
    "gpu_l2_cache_mb",
    "gpu_tdp_w",
    "gpu_in_model_scope",
    "gpu_spec_confidence",
]

# Matches "(x8)", "(x16)", "(x32)", "(x87)" etc.
_COUNT_SUFFIX_RE = re.compile(r"\s*\(x\d+\)\s*$")
# Matches "(Power Cap 1000 W)" and similar parenthetical power notes
_POWER_SUFFIX_RE = re.compile(r"\s*\(Power Cap[^)]*\)\s*$", re.IGNORECASE)


@functools.lru_cache(maxsize=4)
def _build_spec_cache(spec_path_str: str) -> tuple[list, dict, dict]:
    """
    Load the spec file and build the two derived lookup structures.
    Cached by path string so repeated enrich_df calls pay the I/O and
    parse cost only once per unique path.

    Returns (specs, alias_index, id_map).

    The cache key is the literal path string (not resolved), so the
    symlink guard in load_specs still fires on first access.
    Cached objects must not be mutated by callers.
    """
    specs  = load_specs(Path(spec_path_str))
    index  = _build_alias_index(specs)
    # Wrap id_map values in MappingProxyType so callers cannot mutate the
    # cached spec rows.  The proxy supports all read operations (get, iter,
    # subscript) that downstream code uses.
    id_map = {s["id"]: types.MappingProxyType(_spec_row(s)) for s in specs}
    return specs, index, id_map


@functools.lru_cache(maxsize=8)
def load_specs(spec_path: Path | str = _DEFAULT_SPEC_PATH) -> list[dict]:
    """
    Load and return the list of GPU spec dicts from the YAML file.

    Raises ValueError for symlinks (path traversal guard) or oversized files.
    Raises FileNotFoundError / OSError for I/O problems.
    Raises ValueError for malformed YAML that is missing the 'gpus' key.
    """
    path = Path(spec_path)
    try:
        st = path.lstat()
    except OSError as exc:
        raise FileNotFoundError(f"GPU spec DB not found: {path}") from exc
    if _stat.S_ISLNK(st.st_mode):
        raise ValueError(f"GPU spec DB path is a symlink (refused): {path}")
    if st.st_size > _MAX_SPEC_BYTES:
        raise ValueError(
            f"GPU spec DB too large ({st.st_size} bytes > {_MAX_SPEC_BYTES}): {path}"
        )
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or "gpus" not in data:
        raise ValueError(
            f"GPU spec DB at {path} is missing required 'gpus' key "
            f"(got {type(data).__name__})"
        )
    return data["gpus"]


def _build_alias_index(specs: list[dict]) -> dict[str, dict]:
    """Return lowercase-alias → spec dict for fast lookup."""
    index: dict[str, dict] = {}
    for spec in specs:
        for alias in spec.get("aliases", []):
            key = alias.lower()
            if key in index and index[key]["id"] != spec["id"]:
                # Warn only when the same alias maps to two *different* GPU ids.
                # Duplicate aliases within the same GPU (e.g. "HBM3E" vs "HBM3e")
                # are harmless after case-folding and should not produce noise.
                log.warning(
                    "Conflicting alias %r maps to both %r and %r in GPU spec DB",
                    alias, index[key]["id"], spec["id"],
                )
            index[key] = spec
    return index


def _is_heterogeneous(raw_name: str) -> bool:
    """Return True for multi-GPU strings mixing different SKUs."""
    # "A and B" pattern (heterogeneous mix)
    if " and " in raw_name.lower():
        return True
    # "A (xN), B (xM)" pattern
    if "," in raw_name and re.search(r"\(x\d+\)", raw_name):
        return True
    return False


def _strip_suffixes(raw_name: str) -> str:
    """Remove parenthetical count and power suffixes from a GPU name."""
    name = _COUNT_SUFFIX_RE.sub("", raw_name)
    name = _POWER_SUFFIX_RE.sub("", name)
    return name.strip()


def normalize_gpu_name(
    raw_name: str | None,
    specs: list[dict],
    *,
    _index: dict[str, dict] | None = None,
) -> Optional[str]:
    """
    Map a raw MLPerf accelerator_model_name to a canonical GPU id.

    Returns None for:
      - null / "N/A" names
      - heterogeneous multi-GPU strings (e.g. "MI300X ... and MI325X ...")
      - names that don't match any alias in the spec DB (logged at DEBUG)
    """
    if not raw_name or raw_name.strip().upper() in {"N/A", "NA", "NONE", ""}:
        return None

    if _is_heterogeneous(raw_name):
        log.debug("Skipping heterogeneous GPU string: %r", raw_name)
        return None

    index = _index if _index is not None else _build_alias_index(specs)
    clean = _strip_suffixes(raw_name)

    match = index.get(clean.lower())
    if match is None:
        log.debug("No GPU spec match for: %r (stripped: %r)", raw_name, clean)
        return None

    return match["id"]


def _spec_row(spec: dict) -> dict:
    """Flatten one GPU spec dict into the SPEC_COLUMNS schema."""
    pt = spec.get("peak_tflops") or {}
    return {
        "canonical_gpu_id":        spec["id"],
        "gpu_vendor":              spec.get("vendor"),
        "gpu_architecture":        spec.get("architecture"),
        "gpu_memory_type":         spec.get("memory_type"),
        "gpu_vram_gb":             spec.get("vram_gb"),
        "gpu_hbm_bandwidth_tbps":  spec.get("hbm_bandwidth_tbps"),
        "gpu_peak_fp32_tflops":    pt.get("fp32"),
        "gpu_peak_bf16_tflops":    pt.get("bf16"),
        "gpu_peak_fp16_tflops":    pt.get("fp16"),
        "gpu_peak_fp8_tflops":     pt.get("fp8"),
        "gpu_peak_fp6_tflops":     pt.get("fp6"),
        "gpu_peak_fp4_tflops":     pt.get("fp4"),
        "gpu_peak_int8_tops": pt.get("int8"),
        # AMD uses compute_units; NVIDIA uses streaming_multiprocessors.
        # Stored under a unified key so feature engineering is vendor-agnostic.
        "gpu_cu_sm_count": (
            spec.get("compute_units") or spec.get("streaming_multiprocessors")
        ),
        "gpu_l2_cache_mb": spec.get("l2_cache_mb"),
        "gpu_tdp_w": spec.get("tdp_w"),
        "gpu_in_model_scope": spec.get("in_model_scope"),
        "gpu_spec_confidence": spec.get("spec_confidence"),
    }


def enrich_df(
    df: pd.DataFrame,
    spec_path: Path | str = _DEFAULT_SPEC_PATH,
) -> pd.DataFrame:
    """
    Add GPU hardware spec columns to a parsed MLPerf DataFrame.

    Joins on the canonical_gpu_id derived from each row's gpu_name.
    Rows whose gpu_name cannot be normalized receive NaN in all spec columns.
    The original DataFrame is not modified; a new DataFrame is returned.
    """
    # Fix 1: derive lookup structures once per unique path (cached).
    specs, index, id_map = _build_spec_cache(str(Path(spec_path)))

    # Fix 2: normalize once per unique gpu_name, then broadcast to all rows.
    # The corpus has ~33 unique names across ~1223 rows; calling
    # normalize_gpu_name per-row wastes ~1190 redundant lookups.
    unique_names = {n for n in df["gpu_name"].dropna().unique()}
    name_to_id = {
        name: normalize_gpu_name(name, specs, _index=index)
        for name in unique_names
    }
    canonical_ids = df["gpu_name"].map(name_to_id)

    spec_df = pd.DataFrame(
        [id_map.get(cid, {}) for cid in canonical_ids],
        index=df.index,
    )

    # Ensure all expected columns exist even if spec_rows were all empty.
    for col in SPEC_COLUMNS:
        if col not in spec_df.columns:
            spec_df[col] = None

    matched = canonical_ids.notna().sum()
    unmatched = canonical_ids.isna().sum()
    log.info(
        "GPU spec enrichment: %d rows matched, %d unmatched (%.0f%%)",
        matched, unmatched, 100 * matched / max(len(df), 1),
    )
    if unmatched:
        unseen = set(df.loc[canonical_ids.isna(), "gpu_name"].dropna().unique())
        # Use %r so control characters (newlines, ANSI escapes) in GPU names
        # are escaped rather than injected into the log stream.
        for name in sorted(unseen):
            log.debug("Unmatched gpu_name: %r", name)

    result = pd.concat([df, spec_df[SPEC_COLUMNS]], axis=1)

    # Warn when the parser's reported vram_gb (from system JSON
    # accelerator_memory_capacity) differs from the spec DB's gpu_vram_gb.
    # This happens when submitters report total system VRAM rather than
    # per-GPU capacity.  Use gpu_vram_gb for the memory-fit constraint;
    # treat vram_gb as unreliable for multi-GPU submissions.
    if "vram_gb" in result.columns and "gpu_vram_gb" in result.columns:
        conflict = (
            result["vram_gb"].notna()
            & result["gpu_vram_gb"].notna()
            & (result["vram_gb"] != result["gpu_vram_gb"])
        )
        if conflict.any():
            n = conflict.sum()
            examples = (
                result.loc[conflict, ["gpu_name", "num_gpus", "vram_gb", "gpu_vram_gb"]]
                .drop_duplicates()
            )
            log.warning(
                "%d rows: vram_gb (parser) != gpu_vram_gb (spec DB). "
                "Some submitters report total system VRAM. "
                "Use gpu_vram_gb for the memory-fit constraint.",
                n,
            )
            # Fix 3: itertuples ~10x faster than iterrows; sufficient for
            # attribute access.  %r quotes gpu_name to prevent log injection.
            for row in examples.itertuples(index=False):
                log.warning(
                    "  vram conflict: gpu=%r  num_gpus=%s  reported=%s  spec=%s",
                    row.gpu_name, row.num_gpus, row.vram_gb, row.gpu_vram_gb,
                )

    return result
