"""Data manifest — pins source snapshots (git commit or SHA-256) for reproducibility; verification is warn-only (not hard-refuse) since a single-developer project can't enforce refresh-then-relock discipline, but symlinked inputs are always refused outright, same as gpu_spec_db/recommender/predictor's integrity guards."""

from __future__ import annotations

import hashlib
import logging
import stat as _stat
import subprocess
from pathlib import Path
from typing import Optional

import yaml

log = logging.getLogger(__name__)

MLPERF_ROUNDS = ["v4.1", "v5.0", "v5.1", "v6.0"]
MLPERF_ROOT = Path("data/raw/mlperf")
CALIBRATION_CSV = Path("benchmarks/mi300x_calibration_results.csv")
MANIFEST_PATH = Path("data/data_manifest.lock")

# Calibration CSVs/lock files are KB-scale; 1 MB matches gpu_spec_db.py/recommender.py's cap for gpu_specs.yaml/pricing.yaml.
_MAX_HASHED_FILE_BYTES = 1 * 1024 * 1024  # 1 MB

# The training corpus grows with every merged round/batch, so it gets its own larger cap; 20 MB is generous headroom over the current ~115 KB file.
_MAX_CORPUS_BYTES = 20 * 1024 * 1024  # 20 MB


def _refuse_symlink_and_oversize(path: Path, max_bytes: int) -> None:
    """Mirror gpu_spec_db.load_specs / recommender._load_pricing's guard: a single lstat() refuses symlinks and oversized files before this module hashes/parses a trust-anchor file."""
    try:
        st = path.lstat()
    except OSError as exc:
        raise FileNotFoundError(f"Path not found: {path}") from exc
    if _stat.S_ISLNK(st.st_mode):
        raise ValueError(f"Refusing to read a symlinked path: {path}")
    if st.st_size > max_bytes:
        raise ValueError(f"{path} too large ({st.st_size} bytes > {max_bytes})")


def _run_git(repo_dir: Path, *args: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_dir), *args],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _file_sha256(path: Path, max_bytes: Optional[int] = None) -> Optional[str]:
    if not path.is_file():
        return None
    # Resolved inside the call (not a bound default) so tests that monkeypatch _MAX_HASHED_FILE_BYTES still take effect.
    if max_bytes is None:
        max_bytes = _MAX_HASHED_FILE_BYTES
    _refuse_symlink_and_oversize(path, max_bytes)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def corpus_sha256(path: Path) -> Optional[str]:
    """SHA-256 of a built training corpus file for the model sidecar's ``corpus_sha256`` field, using the larger ``_MAX_CORPUS_BYTES`` cap; returns None if the corpus isn't present (normal on a fresh clone)."""
    return _file_sha256(path, max_bytes=_MAX_CORPUS_BYTES)


def _mlperf_source(round_tag: str) -> dict:
    repo_dir = MLPERF_ROOT / round_tag
    # Same as mlperf_parser.py's walk guard — a symlinked round dir could let `git -C` report an arbitrary repo as the pinned mirror.
    if repo_dir.is_symlink():
        raise ValueError(f"Refusing to read a symlinked MLPerf mirror directory: {repo_dir}")
    # One `git log` call gets commit hash + date via \x1f delimiter, saving a subprocess spawn (measured 12 spawns/260ms across 4 rounds before this).
    log_out = _run_git(repo_dir, "log", "-1", "--format=%H\x1f%cI")
    if log_out and "\x1f" in log_out:
        commit, commit_date = log_out.split("\x1f", 1)
    else:
        commit, commit_date = None, None
    url = _run_git(repo_dir, "remote", "get-url", "origin")
    return {
        "type": "git",
        "path": str(repo_dir),
        "url": url,
        "commit": commit,
        "snapshot_date": commit_date[:10] if commit_date else None,
        "present": repo_dir.is_dir(),
    }


def _calibration_source() -> dict:
    return {
        "type": "file",
        "path": str(CALIBRATION_CSV),
        "url": "self_run:amd_dev_cloud",
        "sha256": _file_sha256(CALIBRATION_CSV),
        "snapshot_date": None,
        "present": CALIBRATION_CSV.is_file(),
    }


def compute_manifest() -> dict:
    """Build a manifest reflecting the current state of every known source."""
    sources = {f"mlperf_{tag.replace('.', '_')}": _mlperf_source(tag) for tag in MLPERF_ROUNDS}
    sources["mi300x_calibration"] = _calibration_source()
    return {"sources": sources}


def write_manifest(path: Path = MANIFEST_PATH) -> dict:
    """Compute the current manifest and persist it as the locked snapshot."""
    manifest = compute_manifest()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False, default_flow_style=False)
    log.info("Wrote data manifest → %s", path)
    return manifest


def load_manifest(path: Path = MANIFEST_PATH) -> Optional[dict]:
    if not path.is_file():
        return None
    # The lock file is the trust anchor verify_manifest() checks everything else against, so it gets the same symlink/size guard as _file_sha256.
    _refuse_symlink_and_oversize(path, _MAX_HASHED_FILE_BYTES)
    with path.open() as f:
        return yaml.safe_load(f)


def _identity(entry: dict) -> Optional[str]:
    """The value that must stay stable for a source to be considered unchanged."""
    return entry.get("commit") if entry.get("type") == "git" else entry.get("sha256")


def sources_present(
    path: Path = MANIFEST_PATH, locked: Optional[dict] = None
) -> tuple[int, int]:
    """(present, total) count of locked sources that exist on disk right now — gives context to train_final.py's flattened manifest_verified boolean; cheap presence-only check (no git spawns/hashing); pass an already-loaded ``locked`` dict to avoid re-reading the lock file."""
    if locked is None:
        locked = load_manifest(path)
    if locked is None:
        return 0, 0
    sources = locked.get("sources", {})
    present = 0
    for entry in sources.values():
        p = Path(entry["path"])
        if p.is_symlink():
            continue  # a tampered/symlinked path is not a genuine "present" source
        is_present = p.is_dir() if entry.get("type") == "git" else p.is_file()
        if is_present:
            present += 1
    return present, len(sources)


def verify_manifest(path: Path = MANIFEST_PATH, locked: Optional[dict] = None) -> list[str]:
    """Compare the locked manifest against current source state; returns human-readable mismatch descriptions (absent sources aren't drift) and logs a WARNING per mismatch, but never raises (warn-only by design — see module docstring)."""
    if locked is None:
        locked = load_manifest(path)
    if locked is None:
        log.warning("No data manifest found at %s — skipping source-drift check.", path)
        return []

    current = compute_manifest()
    mismatches: list[str] = []

    for name, locked_entry in locked.get("sources", {}).items():
        current_entry = current["sources"].get(name)
        if current_entry is None:
            continue
        if not current_entry.get("present"):
            continue  # not fetched locally — not a drift, just absent
        locked_id = _identity(locked_entry)
        current_id = _identity(current_entry)
        if locked_id is None or current_id is None:
            continue
        if locked_id != current_id:
            msg = (
                f"data source '{name}' has drifted from data_manifest.lock: "
                f"locked={locked_id[:12]}  current={current_id[:12]}"
            )
            mismatches.append(msg)
            log.warning(msg)

    if mismatches:
        log.warning(
            "%d data source(s) drifted from the locked manifest (%s). "
            "Training will proceed (warn-only by design) — "
            "if this drift wasn't intentional, results may not reproduce prior runs. "
            "Regenerate the lock file with `python -m src.data.manifest` once you've "
            "confirmed the new source state is intentional.",
            len(mismatches), path,
        )
    return mismatches


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    write_manifest()


if __name__ == "__main__":
    main()
