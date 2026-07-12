"""
Data manifest — pins source snapshots for reproducibility.

Records, for every external data source feeding the training corpus, its
origin (git remote + commit for the MLPerf submission mirrors, or a SHA-256
for the flat self-run calibration CSV) and a snapshot date, so any historical
training run can be traced back to the exact source state it was trained on.

Verification is warn-only in v1, not hard-refuse. The original design
called for "refuse to train on hash mismatch," but a hard-refuse would
break routine data refreshes (``fetch_mlperf.sh`` pulling a newer MLPerf round,
or re-running the AMD Dev Cloud calibration) unless every refresh remembers to
regenerate this file first. Chose warn-only deliberately for a single-developer
project where that discipline can't be enforced by a second reviewer; revisit
if this project ever gains a contributor who could refresh data without also
being the one updating the lock file.

Warn-only applies strictly to *drift* (a locked hash and a current hash both
computed honestly, but differing). It does not apply to symlinked inputs —
every file/directory this module reads (the lock file itself, the calibration
CSV, each MLPerf mirror directory) refuses symlinks and oversized files outright
(raises ``ValueError``), matching the same guard ``gpu_spec_db.load_specs``,
``recommender._load_pricing``, and ``predictor``'s model-artifact loader
already apply to their own integrity-sensitive files. A module whose entire
purpose is verifying data integrity has no excuse for being the one place that
skips its own codebase's established symlink guard.
"""

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

# Real calibration CSVs and lock files are small (KB-scale). 1 MB matches the
# cap gpu_spec_db.py/recommender.py already use for gpu_specs.yaml/pricing.yaml.
_MAX_HASHED_FILE_BYTES = 1 * 1024 * 1024  # 1 MB

# The built training corpus (data/processed/mlperf_features.parquet) is a
# different file class than the lock file / calibration CSV above — it grows
# with every new MLPerf round and self-run calibration batch merged in, so it
# gets its own, larger cap rather than sharing _MAX_HASHED_FILE_BYTES. 20 MB
# is generous headroom over the current ~115 KB file.
_MAX_CORPUS_BYTES = 20 * 1024 * 1024  # 20 MB


def _refuse_symlink_and_oversize(path: Path, max_bytes: int) -> None:
    """Mirror gpu_spec_db.load_specs / recommender._load_pricing's guard: a
    single lstat() supplies both the symlink bit and size, so a file this
    module is about to hash or parse as a trust anchor can't be silently
    swapped for an attacker-controlled target via a symlink, nor be an
    unbounded read. Every other integrity-sensitive file this codebase reads
    (gpu_specs.yaml, pricing.yaml, the model artifact) already refuses
    symlinks this way — manifest.py verifies data integrity, so it must hold
    itself to at least the same bar, not a lower one.
    """
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
    # Resolved inside the call, not bound as a mutable default, so tests that
    # monkeypatch module._MAX_HASHED_FILE_BYTES (e.g. to exercise the
    # oversize-refusal path) still take effect — a `= _MAX_HASHED_FILE_BYTES`
    # default would capture the value once at import time and never see the
    # monkeypatch.
    if max_bytes is None:
        max_bytes = _MAX_HASHED_FILE_BYTES
    _refuse_symlink_and_oversize(path, max_bytes)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def corpus_sha256(path: Path) -> Optional[str]:
    """SHA-256 of a built training corpus file (e.g. mlperf_features.parquet),
    for the model sidecar's ``corpus_sha256`` field. Same symlink and
    size guards as every other integrity-sensitive file this module hashes,
    just against the larger ``_MAX_CORPUS_BYTES`` cap. Returns None if the
    corpus isn't present (data/processed/ is gitignored, so this is the normal
    state on a fresh clone before the corpus has been built).
    """
    return _file_sha256(path, max_bytes=_MAX_CORPUS_BYTES)


def _mlperf_source(round_tag: str) -> dict:
    repo_dir = MLPERF_ROOT / round_tag
    # Same principle as mlperf_parser.py's directory-walk guard ("git repos
    # can contain symlinks pointing anywhere on disk") — a symlinked round
    # directory would let `git -C <repo_dir>` silently report an arbitrary
    # repo's commit/remote as if it were the pinned MLPerf mirror.
    if repo_dir.is_symlink():
        raise ValueError(f"Refusing to read a symlinked MLPerf mirror directory: {repo_dir}")
    # One `git log` call for both the commit hash (%H — identical to a
    # separate `git rev-parse HEAD`) and the commit date, using \x1f (unit
    # separator) as a delimiter that can't collide with real git output.
    # Measured 12 subprocess spawns / 260ms for compute_manifest() across 4
    # rounds before this; a fresh process spawn per round is the dominant
    # cost here, not CPU work, so cutting one of the three per-round spawns
    # is the lever that matters. `git remote get-url origin` is a genuinely
    # separate git subcommand and can't be folded into the same call.
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
    # The lock file is the trust anchor verify_manifest() checks everything
    # else against — if *this* read silently followed a symlink, an attacker
    # (or a stray merge artifact) could point data_manifest.lock at a file
    # containing fabricated hashes that always match, silently defeating the
    # whole drift check. Same guard as _file_sha256, applied to the one file
    # where skipping it would matter most.
    _refuse_symlink_and_oversize(path, _MAX_HASHED_FILE_BYTES)
    with path.open() as f:
        return yaml.safe_load(f)


def _identity(entry: dict) -> Optional[str]:
    """The value that must stay stable for a source to be considered unchanged."""
    return entry.get("commit") if entry.get("type") == "git" else entry.get("sha256")


def sources_present(
    path: Path = MANIFEST_PATH, locked: Optional[dict] = None
) -> tuple[int, int]:
    """How many of the locked manifest's sources actually exist on this
    machine right now, out of the total locked.

    A caller can't tell "manifest_verified: true because every source was
    present and matched" apart from "manifest_verified: true because most
    sources were simply absent" from verify_manifest()'s return value alone
    — by design, an absent source is skipped, not counted as drift (see that
    function's docstring), which is the right call for a live console
    warning but loses meaning once flattened into a permanent boolean
    artifact (train_final.py's manifest_verified field). This gives the
    denominator that boolean needs context from.

    Deliberately cheap: reads each source's already-recorded ``path`` from
    the *locked* manifest and does a filesystem presence check only (no git
    subprocess spawns, no re-hashing) — computing this from a fresh
    compute_manifest() call instead would double the git-subprocess cost
    verify_manifest() already pays for the exact same information.

    Pass ``locked`` (an already-loaded manifest dict) when the caller has
    already called load_manifest() itself — e.g. train_final.py, which also
    calls verify_manifest() on the same path — so this doesn't re-read and
    re-parse data_manifest.lock a second time for the same call site.
    """
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
    """Compare the locked manifest against current source state.

    Returns a list of human-readable mismatch descriptions (empty if every
    locked source is either unchanged or simply not present locally yet — a
    source that hasn't been fetched isn't a drift, it's an unfetched
    prerequisite). Logs a WARNING per mismatch. Does not raise — see module
    docstring for why this is deliberately warn-only in v1.

    Pass ``locked`` (an already-loaded manifest dict) to skip re-reading
    data_manifest.lock when the caller has already loaded it itself — see
    sources_present()'s docstring for the call site this avoids doubling up.
    """
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
