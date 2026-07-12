"""
Tests for src/data/manifest.py (data manifest).

Strategy: build tiny synthetic git repos and a synthetic calibration CSV
under tmp_path (never touch the real data/raw/mlperf mirrors or the real
data/data_manifest.lock), and monkeypatch the module's path constants to
point at them.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from src.data import manifest

_TEST_REMOTE_URL = "https://github.com/mlcommons/inference_results_v6.0.git"


def _init_git_repo(repo_dir: Path, remote_url: str) -> str:
    """Create a one-commit git repo and return its commit SHA."""
    repo_dir.mkdir(parents=True, exist_ok=True)
    run = lambda *args: subprocess.run(  # noqa: E731
        ["git", "-C", str(repo_dir), *args],
        check=True, capture_output=True, text=True,
    )
    run("init", "-q")
    run("-c", "user.email=test@example.com", "-c", "user.name=Test", "commit",
        "--allow-empty", "-q", "-m", "init")
    run("remote", "add", "origin", remote_url)
    return run("rev-parse", "HEAD").stdout.strip()


@pytest.fixture
def fixture_env(tmp_path, monkeypatch):
    """One synthetic MLPerf round repo + one calibration CSV, wired into the module."""
    mlperf_root = tmp_path / "mlperf"
    round_dir = mlperf_root / "v6.0"
    commit_sha = _init_git_repo(round_dir, _TEST_REMOTE_URL)

    calibration_csv = tmp_path / "calibration.csv"
    calibration_csv.write_text("gpu_name,throughput\nmi300x,123.4\n")

    lock_path = tmp_path / "data_manifest.lock"

    monkeypatch.setattr(manifest, "MLPERF_ROUNDS", ["v6.0"])
    monkeypatch.setattr(manifest, "MLPERF_ROOT", mlperf_root)
    monkeypatch.setattr(manifest, "CALIBRATION_CSV", calibration_csv)
    monkeypatch.setattr(manifest, "MANIFEST_PATH", lock_path)

    return {
        "round_dir": round_dir,
        "commit_sha": commit_sha,
        "calibration_csv": calibration_csv,
        "lock_path": lock_path,
    }


class TestFileSha256:
    def test_matches_hashlib(self, tmp_path):
        p = tmp_path / "f.txt"
        p.write_text("hello world")
        import hashlib
        expected = hashlib.sha256(b"hello world").hexdigest()
        assert manifest._file_sha256(p) == expected

    def test_missing_file_returns_none(self, tmp_path):
        assert manifest._file_sha256(tmp_path / "does_not_exist.txt") is None

    def test_refuses_symlinked_file(self, tmp_path):
        """A symlinked calibration CSV must not be silently hashed — its
        target could be attacker-controlled content masquerading as the
        real, tracked file."""
        real = tmp_path / "real.csv"
        real.write_text("real calibration content")
        link = tmp_path / "link.csv"
        link.symlink_to(real)
        with pytest.raises(ValueError, match="symlink"):
            manifest._file_sha256(link)

    def test_refuses_oversized_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(manifest, "_MAX_HASHED_FILE_BYTES", 10)
        p = tmp_path / "big.csv"
        p.write_text("this is more than ten bytes of content")
        with pytest.raises(ValueError, match="too large"):
            manifest._file_sha256(p)


class TestCorpusSha256:
    """corpus_sha256() hashes a built training corpus file for the
    model sidecar's corpus_sha256 field — same guards as _file_sha256, just
    against the larger _MAX_CORPUS_BYTES cap instead of _MAX_HASHED_FILE_BYTES."""

    def test_matches_hashlib(self, tmp_path):
        p = tmp_path / "corpus.parquet"
        p.write_bytes(b"fake parquet bytes")
        import hashlib
        expected = hashlib.sha256(b"fake parquet bytes").hexdigest()
        assert manifest.corpus_sha256(p) == expected

    def test_missing_corpus_returns_none(self, tmp_path):
        assert manifest.corpus_sha256(tmp_path / "does_not_exist.parquet") is None

    def test_refuses_symlinked_corpus(self, tmp_path):
        real = tmp_path / "real.parquet"
        real.write_bytes(b"real corpus content")
        link = tmp_path / "link.parquet"
        link.symlink_to(real)
        with pytest.raises(ValueError, match="symlink"):
            manifest.corpus_sha256(link)

    def test_uses_corpus_cap_not_hashed_file_cap(self, tmp_path, monkeypatch):
        """A corpus larger than _MAX_HASHED_FILE_BYTES (1 MB) must still hash
        fine as long as it's under _MAX_CORPUS_BYTES — proves corpus_sha256
        really uses the larger cap, not the lock-file/calibration-CSV one."""
        monkeypatch.setattr(manifest, "_MAX_HASHED_FILE_BYTES", 10)
        p = tmp_path / "corpus.parquet"
        p.write_bytes(b"this is more than ten bytes of fake corpus content")
        assert manifest.corpus_sha256(p) is not None

    def test_refuses_oversized_corpus(self, tmp_path, monkeypatch):
        monkeypatch.setattr(manifest, "_MAX_CORPUS_BYTES", 10)
        p = tmp_path / "corpus.parquet"
        p.write_bytes(b"this is more than ten bytes of fake corpus content")
        with pytest.raises(ValueError, match="too large"):
            manifest.corpus_sha256(p)


class TestRefuseSymlinkAndOversize:
    def test_missing_path_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            manifest._refuse_symlink_and_oversize(
                tmp_path / "nope", manifest._MAX_HASHED_FILE_BYTES
            )

    def test_regular_file_within_size_passes(self, tmp_path):
        p = tmp_path / "ok.txt"
        p.write_text("small")
        manifest._refuse_symlink_and_oversize(p, manifest._MAX_HASHED_FILE_BYTES)  # must not raise


class TestComputeManifest:
    def test_structure_and_present_flags(self, fixture_env):
        m = manifest.compute_manifest()
        assert set(m["sources"]) == {"mlperf_v6_0", "mi300x_calibration"}
        assert m["sources"]["mlperf_v6_0"]["present"] is True
        assert m["sources"]["mlperf_v6_0"]["commit"] == fixture_env["commit_sha"]
        assert m["sources"]["mlperf_v6_0"]["url"] == _TEST_REMOTE_URL
        assert m["sources"]["mi300x_calibration"]["present"] is True
        assert m["sources"]["mi300x_calibration"]["sha256"] is not None

    def test_commit_and_snapshot_date_from_combined_git_log_call(self, fixture_env):
        """commit + snapshot_date are derived from one `git log --format=%H\\x1f%cI`
        call (perf fix: was two separate subprocess spawns — rev-parse HEAD and
        log --format=%cI). Confirms the split still yields the *right* values,
        not just correctly-shaped ones — a regex-only date-format check would
        pass even if snapshot_date were a hardcoded, wrong-but-valid-looking
        date (caught by mutation-testing this exact test: hardcoding
        commit_date to "1970-01-01T..." kept a format-only version of this
        test green)."""
        m = manifest.compute_manifest()
        entry = m["sources"]["mlperf_v6_0"]
        assert entry["commit"] == fixture_env["commit_sha"]
        assert len(entry["commit"]) == 40  # full SHA-1 hex, not truncated by the split

        # Independently derive the expected date via a fresh subprocess call
        # (not manifest._run_git / _mlperf_source) so a bug in the module's
        # own combined-call parsing can't also poison the "expected" value.
        expected_iso = subprocess.run(
            ["git", "-C", str(fixture_env["round_dir"]), "log", "-1", "--format=%cI"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        assert entry["snapshot_date"] == expected_iso[:10]

    def test_malformed_git_log_output_yields_none_fields_not_a_crash(self, tmp_path, monkeypatch):
        """If `_run_git` ever returns output without the \\x1f separator
        (e.g. a git version that ignores the format string), the split must
        degrade to None/None rather than raising or silently truncating.

        Monkeypatches MLPERF_ROOT to an isolated tmp_path (matching every
        other test in this file) rather than relying on the real
        data/raw/mlperf checkout: without it, this test would raise
        ValueError instead of exercising the code under test on any machine
        where that directory happens to be a symlink.
        """
        monkeypatch.setattr(manifest, "MLPERF_ROOT", tmp_path / "mlperf")

        def fake_run_git(repo_dir, *args):
            return "no-separator-here" if args[0] == "log" else "https://example.com/repo.git"

        monkeypatch.setattr(manifest, "_run_git", fake_run_git)
        entry = manifest._mlperf_source("v6.0")
        assert entry["commit"] is None
        assert entry["snapshot_date"] is None

    def test_absent_source_flagged_not_present(self, tmp_path, monkeypatch):
        monkeypatch.setattr(manifest, "MLPERF_ROUNDS", ["v6.0"])
        monkeypatch.setattr(manifest, "MLPERF_ROOT", tmp_path / "nonexistent_root")
        monkeypatch.setattr(manifest, "CALIBRATION_CSV", tmp_path / "nonexistent.csv")
        m = manifest.compute_manifest()
        assert m["sources"]["mlperf_v6_0"]["present"] is False
        assert m["sources"]["mlperf_v6_0"]["commit"] is None

    def test_refuses_symlinked_mlperf_round_dir(self, tmp_path, monkeypatch):
        """A symlinked round directory would let `git -C <dir>` silently
        report an arbitrary repo's commit/remote as the pinned MLPerf mirror."""
        real_repo = tmp_path / "real_repo"
        _init_git_repo(real_repo, _TEST_REMOTE_URL)
        mlperf_root = tmp_path / "mlperf"
        mlperf_root.mkdir()
        (mlperf_root / "v6.0").symlink_to(real_repo)

        monkeypatch.setattr(manifest, "MLPERF_ROUNDS", ["v6.0"])
        monkeypatch.setattr(manifest, "MLPERF_ROOT", mlperf_root)
        monkeypatch.setattr(manifest, "CALIBRATION_CSV", tmp_path / "nonexistent.csv")

        with pytest.raises(ValueError, match="symlink"):
            manifest.compute_manifest()


class TestWriteAndLoadManifest:
    def test_roundtrip(self, fixture_env):
        written = manifest.write_manifest(fixture_env["lock_path"])
        loaded = manifest.load_manifest(fixture_env["lock_path"])
        assert loaded == written

    def test_load_missing_file_returns_none(self, tmp_path):
        assert manifest.load_manifest(tmp_path / "no_such.lock") is None

    def test_refuses_symlinked_lock_file(self, fixture_env, tmp_path):
        """The lock file is the trust anchor verify_manifest() checks
        everything against — silently following a symlink here would let an
        attacker point it at fabricated hashes that always match, defeating
        the drift check entirely."""
        manifest.write_manifest(fixture_env["lock_path"])
        evil_lock = tmp_path / "evil.lock"
        evil_lock.write_text("sources: {mi300x_calibration: {sha256: attacker_controlled}}")

        symlinked_path = tmp_path / "symlinked_data_manifest.lock"
        symlinked_path.symlink_to(evil_lock)

        with pytest.raises(ValueError, match="symlink"):
            manifest.load_manifest(symlinked_path)

    def test_refuses_oversized_lock_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(manifest, "_MAX_HASHED_FILE_BYTES", 10)
        p = tmp_path / "big.lock"
        p.write_text("sources: {this_is_more_than_ten_bytes: true}")
        with pytest.raises(ValueError, match="too large"):
            manifest.load_manifest(p)


class TestSourcesPresent:
    """sources_present() — disambiguates verify_manifest()'s "clean" result
    (every locked source present and matching) from "vacuously clean"
    (most/all locked sources simply absent, so nothing was actually
    compared) — a distinction lost once flattened into a single boolean."""

    def test_all_present(self, fixture_env):
        manifest.write_manifest(fixture_env["lock_path"])
        present, total = manifest.sources_present(fixture_env["lock_path"])
        assert (present, total) == (2, 2)

    def test_partially_absent(self, fixture_env, tmp_path):
        manifest.write_manifest(fixture_env["lock_path"])
        import shutil
        shutil.rmtree(fixture_env["round_dir"])
        present, total = manifest.sources_present(fixture_env["lock_path"])
        assert (present, total) == (1, 2)

    def test_missing_lock_file_returns_zero_zero(self, tmp_path):
        assert manifest.sources_present(tmp_path / "no_such.lock") == (0, 0)

    def test_symlinked_source_path_not_counted_present(self, fixture_env, tmp_path):
        """A symlinked source path must not count as genuinely present —
        matches _refuse_symlink_and_oversize's "don't trust a symlink"
        convention used everywhere else in this module."""
        manifest.write_manifest(fixture_env["lock_path"])
        real = tmp_path / "real_target"
        real.mkdir()
        shadow_round_dir = fixture_env["round_dir"]
        import shutil
        shutil.rmtree(shadow_round_dir)
        shadow_round_dir.symlink_to(real)
        present, total = manifest.sources_present(fixture_env["lock_path"])
        assert (present, total) == (1, 2)

    def test_git_source_path_that_is_a_file_not_counted_present(self, fixture_env):
        """A `type: git` source's path must be a *directory* to count as
        present, matching compute_manifest()'s own `repo_dir.is_dir()`
        semantics — a bare `.exists()` check would (wrongly) count a
        same-path regular file as a present MLPerf mirror."""
        manifest.write_manifest(fixture_env["lock_path"])
        import shutil
        shutil.rmtree(fixture_env["round_dir"])
        fixture_env["round_dir"].write_text("not actually a git repo directory")
        present, total = manifest.sources_present(fixture_env["lock_path"])
        assert (present, total) == (1, 2)

    def test_file_source_path_that_is_a_directory_not_counted_present(self, fixture_env):
        """A `type: file` source's path (the calibration CSV) must be a
        regular *file* to count as present, matching compute_manifest()'s
        own `CALIBRATION_CSV.is_file()` semantics."""
        manifest.write_manifest(fixture_env["lock_path"])
        fixture_env["calibration_csv"].unlink()
        fixture_env["calibration_csv"].mkdir()
        present, total = manifest.sources_present(fixture_env["lock_path"])
        assert (present, total) == (1, 2)

    def test_accepts_preloaded_locked_dict_without_reloading(self, fixture_env, monkeypatch):
        """train_final.py loads data_manifest.lock once and passes it to both
        verify_manifest() and sources_present() rather than each independently
        re-reading and re-parsing the same file — confirms passing `locked`
        both produces the identical result and genuinely skips load_manifest()."""
        manifest.write_manifest(fixture_env["lock_path"])
        preloaded = manifest.load_manifest(fixture_env["lock_path"])

        def _fail_if_called(*args, **kwargs):
            raise AssertionError("load_manifest() should not be called when `locked` is given")

        monkeypatch.setattr(manifest, "load_manifest", _fail_if_called)
        present, total = manifest.sources_present(fixture_env["lock_path"], locked=preloaded)
        assert (present, total) == (2, 2)


class TestVerifyManifest:
    def test_no_mismatch_when_unchanged(self, fixture_env):
        manifest.write_manifest(fixture_env["lock_path"])
        assert manifest.verify_manifest(fixture_env["lock_path"]) == []

    def test_accepts_preloaded_locked_dict_without_reloading(self, fixture_env, monkeypatch):
        """Same contract as sources_present()'s equivalent test — train_final.py
        loads data_manifest.lock once and passes it to both functions."""
        manifest.write_manifest(fixture_env["lock_path"])
        preloaded = manifest.load_manifest(fixture_env["lock_path"])

        def _fail_if_called(*args, **kwargs):
            raise AssertionError("load_manifest() should not be called when `locked` is given")

        monkeypatch.setattr(manifest, "load_manifest", _fail_if_called)
        assert manifest.verify_manifest(fixture_env["lock_path"], locked=preloaded) == []

    def test_detects_git_commit_drift(self, fixture_env, caplog):
        manifest.write_manifest(fixture_env["lock_path"])

        # Simulate the MLPerf mirror being updated to a new commit after locking.
        run = lambda *args: subprocess.run(  # noqa: E731
            ["git", "-C", str(fixture_env["round_dir"]), *args],
            check=True, capture_output=True, text=True,
        )
        run("-c", "user.email=test@example.com", "-c", "user.name=Test", "commit",
            "--allow-empty", "-q", "-m", "second commit")

        with caplog.at_level("WARNING"):
            mismatches = manifest.verify_manifest(fixture_env["lock_path"])
        assert len(mismatches) == 1
        assert "mlperf_v6_0" in mismatches[0]
        assert any("drifted" in rec.message for rec in caplog.records)

    def test_detects_calibration_file_content_drift(self, fixture_env, caplog):
        manifest.write_manifest(fixture_env["lock_path"])
        fixture_env["calibration_csv"].write_text("gpu_name,throughput\nmi300x,999.9\n")

        with caplog.at_level("WARNING"):
            mismatches = manifest.verify_manifest(fixture_env["lock_path"])
        assert len(mismatches) == 1
        assert "mi300x_calibration" in mismatches[0]

    def test_absent_source_is_not_a_mismatch(self, fixture_env):
        """A source that was locked but hasn't been re-fetched locally (e.g. a
        fresh clone before running fetch_mlperf.sh) must not be reported as
        drift — it's an unfetched prerequisite, not a silent corpus change."""
        manifest.write_manifest(fixture_env["lock_path"])

        import shutil
        shutil.rmtree(fixture_env["round_dir"])

        assert manifest.verify_manifest(fixture_env["lock_path"]) == []

    def test_missing_lock_file_returns_empty_and_warns(self, tmp_path, caplog):
        with caplog.at_level("WARNING"):
            mismatches = manifest.verify_manifest(tmp_path / "no_such.lock")
        assert mismatches == []
        assert any("No data manifest found" in rec.message for rec in caplog.records)

    def test_never_raises_even_on_mismatch(self, fixture_env):
        """Warn-only by design (deliberately not hard-refuse,
        see module docstring) — verify_manifest must never raise."""
        manifest.write_manifest(fixture_env["lock_path"])
        fixture_env["calibration_csv"].write_text("completely different content")
        manifest.verify_manifest(fixture_env["lock_path"])  # must not raise

    def test_still_raises_on_symlinked_lock_file(self, fixture_env, tmp_path):
        """Warn-only applies to hash *mismatches*, not to active tampering.
        A symlinked lock file is a different, more severe failure mode
        (see load_manifest's docstring) and must still raise — verify_manifest
        deliberately does not catch/silence it."""
        manifest.write_manifest(fixture_env["lock_path"])
        evil_lock = tmp_path / "evil.lock"
        evil_lock.write_text("sources: {mi300x_calibration: {sha256: attacker_controlled}}")
        symlinked_path = tmp_path / "symlinked_data_manifest.lock"
        symlinked_path.symlink_to(evil_lock)

        with pytest.raises(ValueError, match="symlink"):
            manifest.verify_manifest(symlinked_path)
