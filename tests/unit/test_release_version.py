from __future__ import annotations

from pathlib import Path

import pytest

import release_version


def _write_release_fixture(root: Path) -> None:
    (root / "beamz").mkdir()
    (root / "pyproject.toml").write_text(
        '[project]\nname = "beamz"\nversion = "0.4.3"\n\n'
        '[tool.ruff]\ntarget-version = "py310"\n'
    )
    (root / "beamz" / "__init__.py").write_text('__version__ = "0.4.3"\n')
    (root / "CITATION.cff").write_text(
        'cff-version: 1.2.0\ntitle: "BeamZ"\nversion: "0.4.3"\n'
    )
    (root / "uv.lock").write_text("version = 1\n")
    (root / "Cargo.toml").write_text(
        '[workspace]\nmembers = []\n\n[workspace.package]\nversion = "0.4.3"\n'
    )
    (root / "Cargo.lock").write_text(
        """version = 4

[[package]]
name = "earcutr"
version = "0.4.3"

[[package]]
name = "fdtd-raster-core"
version = "0.4.3"

[[package]]
name = "fdtd-raster-py"
version = "0.4.3"
"""
    )


def test_update_version_keeps_python_and_native_engine_versions_in_sync(
    tmp_path, monkeypatch
):
    _write_release_fixture(tmp_path)
    monkeypatch.chdir(tmp_path)

    assert release_version.update_version("0.5.0")

    pyproject = (tmp_path / "pyproject.toml").read_text()
    assert 'version = "0.5.0"' in pyproject
    assert 'target-version = "py310"' in pyproject
    assert '__version__ = "0.5.0"' in (tmp_path / "beamz" / "__init__.py").read_text()
    citation = (tmp_path / "CITATION.cff").read_text()
    assert 'version: "0.5.0"' in citation
    assert "cff-version: 1.2.0" in citation
    assert 'title: "BeamZ"' in citation
    assert (
        '[workspace.package]\nversion = "0.5.0"'
        in (tmp_path / "Cargo.toml").read_text()
    )
    lockfile = (tmp_path / "Cargo.lock").read_text()
    assert 'name = "fdtd-raster-core"\nversion = "0.5.0"' in lockfile
    assert 'name = "fdtd-raster-py"\nversion = "0.5.0"' in lockfile
    assert 'name = "earcutr"\nversion = "0.4.3"' in lockfile
    assert not release_version.update_version("0.5.0")


def test_update_version_rejects_a_lockfile_missing_a_workspace_package(
    tmp_path, monkeypatch
):
    _write_release_fixture(tmp_path)
    lockfile = tmp_path / "Cargo.lock"
    lockfile.write_text(
        lockfile.read_text().replace(
            '[[package]]\nname = "fdtd-raster-py"\nversion = "0.4.3"\n', ""
        )
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(RuntimeError, match="fdtd-raster-py"):
        release_version.update_version("0.5.0")


def test_verify_version_sync_rejects_python_and_rust_version_mismatch(
    tmp_path, monkeypatch
):
    _write_release_fixture(tmp_path)
    monkeypatch.chdir(tmp_path)

    assert release_version.verify_version_sync("0.4.3") == "0.4.3"
    (tmp_path / "beamz" / "__init__.py").write_text('__version__ = "0.4.2"\n')

    with pytest.raises(RuntimeError, match="beamz/__init__.py=0.4.2"):
        release_version.verify_version_sync("0.4.3")


def test_repository_python_and_rust_version_metadata_stay_in_sync():
    assert release_version.verify_version_sync()


def test_verify_version_sync_rejects_stale_citation(tmp_path, monkeypatch):
    _write_release_fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "CITATION.cff").write_text('version: "0.4.2"\n')

    with pytest.raises(RuntimeError, match="CITATION.cff=0.4.2"):
        release_version.verify_version_sync("0.4.3")


@pytest.mark.parametrize("citation", [None, "cff-version: 1.2.0\n"])
def test_update_version_requires_citation_before_modifying_files(
    tmp_path, monkeypatch, citation
):
    _write_release_fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    if citation is None:
        (tmp_path / "CITATION.cff").unlink()
    else:
        (tmp_path / "CITATION.cff").write_text(citation)
    original = (tmp_path / "pyproject.toml").read_text()

    with pytest.raises((FileNotFoundError, RuntimeError), match="citation"):
        release_version.update_version("0.5.0")

    assert (tmp_path / "pyproject.toml").read_text() == original


def test_commit_version_changes_stages_python_and_rust_metadata(tmp_path, monkeypatch):
    _write_release_fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    calls = []

    def record(command, *, check):
        calls.append((command, check))

    monkeypatch.setattr(release_version.subprocess, "run", record)

    release_version.commit_version_changes("0.5.0")

    staged = [command[-1] for command, _check in calls if command[:2] == ["git", "add"]]
    assert staged == [
        "pyproject.toml",
        "uv.lock",
        "beamz/__init__.py",
        "CITATION.cff",
        "Cargo.toml",
        "Cargo.lock",
    ]
    assert calls[-1] == (["git", "commit", "-m", "Bump version to 0.5.0"], True)
