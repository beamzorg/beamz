"""Contract tests for the wheel and sdist content gate used in CI."""

from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from scripts.check_distributions import REQUIRED, main

_ROOT = "beamz-0.6.2"
_NATIVE = "beamz/design/raster/_native"
_METADATA = f"{_ROOT}.dist-info/METADATA"


def _sdist_members(*extra: str, omit: str | None = None) -> list[str]:
    names = [f"{_ROOT}/{name}" for name in sorted(REQUIRED["sdist"]) if name != omit]
    names.extend(extra)
    return names


def _wheel_members(
    extension: str = ".so",
    *extra: str,
    omit: frozenset[str] | None = None,
) -> list[str]:
    names = set(REQUIRED["wheel"])
    names.add(_METADATA)
    names.add(f"{_NATIVE}{extension}")
    names.update(extra)
    return sorted(names - (omit or frozenset()))


def _write_sdist(path: Path, names: list[str]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for name in names:
            info = tarfile.TarInfo(name)
            info.size = 0
            archive.addfile(info, io.BytesIO(b""))


def _write_wheel(path: Path, names: list[str]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name in names:
            archive.writestr(name, b"")


def _populate(dist: Path, *, extension: str = ".so") -> tuple[Path, Path]:
    sdist = dist / f"{_ROOT}.tar.gz"
    wheel = dist / f"{_ROOT}-py3-none-any.whl"
    _write_sdist(sdist, _sdist_members())
    _write_wheel(wheel, _wheel_members(extension))
    return sdist, wheel


def test_missing_archives_are_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    assert main(tmp_path) == 1

    error = capsys.readouterr().err
    assert f"Missing sdist in {tmp_path}" in error
    assert f"Missing wheel in {tmp_path}" in error


@pytest.mark.parametrize("extension", [".so", ".pyd", ".dylib"])
def test_complete_archives_pass(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], extension: str
):
    sdist, wheel = _populate(tmp_path, extension=extension)

    assert main(tmp_path) == 0

    output = capsys.readouterr().out
    assert f"Checked {sdist}:" in output
    assert f"Checked {wheel}:" in output


def test_sdist_alone_still_requires_a_wheel(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    sdist = tmp_path / f"{_ROOT}.tar.gz"
    _write_sdist(sdist, _sdist_members())

    assert main(tmp_path) == 1

    captured = capsys.readouterr()
    assert f"Checked {sdist}:" in captured.out
    assert f"Missing wheel in {tmp_path}" in captured.err
    assert "Missing sdist" not in captured.err


@pytest.mark.parametrize(
    "relative",
    [
        "beamz/__pycache__/probe.pyc",
        "beamz/.cache/probe",
        "beamz/.beamz_cache/probe",
        ".coverage",
        ".coverage.3.11",
        ".DS_Store",
        "beamz.egg-info/PKG-INFO",
        "htmlcov/index.html",
        "build/temp.py",
        "beamz/site/probe",
        "beamz/module.pyo",
    ],
)
def test_sdist_rejects_workspace_state(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], relative: str
):
    sdist = tmp_path / f"{_ROOT}.tar.gz"
    _write_sdist(sdist, _sdist_members(f"{_ROOT}/{relative}"))
    _write_wheel(tmp_path / "ok.whl", _wheel_members())

    assert main(tmp_path) == 1

    error = capsys.readouterr().err
    assert f"{sdist}: forbidden entries: {relative}" in error


@pytest.mark.parametrize(
    "member",
    [
        "/tmp/outside",
        f"{_ROOT}/../escape",
        f"{_ROOT}/beamz\\probe.py",
    ],
)
def test_sdist_rejects_unsafe_paths(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], member: str
):
    sdist = tmp_path / f"{_ROOT}.tar.gz"
    _write_sdist(sdist, _sdist_members(member))
    _write_wheel(tmp_path / "ok.whl", _wheel_members())

    assert main(tmp_path) == 1
    assert f"{sdist}: unsafe archive paths: {member}" in capsys.readouterr().err


def test_sdist_rejects_multiple_archive_roots(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    sdist = tmp_path / f"{_ROOT}.tar.gz"
    _write_sdist(
        sdist,
        [
            f"{_ROOT}/pyproject.toml",
            "other-root/pyproject.toml",
        ],
    )
    _write_wheel(tmp_path / "ok.whl", _wheel_members())

    assert main(tmp_path) == 1
    assert (
        f"{sdist}: sdist must have one archive root, found ['{_ROOT}', 'other-root']"
        in capsys.readouterr().err
    )


def test_sdist_reports_forbidden_and_missing_entries_together(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    sdist = tmp_path / f"{_ROOT}.tar.gz"
    _write_sdist(
        sdist,
        _sdist_members(f"{_ROOT}/.pytest_cache/cache", omit="pyproject.toml"),
    )
    _write_wheel(tmp_path / "ok.whl", _wheel_members())

    assert main(tmp_path) == 1

    error = capsys.readouterr().err
    assert f"{sdist}: forbidden entries: .pytest_cache/cache" in error
    assert f"{sdist}: missing entries: pyproject.toml" in error


def test_wheel_requires_metadata_and_native_extension(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    _write_sdist(tmp_path / f"{_ROOT}.tar.gz", _sdist_members())
    wheel = tmp_path / "incomplete.whl"
    _write_wheel(
        wheel,
        _wheel_members(omit=frozenset({_METADATA, f"{_NATIVE}.so"})),
    )

    assert main(tmp_path) == 1
    error = capsys.readouterr().err
    assert "*.dist-info/METADATA" in error
    assert f"{_NATIVE}.<platform-extension>" in error
    assert error.count(f"{wheel}: missing entries:") == 1


def test_wheel_rejects_native_extension_outside_the_raster_package(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    _write_sdist(tmp_path / f"{_ROOT}.tar.gz", _sdist_members())
    wheel = tmp_path / "misplaced.whl"
    _write_wheel(
        wheel,
        _wheel_members(
            ".so",
            f"vendor/{_NATIVE}.so",
            omit=frozenset({f"{_NATIVE}.so"}),
        ),
    )

    assert main(tmp_path) == 1
    assert f"{_NATIVE}.<platform-extension>" in capsys.readouterr().err


def test_wheel_rejects_cached_bytecode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    _write_sdist(tmp_path / f"{_ROOT}.tar.gz", _sdist_members())
    wheel = tmp_path / "cached.whl"
    _write_wheel(wheel, _wheel_members(".so", "beamz/__pycache__/api.pyc"))

    assert main(tmp_path) == 1
    assert (
        f"{wheel}: forbidden entries: beamz/__pycache__/api.pyc"
        in capsys.readouterr().err
    )
