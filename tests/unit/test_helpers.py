"""Contracts for dependency-free helpers outside the FDTD update."""

from __future__ import annotations

import io
import logging

import numpy as np
import pytest

from beamz import (
    LIGHT_SPEED,
    calc_optimal_fdtd_params,
    create_plain_progress,
    display_status,
    dxdt,
    get_si_scale_and_label,
)
from beamz._helpers import (
    PlainProgress,
    _finish_inline_progress,
    _print_inline_progress,
    _print_inline_status,
    env_bool,
    positive_float,
    positive_integer,
)


class _Tty(io.StringIO):
    def isatty(self) -> bool:
        return True


class _NotTty(io.StringIO):
    def isatty(self) -> bool:
        return False


def test_positive_numbers_reject_non_finite_and_non_integral_values():
    assert positive_float("1.5", name="wavelength") == 1.5
    assert positive_float(np.float64(0.25), name="wavelength") == 0.25
    assert positive_integer(np.int64(4), name="points") == 4

    for value in (0, -1.0, float("nan"), float("inf"), "no", None):
        with pytest.raises(ValueError, match="wavelength must be positive and finite"):
            positive_float(value, name="wavelength")
    for value in (True, False, 1.2, "3", 0, -2, np.int64(0)):
        with pytest.raises(ValueError, match="points must be a positive integer"):
            positive_integer(value, name="points")


def test_env_bool_uses_the_default_only_when_the_variable_is_unset(monkeypatch):
    monkeypatch.delenv("BEAMZ_HELPER_FLAG", raising=False)
    assert env_bool("BEAMZ_HELPER_FLAG") is False
    assert env_bool("BEAMZ_HELPER_FLAG", default=True) is True

    for raw in ("1", " TRUE ", "yes", "on"):
        monkeypatch.setenv("BEAMZ_HELPER_FLAG", raw)
        assert env_bool("BEAMZ_HELPER_FLAG") is True
    for raw in ("0", "false", "no", "off", "", "  ", "maybe"):
        monkeypatch.setenv("BEAMZ_HELPER_FLAG", raw)
        assert env_bool("BEAMZ_HELPER_FLAG") is False


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1e-3, (1e3, "mm")),
        (1.0, (1e3, "mm")),
        (1e-6, (1e6, "um")),
        (9.999e-4, (1e6, "um")),
        (1e-9, (1e9, "nm")),
        (9.999e-7, (1e9, "nm")),
        (9.999e-10, (1e12, "pm")),
    ],
)
def test_si_length_scale_uses_millimetre_to_picometre_thresholds(value, expected):
    assert get_si_scale_and_label(value) == expected


def test_fdtd_step_estimate_matches_the_courant_formula_and_grid_warning(caplog):
    wavelength = 1.55e-6
    n_max = 3.5
    points = 12
    safety = 0.9
    resolution = wavelength / (n_max * points)
    dt_2d = safety * resolution / (LIGHT_SPEED * np.sqrt(2.0))
    small = 10 * resolution

    with caplog.at_level(logging.WARNING, logger="beamz._helpers"):
        assert calc_optimal_fdtd_params(
            wavelength,
            n_max,
            dims=2,
            safety_factor=safety,
            points_per_wavelength=points,
            width=small,
            height=small,
        ) == (resolution, dt_2d)
        assert calc_optimal_fdtd_params(
            wavelength,
            n_max,
            dims=3,
            safety_factor=safety,
            points_per_wavelength=points,
            width=small,
        ) == (resolution, safety * resolution / (LIGHT_SPEED * np.sqrt(3.0)))
    assert caplog.records == []

    huge = 3000 * resolution
    with caplog.at_level(logging.WARNING, logger="beamz._helpers"):
        estimated = dxdt(
            wavelength,
            n_max=n_max,
            dims=3,
            safety_factor=safety,
            points_per_wavelength=points,
            width=huge,
            height=huge,
            depth=huge,
        )
    assert estimated == calc_optimal_fdtd_params(
        wavelength,
        n_max,
        dims=3,
        safety_factor=safety,
        points_per_wavelength=points,
        width=huge,
        height=huge,
        depth=huge,
    )
    assert len(caplog.records) == 2
    assert {record.levelname for record in caplog.records} == {"WARNING"}
    assert all(
        "Large simulation grid detected" in record.message for record in caplog.records
    )


def test_display_status_maps_categories_onto_stdlib_log_levels(caplog):
    with caplog.at_level(logging.INFO, logger="beamz._helpers"):
        display_status("saved", "success")
        display_status("coarse", "warning")
        display_status("failed", "error")
        display_status("note", "other")

    assert [(record.levelname, record.message) for record in caplog.records] == [
        ("INFO", "saved"),
        ("WARNING", "coarse"),
        ("ERROR", "failed"),
        ("INFO", "note"),
    ]


def test_plain_progress_is_silent_unless_enabled_and_reports_task_completion():
    silent = io.StringIO()
    with PlainProgress(file=silent, enabled=False) as progress:
        task_id = progress.add_task("hidden", total=3)
        progress.update(task_id, advance=1)
    assert silent.getvalue() == ""

    plain = _NotTty()
    with PlainProgress(file=plain, inline=True, enabled=True) as progress:
        task_id = progress.add_task("Rasterizing.", total=4)
        progress.update(task_id, completed=2)
        open_task = progress.add_task("compile")
        progress.update(open_task, advance=1)
    assert plain.getvalue() == (
        "* Rasterizing.\n* compile\n* Rasterizing. done (2/4)\n* compile done\n"
    )

    rewound = io.StringIO()
    with PlainProgress(file=rewound, inline=False, enabled=True) as progress:
        task_id = progress.add_task("scan", total=4)
        progress.update(task_id, completed=2)
        progress.update(task_id, advance=-5)
    assert rewound.getvalue() == "* scan\n* scan\n* scan done (0/4)\n"

    failed = io.StringIO()
    with (
        pytest.raises(RuntimeError, match="boom"),
        PlainProgress(file=failed, inline=False, enabled=True) as progress,
    ):
        progress.add_task("compile")
        raise RuntimeError("boom")
    assert failed.getvalue() == "* compile\n* compile failed\n"


def test_inline_progress_rewrites_the_terminal_line_and_clamps_the_bar():
    terminal = _Tty()
    with PlainProgress(file=terminal, inline=True, enabled=True) as progress:
        task_id = progress.add_task("Running.", total=4)
        progress.update(task_id, advance=-3)
        progress.update(task_id, advance=2)
        progress.update(task_id, completed=10)
    text = terminal.getvalue()
    assert "\r* Running: [----------" in text
    assert "\r* Running: [##########----------] 50% (2/4 items)" in text
    assert text.endswith("\r* Running. done (4/4)\n")

    empty = _Tty()
    with PlainProgress(file=empty, enabled=True) as progress:
        progress.add_task("empty", total=0)
    assert "\r* empty: [--------------------] 0% (0/1 items)" in empty.getvalue()


def test_progress_helpers_fall_back_to_stdout(monkeypatch):
    captured = _Tty()
    monkeypatch.setattr("sys.stdout", captured)
    reporter = create_plain_progress(enabled=True)
    assert isinstance(reporter, PlainProgress)
    with reporter:
        task_id = reporter.add_task("boot", total=1)
        reporter.update(task_id, completed=1)
    _print_inline_status("Compiling simulation...")
    _print_inline_progress(1, 2)
    _finish_inline_progress()

    text = captured.getvalue()
    assert "\r* boot: [####################] 100% (1/1 items)" in text
    assert text.endswith(
        "\r* Compiling simulation...\r* Progress: [##########----------] 50% (1/2 steps)\n"
    )
