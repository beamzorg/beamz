"""Progress callbacks report completed chunks and the convergence residuals."""

import numpy as np
import pytest

from beamz import (
    LIGHT_SPEED,
    PML,
    AutoTermination,
    Design,
    FieldMonitor,
    GaussianSource,
    Material,
    RunProgress,
    Simulation,
    calc_optimal_fdtd_params,
    um,
)

pytestmark = [pytest.mark.compiled, pytest.mark.component]


def _pulse_simulation(num_steps=420):
    wavelength = 1.0 * um
    resolution, dt = calc_optimal_fdtd_params(
        wavelength, 1.0, dims=2, points_per_wavelength=10
    )
    time = np.arange(num_steps, dtype=float) * dt
    step = np.arange(num_steps)
    frequency = LIGHT_SPEED / wavelength
    signal = np.exp(-0.5 * ((step - 24) / 7) ** 2) * np.cos(
        2.0 * np.pi * frequency * time
    )
    signal[65:] = 0.0
    return Simulation(
        design=Design(width=8 * um, height=8 * um, material=Material(permittivity=1.0)),
        sources=(
            GaussianSource(position=(2 * um, 4 * um), width=0.3 * um, signal=signal),
        ),
        monitors=(
            FieldMonitor(
                center=(6 * um, 4 * um, 0.0),
                size=(0.0, 4 * um, 0.0),
                freqs=(frequency,),
                fields=("Ez",),
                name="output",
            ),
        ),
        boundaries=(PML(thickness=wavelength),),
        time=time,
        resolution=resolution,
    )


def test_termination_callback_reports_each_check(capsys):
    simulation = _pulse_simulation()
    policy = AutoTermination(chunk_steps=30, monitor_names=("output",))
    events: list[RunProgress] = []

    results = simulation.run(
        termination=policy, performance=False, progress_callback=events.append
    )

    report = results.termination
    assert capsys.readouterr().out == ""
    assert [e.step for e in events] == list(
        range(policy.chunk_steps, report.steps + 1, policy.chunk_steps)
    )
    assert all(e.total_steps == simulation.num_steps for e in events)
    assert np.all(np.diff([e.elapsed for e in events]) >= 0)
    last = events[-1]
    assert last.step == report.steps
    assert last.field_decay == report.field_decay
    assert last.monitor_change == report.monitor_change
    assert last.source_decay == report.source_decay
    assert last.consecutive_checks == report.consecutive_checks
    assert last.time == pytest.approx(report.time)
    assert events[0].monitor_change is None


def test_fixed_run_callback_reports_completed_steps(capsys):
    simulation = _pulse_simulation(num_steps=203)
    events: list[RunProgress] = []

    results = simulation.run(performance=False, progress_callback=events.append)

    assert capsys.readouterr().out == ""
    steps = [e.step for e in events]
    assert steps == sorted(steps) and steps[-1] == simulation.num_steps
    assert all(e.total_steps == simulation.num_steps for e in events)
    assert all(e.field_decay is None for e in events)
    assert events[-1].fraction == 1.0
    reference = simulation.run(performance=False)
    np.testing.assert_allclose(
        results["output"].get_dft_component("Ez"),
        reference["output"].get_dft_component("Ez"),
    )


def test_advance_callback_counts_from_continuation_state():
    simulation = _pulse_simulation(num_steps=200)
    first = simulation.advance(num_steps=80, performance=False)
    events: list[RunProgress] = []

    simulation.advance(
        state=first.state,
        num_steps=60,
        performance=False,
        progress_callback=events.append,
    )

    assert events[-1].step == 140
    assert all(e.total_steps == 140 for e in events)
