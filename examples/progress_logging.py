"""Log run progress and automatic-termination residuals while a simulation runs.

Each completed chunk prints one line: completed and maximum timesteps, physical
time, wall time with an upper-bound estimate of the time remaining, and the
convergence residuals that decide when the run stops. The estimate assumes the run
reaches the end of the time grid; a converging run finishes sooner.
"""

import time

import numpy as np

import beamz as bz

um = bz.um


def _clock(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes}:{seconds:02d}"


def _sci(value: float | None) -> str:
    return "--" if value is None else f"{value:.1e}"


class ProgressLog:
    """Format one line per chunk; the rate excludes the first chunk's compile."""

    def __init__(self, policy: bz.AutoTermination):
        self.policy = policy
        self.start: tuple[float, int] | None = None

    def __call__(self, p: bz.RunProgress) -> None:
        now = time.perf_counter()
        if self.start is None:
            self.start = (now, p.step)
        start_time, start_step = self.start
        steps_done = p.step - start_step
        rate = (now - start_time) / steps_done if steps_done else None
        remaining = "--" if rate is None else _clock(rate * (p.total_steps - p.step))
        # Residuals only drive the decision once the source has switched off.
        source_off = (p.source_decay or 0.0) <= self.policy.source_decay
        print(
            f"step {p.step:>6}/{p.total_steps} ({100 * p.fraction:3.0f}%)  "
            f"t={p.time * 1e15:7.1f} fs  elapsed {_clock(p.elapsed)}  "
            f"max remaining {remaining}  "
            f"source {_sci(p.source_decay)}  "
            f"field decay {_sci(p.field_decay) if source_off else '--'}  "
            f"monitor change {_sci(p.monitor_change) if source_off else '--'}  "
            f"checks {p.consecutive_checks}/{self.policy.consecutive_checks}",
            flush=True,
        )


def waveguide_pulse() -> bz.Simulation:
    """Return a 2D silicon strip excited by a short pulse at 1550 nm."""
    wavelength = 1.55 * um
    n_clad, n_core = 1.44, 3.48
    dx, dt = bz.dxdt(wavelength, n_core, dims=2, points_per_wavelength=12)
    # Upper bound on the run; automatic termination stops earlier.
    run_time = 400 * wavelength / bz.LIGHT_SPEED

    design = bz.Design(12 * um, 6 * um, material=bz.Material(n_clad**2))
    design += bz.Rectangle(
        position=(0, 2.75 * um),
        width=12 * um,
        height=0.5 * um,
        material=bz.Material(n_core**2),
    )

    t = np.arange(0, run_time, dt)
    frequency = bz.LIGHT_SPEED / wavelength
    center, width = 30 * wavelength / bz.LIGHT_SPEED, 8 * wavelength / bz.LIGHT_SPEED
    signal = np.exp(-0.5 * ((t - center) / width) ** 2) * np.cos(
        2 * np.pi * frequency * t
    )
    signal[t > 2 * center] = 0.0

    return bz.Simulation(
        design=design,
        sources=[
            bz.GaussianSource(position=(2 * um, 3 * um), width=0.4 * um, signal=signal)
        ],
        monitors=[
            bz.FieldMonitor(
                center=(10 * um, 3 * um, 0.0),
                size=(0.0, 2 * um, 0.0),
                freqs=(frequency,),
                fields=("Ez",),
                name="output",
            )
        ],
        boundaries=[bz.PML(edges="all", thickness=wavelength)],
        time=t,
        resolution=dx,
    )


def main() -> None:
    sim = waveguide_pulse()
    policy = bz.AutoTermination(field_decay=1e-5, monitor_change=1e-3)

    results = sim.run(termination=policy, progress_callback=ProgressLog(policy))

    report = results.termination
    print(
        f"stopped: {report.reason} at step {report.steps}/{sim.num_steps} "
        f"({100 * report.steps / sim.num_steps:.0f}% of the time grid), "
        f"field decay {_sci(report.field_decay)}, "
        f"monitor change {_sci(report.monitor_change)}"
    )


if __name__ == "__main__":
    main()
