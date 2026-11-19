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

WL = 1.55 * um
N_CLAD, N_CORE = 1.44, 3.48
DX, DT = bz.dxdt(WL, N_CORE, dims=2, points_per_wavelength=12)
RUN_TIME = 400 * WL / bz.LIGHT_SPEED  # upper bound; termination stops earlier

design = bz.Design(12 * um, 6 * um, material=bz.Material(N_CLAD**2))
design += bz.Rectangle(
    position=(0, 2.75 * um),
    width=12 * um,
    height=0.5 * um,
    material=bz.Material(N_CORE**2),
)

t = np.arange(0, RUN_TIME, DT)
frequency = bz.LIGHT_SPEED / WL
pulse_center, pulse_width = 30 * WL / bz.LIGHT_SPEED, 8 * WL / bz.LIGHT_SPEED
signal = np.exp(-0.5 * ((t - pulse_center) / pulse_width) ** 2) * np.cos(
    2 * np.pi * frequency * t
)
signal[t > 2 * pulse_center] = 0.0

sim = bz.Simulation(
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
    boundaries=[bz.PML(edges="all", thickness=WL)],
    time=t,
    resolution=DX,
)
policy = bz.AutoTermination(field_decay=1e-5, monitor_change=1e-3)


def _clock(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes}:{seconds:02d}"


def _sci(value: float | None) -> str:
    return "--" if value is None else f"{value:.1e}"


class ProgressLog:
    """Format one line per chunk; the clock starts after the first compile."""

    def __init__(self, policy: bz.AutoTermination):
        self.policy = policy
        self.start: tuple[float, int] | None = None

    def __call__(self, p: bz.RunProgress) -> None:
        now = time.perf_counter()
        if self.start is None:
            self.start = (now, 0)
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


results = sim.run(termination=policy, progress_callback=ProgressLog(policy))

report = results.termination
print(
    f"stopped: {report.reason} at step {report.steps}/{sim.num_steps} "
    f"({100 * report.steps / sim.num_steps:.0f}% of the time grid), "
    f"field decay {_sci(report.field_decay)}, "
    f"monitor change {_sci(report.monitor_change)}"
)
