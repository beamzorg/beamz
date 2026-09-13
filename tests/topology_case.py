"""Deterministic two-dimensional full-solver topology fixture."""

import numpy as np

from beamz import (
    LIGHT_SPEED,
    PML,
    Design,
    GaussianPulse,
    Material,
    ModeMonitor,
    ModeSource,
    ModeSpec,
    Rectangle,
    Simulation,
)
from beamz.optimization import ModePower, TopologyProblem, TopologySpec


def make_problem(
    dx=0.1e-6,
    run_time=160e-15,
    checkpoint_interval=32,
    target_mode=1,
    polarization="tm",
    reference=True,
):
    width, height = 6e-6, 4e-6
    wavelength = 1.55e-6
    frequency = LIGHT_SPEED / wavelength
    design = Design(width=width, height=height, material=Material(permittivity=1))
    design += Rectangle(
        position=(0, 1.6e-6),
        width=2e-6,
        height=0.8e-6,
        material=Material(permittivity=4),
    )
    design += Rectangle(
        position=(4e-6, 1.4e-6),
        width=2e-6,
        height=1.2e-6,
        material=Material(permittivity=4),
    )
    mode_spec = ModeSpec(num_modes=2, polarization=polarization)
    source = ModeSource(
        center=(1e-6, 2e-6, 0),
        size=(0, 2.8e-6, 1e-6),
        direction="+",
        mode_spec=ModeSpec(num_modes=1, polarization=polarization),
        source_time=GaussianPulse(freq0=frequency, fwidth=0.15 * frequency, offset=1.5),
    )
    monitor = ModeMonitor(
        center=(5e-6, 2e-6, 0),
        size=(0, 2.8e-6, 1e-6),
        freqs=[frequency],
        mode_spec=mode_spec,
        name="output",
    )
    incoming = monitor.updated_copy(
        center=(1.6e-6, 2e-6, 0),
        name="input",
        mode_spec=ModeSpec(num_modes=1, polarization=polarization),
    )
    sim = Simulation(
        polarization=polarization,
        design=design,
        resolution=dx,
        run_time=run_time,
        sources=[source],
        monitors=[incoming, monitor],
        boundaries=[PML(thickness=0.6e-6, formulation="cpml")],
    )
    grid = sim.compile(backend="jax").grid
    y, x = np.indices(grid.material_grid.shape)
    mask = (
        (x * dx >= 2e-6 - 1e-15)
        & (x * dx < 4e-6 - 1e-15)
        & (y * dx >= 1.2e-6 - 1e-15)
        & (y * dx < 2.8e-6 - 1e-15)
    )
    topo = TopologySpec(
        design=sim.design,
        region_mask=mask,
        resolution=dx,
        filter_radius=0.2e-6,
        eps_min=1,
        eps_max=4,
        learning_rate=0.04,
        beta_schedule=(1, 8),
    )
    return TopologyProblem(
        sim,
        topo,
        ModePower(
            "output",
            mode_index=target_mode,
            reference_monitor="input" if reference else None,
        ),
        checkpoint_interval=checkpoint_interval,
    )


def make_wdm_problem(dx=100e-9, run_time=220e-15, polarization="tm"):
    low, high = Material(permittivity=1), Material(permittivity=4)
    design = Design(width=8e-6, height=5e-6, material=low)
    design += Rectangle(
        position=(0, 2.25e-6), width=2.5e-6, height=0.5e-6, material=high
    )
    for center in [1.6e-6, 3.4e-6]:
        design += Rectangle(
            position=(5.5e-6, center - 0.25e-6),
            width=2.5e-6,
            height=0.5e-6,
            material=high,
        )
    wavelengths = np.array([1.28, 1.30, 1.32, 1.53, 1.55, 1.57]) * 1e-6
    freqs = LIGHT_SPEED / wavelengths
    freq0 = (freqs.max() + freqs.min()) / 2
    source = ModeSource(
        center=(1e-6, 2.5e-6, 0),
        size=(0, 1.6e-6, 1e-6),
        direction="+",
        mode_spec=ModeSpec(num_modes=1, polarization=polarization, num_freqs=5),
        source_time=GaussianPulse(freq0=freq0, fwidth=0.3 * freq0, offset=2),
    )
    incoming = ModeMonitor(
        center=(1.7e-6, 2.5e-6, 0),
        size=(0, 1.6e-6, 1e-6),
        freqs=freqs,
        mode_spec=ModeSpec(num_modes=1, polarization=polarization),
        name="input",
    )
    monitors = [incoming] + [
        incoming.updated_copy(center=(6.6e-6, center, 0), name=name)
        for center, name in [(1.6e-6, "lower"), (3.4e-6, "upper")]
    ]
    sim = Simulation(
        design=design,
        resolution=dx,
        run_time=run_time,
        sources=[source],
        monitors=monitors,
        polarization=polarization,
        boundaries=[PML(thickness=0.6e-6, formulation="cpml")],
    )
    grid = sim.compile(backend="jax").grid
    y, x = np.indices(grid.material_grid.shape)
    mask = (
        (x * dx >= 2.5e-6 - 1e-15)
        & (x * dx < 5.5e-6 - 1e-15)
        & (y * dx >= 1e-6 - 1e-15)
        & (y * dx < 4e-6 - 1e-15)
    )
    topo = TopologySpec(
        design=sim.design,
        region_mask=mask,
        resolution=dx,
        eps_min=1,
        eps_max=4,
        filter_radius=0.15e-6,
        learning_rate=0.035,
        beta_schedule=(1, 16),
    )

    def band(port, which):
        return ModePower(port, reference_monitor="input", frequencies=freqs[which])

    desired = 0.5 * (band("upper", slice(0, 3)) + band("lower", slice(3, 6)))
    leakage = 0.5 * (band("lower", slice(0, 3)) + band("upper", slice(3, 6)))
    reflection = ModePower("input", direction="-", reference_monitor="input")
    p = TopologyProblem(sim, topo, desired - 0.4 * leakage - 0.1 * reflection)
    return p
