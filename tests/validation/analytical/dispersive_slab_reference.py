"""Broadband slab fixtures and independent thin-film optics for regression tests."""

import time

import numpy as np

import beamz as bz


def analytic_slab(epsilon, wavelength, thickness):
    n = np.sqrt(epsilon)
    r = (1 - n) / (1 + n)
    phase = np.exp(2j * np.pi * n * thickness / wavelength)
    reflection = r * (1 - phase**2) / (1 - r * r * phase**2)
    transmission = (1 - r * r) * phase / (1 - r * r * phase**2)
    return abs(reflection) ** 2, abs(transmission) ** 2


def run_slab(material, dx, wavelengths, thickness, courant=0.7, run_time=90e-15):
    # Narrow periodic cross section is sufficient for a normal plane wave.
    clearance = max(0, thickness / 2 - 0.2e-6)
    size = (4 * dx, 4 * dx, 2.4e-6 + 2 * clearance)
    f = bz.LIGHT_SPEED / wavelengths
    pulse = bz.GaussianPulse(
        float((f.min() + f.max()) / 2), float((f.max() - f.min()) * 0.9)
    )
    monitors = [
        bz.FluxMonitor(center=(0, 0, z), size=(size[0], size[1], 0), freqs=f, name=name)
        for name, z in [
            ("input", 0.45e-6 + clearance),
            ("output", -0.45e-6 - clearance),
        ]
    ]
    source = bz.PlaneWaveSource(
        center=(0, 0, 0.7e-6 + clearance),
        size=(size[0], size[1], 0),
        source_time=pulse,
        direction="-z",
        power=1.0,
    )
    design = bz.Design(background=bz.Material(1.0))
    if material is not None:
        design += bz.Box(
            center=(0, 0, 0), size=(size[0], size[1], thickness), material=material
        )
    sim = bz.Simulation(
        design=design,
        size=size,
        sources=[source],
        monitors=monitors,
        boundaries=[
            bz.Periodic(axes=("x", "y")),
            bz.PML(edges=("front", "back"), thickness=0.3e-6),
        ],
        grid_spec=bz.GridSpec.uniform(dx, courant=courant),
        run_time=run_time,
    )
    t = time.perf_counter()
    result = sim.run(backend="jax", progress=False)
    return {
        name: np.asarray(result[name].flux) for name in ("input", "output")
    }, time.perf_counter() - t
