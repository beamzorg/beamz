"""One-way surface magnetoplasmons on a magnetized plasma (gyrotropic media).

A Drude plasma biased by a static magnetic field B along z has a
non-symmetric (gyrotropic) permittivity. For TE waves (E in the xy plane,
Hz out of plane) the interface between vacuum and the plasma carries surface
magnetoplasmons whose dispersion is non-reciprocal. Their large-k cutoffs are

    omega_sp(+-) = (+-omega_c + sqrt(omega_c^2 + 2 omega_p^2)) / 2,

and between them only ONE propagation direction exists. Driven inside that
window, and inside the bulk gap where the plasma is opaque, a point source on
the interface launches a surface wave running a single way. Flipping B
reverses it. No reciprocal (symmetric-tensor) medium can do this.

The plasma touches the domain edges, so the run uses ``bz.Absorber`` (graded
loss), which absorbs the backward waves of indefinite media that a PML would
amplify.

Output (next to this script):
    gyrotropic_oneway_surface_wave.png  |Hz| for +B and -B with direction ratios
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import beamz as bz

um = bz.um
C0 = bz.LIGHT_SPEED

WAVELENGTH = 1.5 * um
FREQUENCY = C0 / WAVELENGTH
OMEGA = 2.0 * np.pi * FREQUENCY
OMEGA_P = OMEGA / 0.75  # drive at 0.75 omega_p
OMEGA_C = 0.4 * OMEGA_P

RESOLUTION = WAVELENGTH / 30
CELLS = (600, 300)  # (x, y): 30 um x 15 um
DOMAIN = (CELLS[0] * RESOLUTION, CELLS[1] * RESOLUTION)
INTERFACE_Y = DOMAIN[1] / 2
SOURCE = (DOMAIN[0] / 2, INTERFACE_Y + RESOLUTION)
RUN_TIME = 70 * WAVELENGTH / C0


def surface_cutoffs():
    root = np.sqrt(OMEGA_C**2 + 2 * OMEGA_P**2)
    return (root - OMEGA_C) / 2, (root + OMEGA_C) / 2


def plasma(sign: float) -> bz.AnisotropicMedium:
    return bz.AnisotropicMedium(
        electric_poles=[
            bz.GyrotropicPole.magnetized_plasma(
                OMEGA_P, (0.0, 0.0, sign * OMEGA_C), damping=0.005 * OMEGA_P
            )
        ]
    )


def simulate(sign: float) -> np.ndarray:
    design = bz.Design(width=DOMAIN[0], height=DOMAIN[1], material=bz.Material(1.0))
    design += bz.Rectangle(
        position=(0.0, 0.0), width=DOMAIN[0], height=INTERFACE_Y, material=plasma(sign)
    )
    dt = 0.5 * RESOLUTION / (C0 * np.sqrt(2.0))
    time = np.arange(0.0, RUN_TIME, dt)
    ramp = 0.5 * (1 - np.cos(np.pi * np.clip(time / (10 * WAVELENGTH / C0), 0, 1)))
    simulation = bz.Simulation(
        design=design,
        sources=[
            bz.GaussianSource(
                position=SOURCE,
                width=3 * RESOLUTION,
                signal=ramp * np.sin(OMEGA * time),
            )
        ],
        monitors=[
            bz.FieldMonitor(
                center=(DOMAIN[0] / 2, DOMAIN[1] / 2, 0.0),
                size=(DOMAIN[0], DOMAIN[1], 0.0),
                freqs=[FREQUENCY],
                fields=("Hz",),
                name="field",
            )
        ],
        boundaries=[bz.Absorber(edges="all", thickness=2 * WAVELENGTH)],
        time=time,
        resolution=RESOLUTION,
        polarization="te",
    )
    hz = simulation.run(backend="jax").monitor("field").dft_fields["Hz"]
    return np.asarray(hz).reshape(CELLS[::-1])


def direction_ratio(hz: np.ndarray) -> float:
    """Right-going over left-going surface intensity along the interface."""
    x = (np.arange(CELLS[0]) + 0.5) * RESOLUTION
    rows = slice(
        int((INTERFACE_Y - 1 * um) / RESOLUTION),
        int((INTERFACE_Y + 1 * um) / RESOLUTION),
    )
    band = np.sum(np.abs(hz[rows]) ** 2, axis=0)
    near = 2 * WAVELENGTH
    far = DOMAIN[0] / 2 - 2.5 * WAVELENGTH  # stay clear of the absorbers
    right = band[(x > SOURCE[0] + near) & (x < SOURCE[0] + far)].sum()
    left = band[(x < SOURCE[0] - near) & (x > SOURCE[0] - far)].sum()
    return float(right / left)


def main():
    low, high = surface_cutoffs()
    eps = plasma(+1).permittivity_tensor(FREQUENCY)
    voigt = ((eps[0, 0] ** 2 + eps[0, 1] ** 2) / eps[0, 0]).real
    print(
        f"one-way window {low / OMEGA_P:.3f}-{high / OMEGA_P:.3f} omega_p, drive "
        f"{OMEGA / OMEGA_P:.3f} omega_p; bulk Voigt eps = {voigt:.3f} (opaque)"
    )
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True)
    extent = (0, DOMAIN[0] / um, 0, DOMAIN[1] / um)
    for ax, sign in zip(axes, (+1.0, -1.0), strict=True):
        hz = simulate(sign)
        ratio = direction_ratio(hz)
        print(
            f"B {'+z' if sign > 0 else '-z'}: right/left surface intensity = {ratio:.3g}"
        )
        image = np.abs(hz) / np.abs(hz).max()
        ax.imshow(
            image**0.5, origin="lower", extent=extent, cmap="magma", aspect="equal"
        )
        ax.axhline(INTERFACE_Y / um, color="w", lw=0.6)
        ax.plot(SOURCE[0] / um, SOURCE[1] / um, "c*", ms=8)
        ax.set(
            ylabel="y (um)",
            title=f"B along {'+z' if sign > 0 else '-z'}: right/left intensity {ratio:.3g}",
        )
    axes[-1].set_xlabel("x (um)")
    fig.suptitle("One-way surface magnetoplasmon (sqrt |Hz|); plasma below the line")
    fig.tight_layout()
    fig.savefig(Path(__file__).with_suffix(".png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
