"""Iso-frequency contours of tilted hyperbolic permittivity and permeability.

A point source inside a crystal excites every wavevector the medium supports
at the drive frequency, so the spatial Fourier transform of its CW field
traces the medium's iso-frequency contour. For a hyperbolic tensor that
contour is open, and tilting the crystal axes rotates the hyperbola.

The script runs the same tilted Drude tensor twice, once as a permittivity
and once as a permeability, which are electromagnetic duals:

* TE (Hz out of plane) in an electric hyperbolic crystal: q^T eps^-1 q = k0^2.
* TM (Ez out of plane) in a magnetic hyperbolic crystal: q^T mu^-1 q = k0^2.

Here q = (ky, -kx). Both simulated contours should land on the same analytic
hyperbola, which checks the tensor-pole updates of the E and H families
independently.

Output (next to this script):
    hyperbolic_isofrequency.png  CW fields and k-space contours with analytic curve
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
TILT = np.radians(30.0)

# Crystal frame: a Drude response along x makes eps_x = 2 - (wp / w)^2 = -2 at
# the drive frequency, while eps_y = 2: a hyperbolic in-plane tensor.
POLE = bz.TensorPole.drude((2.0 * OMEGA, 0.0, 0.0), damping=0.01 * OMEGA)
ELECTRIC = bz.AnisotropicMedium(permittivity=2.0, electric_poles=[POLE]).rotated(
    (0.0, 0.0, TILT)
)
MAGNETIC = bz.AnisotropicMedium(permeability=2.0, magnetic_poles=[POLE]).rotated(
    (0.0, 0.0, TILT)
)

RESOLUTION = WAVELENGTH / 20
CELLS = 320
DOMAIN = CELLS * RESOLUTION
PML_THICKNESS = 16 * RESOLUTION
# The crystal stops short of the absorber: a PML amplifies the backward waves
# that hyperbolic media carry.
CRYSTAL_MARGIN = PML_THICKNESS + 1.0 * um
RUN_TIME = 60 * WAVELENGTH / C0


def simulate(medium, polarization):
    design = bz.Design(width=DOMAIN, height=DOMAIN, material=bz.Material(1.0))
    design += bz.Rectangle(
        position=(CRYSTAL_MARGIN, CRYSTAL_MARGIN),
        width=DOMAIN - 2 * CRYSTAL_MARGIN,
        height=DOMAIN - 2 * CRYSTAL_MARGIN,
        material=medium,
    )
    dt = 0.5 * RESOLUTION / (C0 * np.sqrt(2.0))
    time = np.arange(0.0, RUN_TIME, dt)
    ramp = 0.5 * (1.0 - np.cos(np.pi * np.clip(time / (10 * WAVELENGTH / C0), 0, 1)))
    component = "Hz" if polarization == "te" else "Ez"
    simulation = bz.Simulation(
        design=design,
        sources=[
            bz.GaussianSource(
                position=(DOMAIN / 2, DOMAIN / 2),
                width=RESOLUTION,
                signal=ramp * np.sin(OMEGA * time),
            )
        ],
        monitors=[
            bz.FieldMonitor(
                center=(DOMAIN / 2, DOMAIN / 2, 0.0),
                size=(DOMAIN, DOMAIN, 0.0),
                freqs=[FREQUENCY],
                fields=(component,),
                name="field",
            )
        ],
        boundaries=[bz.PML(edges="all", thickness=PML_THICKNESS, formulation="cpml")],
        time=time,
        resolution=RESOLUTION,
        polarization=polarization,
    )
    field = simulation.run(backend="jax").monitor("field").dft_fields[component]
    field = np.asarray(field).reshape(-1)
    side = int(round(np.sqrt(field.size)))
    return field.reshape(side, side)


def k_space(field):
    """|FFT| of the field inside the crystal, axes in units of k0."""
    start = int(np.ceil(CRYSTAL_MARGIN / RESOLUTION)) + 4
    window = field[start:-start, start:-start]
    window = window * np.outer(np.hanning(window.shape[0]), np.hanning(window.shape[1]))
    size = 4 * window.shape[0]
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(window, s=(size, size))))
    k = np.fft.fftshift(np.fft.fftfreq(size, RESOLUTION)) * WAVELENGTH
    return k, spectrum / spectrum.max()


def analytic_contour(tensor, k):
    """Evaluate q^T T^-1 q on a k grid (contour level 1 is the dispersion)."""
    inverse = np.linalg.inv(np.real(tensor[:2, :2]))
    kx, ky = np.meshgrid(k, k)
    return inverse[0, 0] * ky**2 - 2 * inverse[0, 1] * kx * ky + inverse[1, 1] * kx**2


def main():
    tensor = ELECTRIC.permittivity_tensor(FREQUENCY)
    np.testing.assert_allclose(tensor, MAGNETIC.permeability_tensor(FREQUENCY))
    print("in-plane tensor at the drive frequency (eps for TE, mu for TM):")
    print(np.array2string(tensor[:2, :2], precision=3))

    cases = (
        ("TE, hyperbolic permittivity", ELECTRIC, "te", "Hz"),
        ("TM, hyperbolic permeability", MAGNETIC, "tm", "Ez"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    extent = (0, DOMAIN / um, 0, DOMAIN / um)
    for column, (title, medium, polarization, component) in enumerate(cases):
        field = simulate(medium, polarization)
        scale = np.percentile(np.abs(field), 99.5)
        axes[0, column].imshow(
            np.real(field),
            origin="lower",
            extent=extent,
            cmap="RdBu",
            vmin=-scale,
            vmax=scale,
        )
        axes[0, column].set(
            title=f"{title}\nRe {component} (CW)", xlabel="x (um)", ylabel="y (um)"
        )
        k, spectrum = k_space(field)
        axes[1, column].imshow(
            np.log10(spectrum + 1e-4),
            origin="lower",
            extent=(k[0], k[-1], k[0], k[-1]),
            vmin=-2.5,
            cmap="viridis",
        )
        kx, ky = np.meshgrid(k, k)
        axes[1, column].contour(
            kx,
            ky,
            analytic_contour(tensor, k),
            levels=[1.0],
            colors="r",
            linewidths=0.9,
        )
        axes[1, column].set(
            xlim=(-4, 4),
            ylim=(-4, 4),
            xlabel="kx / k0",
            ylabel="ky / k0",
            title="|FFT| inside crystal, red: analytic",
        )
    fig.tight_layout()
    fig.savefig(Path(__file__).with_suffix(".png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
