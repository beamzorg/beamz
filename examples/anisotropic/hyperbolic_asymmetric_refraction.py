"""Asymmetric refraction by a tilted hyperbolic crystal (oriented tensor poles).

Calcite driven inside its ordinary Reststrahlen band (1470 cm^-1, lambda =
6.8 um) has Re(eps) < 0 across the ordinary plane and Re(eps) > 0 along the
optic axis: it is a natural hyperbolic medium. Tilting the optic axis 25
degrees above the slab surface, inside the plane of incidence, adds a
frequency-dependent off-diagonal eps_xy(omega) to the lab-frame tensor.

Both ingredients need tensor dispersion. The negative principal value must
come from a resonance (a static negative permittivity is unconditionally
unstable in FDTD), and the tilt needs pole strengths that are full tensors,
because per-axis poles can only describe grid-aligned optical axes.

The tilt shears the hyperbolic iso-frequency curve and breaks the kx -> -kx
mirror symmetry of the interface. Two mirrored 45-degree beams therefore
refract differently: one bends forward (positive refraction) and the other
bends backward (negative refraction), which no grid-aligned tensor can do.

This 2D TE run (E in the xy plane, Hz out of plane) launches both beams at
once and compares each beam's energy direction inside the slab with the
analytic group-velocity direction of the tilted tensor.

Outputs (next to this script):
    hyperbolic_asymmetric_refraction_tensor.png     lab-frame eps components
    hyperbolic_asymmetric_refraction_field.png      |Hz| with analytic energy rays
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import beamz as bz
from beamz.design.raster import RasterOptions

um = bz.um
C0 = bz.LIGHT_SPEED


def wavenumber_to_omega(wavenumber_cm: float) -> float:
    """Angular frequency (rad/s) of a spectroscopic wavenumber in 1/cm."""
    return 2.0 * np.pi * C0 * wavenumber_cm * 100.0


DRIVE_CM = 1470.17  # inside calcite's ordinary Reststrahlen band
FREQUENCY = C0 * DRIVE_CM * 100.0
WAVELENGTH = C0 / FREQUENCY  # 6.80 um
OMEGA = 2.0 * np.pi * FREQUENCY

# Optic axis 25 degrees above the slab surface (the slab normal is y).
OPTIC_AXIS_ELEVATION = np.radians(25.0)
THETA_INCIDENT = np.radians(45.0)

# Two-pole Lorentz model of calcite. Crystal frame: the optic (extraordinary)
# axis is y, the slab normal; x and z are ordinary.
CRYSTAL = bz.AnisotropicMedium(
    permittivity=(2.5898, 2.4691, 2.5898),
    electric_poles=[
        bz.TensorPole.lorentz(
            (0.5416, 0.0, 0.5416),
            resonance=wavenumber_to_omega(1410.0),
            damping=wavenumber_to_omega(10.0),
        ),
        bz.TensorPole.lorentz(
            (0.0, 0.3745, 0.0),
            resonance=wavenumber_to_omega(871.0),
            damping=wavenumber_to_omega(3.0),
        ),
    ],
)
# Rotate about z so the optic axis leaves the normal by 65 degrees, i.e. sits
# 25 degrees above the surface (towards +x) in the xy plane of incidence.
CALCITE = CRYSTAL.rotated((0.0, 0.0, -(np.pi / 2 - OPTIC_AXIS_ELEVATION)))

RESOLUTION = 0.17 * um
CELLS = (1000, 188)  # (x, y): 170 um x 32 um
DOMAIN = (CELLS[0] * RESOLUTION, CELLS[1] * RESOLUTION)
PML_THICKNESS = 12 * RESOLUTION
SLAB_Y = (11.0 * um, 20.0 * um)
SLAB_X = (PML_THICKNESS + 3.0 * um, DOMAIN[0] - PML_THICKNESS - 3.0 * um)
SOURCE_Y = DOMAIN[1] - PML_THICKNESS - 6 * RESOLUTION
BEAM_RADIUS = 17.0 * um
# Beam A (+45 deg, travelling +x) and beam B (-45 deg, travelling -x) hit the
# slab far enough apart that their incident, reflected, and refracted paths
# never overlap.
BEAMS = {"A": (31.1 * um, +THETA_INCIDENT), "B": (133.9 * um, -THETA_INCIDENT)}
# The negatively refracted wave rides the slow high-k branch of the hyperbola;
# it needs about twice the transit time of the positive beam to settle.
RUN_TIME = 1600e-15


def in_plane_permittivity(medium, frequency):
    """Lab-frame 2x2 (x, y) block of the permittivity tensor."""
    return medium.permittivity_tensor(frequency)[:2, :2]


def analytic_energy_angle(theta_incident: float) -> float:
    """TE energy-refraction angle in the crystal, in degrees from -y.

    With Hz out of plane the dispersion is q^T eps^-1 q = k0^2 for
    q = (ky, -kx). The tangential kx = k0 sin(theta) is conserved; of the two
    ky roots the physical one carries energy into the slab, and the energy
    direction is the gradient of the quadratic form (dispersive calcite has
    d(eps)/d(omega) > 0, so the group velocity is parallel to it).
    """
    inverse = np.linalg.inv(np.real(in_plane_permittivity(CALCITE, FREQUENCY)))
    kx = np.sin(theta_incident)
    a = inverse[0, 0]
    b = -2.0 * inverse[0, 1] * kx
    c = inverse[1, 1] * kx**2 - 1.0
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        raise ValueError("The TE wave is evanescent in the crystal at this angle.")
    for sign in (+1.0, -1.0):
        ky = (-b + sign * np.sqrt(discriminant)) / (2 * a)
        gx = inverse[1, 1] * kx - inverse[0, 1] * ky
        gy = inverse[0, 0] * ky - inverse[0, 1] * kx
        if gy < 0:
            return float(np.degrees(np.arctan2(gx, -gy)))
    raise ValueError("No root carries energy into the slab.")


def beam_sources(time, hz_shape):
    """Tilted CW Gaussian beams as two quadrature Hz line currents each.

    A line current J(x, t) = A(x) cos(omega t - kx x) radiates a beam with
    tangential wavenumber kx. Expanding the cosine gives two separable
    sources: A cos(kx x) cos(omega t) + A sin(kx x) sin(omega t).
    """
    row = int(round(SOURCE_Y / RESOLUTION))
    x = (np.arange(hz_shape[1]) + 0.5) * RESOLUTION
    ramp = 0.5 * (1.0 - np.cos(np.pi * np.clip(time / 150e-15, 0.0, 1.0)))
    sources = []
    for center, theta in BEAMS.values():
        kx = 2.0 * np.pi / WAVELENGTH * np.sin(theta)
        envelope = np.exp(-(((x - center) / BEAM_RADIUS) ** 2))
        for spatial, temporal in (
            (envelope * np.cos(kx * (x - center)), np.cos(OMEGA * time)),
            (envelope * np.sin(kx * (x - center)), np.sin(OMEGA * time)),
        ):
            sources.append(
                bz.CustomSource(
                    component="Hz",
                    timing="h",
                    index=(row, slice(None)),
                    coeff=spatial,
                    waveform=ramp * temporal,
                    target_shape=hz_shape,
                )
            )
    return sources


def build_simulation():
    design = bz.Design(width=DOMAIN[0], height=DOMAIN[1], material=bz.Material(1.0))
    # Stop the crystal short of the x absorbers. Hyperbolic media carry backward
    # waves (phase and energy opposed along x) that a PML amplifies rather than
    # absorbs, so the slab must terminate in vacuum.
    design += bz.Rectangle(
        position=(SLAB_X[0], SLAB_Y[0]),
        width=SLAB_X[1] - SLAB_X[0],
        height=SLAB_Y[1] - SLAB_Y[0],
        material=CALCITE,
    )
    dt = 0.5 * RESOLUTION / (C0 * np.sqrt(2.0))
    time = np.arange(0.0, RUN_TIME, dt)
    hz_shape = (CELLS[1], CELLS[0])  # TE Hz sits at cell centres, (y, x) storage
    monitor = bz.FieldMonitor(
        center=(DOMAIN[0] / 2, DOMAIN[1] / 2, 0.0),
        size=(DOMAIN[0], DOMAIN[1], 0.0),
        freqs=[FREQUENCY],
        fields=("Hz",),
        name="field",
    )
    return bz.Simulation(
        design=design,
        sources=beam_sources(time, hz_shape),
        monitors=[monitor],
        boundaries=[bz.PML(edges="all", thickness=PML_THICKNESS, formulation="cpml")],
        time=time,
        resolution=RESOLUTION,
        polarization="te",
        # A full permittivity tensor needs node-centred off-diagonal coefficients.
        raster_options=RasterOptions(smoothing="farjadpour_full"),
    )


def measured_energy_angle(intensity, x, y, entry_x):
    """Energy-corridor angle from the intensity ridge inside the slab, in degrees.

    Follows the ridge row by row from the entry point downwards inside a
    +-3 um window, weighting by the squared above-median intensity, then fits
    a line through the mid-slab rows: the first 1.5 um below the face carry a
    bright interface wave and the deepest 2 um have little signal left.
    """
    window = round(3.0 * um / RESOLUTION)
    rows = np.flatnonzero((y > SLAB_Y[0]) & (y < SLAB_Y[1]))[::-1]  # top to bottom
    previous = int(round(entry_x / RESOLUTION))
    depths, ridge = [], []
    for row in rows:
        lo, hi = max(previous - window, 0), min(previous + window, x.size)
        values = intensity[row, lo:hi]
        weights = np.clip(values - np.median(values), 0.0, None) ** 2
        if weights.sum() <= 0:
            continue
        center = float(np.average(x[lo:hi], weights=weights))
        previous = int(round(center / RESOLUTION))
        if SLAB_Y[0] + 2.0 * um <= y[row] <= SLAB_Y[1] - 1.5 * um:
            depths.append(y[row])
            ridge.append(center)
    slope = np.polyfit(depths, ridge, 1)[0]  # dx/dy, y decreasing into the slab
    return float(np.degrees(np.arctan(-slope)))


def plot_tensor(path):
    frequencies = C0 * np.linspace(1200.0, 1650.0, 600) * 100.0
    eps = CALCITE.permittivity_tensor(frequencies)
    wavenumber = frequencies / (C0 * 100.0)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for (i, j), label in {(0, 0): "xx", (1, 1): "yy", (0, 1): "xy"}.items():
        ax.plot(wavenumber, eps[:, i, j].real, label=f"Re eps_{label}")
    ax.axvline(DRIVE_CM, color="0.5", ls="--", lw=1)
    ax.axhline(0.0, color="0.7", lw=0.8)
    ax.set(xlabel="wavenumber (1/cm)", ylabel="lab-frame permittivity", ylim=(-25, 25))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    out = Path(__file__).with_suffix("")
    eps = in_plane_permittivity(CALCITE, FREQUENCY)
    print("lab-frame eps (x, y block) at the drive frequency:")
    print(np.array2string(eps, precision=3))
    principal = np.linalg.eigvalsh(np.real(eps))
    print(f"principal Re(eps): {principal.round(3)} -> hyperbolic")
    plot_tensor(out.with_name(out.name + "_tensor.png"))

    simulation = build_simulation()
    results = simulation.run(backend="jax")
    hz = np.asarray(results.monitor("field").dft_fields["Hz"]).reshape(CELLS[::-1])
    intensity = np.abs(hz) ** 2
    y = (np.arange(intensity.shape[0]) + 0.5) * RESOLUTION
    x = (np.arange(intensity.shape[1]) + 0.5) * RESOLUTION

    magnitude = np.abs(hz) / np.abs(hz).max()
    fig, ax = plt.subplots(figsize=(11, 3.4))
    extent = (0, DOMAIN[0] / um, 0, DOMAIN[1] / um)
    image = ax.imshow(
        magnitude, origin="lower", extent=extent, cmap="magma", aspect="equal"
    )
    for boundary in SLAB_Y:
        ax.axhline(boundary / um, color="w", lw=0.8)
    for (name, (center, theta)), color in zip(
        BEAMS.items(), ("lime", "cyan"), strict=True
    ):
        predicted = analytic_energy_angle(theta)
        hit = center + (SOURCE_Y - SLAB_Y[1]) * np.tan(theta)
        measured = measured_energy_angle(magnitude**2, x, y, hit)
        kind = "positive" if np.sign(predicted) == np.sign(theta) else "negative"
        print(
            f"beam {name}: incidence {np.degrees(theta):+.0f} deg -> energy "
            f"{measured:+.1f} deg measured, {predicted:+.1f} deg analytic ({kind} refraction)"
        )
        exit_x = hit + (SLAB_Y[1] - SLAB_Y[0]) * np.tan(np.radians(predicted))
        ax.plot(
            [center / um, hit / um],
            [SOURCE_Y / um, SLAB_Y[1] / um],
            color="w",
            lw=0.8,
            ls=":",
        )
        ax.plot(
            [hit / um, exit_x / um],
            [SLAB_Y[1] / um, SLAB_Y[0] / um],
            color=color,
            lw=1.4,
            ls="--",
            label=f"{np.degrees(theta):+.0f} deg in: {predicted:+.1f} deg analytic, "
            f"{measured:+.1f} deg measured ({kind})",
        )
    ax.legend(loc="lower left", fontsize=7, framealpha=0.85)
    fig.colorbar(image, ax=ax, label="|Hz| (normalized)", fraction=0.025)
    ax.set(
        xlabel="x (um)",
        ylabel="y (um)",
        title="Tilted hyperbolic calcite: mirrored beams refract asymmetrically",
    )
    fig.tight_layout()
    fig.savefig(out.with_name(out.name + "_field.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
