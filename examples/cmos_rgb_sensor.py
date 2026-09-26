"""Broadband RGGB reference sensor, shared by the notebook and local runner."""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

import beamz as bz
from beamz.design.discretization import MaterialGrid

um = bz.um
DATA = Path(__file__).resolve().parent / "data/cmos_rgb"


@lru_cache(maxsize=1)
def load_materials():
    raw = json.loads((DATA / "materials.json").read_text())
    filters = json.loads((DATA / "filters.json").read_text())
    return {
        "air": bz.Material(1.0),
        "silica": bz.PoleResidue.from_spec(raw["SiO2_Palik_LowLoss"]),
        "sin": bz.PoleResidue.from_spec(raw["SiN_Horiba"]),
        "silicon": bz.PoleResidue.from_spec(raw["aSi_Horiba"]),
        "metal": bz.PoleResidue.from_spec(raw["Al_Rakic1995"]),
        **{name: bz.PoleResidue.from_spec(spec) for name, spec in filters.items()},
    }


Lx = Ly = 3.0 * um
d_absorber_space = 1.5 * um

pixel_width = Lx / 2
lens_radius = 1.0 * um
lens_height = (1 - np.cos(np.arcsin(pixel_width / (2 * lens_radius)))) * lens_radius

d_lens_to_filter = 0.2 * um
t_filter = 1.0 * um
d_filter_to_shield = 0.5 * um
t_shield = 0.10 * um
shield_hole_radius = 0.95 * pixel_width
d_shield_to_connect = 0.5 * um
t_connect = 0.10 * um
connect_width = 0.15 * um
connect_gap = 0.2 * um
connect_distance = 1.8 * pixel_width
d_connect_to_ar = 0.5 * um
t_silicon = 2.0 * um

lam_blue = 0.45 * um
lam_green = 0.55 * um
lam_red = 0.65 * um
n_sin = float(
    np.sqrt(load_materials()["sin"].eps_model(bz.LIGHT_SPEED / lam_green)).real
)
t_ar = lam_green / (4 * n_sin)

Lz = (
    d_absorber_space
    + lens_height
    + d_lens_to_filter
    + t_filter
    + d_filter_to_shield
    + t_shield
    + d_shield_to_connect
    + 2 * t_connect
    + connect_gap
    + d_connect_to_ar
    + t_ar
    + t_silicon
)

pixel_signs = {
    "blue": (+1, +1),
    "red": (-1, -1),
    "green1": (-1, +1),
    "green2": (+1, -1),
}


def box_from_bounds(lower, upper, material):
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    return bz.Box(
        center=tuple(0.5 * (lower + upper)),
        size=tuple(upper - lower),
        material=material,
    )


def make_sensor_design():
    mat = load_materials()
    # The reference geometry is centered at zero; Simulation supplies its size.
    design = bz.Design(background=mat["air"])
    z_bottom = -Lz / 2
    z_silicon_top = z_bottom + t_silicon
    z_silica_top = Lz / 2 - d_absorber_space - lens_height
    z_filter_top = z_silica_top - d_lens_to_filter
    z_filter_bottom = z_filter_top - t_filter
    z_shield = z_filter_bottom - d_filter_to_shield - t_shield / 2
    z_connect0 = z_shield - t_shield / 2 - d_shield_to_connect - t_connect / 2
    z_connect1 = z_connect0 - connect_gap - t_connect
    z_ar = z_connect1 - t_connect / 2 - d_connect_to_ar - t_ar / 2

    # Full spheres come first; later layers clip them into spherical caps.
    lens_center_z = Lz / 2 - d_absorber_space - lens_radius
    for sign_x, sign_y in pixel_signs.values():
        design += bz.Sphere(
            position=(
                sign_x * pixel_width / 2,
                sign_y * pixel_width / 2,
                lens_center_z,
            ),
            radius=lens_radius,
            material=mat["silica"],
        )

    # The silica host clips each sphere at the cap base and fills the stack.
    design += box_from_bounds(
        (-Lx / 2, -Ly / 2, z_silicon_top),
        (+Lx / 2, +Ly / 2, z_silica_top),
        mat["silica"],
    )

    # Silicon photodetector and quarter-wave antireflection layer.
    design += box_from_bounds(
        (-Lx / 2, -Ly / 2, z_bottom),
        (+Lx / 2, +Ly / 2, z_silicon_top),
        mat["silicon"],
    )
    design += bz.Box(center=(0, 0, z_ar), size=(Lx, Ly, t_ar), material=mat["sin"])

    # Two interconnect planes and their central ground bars.
    for center_x in (-connect_distance / 2, +connect_distance / 2):
        design += bz.Box(
            center=(center_x, 0, z_connect1),
            size=(connect_width, Ly, t_connect),
            material=mat["metal"],
        )
    for center_y in (-connect_distance / 2, +connect_distance / 2):
        design += bz.Box(
            center=(0, center_y, z_connect0),
            size=(Lx, connect_width, t_connect),
            material=mat["metal"],
        )
    design += bz.Box(
        center=(0, 0, z_connect1),
        size=(2 * connect_width, Ly, t_connect),
        material=mat["metal"],
    )
    design += bz.Box(
        center=(0, 0, z_connect0),
        size=(Lx, 2 * connect_width, t_connect),
        material=mat["metal"],
    )

    # Metal shield, then its circular silica opening.
    design += bz.Box(
        center=(0, 0, z_shield), size=(Lx, Ly, t_shield), material=mat["metal"]
    )
    design += bz.Circle(
        position=(0, 0),
        z=z_shield - t_shield / 2,
        radius=shield_hole_radius,
        depth=t_shield,
        points=96,
        material=mat["silica"],
    )

    # RGGB filter quadrants.
    for pixel, (sign_x, sign_y) in pixel_signs.items():
        channel = "green" if pixel.startswith("green") else pixel
        design += bz.Box(
            center=(
                sign_x * pixel_width / 2,
                sign_y * pixel_width / 2,
                0.5 * (z_filter_bottom + z_filter_top),
            ),
            size=(pixel_width, pixel_width, t_filter),
            material=mat[channel],
        )
    return design, z_silicon_top, z_ar


def make_sim(
    *,
    dx=25e-9,
    run_time=300e-15,
    reference=False,
    fields=True,
    courant=0.7,
    grid_spec=None,
    absorber_thickness=0.4e-6,
    num_freqs=None,
):
    design, z_silicon_top, _ = make_sensor_design()
    if reference and grid_spec is None:
        design = bz.Design(background=bz.Material(1.0))
    frequencies = (
        bz.LIGHT_SPEED / np.linspace(400e-9, 700e-9, 31)
        if num_freqs is None
        else np.linspace(bz.LIGHT_SPEED / 700e-9, bz.LIGHT_SPEED / 400e-9, num_freqs)
    )
    center_frequency = 0.5 * (frequencies.min() + frequencies.max())
    pulse = bz.GaussianPulse(freq0=center_frequency, fwidth=center_frequency / 3)
    source = bz.PlaneWaveSource(
        center=(0, 0, (Lz - d_absorber_space) / 2),
        size=(Lx, Ly, 0),
        source_time=pulse,
        direction="-z",
        power=1.0,
    )
    monitors = [
        bz.FluxMonitor(
            center=(0, 0, Lz / 2 - d_absorber_space - 0.05 * um),
            size=(Lx, Ly, 0),
            freqs=frequencies,
            name="incident_plane",
        )
    ]
    for pixel, (sign_x, sign_y) in pixel_signs.items():
        monitors.append(
            bz.FluxMonitor(
                center=(
                    sign_x * pixel_width / 2,
                    sign_y * pixel_width / 2,
                    z_silicon_top,
                ),
                size=(0.7 * pixel_width, 0.7 * pixel_width, 0),
                freqs=frequencies,
                name=f"flux_si_{pixel}",
            )
        )
    if reference:
        monitors.append(
            bz.FieldMonitor(
                center=(0, 0, Lz / 2 - d_absorber_space - 0.05 * um),
                size=(Lx, Ly, 0),
                freqs=bz.LIGHT_SPEED / np.array([650e-9, 550e-9, 450e-9]),
                fields=("Ex", "Ey", "Ez"),
                name="incident_field",
            )
        )
    if fields and not reference:
        ff = bz.LIGHT_SPEED / np.array([650e-9, 550e-9, 450e-9])
        monitors += [
            bz.FieldMonitor(
                center=(0, 0, z_silicon_top - 0.001 * um),
                size=(Lx, Ly, 0),
                freqs=ff,
                fields=("Ex", "Ey", "Ez"),
                name="field_xy_silicon",
            ),
            bz.FieldMonitor(
                center=(pixel_width / 2, 0, 0),
                size=(0, Ly, Lz),
                freqs=ff,
                fields=("Ex", "Ey", "Ez"),
                name="field_yz_BG",
            ),
            bz.FieldMonitor(
                center=(-pixel_width / 2, 0, 0),
                size=(0, Ly, Lz),
                freqs=ff,
                fields=("Ex", "Ey", "Ez"),
                name="field_yz_GR",
            ),
        ]
    simulation = bz.Simulation(
        design=design,
        size=(Lx, Ly, Lz),
        sources=[source],
        monitors=monitors,
        boundaries=[
            bz.Periodic(axes=("x", "y")),
            bz.Absorber(edges=("front", "back"), thickness=absorber_thickness),
        ],
        grid_spec=grid_spec or bz.GridSpec.uniform(dx, courant=courant),
        run_time=run_time,
    )
    if reference and grid_spec is not None:
        # Auto meshing an empty design would change both the mesh and timestep.
        # Reuse the sensor's realized grid with homogeneous air coefficients.
        shape = tuple(reversed(simulation.grid.shape))
        air = MaterialGrid(
            permittivity=np.ones(shape),
            conductivity=np.zeros(shape),
            permeability=np.ones(shape),
            resolution=simulation.grid.minimum_spacing,
            shape=shape,
            grid=simulation.grid,
        )
        return bz.Simulation(
            material_grid=air,
            sources=simulation.sources,
            monitors=simulation.monitors,
            boundaries=simulation.boundaries,
            time=simulation.time,
        )
    return simulation


def sensor_grid(*, lateral_nm=20, metal_nm=5, bulk_nm=30, courant=0.7):
    """Graded z mesh with explicit interface refinement and lateral resolution.

    Overrides cap memory on a 24 GB GPU. The lateral spacing remains an explicit
    convergence parameter; resolving metal depth alone does not converge edges.
    """
    snapshot = json.loads((DATA / "tidy3d_reference_simulation.json").read_text())
    overrides = [
        bz.MeshOverride(
            center=(0, 0, 0),
            size=(Lx, Ly, Lz),
            dl=(lateral_nm * 1e-9, lateral_nm * 1e-9, bulk_nm * 1e-9),
            enforced=True,
        )
    ]
    for structure in snapshot["structures"]:
        name = structure["name"]
        if (
            name != "metal shield"
            and not name.startswith("interconnect")
            and name != "anti-reflection"
        ):
            continue
        geometry = structure["geometry"]
        z = geometry["center"][2] * um
        thickness = geometry["size"][2] * um
        overrides.append(
            bz.MeshOverride(
                center=(0, 0, z),
                size=(Lx, Ly, thickness + 4 * metal_nm * 1e-9),
                dl=(None, None, metal_nm * 1e-9),
                enforced=True,
            )
        )
    z_si = -Lz / 2 + t_silicon
    overrides.append(
        bz.MeshOverride(
            center=(0, 0, z_si - 0.25 * um),
            size=(Lx, Ly, 0.5 * um),
            dl=(None, None, metal_nm * 1e-9),
            enforced=True,
        )
    )
    return bz.GridSpec.auto(
        wavelength=550e-9,
        courant=courant,
        max_scale=1.3,
        overrides=tuple(overrides),
        max_total_cells=30_000_000,
    )


def plot_rgb_fields(result, reference):
    """Total colocated |E|² / |E_inc|², with a shared scale per monitor plane."""
    import matplotlib.pyplot as plt

    incident = reference["incident_field"]
    incident_intensity = sum(
        abs(incident.dft_fields[c]) ** 2 for c in ("Ex", "Ey", "Ez")
    )
    incident_intensity = incident_intensity.reshape(3, -1).mean(axis=1)
    figures = []
    for name, axes_names in (
        ("field_xy_silicon", ("x", "y")),
        ("field_yz_BG", ("y", "z")),
        ("field_yz_GR", ("y", "z")),
    ):
        monitor = result[name]
        intensity = sum(abs(monitor.dft_fields[c]) ** 2 for c in ("Ex", "Ey", "Ez"))
        grid = result.metadata.grid
        coordinates = []
        for axis in axes_names:
            interval = monitor.sample_region.axis_interval(axis)
            edges = grid.axis_edges(axis)[interval.start : interval.stop + 1]
            offset = result.metadata.coordinate_offset["xyz".index(axis)]
            coordinates.append((edges - offset) / um)
        nx, ny = len(coordinates[0]) - 1, len(coordinates[1]) - 1
        intensity = intensity.reshape(3, ny, nx) / incident_intensity[:, None, None]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        vmax = float(intensity.max())
        for i, (ax, wl) in enumerate(zip(axes, (650, 550, 450), strict=True)):
            mesh = ax.pcolormesh(
                *coordinates,
                intensity[i],
                shading="flat",
                cmap="magma",
                vmin=0,
                vmax=vmax,
            )
            ax.set(
                title=f"{wl} nm",
                xlabel=f"{axes_names[0]} (µm)",
                ylabel=f"{axes_names[1]} (µm)",
            )
            ax.set_aspect("equal")
        fig.colorbar(mesh, ax=axes, label="|E|² / |E incident|²")
        fig.suptitle(name)
        figures.append(fig)
    return figures
