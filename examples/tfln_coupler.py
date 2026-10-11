"""Shared setup for the TFLN example and benchmark notebooks.

Lengths in CouplerConfig are micrometres; BeamZ receives SI lengths.
Importing this module neither builds a simulation nor starts a solver.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

import beamz as bz
from beamz.design.raster import RasterOptions

REFERENCE_URL = (
    "https://www.flexcompute.com/tidy3d/examples/notebooks/AdiabaticCouplerLN/"
)
NORMALIZATION = "forward selected output mode power / forward input mode power"


@dataclass(frozen=True)
class CouplerConfig:
    length_um: float = 2000.0
    width_um: float = 1.0
    height_um: float = 0.6
    d12_um: float = 2.0  # Centre-to-centre distances at the entrance.
    d23_um: float = 1.0
    buffer_um: float = 10.0
    n_o: float = 2.211
    n_e: float = 2.138
    n_clad: float = 1.44
    wavelength_min_um: float = 1.45
    wavelength_max_um: float = 1.65
    frequency_count: int = 101
    dx_um: float = 0.06
    domain_y_um: float = 8.0
    domain_z_um: float = 5.2
    pml_um: float = 0.6
    run_time_s: float = 40e-12

    def __post_init__(self):
        if not all(np.isfinite(v) and v > 0 for v in asdict(self).values()):
            raise ValueError("All configuration values must be positive and finite.")
        if (
            self.frequency_count < 2
            or int(self.frequency_count) != self.frequency_count
        ):
            raise ValueError("frequency_count must be an integer >= 2.")
        if self.wavelength_min_um >= self.wavelength_max_um:
            raise ValueError("Wavelength endpoints must be strictly increasing.")
        if min(self.d12_um, self.d23_um) < self.width_um:
            raise ValueError("The reference requires non-overlapping waveguides.")
        pml = self.pml_cells * self.dx_um
        if self.buffer_um * 0.75 <= pml + 2 * self.dx_um:
            raise ValueError("Source is too close to the x PML.")
        required_y = self.d12_um + self.d23_um + 3 * self.width_um
        required_z = 6 * self.height_um
        if (
            min(
                (self.domain_y_um - required_y) / 2,
                (self.domain_z_um - required_z) / 2,
            )
            <= pml + self.dx_um
        ):
            raise ValueError("Mode apertures must fit inside PML with grid clearance.")

    @property
    def frequencies(self):
        return np.linspace(
            bz.LIGHT_SPEED / (self.wavelength_max_um * bz.um),
            bz.LIGHT_SPEED / (self.wavelength_min_um * bz.um),
            self.frequency_count,
        )

    @property
    def freq0(self):
        return float(np.mean(self.frequencies[[0, -1]]))

    @property
    def fwidth(self):
        return float(np.ptp(self.frequencies) / 2)

    @property
    def pml_cells(self):
        return int(np.ceil(self.pml_um / self.dx_um - 1e-10))

    @property
    def shape_xyz(self):
        sizes = (
            self.length_um + 2 * self.buffer_um,
            self.domain_y_um,
            self.domain_z_um,
        )
        return tuple(int(np.ceil(s / self.dx_um - 1e-10)) for s in sizes)

    @property
    def domain_um(self):
        return tuple(n * self.dx_um for n in self.shape_xyz)

    @property
    def identity(self):
        payload = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:12]


def waveguide_vertices(config):
    """Three constant-normal-width rectangles, translated to a centred domain."""
    c = config
    half_separation = (c.d12_um + c.d23_um) / 2
    outer_extent = c.length_um / 2 + 2 * c.buffer_um
    segments = (
        ((-outer_extent, -half_separation), (outer_extent, -half_separation)),
        (
            (-c.length_um / 2, c.d12_um - half_separation),
            (c.length_um / 2, c.d23_um - half_separation),
        ),
        ((-outer_extent, half_separation), (outer_extent, half_separation)),
    )
    polygons = []
    for start, stop in segments:
        start, stop = np.asarray(start), np.asarray(stop)
        tangent = stop - start
        normal = np.array([-tangent[1], tangent[0]]) / np.linalg.norm(tangent)
        offset = normal * c.width_um / 2
        polygons.append(
            np.array([start - offset, stop - offset, stop + offset, start + offset])
        )
    return tuple(polygons)


def plane_centres(config):
    c = config
    rail = (c.d12_um + c.d23_um) / 2
    source_x = -c.length_um / 2 - c.buffer_um / 4
    output_x = c.length_um / 2 + c.buffer_um / 4
    return {
        "source": (source_x, -rail, 0.0),
        "input": (source_x + c.buffer_um / 10, -rail, 0.0),
        "cross": (output_x, rail, 0.0),
        "through": (output_x, -rail, 0.0),
    }


def resource_estimate(config):
    """Arithmetic only. Six-field storage is a lower bound, not peak memory."""
    cells = int(np.prod(config.shape_xyz))
    dt = 0.99 * config.dx_um * bz.um / (bz.LIGHT_SPEED * np.sqrt(3))
    return {
        "shape_xyz": config.shape_xyz,
        "cells": cells,
        "six_float32_fields_GiB_lower_bound": cells * 6 * 4 / 2**30,
        "vacuum_CFL_steps_estimate": int(np.ceil(config.run_time_s / dt)),
    }


def build_simulation(config, polarization, *, include_field=False):
    c = config
    if polarization not in {"te", "tm"}:
        raise ValueError("polarization must be 'te' or 'tm'.")
    ln = bz.Material(permittivity=(c.n_o**2, c.n_e**2, c.n_o**2))
    design = bz.Design(background=bz.Material(permittivity=c.n_clad**2))
    for vertices in waveguide_vertices(c):
        design += bz.Polygon(
            vertices=tuple(map(tuple, vertices * bz.um)),
            material=ln,
            depth=c.height_um * bz.um,
            z=-c.height_um * bz.um / 2,
        )
    centres = plane_centres(c)
    aperture = (0.0, 3 * c.width_um * bz.um, 6 * c.height_um * bz.um)
    spec = bz.ModeSpec(
        num_modes=4, mode_index=0, polarization=polarization, target_neff=c.n_e
    )
    source = bz.ModeSource(
        center=tuple(np.asarray(centres["source"]) * bz.um),
        size=aperture,
        direction="+",
        mode_spec=spec,
        source_time=bz.GaussianPulse(freq0=c.freq0, fwidth=c.fwidth),
    )
    monitors = [
        bz.ModeMonitor(
            center=tuple(np.asarray(centres[name]) * bz.um),
            size=aperture,
            freqs=c.frequencies,
            mode_spec=spec,
            name=name,
        )
        for name in ("input", "cross", "through")
    ]
    if include_field:
        monitors.append(
            bz.FieldMonitor(
                center=(0.0, 0.0, 0.0),
                size=(c.length_um * bz.um, 4 * bz.um, 0.0),
                freqs=[c.freq0],
                fields=("Ex", "Ey", "Ez"),
                name="field",
            )
        )
    return bz.Simulation(
        domain=tuple(np.nextafter(np.asarray(c.domain_um) * bz.um, np.inf)),
        design=design,
        sources=(source,),
        monitors=monitors,
        boundaries=(
            bz.PML(thickness=c.pml_cells * c.dx_um * bz.um, formulation="cpml"),
        ),
        grid_spec=bz.GridSpec(resolution=c.dx_um * bz.um, max_total_cells=None),
        run_time=c.run_time_s,
        setup_device="cpu",
        raster_options=RasterOptions(smoothing="farjadpour_diagonal"),
    )


def modal_spectrum(mode_getter, config):
    """Common BeamZ/Tidy3D extraction; retain weak-source bins as NaN."""

    def power(name, direction):
        amps = mode_getter(name).amps.sel(direction=direction, mode_index=0)
        amps = amps.sel(f=config.frequencies).transpose("f")
        return np.abs(np.asarray(amps)) ** 2

    incident = power("input", "+")
    if not np.all(np.isfinite(incident)) or np.max(incident) <= 0:
        raise ValueError("Incident spectrum is non-finite or empty.")
    valid = incident > np.max(incident) * 1e-4
    denominator = np.where(valid, incident, np.nan)
    return {
        "frequency_hz": config.frequencies,
        "wavelength_um": bz.LIGHT_SPEED / config.frequencies / bz.um,
        "incident_power": incident,
        "valid": valid,
        "T_cross": power("cross", "+") / denominator,
        "T_through": power("through", "+") / denominator,
        "R_input": power("input", "-") / denominator,
    }


def save_spectrum(directory, solver, polarization, config, spectrum, **metadata):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{solver}_{polarization}_{config.identity}.npz"
    manifest = dict(
        config=asdict(config),
        solver=solver,
        polarization=polarization,
        reference_url=REFERENCE_URL,
        normalization=NORMALIZATION,
        **metadata,
    )
    np.savez_compressed(path, **spectrum, manifest=json.dumps(manifest))
    return path


def load_spectrum(directory, solver, polarization, config):
    path = Path(directory) / f"{solver}_{polarization}_{config.identity}.npz"
    with np.load(path, allow_pickle=False) as saved:
        manifest = json.loads(str(saved["manifest"]))
        if (
            manifest["config"] != asdict(config)
            or manifest["normalization"] != NORMALIZATION
            or manifest["solver"] != solver
            or manifest["polarization"] != polarization
        ):
            raise ValueError(f"Reference metadata mismatch: {path}")
        spectrum = {key: saved[key].copy() for key in saved.files if key != "manifest"}
    np.testing.assert_allclose(spectrum["frequency_hz"], config.frequencies, rtol=1e-12)
    return spectrum, manifest


def build_tidy3d_simulation(config, polarization):
    """Optional matched-physics reference; importing Tidy3D is deferred."""
    import tidy3d as td

    c = config
    if polarization not in {"te", "tm"}:
        raise ValueError("polarization must be 'te' or 'tm'.")
    ordinary = td.Medium(permittivity=c.n_o**2)
    extraordinary = td.Medium(permittivity=c.n_e**2)
    ln = td.AnisotropicMedium(xx=ordinary, yy=extraordinary, zz=ordinary)
    structures = [
        td.Structure(
            geometry=td.PolySlab(
                vertices=vertices,
                axis=2,
                slab_bounds=(-c.height_um / 2, c.height_um / 2),
            ),
            medium=ln,
        )
        for vertices in waveguide_vertices(c)
    ]
    aperture = (0.0, 3 * c.width_um, 6 * c.height_um)
    centres = plane_centres(c)
    spec = td.ModeSpec(num_modes=4, target_neff=c.n_e, filter_pol=polarization)
    source = td.ModeSource(
        center=centres["source"],
        size=aperture,
        direction="+",
        mode_spec=spec,
        mode_index=0,
        source_time=td.GaussianPulse(freq0=c.freq0, fwidth=c.fwidth),
    )
    monitors = [
        td.ModeMonitor(
            center=centres[name],
            size=aperture,
            freqs=c.frequencies,
            mode_spec=spec,
            name=name,
        )
        for name in ("input", "cross", "through")
    ]
    # Tidy3D extends its PML outside `size`; BeamZ includes CPML inside its domain.
    # Match the non-PML interior, with the same number of absorbing grid layers.
    interior = tuple(s - 2 * c.pml_cells * c.dx_um for s in c.domain_um)
    return td.Simulation(
        size=interior,
        structures=structures,
        medium=td.Medium(permittivity=c.n_clad**2),
        sources=[source],
        monitors=monitors,
        run_time=c.run_time_s,
        grid_spec=td.GridSpec.uniform(dl=c.dx_um),
        boundary_spec=td.BoundarySpec.all_sides(td.PML(num_layers=c.pml_cells)),
        symmetry=(0, 0, 0),
        shutoff=0,
    )
