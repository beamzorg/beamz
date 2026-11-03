"""Sweep 3D dielectric-port resolution, aperture, and mirror symmetry locally."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import beamz as bz
from beamz.design.discretization import MaterialGrid
from beamz.design.grid import RectilinearGrid
from beamz.lattice import (
    component_shape_3d,
    sample_voxel_grid_at_e_component_3d_centered,
)


def axis_edges(extent, core, spacing):
    """Keep both physical interfaces exact while refining a mirrored mesh."""
    inside = int(np.ceil(core / spacing))
    outside = int(np.ceil((extent - core) / (2 * spacing)))
    return (
        np.concatenate(
            (
                np.linspace(-extent / 2, -core / 2, outside + 1)[:-1],
                np.linspace(-core / 2, core / 2, inside + 1),
                np.linspace(core / 2, extent / 2, outside + 1)[1:],
            )
        )
        + extent / 2
    )


def simulation(ppw, aperture, polarization):
    spacing = 1.55e-6 / (3.48 * ppw)
    width, height = np.asarray(aperture) * bz.um
    y = axis_edges(width, 0.43e-6, spacing)
    z = axis_edges(height, 0.22e-6, spacing)
    yc, zc = 0.5 * (y[:-1] + y[1:]), 0.5 * (z[:-1] + z[1:])
    plane = np.where(
        (abs(yc[None, :] - width / 2) < 0.215e-6)
        & (abs(zc[:, None] - height / 2) < 0.11e-6),
        3.48**2,
        1.444**2,
    )
    eps = np.repeat(plane[..., None], 4, axis=2).astype(np.float32)
    yee = {
        "eps_" + a: np.asarray(
            sample_voxel_grid_at_e_component_3d_centered(eps, "E" + a)
        )
        for a in "xyz"
    }
    yee.update(
        {"mu_h" + a: np.ones(component_shape_3d("H" + a, eps.shape)) for a in "xyz"}
    )
    material = MaterialGrid(
        eps,
        0.0,
        1.0,
        spacing,
        eps.shape,
        grid=RectilinearGrid(np.linspace(0, 0.4e-6, 5), y, z),
        yee_materials=yee,
    )
    frequency = bz.LIGHT_SPEED / 1.55e-6
    source = bz.ModeSource(
        center=(0.2e-6, width / 2, height / 2),
        size=(0.0, width, height),
        direction="+",
        mode_spec=bz.ModeSpec(num_modes=1, target_neff=2.3, polarization=polarization),
        source_time=bz.GaussianPulse(freq0=frequency, fwidth=frequency / 10),
    )
    return bz.Simulation(
        material_grid=material, sources=[source], run_time=1e-15
    ), source


def sample_case(ppw, aperture, polarization):
    sim, source = simulation(ppw, aperture, polarization)
    preview = source.solve_modes(sim)
    material = sim.to_request(num_steps=1).materials
    y, z = preview.grid_edges[1], preview.grid_edges[0]
    yc, zc = 0.5 * (y[:-1] + y[1:]), 0.5 * (z[:-1] + z[1:])
    field = abs(preview.e_fields[0, 0, 1 if polarization == "te" else 2])
    mirror = [
        float(np.linalg.norm(field - np.flip(field, d)) / np.linalg.norm(field))
        for d in (0, 1)
    ]
    # Retain a normalized field on common physical coordinates for refinement comparisons.
    yy, zz = np.meshgrid(
        np.linspace(-0.5e-6, 0.5e-6, 101), np.linspace(-0.3e-6, 0.3e-6, 61)
    )
    samples = RegularGridInterpolator(
        (zc - aperture[1] * bz.um / 2, yc - aperture[0] * bz.um / 2),
        field,
        bounds_error=False,
        fill_value=0,
    )(np.stack((zz, yy), axis=-1))
    samples /= samples.max()
    record = {
        "ppw": ppw,
        "aperture_um": list(aperture),
        "polarization": polarization,
        "shape_zyx": list(material.shape),
        "neff": float(preview.neffs[0, 0].real),
        "mirror_error_z_y": mirror,
        "edge_intensity_fraction": float(
            preview.diagnostics[0]["edge_intensity_fraction"]
        ),
    }
    assert max(mirror) < 1e-8
    print(json.dumps(record), flush=True)
    return record, samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records, fields = [], []
    for polarization in ("te", "tm"):
        for ppw in (10, 15, 20, 30, 40, 60, 96):
            record, field = sample_case(ppw, (1.6, 1.2), polarization)
            records.append(record)
            fields.append(field)
        for aperture in ((0.8, 0.6), (1.2, 1.0), (2.4, 1.8), (3.2, 2.4), (4.8, 3.6)):
            record, field = sample_case(20, aperture, polarization)
            records.append(record)
            fields.append(field)
        for ppw in (15, 30, 60):
            record, field = sample_case(ppw, (3.2, 2.4), polarization)
            records.append(record)
            fields.append(field)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "sweeps.json").write_text(
        json.dumps(
            {
                "beamz_version": bz.__version__,
                "source_sha256": {
                    name: hashlib.sha256(
                        (Path(__file__).resolve().parents[1] / name).read_bytes()
                    ).hexdigest()
                    for name in (
                        "scripts/validate_mode_sampling.py",
                        "beamz/devices/modes/_sampling.py",
                        "beamz/devices/modes/discrete.py",
                        "beamz/devices/modes/solver.py",
                        "beamz/simulation/api.py",
                    )
                },
                "python_version": sys.version.split()[0],
                "numpy_version": np.__version__,
                "accuracy_notes": "Mirror symmetry does not establish continuum accuracy. Resolution and finite-aperture errors are assessed separately. Aperture changes at fixed nominal PPW also slightly change fitted cladding cell widths.",
                "records": records,
            },
            indent=2,
        )
        + "\n"
    )
    np.savez_compressed(args.output / "fields.npz", fields=np.asarray(fields))


if __name__ == "__main__":
    main()
