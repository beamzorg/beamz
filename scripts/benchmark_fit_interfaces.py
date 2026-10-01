"""Convergence of TE cavity modes at translated/rotated dielectric interfaces.

The FIT operator is evaluated on explicitly selected CUDA JAX. A separate
body-fitted linear-triangle FEM solves the continuum Hz problem on the host.
This isolates spatial/interface error from time dispersion and PML error.
SciPy is used for reference/eigenanalysis, not for the FIT stepping operator.
"""

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

ROOT = Path(__file__).resolve().parents[1]


def fem_reference(plane, cells, modes=3, polarization="te"):
    """Body-fitted P1 FEM for -div(1/epsilon grad Hz) = lambda Hz.

    TM instead solves -laplacian Ez = lambda epsilon Ez with Dirichlet walls.
    TE PEC implies Neumann Hz boundaries. Each square is split at the exact
    material plane, then each resulting convex polygon is triangulated.
    Use consistent mass, and remove the constant zero-frequency mode.
    """
    from beamz.simulation.fit.interfaces import _clip_polygon

    if polarization not in {"te", "tm"}:
        raise ValueError("polarization must be te or tm")

    nodes, lookup, triangles, permittivities = [], {}, [], []

    def node(vertex):
        key = tuple(np.round(vertex, 14))
        if key not in lookup:
            lookup[key] = len(nodes)
            nodes.append(vertex)
        return lookup[key]

    normal = np.asarray(plane.normal, dtype=float)
    for y in range(cells):
        for x in range(cells):
            square = (
                np.array([[x, y], [x + 1, y], [x + 1, y + 1], [x, y + 1]], dtype=float)
                / cells
            )
            for sign, eps in [
                (1, plane.permittivity_minus),
                (-1, plane.permittivity_plus),
            ]:
                polygon = _clip_polygon(square, sign * normal, sign * plane.offset)
                if len(polygon) < 3:
                    continue
                for k in range(1, len(polygon) - 1):
                    vertices = polygon[[0, k, k + 1]]
                    area = abs(np.linalg.det(vertices[1:] - vertices[0])) / 2
                    if area < 1e-18:
                        continue
                    triangles.append([node(v) for v in vertices])
                    permittivities.append(eps)
    nodes, triangles = np.asarray(nodes), np.asarray(triangles)
    vertices = nodes[triangles]
    coordinates = np.concatenate([np.ones((*vertices.shape[:2], 1)), vertices], axis=-1)
    area = abs(np.linalg.det(coordinates)) / 2
    gradients = np.linalg.inv(coordinates)[:, 1:, :]
    stiffness = (
        np.einsum("tki,tkj->tij", gradients, gradients)
        * (area / np.asarray(permittivities) if polarization == "te" else area)[
            :, None, None
        ]
    )
    mass = (np.ones((3, 3)) + np.eye(3)) * area[:, None, None] / 12
    if polarization == "tm":
        mass = mass * np.asarray(permittivities)[:, None, None]
    row = np.repeat(triangles, 3, axis=1).ravel()
    col = np.tile(triangles, (1, 3)).ravel()
    shape = (len(nodes), len(nodes))
    k = coo_matrix((stiffness.ravel(), (row, col)), shape=shape).tocsr()
    m = coo_matrix((mass.ravel(), (row, col)), shape=shape).tocsr()
    if polarization == "tm":
        free = np.all((nodes > 1e-12) & (nodes < 1 - 1e-12), axis=1)
        k, m = k[free][:, free], m[free][:, free]
    skip = 1 if polarization == "te" else 0
    values = np.sort(
        eigsh(
            k,
            M=m,
            k=modes + skip,
            sigma=-1e-5,
            which="LM",
            tol=1e-9,
            v0=np.linspace(1, 2, k.shape[0]),
            return_eigenvectors=False,
        )
    )[skip:]
    return np.sqrt(values)


def fit_modes(plane, cells, kind, modes=3):
    import jax
    import jax.numpy as jnp

    from beamz import FITSimulation, UniformFITMesh
    from beamz.const import EPS_0
    from beamz.simulation.fit import topology

    mesh = UniformFITMesh((cells, cells), 1 / cells, "te")
    material = plane.material(mesh)
    fractions = material.fraction_minus
    if kind == "tensor":
        sim = FITSimulation(mesh, interface=material, precision="float64")
    else:
        if kind == "staircase":
            y, x = np.meshgrid(
                (np.arange(cells) + 0.5) / cells,
                (np.arange(cells) + 0.5) / cells,
                indexing="ij",
            )
            fraction = (
                plane.normal[0] * x + plane.normal[1] * y <= plane.offset
            ).astype(float)
        else:
            fraction = fractions
        eps = (
            fraction * plane.permittivity_minus
            + (1 - fraction) * plane.permittivity_plus
        )
        sim = FITSimulation(mesh, permittivity=eps, precision="float64")

    def operator(h):
        d = topology.curl_transpose({"Bz": h}, dimension=2, polarization="te")
        d = {
            name: jnp.where(sim.e_masks[name], 0.0, value) for name, value in d.items()
        }
        if sim.electric_operator is None:
            e = {name: value * EPS_0 / sim.m_epsilon[name] for name, value in d.items()}
        else:
            e = sim.electric_operator.apply_relative(d)
        return topology.curl(e, dimension=2, polarization="te")["Bz"] / mesh.spacing**2

    start = time.perf_counter()
    basis = jnp.eye(cells**2, dtype=jnp.float64).reshape(cells**2, cells, cells)
    applied = jax.jit(jax.vmap(operator))(basis)
    matrix = np.asarray(applied).reshape(cells**2, cells**2).T
    elapsed = time.perf_counter() - start
    defect = np.max(abs(matrix - matrix.T)) / np.max(abs(matrix))
    if defect > 1e-11:
        raise RuntimeError(f"Assembled operator is not symmetric: {defect}")
    values = eigh(
        0.5 * (matrix + matrix.T), subset_by_index=(1, modes), eigvals_only=True
    )
    return np.sqrt(values), {
        "operator_device": str(next(iter(applied.devices()))),
        "assembly_seconds": elapsed,
        "relative_symmetry_defect": float(defect),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "docs/fit_interface_convergence.json"
    )
    parser.add_argument("--resolutions", type=int, nargs="+", default=[12, 20, 28])
    parser.add_argument("--reference-resolutions", type=int, nargs=2, default=[96, 144])
    args = parser.parse_args()
    os.environ["JAX_PLATFORMS"] = args.platform
    os.environ["JAX_ENABLE_X64"] = "1"
    sys.path.insert(0, str(ROOT))
    import jax

    from beamz import PlanarDielectricInterface

    devices = jax.devices("gpu" if args.platform == "cuda" else "cpu")
    cases = [(0, 0.43), (0, 0.47), (30, 0.65), (30, 0.69), (60, 0.65), (60, 0.69)]
    records = []
    for angle, offset in cases:
        normal = (np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle)))
        plane = PlanarDielectricInterface(normal, offset, 2.0, 12.0)
        references = [fem_reference(plane, n) for n in args.reference_resolutions]
        reference = references[-1]
        reference_drift = abs(references[0] / reference - 1)
        for cells in args.resolutions:
            for kind in ["staircase", "scalar_fraction", "tensor"]:
                frequencies, details = fit_modes(plane, cells, kind)
                records.append(
                    {
                        "angle_deg": angle,
                        "offset_m": offset,
                        "cells_per_axis": cells,
                        "model": kind,
                        "omega_over_c0": frequencies.tolist(),
                        "reference_omega_over_c0": reference.tolist(),
                        "relative_frequency_error": abs(
                            frequencies / reference - 1
                        ).tolist(),
                        "reference_relative_drift": reference_drift.tolist(),
                        **details,
                    }
                )
        print(
            f"Completed angle={angle}, offset={offset}; FEM relative drift={reference_drift.max():.3g}",
            flush=True,
        )
    report = {
        "environment": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            "devices": [str(device) for device in devices],
            "precision": "float64",
        },
        "problem": "Unit-square PEC TE cavity, nonmagnetic epsilon=2/12, exact planar interfaces",
        "reference": "Independent body-fitted P1 triangle FEM with consistent mass and Neumann Hz boundaries",
        "reference_resolutions": args.reference_resolutions,
        "records": records,
        "timing_workloads": timing_workloads(),
        "limits": [
            "Spatial mode eigenvalues, not a propagation/DFT/PML test.",
            "Only three lowest nonzero cavity modes; no universal convergence-order claim.",
            "FIT operator assembly runs on selected JAX device; SciPy reference/eigenanalysis runs on CPU.",
            "Staircase comparator uses center-assigned cells and the initial FIT diagonal native averaging; scalar_fraction uses exact fractions and scalar arithmetic averaging.",
            "Dense matrices are assembled only for these small validation eigenproblems; production time stepping remains matrix-free.",
        ],
        "reproduce": "OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/benchmark_fit_interfaces.py --platform cuda",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {args.output}")


def timing_workloads():
    """Synchronized first/warm runs; include constitutive solve overhead."""
    import jax

    from beamz import FITSimulation, PlanarDielectricInterface, UniformFITMesh

    observations = []
    for shape in [(48, 64), (12, 16, 20)]:
        mesh = UniformFITMesh(shape, 0.1e-6, "te")
        normal = np.array([1, 0.7] if mesh.dimension == 2 else [1, 0.7, 0.4])
        offset = 0.5 * normal @ (np.array(shape[::-1]) * mesh.spacing)
        plane = PlanarDielectricInterface(tuple(normal), offset, 2, 12)
        material = plane.material(mesh)
        parallel = material.tensors(mesh)[1]
        for model in ["scalar_fraction", "tensor"]:
            sim = (
                FITSimulation(mesh, interface=material)
                if model == "tensor"
                else FITSimulation(mesh, permittivity=parallel)
            )
            fields = {}
            for name, field in sim.fields.items():
                if not name.startswith("E"):
                    continue
                seed = np.ones(field.shape)
                for axis, coordinate in enumerate(mesh.coordinates(name)):
                    broadcast = [1] * mesh.dimension
                    broadcast[axis] = len(coordinate)
                    seed *= np.sin(
                        np.pi
                        * coordinate.reshape(broadcast)
                        / (shape[axis] * mesh.spacing)
                    )
                fields[name] = seed
            sim.set_fields(**fields)
            initial_energy = float(sim.energy(conserved=True))
            start = time.perf_counter()
            sim.run(32)
            jax.block_until_ready(sim.state)
            first = time.perf_counter() - start
            warm = []
            for _ in range(3):
                start = time.perf_counter()
                sim.run(32)
                jax.block_until_ready(sim.state)
                warm.append(time.perf_counter() - start)
            observations.append(
                {
                    "model": model,
                    "cell_shape": shape,
                    "dt_s": sim.dt,
                    "precision": "float32",
                    "steps_per_run": 32,
                    "first_run_seconds": first,
                    "warm_run_seconds": warm,
                    "warm_median_seconds": float(np.median(warm)),
                    "relative_conserved_energy_drift": abs(
                        float(sim.energy(conserved=True)) / initial_energy - 1
                    ),
                    "maximum_constitutive_residual": float(sim.state.max_solve_residual)
                    if model == "tensor"
                    else None,
                    "material_coefficient_bytes": sum(
                        block.nbytes for _, block in sim.electric_operator._corners
                    )
                    if model == "tensor"
                    else sum(value.nbytes for value in sim.m_epsilon.values()),
                }
            )
    return observations


if __name__ == "__main__":
    main()
