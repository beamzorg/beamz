"""Bounded, sequential mesh import/raster benchmark with volume verification.

Run with a freshly built BeamZ extension and single-thread environment settings.
Timings are evidence, not hardware-dependent pass/fail thresholds.
"""

import argparse
import json
import platform
import resource
import tempfile
from pathlib import Path
from time import perf_counter

import meshio
import numpy as np

from beamz.design import raster
from beamz.design.raster.importers import from_mesh
from beamz.design.raster.importers.mesh_import import _tetra_boundary


def tetrahedral_cube(count):
    indices = np.indices((count + 1,) * 3).reshape(3, -1).T
    points = indices.astype(float) / count
    origins = np.indices((count,) * 3).reshape(3, -1).T
    corners = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ]
    )
    addresses = origins[:, None, :] + corners
    ids = np.ravel_multi_index(addresses.transpose(2, 0, 1), (count + 1,) * 3)
    pattern = np.array(
        [
            [0, 1, 2, 6],
            [0, 2, 3, 6],
            [0, 3, 7, 6],
            [0, 7, 4, 6],
            [0, 4, 5, 6],
            [0, 5, 1, 6],
        ]
    )
    return points, ids[:, pattern].reshape(-1, 4).astype(np.uint32)


def reference_boundary(points, cells):
    """Previous per-tetrahedron algorithm, retained only for this benchmark."""
    counts, oriented = {}, {}
    for tetra in cells:
        center = points[tetra].mean(axis=0)
        for face in (
            tetra[[1, 2, 3]],
            tetra[[0, 3, 2]],
            tetra[[0, 1, 3]],
            tetra[[0, 2, 1]],
        ):
            vertices = points[face]
            normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
            if np.dot(normal, vertices.mean(axis=0) - center) < 0:
                face = face[[0, 2, 1]]
            key = tuple(sorted(face))
            counts[key] = counts.get(key, 0) + 1
            oriented[key] = face
    return np.array([oriented[key] for key, count in counts.items() if count == 1])


def timed(function, *args):
    start = perf_counter()
    result = function(*args)
    return result, perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 16, 24])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(size < 1 or size > 32 for size in args.sizes):
        parser.error("sizes must be between 1 and 32 to bound memory use")
    rows = []
    with tempfile.TemporaryDirectory() as directory:
        for size in args.sizes:
            points, cells = tetrahedral_cube(size)
            faces, boundary_seconds = timed(_tetra_boundary, points, cells)
            row = {
                "subdivisions": size,
                "tetrahedra": len(cells),
                "surface_triangles": len(faces),
                "boundary_seconds": boundary_seconds,
            }
            if size == min(args.sizes):
                reference, elapsed = timed(reference_boundary, points, cells)
                # Compare unoriented faces plus native manifold/orientation validation.
                assert sorted(map(tuple, np.sort(reference, axis=1))) == sorted(
                    map(tuple, np.sort(faces, axis=1))
                )
                assert raster.inspect_mesh(points, faces).valid_for_rasterization
                row.update(
                    reference_boundary_seconds=elapsed,
                    boundary_speedup=elapsed / boundary_seconds,
                )
            path = Path(directory) / "cube.msh"
            tags = np.ones(len(cells), dtype=int)
            meshio.write(
                path,
                meshio.Mesh(
                    points,
                    [("tetra", cells)],
                    cell_data={"gmsh:physical": [tags], "gmsh:geometrical": [tags]},
                ),
                file_format="gmsh22",
            )
            scene, row["import_seconds"] = timed(
                lambda path=path: from_mesh(path, material=raster.Material(4))
            )
            compiled, row["compile_seconds"] = timed(raster.compile_scene, scene)
            result, row["raster_seconds"] = timed(
                lambda compiled=compiled: compiled.rasterize(
                    raster.Grid.uniform((-0.1, -0.1, -0.1), (1.1, 1.1, 1.1), (8, 8, 8)),
                    options=raster.RasterOptions(smoothing="volume", quality="fast"),
                )
            )
            volume = float(
                np.mean((result.tensors["epsilon"][0].astype(float) - 1) / 3) * 1.2**3
            )
            assert abs(volume - 1) < 2e-6
            row["absolute_volume_error"] = abs(volume - 1)
            row["peak_process_rss_mib"] = resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss / (1024**2 if platform.system() == "Darwin" else 1024)
            rows.append(row)
            print(json.dumps(row), flush=True)
    args.output.write_text(
        json.dumps(
            {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "cases": rows,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
