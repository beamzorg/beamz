"""Rasterize dispersive material ownership on electric Yee supports.

A separate indicator raster for each medium preserves painter order and native
geometry integration. Tangential response uses volume-averaged poles. Sparse native interface data
provide the harmonic normal response through a coupled constituent-field ADE.
"""

from dataclasses import replace

import numpy as np

from beamz.design.discretization import MaterialGrid
from beamz.design.dispersion import PoleResidue
from beamz.design.dispersive_interfaces import DispersiveInterface
from beamz.design.materials import Material
from beamz.lattice import component_shapes

from . import _native  # type: ignore[attr-defined]
from .engine import RasterOptions, rasterize, staircase_materials


def rasterize_dispersion(
    scene,
    grid,
    *,
    kind,
    polarization,
    quality,
    resolution,
    cache_directory,
    smoothing="volume",
    metal_smoothing="staircase",
    reference_frequency=None,
    periodic_axes=(),
):
    metals = staircase_materials(
        scene,
        RasterOptions(
            metal_smoothing=metal_smoothing, reference_frequency=reference_frequency
        ),
    )
    interfaces = []
    normal_weights = {}
    if smoothing == "farjadpour_full":
        raise ValueError(
            "Dispersive full-tensor interfaces are not supported; use farjadpour_diagonal or volume."
        )
    if smoothing == "farjadpour_diagonal":
        # Geometry integration classifies laminar supports even when all material
        # epsilon-infinity values are equal. No permittivity-gradient normal guess.
        native = _native.compile_scene(scene.to_json())
        samples = native.interface_samples(
            tuple(e.tolist() for e in grid.edges),
            quality,
            "all" if kind == "3d" else f"two_dimensional_{polarization}",
            metals,
            tuple(axis in periodic_axes for axis in "xyz"),
        )
        shape = grid.shape_zyx if kind == "3d" else grid.shape_zyx[1:]
        shapes = component_shapes(shape, polarization)
        for component, record in samples.items():
            indices = np.asarray(record["indices"], dtype=np.int64)
            fractions = np.asarray(record["fractions"]).reshape(
                -1, len(scene.materials)
            )
            normals = np.asarray(record["normal_squared"])
            # A 2D electric support uses the first of the identical extruded z planes.
            keep = indices < np.prod(shapes[component])
            indices, fractions, normals = indices[keep], fractions[keep], normals[keep]
            groups = {}
            for row, f in enumerate(fractions):
                owners = tuple(np.flatnonzero(f > 1e-12))
                if len(owners) > 1 and any(
                    isinstance(scene.materials[j], PoleResidue) for j in owners
                ):
                    groups.setdefault(owners, []).append(row)
            mask = np.zeros(shapes[component], dtype=float)
            for owners, rows in groups.items():
                f = fractions[rows][:, owners].T
                f /= f.sum(axis=0)
                interface = DispersiveInterface(
                    component,
                    indices[rows],
                    f,
                    normals[rows],
                    tuple(scene.materials[j] for j in owners),
                )
                interfaces.append(interface)
                mask.reshape(-1)[interface.indices] = interface.normal_squared
            normal_weights[component] = mask
    regions = []
    for index, medium in enumerate(scene.materials):
        if not isinstance(medium, PoleResidue):
            continue
        indicators = tuple(
            Material(1.0, conductivity=float(i == index))
            for i in range(len(scene.materials))
        )
        result = rasterize(
            replace(scene, materials=indicators),
            grid,
            options=RasterOptions(
                quality=quality,
                smoothing="volume",
                periodic_axes=periodic_axes,
                components="all" if kind == "3d" else f"two_dimensional_{polarization}",
            ),
            cache_directory=cache_directory,
            _staircase_materials=metals,
        )
        material_grid = MaterialGrid.from_raster_result(
            result,
            dimensions=3 if kind == "3d" else 2,
            polarization=polarization,
            resolution=resolution,
        )
        supports = {
            "E" + axis: np.clip(
                np.asarray(material_grid.yee_materials["sig_" + axis]), 0.0, 1.0
            )
            for axis in "xyz"
            if "sig_" + axis in material_grid.yee_materials
        }
        for component, weight in supports.items():
            if component in normal_weights:
                supports[component] = weight * (1 - normal_weights[component])
        regions.append((medium, supports))
    return tuple(regions), tuple(interfaces)
