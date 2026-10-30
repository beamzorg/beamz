"""Public PMC specifications, face precedence, and backend support."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import beamz as bz
from beamz.devices._boundary_compile import (
    compile_metallic_masks,
    resolve_magnetic_edges,
    resolve_metallic_edges,
)
from beamz.lattice import component_shapes
from beamz.simulation import backend as backend_runtime


def test_pmc_is_immutable_zero_thickness_and_exported():
    from beamz.devices import PMC as device_pmc
    from beamz.simulation import PMC as simulation_pmc

    assert bz.PMC is device_pmc is simulation_pmc
    boundary = bz.PMC(edges=["LEFT"], thickness=3.0)
    assert boundary.edges == ("left",)
    assert boundary.thickness == 0.0
    assert len(bz.PMC()._get_edges_for_dimensionality(False)) == 4
    assert len(bz.PMC()._get_edges_for_dimensionality(True)) == 6
    with pytest.raises(FrozenInstanceError):
        boundary.edges = "all"
    copied = boundary.updated_copy(edges=["right"])
    assert copied.edges == ("right",)
    assert boundary.edges == ("left",)


@pytest.mark.parametrize("absorber", [bz.PML, bz.Absorber])
def test_last_specification_selects_face_type(absorber):
    boundaries = [
        bz.PEC(),
        bz.PMC(edges=("left", "right", "top")),
        absorber(edges="right"),
        bz.PEC(edges="top"),
    ]
    assert resolve_magnetic_edges(boundaries, True) == {"left"}
    assert resolve_metallic_edges(boundaries, True) == {
        "top",
        "bottom",
        "front",
        "back",
    }
    assert resolve_magnetic_edges([bz.PMC(), bz.PEC()], True) == set()
    assert resolve_metallic_edges([bz.PEC(), bz.PMC()], True) == set()


@pytest.mark.parametrize(
    "shape,polarization", [((3, 4), "tm"), ((3, 4), "te"), ((2, 3, 4), "tm")]
)
@pytest.mark.parametrize("region_setup", [False, True])
def test_pmc_does_not_zero_half_cell_interior_samples(
    shape, polarization, region_setup
):
    masks = compile_metallic_masks(
        component_shapes(shape, polarization),
        shape,
        [bz.PMC()],
        polarization_2d=polarization,
        region_setup=region_setup,
    )
    assert all(not np.any(np.asarray(mask)) for mask in masks.values())


def _simulation(boundaries):
    return bz.Simulation(
        domain=(0.4 * bz.um, 0.3 * bz.um, 0.2 * bz.um),
        resolution=0.1 * bz.um,
        time=np.arange(3) * 1e-17,
        boundaries=boundaries,
    )


@pytest.mark.parametrize("backend", ["auto", "cuda_streamed"])
def test_pmc_compiles_for_cuda_when_available(backend, monkeypatch):
    monkeypatch.setattr(backend_runtime, "resolve_backend", lambda _: "cuda_streamed")
    program = _simulation([bz.PMC()]).compile(backend=backend)
    assert program.config.backend == "cuda_streamed"
    assert len(program.boundary.cpml.magnetic_edges) == 6
    assert not program.boundary.cpml.metallic_edges


def test_2d_pmc_keeps_existing_cuda_dimensionality_limit():
    simulation = bz.Simulation(
        domain=(0.4 * bz.um, 0.3 * bz.um),
        resolution=0.1 * bz.um,
        time=np.arange(3) * 1e-17,
        boundaries=[bz.PMC()],
    )
    with pytest.raises(backend_runtime.CudaBackendUnavailable, match="3D simulation"):
        simulation.compile(backend="cuda_streamed")


@pytest.mark.parametrize("axes,conflict", [("x", True), ("y", False)])
def test_pmc_respects_periodic_face_ownership(axes, conflict):
    from beamz.devices.boundaries import validate_boundary_compatibility

    boundaries = [bz.PMC(edges="left"), bz.Periodic(axes=axes)]
    if conflict:
        with pytest.raises(ValueError, match="Periodic axes cannot also use"):
            validate_boundary_compatibility(boundaries, is_3d=True)
    else:
        validate_boundary_compatibility(boundaries, is_3d=True)
