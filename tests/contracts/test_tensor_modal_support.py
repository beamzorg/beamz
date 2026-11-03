"""Unsupported tensor ports must never silently acquire a diagonal basis."""

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from beamz.devices.modes.constitutive import crop_tensor_support
from beamz.devices.modes.plane import solve_mode_plane_3d
from beamz.simulation.results import _snapshot_material_region


@dataclass(frozen=True)
class _Captured:
    """Stand-in mode carrying the spec the solver received."""

    spec: Any
    diagnostics: dict = field(default_factory=dict)


def _solve(tensors, **kwargs):
    shape = (12, 12, 40)

    def capture(spec):
        return _Captured(spec)

    return solve_mode_plane_3d(
        np.ones(shape),
        np.ones(shape),
        material_tensors=tensors,
        frequency=1e8,
        resolution=1.0,
        dt=1e-10,
        axis="x",
        direction="+x",
        grid_shape=shape,
        center=(10.0, 6.0, 6.0),
        width=4.0,
        height=4.0,
        plane_index=10,
        offset_index=9,
        mode_index=0,
        polarization=None,
        target_neff=1.0,
        num_modes=1,
        snapped_region=None,
        solver=capture,
        **kwargs,
    )


def _tensors():
    eps = np.zeros((6, 12, 12, 40))
    eps[:3] = 2.0
    return {"epsilon": eps, "mu": np.ones((1, 12, 12, 40))}


def test_full_tensor_plane_rejected_before_solver():
    tensors = _tensors()
    tensors["epsilon"][3, :, :, 10] = 0.2
    with pytest.raises(ValueError, match="constitutive support"):
        _solve(tensors)


def test_neighbor_cross_term_rejected():
    tensors = _tensors()
    tensors["epsilon"][4, :, :, 9] = 0.2
    with pytest.raises(ValueError, match="constitutive support"):
        _solve(tensors)


def test_tensor_outside_local_normal_support_does_not_reject(monkeypatch):
    import beamz.devices.modes.plane as plane

    monkeypatch.setattr(plane, "_shift_discrete_mode_to_global", lambda mode, **_: mode)
    tensors = _tensors()
    tensors["epsilon"][3, :, :, 30] = 0.2
    result = _solve(tensors)
    assert set(result.spec.diagonal_permittivity) == {"xx", "yy", "zz"}


def test_incomplete_context_does_not_wrap_negative_indices():
    with pytest.raises(ValueError, match="required halo"):
        crop_tensor_support(
            {"eps_node": np.ones((6, 5, 5, 5))}, origin=(0, 0, -1), shape=(2, 2, 2)
        )


@pytest.mark.parametrize("ndim", [2, 3])
def test_monitor_snapshot_retains_shared_node_tensors(ndim):
    shape = (8,) * ndim
    node_shape = (9,) * ndim
    tensor = np.arange(6 * np.prod(node_shape)).reshape(6, *node_shape)
    materials = SimpleNamespace(
        tensors={}, uses_direct_yee_materials=False, yee_tensors={"eps_node": tensor}
    )
    fields = SimpleNamespace(
        permittivity=np.ones(shape),
        permeability=np.ones(shape),
        material_grid=materials,
    )
    crop = (slice(2, 5),) * ndim
    result = _snapshot_material_region(fields, crop, (2,) * ndim)
    np.testing.assert_array_equal(
        result.yee_tensors["eps_node"], tensor[(slice(None), *(slice(2, 6),) * ndim)]
    )
    assert not result.yee_tensors["eps_node"].flags.writeable


def test_port_preserves_explicit_mixed_polarization_selection():
    import beamz as bz

    geometry = dict(center=(0, 0, 0), size=(0, 1, 1), name="port", direction="+")
    assert bz.Port(**geometry).polarization == "te"
    port = bz.Port(**geometry, mode_spec=bz.ModeSpec(polarization=None))
    assert port.polarization is None
    assert port.to_monitor([1e14]).mode_spec.polarization is None


def test_tensor_interior_with_diagonal_port_uses_local_operator():
    from dataclasses import replace

    import beamz as bz
    from beamz.analysis.sparameters import _extract_port_waves_dft
    from tests.validation.analytical.test_tensor_port_maxwell import guide

    original, d, dt, f = guide()
    tensors = {
        name: np.array(value, copy=True) for name, value in original.yee_tensors.items()
    }
    for values in tensors.values():
        values[3:, ..., :17] = 0
        values[3:, ..., 24:] = 0
    cells = {
        name: np.array(value, copy=True) for name, value in original.tensors.items()
    }
    cells["epsilon"][3:, ..., :17] = 0
    cells["epsilon"][3:, ..., 24:] = 0
    material = replace(original, yee_tensors=tensors, tensors=cells)
    assert material.uses_full_permittivity
    source = bz.ModeSource(
        center=(1e-6, 1e-6, 1e-6),
        size=(0, 1.6e-6, 1.6e-6),
        direction="+",
        source_time=bz.GaussianPulse(freq0=f, fwidth=0.2 * f),
        mode_spec=bz.ModeSpec(target_neff=2.0),
    )
    port = bz.Port(
        center=(3e-6, 1e-6, 1e-6),
        size=source.size,
        direction="+",
        name="out",
        mode_spec=source.mode_spec,
    )
    sim = bz.Simulation(
        material_grid=material,
        sources=[source],
        monitors=[port.to_monitor([f])],
        run_time=30e-15,
        setup_device="cpu",
    )
    preview = source.solve_modes(sim, freqs=[f])
    assert not preview.diagnostics[0]["modes"][0]["full_tensor_operator"]
    result = sim.run(backend="jax", progress=False)
    waves = _extract_port_waves_dft(
        result, ports=[port], frequencies=[f], return_power=True
    )
    assert np.isfinite(waves["out"]["a_minus"]).all()
    assert np.isfinite(waves["out"]["a_plus"]).all()


def test_tensor_preview_is_invariant_under_imported_grid_translation():
    from dataclasses import replace

    import beamz as bz
    from tests.validation.analytical.test_tensor_port_maxwell import guide

    material, _, _, frequency = guide()
    shift = np.array([-2e-6, 3e-6, -4e-6])
    previews = []
    for translation in (np.zeros(3), shift):
        translated = replace(
            material, grid=material.grid.translated(tuple(translation))
        )
        source = bz.ModeSource(
            center=tuple(np.full(3, 1e-6) + translation),
            size=(0, 1.6e-6, 1.6e-6),
            direction="+",
            source_time=bz.GaussianPulse(freq0=frequency, fwidth=0.2 * frequency),
            mode_spec=bz.ModeSpec(target_neff=2.0),
        )
        simulation = bz.Simulation(
            material_grid=translated,
            sources=[source],
            run_time=30e-15,
            setup_device="cpu",
        )
        previews.append(source.solve_modes(simulation, freqs=[frequency]))
    np.testing.assert_allclose(previews[0].neffs, previews[1].neffs, rtol=1e-8)
    for axis, original, translated in zip(
        previews[0].transverse_axes,
        previews[0].grid_edges,
        previews[1].grid_edges,
        strict=True,
    ):
        np.testing.assert_allclose(
            translated, original + shift["xyz".index(axis)], atol=1e-18
        )


def test_tensor_radiation_can_be_inspected_but_not_prepared_for_launch():
    import beamz as bz
    from beamz.design.discretization import MaterialGrid
    from beamz.design.raster import Grid, Material, RasterOptions, Scene, rasterize

    material = MaterialGrid.from_raster_result(
        rasterize(
            Scene(
                (
                    Material(
                        epsilon_r=((3.0, 0.2, 0.0), (0.2, 2.0, 0.0), (0.0, 0.0, 1.5))
                    ),
                )
            ),
            Grid.uniform((0, 0, 0), (1, 1, 1), (8, 8, 8)),
            options=RasterOptions(smoothing="farjadpour_full"),
        )
    )
    simulation = bz.Simulation(
        material_grid=material, run_time=1e-8, setup_device="cpu"
    )
    source = bz.ModeSource(
        center=(0.5, 0.5, 0.5),
        size=(0.5, 0.5, 0),
        direction="+",
        source_time=bz.GaussianPulse(freq0=3e8, fwidth=1e8),
        mode_spec=bz.ModeSpec(),
    )
    preview = source.solve_modes(simulation, freqs=[3e8])
    assert not preview.diagnostics[0]["fields_decay_at_boundary"]
    with pytest.raises(ValueError, match="confined guided mode"):
        source.solve_modes(simulation, freqs=[3e8], field_representation="launch")


@pytest.mark.parametrize(
    "change,message",
    [
        ({"polarization": "te"}, "mixed-polarization"),
        ({"force_symmetry": True}, "force_symmetry"),
        ({"dt": None}, "propagation timestep"),
    ],
)
def test_tensor_port_rejects_unimplemented_combinations(change, message):
    from tests.validation.analytical.test_tensor_port_maxwell import guide

    material, d, dt, f = guide()
    args = dict(
        material_tensors=material.tensors,
        yee_materials=material.yee_materials,
        yee_tensors=material.yee_tensors,
        frequency=f,
        resolution=d,
        dt=dt,
        axis="x",
        direction="+x",
        grid_shape=material.shape,
        center=(1e-6,) * 3,
        width=1.6e-6,
        height=1.6e-6,
        plane_index=10,
        offset_index=9,
        mode_index=0,
        polarization=None,
        target_neff=2.0,
        num_modes=1,
        snapped_region=None,
    )
    args.update(change)
    with pytest.raises(ValueError, match=message):
        solve_mode_plane_3d(material.permittivity, material.permeability, **args)


def test_tensor_port_with_varying_normal_spacing_is_rejected():
    from beamz.design.grid import RectilinearGrid
    from tests.validation.analytical.test_tensor_port_maxwell import guide

    material, d, dt, f = guide()
    x_edges = np.linspace(0, 8e-6, 41)
    x_edges[10] += 0.05e-6
    grid = RectilinearGrid(x_edges, np.linspace(0, 2e-6, 21), np.linspace(0, 2e-6, 21))
    with pytest.raises(ValueError, match="uniform normal-axis spacing"):
        solve_mode_plane_3d(
            material.permittivity,
            material.permeability,
            material_tensors=material.tensors,
            yee_materials=material.yee_materials,
            yee_tensors=material.yee_tensors,
            frequency=f,
            resolution=d,
            dt=dt,
            axis="x",
            direction="+x",
            grid_shape=material.shape,
            center=(2e-6, 1e-6, 1e-6),
            width=1.6e-6,
            height=1.6e-6,
            plane_index=10,
            offset_index=9,
            mode_index=0,
            polarization=None,
            target_neff=2.0,
            num_modes=1,
            snapped_region=None,
            grid=grid,
        )


@pytest.mark.parametrize("unsupported", ["periodic", "bloch", "sharding", "cuda_mstep"])
def test_rectilinear_tensor_propagation_rejects_unimplemented_lowerings(unsupported):
    import beamz as bz
    from tests.validation.analytical.test_rectilinear_tensor_modes import (
        rectilinear_guide,
    )

    material, dt, _ = rectilinear_guide()
    boundary = {"periodic": bz.Periodic, "bloch": bz.Bloch}.get(unsupported)
    sim = bz.Simulation(
        material_grid=material,
        boundaries=[] if boundary is None else [boundary(axes="y")],
        time=np.arange(3) * dt,
        setup_device="cpu",
    )
    with pytest.raises(ValueError, match="Rectilinear full-tensor updates"):
        sim.compile(
            backend="cuda_mstep" if unsupported == "cuda_mstep" else "jax",
            sharding=unsupported == "sharding",
        )
