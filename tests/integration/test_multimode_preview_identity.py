"""Real multimode previews reuse eigenpairs and match launch/projection modes."""

from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

import beamz as bz
from beamz.design.discretization import MaterialGrid
from beamz.design.grid import RectilinearGrid
from beamz.devices.modes import discrete, plane
from beamz.devices.modes.plane import bind_material_mode_plane


@pytest.fixture
def preview_fixture():
    def make(num_modes=3, *, confinement=None, axis="x", direction="+"):
        eps = np.full((40, 80, 4), 1.444**2, dtype=np.float32)
        eps[12:28, 22:58, :] = 3.476**2
        edges = (
            np.linspace(0, 0.2e-6, 5),
            np.linspace(0, 4e-6, 81),
            np.linspace(0, 2e-6, 41),
        )
        center, size = (0.1e-6, 2e-6, 1e-6), (0, 2.8e-6, 1.8e-6)
        if axis == "y":
            eps = eps.swapaxes(1, 2)
            edges = edges[1], edges[0], edges[2]
            center, size = (
                (center[1], center[0], center[2]),
                (size[1], size[0], size[2]),
            )
        material = MaterialGrid(
            eps,
            np.zeros_like(eps),
            np.ones_like(eps),
            0.05e-6,
            eps.shape,
            grid=RectilinearGrid(*edges),
        )
        frequency = bz.LIGHT_SPEED / 1.55e-6
        source = bz.ModeSource(
            center=center,
            size=size,
            direction=direction,
            source_time=bz.GaussianPulse(freq0=frequency, fwidth=frequency / 10),
            mode_spec=bz.ModeSpec(
                num_modes=num_modes,
                polarization="te",
                target_neff=2.4,
                min_aperture_fraction=confinement,
            ),
        )
        return bz.Simulation(
            material_grid=material, sources=[source], run_time=1e-15
        ), source

    return make


def discrete_overlap(left, right):
    vectors = [
        np.concatenate(
            [
                (mode.profiles[n] * np.sqrt(mode.integration_weights[n])).ravel()
                for n in ("Ex", "Ey", "Ez")
            ]
        )
        for mode in (left, right)
    ]
    a, b = vectors
    return abs(np.vdot(a, b)) ** 2 / (np.vdot(a, a).real * np.vdot(b, b).real)


@pytest.mark.parametrize("num_modes", [1, 2, 3])
def test_preview_solves_each_candidate_pool_once(preview_fixture, num_modes):
    sim, source = preview_fixture(num_modes)
    with (
        patch.object(discrete, "solve_grid", wraps=discrete.solve_grid) as solve,
        patch.object(
            plane, "_local_component_materials", wraps=plane._local_component_materials
        ) as supports,
    ):
        data = source.solve_modes(sim)
    assert supports.call_count == 1
    assert [call.kwargs["num_modes"] for call in solve.call_args_list] == [
        2 * num_modes + 5
    ]
    vectors = data.e_fields[0, :, 1].reshape(num_modes, -1)
    vectors = vectors / np.linalg.norm(vectors, axis=1)[:, None]
    assert np.linalg.eigvalsh(vectors.conj() @ vectors.T)[0] > 0.9


def test_preparing_modes_does_not_mutate_the_shared_candidates(
    monkeypatch, preview_fixture
):
    sim, source = preview_fixture()
    captured = []
    resolve = discrete._resolve_mode_basis

    def observe(*args, **kwargs):
        basis = resolve(*args, **kwargs)
        captured.append(
            (
                basis,
                [
                    {n: f.copy() for n, f in c["fields"].items()}
                    for c in basis.candidates
                ],
            )
        )
        return basis

    monkeypatch.setattr(discrete, "_resolve_mode_basis", observe)
    source.solve_modes(sim)
    assert len(captured) == 1
    basis, original = captured[0]
    for candidate, fields in zip(basis.candidates, original, strict=True):
        for name, reference in fields.items():
            np.testing.assert_array_equal(candidate["fields"][name], reference)


def test_preview_resolves_a_separate_basis_at_each_frequency(preview_fixture):
    sim, source = preview_fixture()
    with patch.object(discrete, "solve_grid", wraps=discrete.solve_grid) as solve:
        preview = source.solve_modes(
            sim, freqs=[source.frequency, 1.01 * source.frequency]
        )
    assert preview.neffs.shape == (2, 3)
    assert [call.kwargs["num_modes"] for call in solve.call_args_list] == [11, 11]
    assert (
        solve.call_args_list[0].kwargs["freqs"]
        != solve.call_args_list[1].kwargs["freqs"]
    )


def test_unrestricted_preview_retains_radiation_candidates_for_inspection(
    preview_fixture,
):
    sim, source = preview_fixture()
    material = sim._material_grid()
    eps = np.ones(material.shape, dtype=np.float32)
    eps[:20, :, :] = 1.444**2
    eps[20:24, 35:44, :] = 3.476**2
    sim = sim.updated_copy(material_grid=replace(material, permittivity=eps))
    source = source.updated_copy(
        size=(0, 0.8e-6, 0.8e-6),
        mode_spec=replace(
            source.mode_spec,
            polarization=None,
            target_neff=3.4,
            min_aperture_fraction=None,
        ),
    )
    preview = source.solve_modes(sim)
    assert preview.neffs.shape == (1, 3)
    # Tutorials may deliberately inspect alternatives below the light line.
    assert np.any(preview.neffs[0].real < 1.444)
    assert any(
        mode["aperture_fraction"] < 0.5 for mode in preview.diagnostics[0]["modes"]
    )


@pytest.mark.parametrize("padding", [None, 0, (2, 4)])
@pytest.mark.parametrize("axis,direction", [("x", "+"), ("y", "-")])
def test_preview_launch_and_projection_share_the_mode_family(
    preview_fixture, axis, direction, padding
):
    sim, source = preview_fixture(
        num_modes=3, confinement=0.5, axis=axis, direction=direction
    )
    source = source.updated_copy(
        mode_spec=replace(source.mode_spec, solver_padding_cells=padding)
    )
    sim = sim.updated_copy(sources=[source])
    monitor = bz.ModeMonitor(
        center=source.center,
        size=source.size,
        freqs=[source.frequency],
        name="review",
        mode_spec=source.mode_spec,
    )
    sim = sim.updated_copy(monitors=[monitor])
    preview = source.solve_modes(sim)
    launched = []

    def launch(*args, **kwargs):
        family = bind_material_mode_plane(*args, **kwargs)
        solve = family.solve

        class CapturedFamily:
            def solve(self, index):
                mode = solve(index)
                launched.append(mode)
                return mode

        return CapturedFamily()

    for index in range(3):
        src = source.updated_copy(mode_spec=replace(source.mode_spec, mode_index=index))
        with patch(
            "beamz.devices.sources.mode_launch.bind_material_mode_plane", launch
        ):
            sim.updated_copy(sources=[src]).compile(backend="jax")
    assert len(launched) == 3
    for mode in launched:
        assert (
            mode.diagnostics["solver_padding_cells"]
            == source.mode_spec.solver_padding_cells
        )
        assert (
            mode.diagnostics["solver_bounds"] == preview.diagnostics[0]["solver_bounds"]
        )
    np.testing.assert_allclose(preview.neffs[0], [m.neff for m in launched], rtol=1e-9)
    # The short run supplies real stored monitor/material metadata; these checks
    # concern basis identity, not converged transmission or propagated power.
    result = sim.run(backend="jax", progress=False)
    projected = []

    def projection(*args, **kwargs):
        family = bind_material_mode_plane(*args, **kwargs)
        solve = family.solve

        class CapturedFamily:
            def solve(self, index):
                mode = solve(index)
                projected.append(mode)
                return mode

        return CapturedFamily()

    with patch("beamz.analysis.mode_projection.bind_material_mode_plane", projection):
        result.mode("review")
    assert len(projected) == 3
    for mode in projected:
        assert (
            mode.diagnostics["solver_bounds"] == preview.diagnostics[0]["solver_bounds"]
        )
    projected.sort(key=lambda m: m.diagnostics["mode_index"])
    np.testing.assert_allclose([m.neff for m in projected], preview.neffs[0], rtol=1e-9)
    # Public modal extraction uses the positive-direction basis and reports
    # both amplitudes. Reverse launch is its time-reversed, lossless partner.
    projected = [
        mode
        if mode.direction == launched[0].direction
        else replace(
            mode,
            profiles={
                n: (1 if n.startswith("E") else -1) * np.conj(v)
                for n, v in mode.profiles.items()
            },
        )
        for mode in projected
    ]
    assert all(
        discrete_overlap(a, b) > 0.999 for a, b in zip(launched, projected, strict=True)
    )


@pytest.mark.parametrize("padding", [None, 0, (2, 2)])
@pytest.mark.parametrize("outside", ["neighbor", "substrate"])
def test_preview_and_launch_share_material_window_and_select_wire(
    preview_fixture, outside, padding
):
    from scipy.ndimage import label

    sim, source = preview_fixture(num_modes=1, confinement=0.5)
    material = sim._material_grid()
    eps = np.full(material.shape, 1.444**2, dtype=np.float32)
    if outside == "neighbor":
        eps[18:22, 35:44, :] = 3.476**2
        eps[18:22, 50:60, :] = 3.476**2
        source = source.updated_copy(size=(0, 0.8e-6, 1.2e-6))
        expected_peak = (0.8e-6, 1.2e-6)
    else:
        eps[26:, :, :] = 1
        eps[:16, :, :] = 3.476**2
        eps[26:30, 35:45, :] = 3.476**2
        source = source.updated_copy(
            center=(0.1e-6, 2e-6, 1.4e-6), size=(0, 0.8e-6, 0.8e-6)
        )
        expected_peak = (1.2e-6, 1.6e-6)
    source = source.updated_copy(
        mode_spec=replace(source.mode_spec, solver_padding_cells=padding)
    )
    sim = sim.updated_copy(material_grid=replace(material, permittivity=eps))
    with patch.object(discrete, "solve_grid", wraps=discrete.solve_grid) as solve:
        preview = source.solve_modes(sim)
    preview_material = solve.call_args.kwargs["material_grid"].eps_tensor[0, 0]
    source = source.updated_copy(
        mode_spec=replace(source.mode_spec, target_neff=float(preview.neffs[0, 0].real))
    )
    with patch.object(discrete, "solve_grid", wraps=discrete.solve_grid) as solve:
        program = sim.updated_copy(sources=[source]).compile(backend="jax")
    np.testing.assert_array_equal(
        preview_material, solve.call_args.kwargs["material_grid"].eps_tensor[0, 0]
    )
    assert label(preview_material > 5)[1] == (2 if padding is None else 1)
    assert label(preview.eps_profiles[0] > 5)[1] == 1
    peaks = [
        sim.grid.z_edges[
            term.slab_starts[0]
            + np.unravel_index(
                np.argmax(abs(np.asarray(term.coeff))), term.coeff.shape
            )[0]
        ]
        for term in program.sources
        if term.component == "Ey"
    ]
    assert peaks and all(expected_peak[0] < z < expected_peak[1] for z in peaks)
