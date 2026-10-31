"""Independent synthetic fields exercise physical mirrors without geometry files."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from beamz import ModeSpec
from beamz._cache_tokens import cache_token
from beamz.devices.modes import discrete
from beamz.devices.modes._symmetry import project_mode_symmetry
from beamz.devices.modes.plane import solve_modes
from beamz.lattice import component_shape_3d
from tests.mode_family_fixture import mode_family

COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


def synthetic_result(edges, skew=0.3):
    z, y = ((e[:-1] + e[1:]) / 2 for e in edges)
    zz, yy = np.meshgrid(z, y, indexing="ij")
    envelope = np.exp(
        -(((yy - y.mean()) / np.ptp(y)) ** 2) - ((zz - z.mean()) / np.ptp(z)) ** 2
    )
    envelope *= 1 + skew * (yy - y.mean()) / np.ptp(y)
    fields = {n: np.zeros_like(envelope, dtype=complex) for n in COMPONENTS}
    fields["Ey"] = envelope.astype(complex)
    fields["Hz"] = envelope.astype(complex) / 377
    return mode_family(
        n_complex=xr.DataArray([[2.0 + 0j]], dims=("f", "mode_index")),
        field_components={
            name: xr.DataArray(
                value.T[..., None, None, None],
                dims=("y", "z", "x", "f", "mode_index"),
                coords={"z": z / 1e-6, "y": y / 1e-6},
            )
            for name, value in fields.items()
        },
        solver_info={},
    ), fields


def preview(monkeypatch, eps, edges, **options):
    result, original = synthetic_result(edges)
    monkeypatch.setattr("beamz.devices.modes.plane.solve_grid", lambda **kw: result)
    diagnostics = {}
    _, electric, magnetic, _ = solve_modes(
        eps,
        omega=2 * np.pi * 2e14,
        dL=1e-7,
        grid_edges=edges,
        diagnostics=diagnostics,
        **options,
    )
    return (
        dict(zip(COMPONENTS, (*electric[0], *magnetic[0]), strict=True)),
        original,
        diagnostics,
    )


def test_default_preserves_displaced_inclusion_and_field_centroid(monkeypatch):
    edges = (np.linspace(0, 1e-6, 12), np.linspace(0, 1e-6, 14))
    eps = np.full((11, 13), 100.0)
    eps[3, 4] += 1
    actual, original, diag = preview(monkeypatch, eps, edges)
    for name in COMPONENTS:
        np.testing.assert_array_equal(actual[name], original[name])
    y = (edges[1][:-1] + edges[1][1:]) / 2

    def centroid(field):
        power = abs(field) ** 2
        return np.sum(power * y) / power.sum()

    assert centroid(actual["Ey"]) == centroid(original["Ey"])
    assert not diag["force_symmetry_requested"]
    assert not diag["force_symmetry_applied"]


@pytest.mark.parametrize("count", [12, 13])
def test_default_does_not_switch_at_old_correlation_threshold(monkeypatch, count):
    edges = (np.linspace(0, 1e-6, count + 1), np.linspace(0, 1e-6, 16))
    ratio = (1 - 0.995) / 0.995
    critical = ratio + np.sqrt(ratio**2 + ratio * count * 15)
    correlations = []
    for factor in (0.98, 1.02):
        eps = np.ones((count, 15))
        eps[2, 4] += critical * factor
        correlations.append(np.sum(eps * eps[:, ::-1]) / np.sum(eps**2))
        actual, original, _ = preview(monkeypatch, eps, edges)
        np.testing.assert_array_equal(actual["Ey"], original["Ey"])
    assert correlations[0] > 0.995 > correlations[1]


@pytest.mark.parametrize("graded", [False, True])
def test_opt_in_uses_physical_mirrors_on_symmetric_grids(monkeypatch, graded):
    edge = (
        np.array([0, 0.1, 0.3, 0.6, 0.8, 0.9]) * 1e-6
        if graded
        else np.linspace(0, 0.9e-6, 6)
    )
    actual, original, diag = preview(
        monkeypatch, np.ones((5, 5)), (edge, edge), force_symmetry=True
    )
    assert diag["force_symmetry_requested"] and diag["force_symmetry_applied"]
    assert diag["force_symmetry_axes"] == ("z", "y")
    np.testing.assert_allclose(
        actual["Ey"], (original["Ey"] + original["Ey"][:, ::-1]) / 2
    )
    np.testing.assert_allclose(actual["Hz"] * 377, actual["Ey"])
    assert not np.array_equal(actual["Ey"], original["Ey"])


def test_opt_in_rejects_nonmirror_graded_grid_and_wrong_center(monkeypatch):
    edge = np.array([0, 0.1, 0.3, 0.6, 0.7, 0.9]) * 1e-6
    actual, original, diag = preview(
        monkeypatch, np.ones((5, 5)), (edge, edge), force_symmetry=True
    )
    assert not diag["force_symmetry_applied"]
    np.testing.assert_array_equal(actual["Ey"], original["Ey"])
    edge = np.linspace(0, 1e-6, 6)
    _, _, diag = preview(
        monkeypatch,
        np.ones((5, 5)),
        (edge, edge),
        force_symmetry=True,
        mirror_center=(0.45e-6, 0.45e-6),
    )
    assert not diag["force_symmetry_applied"]


@pytest.mark.parametrize("tensor", ["eps_yy", "eps_zz", "mu_xx", "mu_yy", "mu_zz"])
def test_opt_in_checks_each_complex_material_tensor(monkeypatch, tensor):
    edge = np.linspace(0, 1e-6, 6)
    material = np.ones((5, 5), complex)
    material[1, 1] += 0.01j
    _, _, diag = preview(
        monkeypatch,
        np.ones((5, 5)),
        (edge, edge),
        force_symmetry=True,
        **{tensor: material},
    )
    assert not diag["force_symmetry_applied"]
    assert all(
        "material" in reason for reason in diag["force_symmetry_skipped"].values()
    )


def test_opt_in_rejects_displaced_inclusion_even_with_large_background(monkeypatch):
    edge = np.linspace(0, 1e-6, 6)
    eps = np.full((5, 5), 100.0)
    eps[1, 1] += 0.01
    actual, original, diag = preview(
        monkeypatch, eps, (edge, edge), force_symmetry=True
    )
    assert not diag["force_symmetry_applied"]
    np.testing.assert_array_equal(actual["Ey"], original["Ey"])


def test_staggered_component_must_have_mirror_closed_coordinates():
    edge = np.linspace(0, 1, 6)
    centers = (edge[:-1] + edge[1:]) / 2
    # Same array shape, different physical supports: this must reject a mirror.
    fields = {"Ey": np.ones((5, 5)), "Hz": np.ones((5, 5))}
    supports = {"Ey": (centers, centers), "Hz": (centers, edge[:-1])}
    _, diag = project_mode_symmetry(
        fields,
        requested=True,
        edges=(edge, edge),
        axes=("z", "y"),
        center=(0.5, 0.5),
        materials=[np.ones((5, 5))],
        supports=supports,
    )
    assert diag["force_symmetry_axes"] == ("z",)
    assert "component supports" in diag["force_symmetry_skipped"]["y"]


def discrete_fixture():
    size, resolution = 7, 1e-7
    eps = np.ones((size, size))
    eps[2:5, 2:5] = 4
    shape = (size, size, 4)
    return discrete.ModePlaneSpec(
        eps,
        2e14,
        resolution,
        None,
        "x",
        "+x",
        ("z", "y"),
        shape,
        (2e-7, 0.35e-6, 0.35e-6),
        0.5e-6,
        0.5e-6,
        2,
        1,
        component_permittivity={
            n: np.ones(component_shape_3d(n, shape)) for n in COMPONENTS[:3]
        },
        component_permeability={
            n: np.ones(component_shape_3d(n, shape)) for n in COMPONENTS[3:]
        },
        aperture_window_alpha=0,
    )


@pytest.mark.parametrize("force_symmetry", [False, True])
@pytest.mark.parametrize("accepted", [False, True])
def test_discrete_projection_refinement_and_normalization_remain_separate(
    monkeypatch, force_symmetry, accepted
):
    spec = replace(discrete_fixture(), force_symmetry=force_symmetry)
    edge = np.arange(8) * spec.resolution
    result, _ = synthetic_result((edge, edge))
    monkeypatch.setattr(discrete, "solve_grid", lambda **kw: result)
    seeds = []

    def refine(profiles, *args, **kw):
        seeds.append({n: v.copy() for n, v in profiles.items()})
        return {n: 2 * v for n, v in profiles.items()}, 1e-8, 1.0, kw["k_num"], 1.0

    monkeypatch.setattr(discrete, "refine_x_mode_at_fixed_beta", refine)
    monkeypatch.setattr(
        discrete,
        "validate_x_mode_refinement",
        lambda *a, **kw: (
            accepted,
            {"rejection_reason": "" if accepted else "synthetic rejection"},
        ),
    )
    mode = discrete.solve_beamz_mode(spec)
    assert len(seeds) == 1
    assert mode.diagnostics["force_symmetry_requested"] is force_symmetry
    assert mode.diagnostics["force_symmetry_applied"] is force_symmetry
    assert mode.diagnostics["yee_refinement_accepted"] is accepted
    assert mode.diagnostics["yee_residual"] == 1e-8
    assert mode.diagnostics["power_after_phase_reference"] == pytest.approx(
        1.0, abs=1e-12
    )
    assert np.isfinite(mode.power_scale)
    assert all(np.all(np.isfinite(v)) for v in mode.profiles.values())
    # A rejected candidate restores exactly the seed; an accepted rescaling
    # has the same fields after the independent unit-power normalization.
    reference = discrete.solve_beamz_mode(replace(spec, yee_refinement=False))
    for name in COMPONENTS:
        np.testing.assert_allclose(
            mode.profiles[name], reference.profiles[name], atol=1e-12
        )


def test_immutable_specs_and_cache_tokens_include_option():
    assert ModeSpec().force_symmetry is False
    assert ModeSpec() != ModeSpec(force_symmetry=True)
    spec = discrete_fixture()
    assert cache_token(spec) != cache_token(replace(spec, force_symmetry=True))
    with pytest.raises(TypeError, match="force_symmetry"):
        ModeSpec(force_symmetry="false")


@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize("force_symmetry", [False, True])
@pytest.mark.parametrize("reject_refinement", [False, True])
def test_real_symmetric_guide_preserves_maxwell_residual_and_unit_power(
    monkeypatch, force_symmetry, reject_refinement, axis
):
    from beamz.lattice import sample_voxel_grid_at_e_component_3d_centered

    size = 11
    eps = np.ones((size, size))
    eps[3:8, 3:8] = 12
    shape = (size, size, 4) if axis == "x" else (size, 4, size)
    voxel = np.broadcast_to(eps[..., None] if axis == "x" else eps[:, None, :], shape)
    spec = replace(
        discrete_fixture(),
        scalar_permittivity=eps,
        grid_shape=shape,
        axis=axis,
        direction="+" + axis,
        solver_direction="+" + axis,
        transverse_axes=("z", "y" if axis == "x" else "x"),
        center=(2e-7, size * 0.5e-7, size * 0.5e-7)
        if axis == "x"
        else (size * 0.5e-7, 2e-7, size * 0.5e-7),
        width=1.1e-6,
        height=1.1e-6,
        target_neff=2.5,
        force_symmetry=force_symmetry,
        component_permittivity={
            n: sample_voxel_grid_at_e_component_3d_centered(
                voxel, n, stored_shape=component_shape_3d(n, shape)
            )
            for n in COMPONENTS[:3]
        },
        component_permeability={
            n: np.ones(component_shape_3d(n, shape)) for n in COMPONENTS[3:]
        },
    )
    validate = discrete.validate_x_mode_refinement
    validations = []

    def validation(*args, **kwargs):
        accepted, diagnostics = validate(*args, **kwargs)
        validations.append(accepted)
        assert accepted, diagnostics
        if reject_refinement:
            return False, {
                **diagnostics,
                "rejection_reason": "test rejection after physical validation",
            }
        return accepted, diagnostics

    monkeypatch.setattr(discrete, "validate_x_mode_refinement", validation)
    mode = discrete.solve_beamz_mode(spec)
    assert mode.diagnostics["force_symmetry_applied"] is force_symmetry
    assert mode.diagnostics["force_symmetry_axes"] == (
        ("z", "y" if axis == "x" else "x") if force_symmetry else ()
    )
    assert validations == [True]
    assert mode.diagnostics["yee_refinement_attempted"]
    assert mode.diagnostics["yee_refinement_accepted"] is not reject_refinement
    assert mode.diagnostics["yee_residual"] < 1e-10
    assert mode.diagnostics["power_after_phase_reference"] == pytest.approx(
        1, abs=1e-12
    )
    if reject_refinement:
        seed = discrete.solve_beamz_mode(replace(spec, yee_refinement=False))
        for name in COMPONENTS:
            np.testing.assert_allclose(
                mode.profiles[name], seed.profiles[name], atol=1e-10
            )


def test_projection_cache_distinguishes_public_symmetry_option(monkeypatch):
    from beamz.analysis import mode_projection as mp

    monkeypatch.setattr(mp, "_mode_components_for_port", lambda spec: {})
    monkeypatch.setattr(
        mp,
        "_build_port_projection_2d",
        lambda sim, **kw: {"requested": kw["monitor"].mode_spec.force_symmetry},
    )
    cache = {}
    sim = SimpleNamespace(is_3d=False)
    port = SimpleNamespace(name="port", mode_spec=ModeSpec())
    for flag in (False, True, False):
        monitor = SimpleNamespace(
            name="monitor", mode_spec=ModeSpec(force_symmetry=flag)
        )
        result = mp._build_port_projection(sim, port, monitor, 2e14, cache)
        assert result["requested"] is flag
    assert len(cache) == 2


def test_public_preview_records_immutable_processing_diagnostics(monkeypatch):
    from beamz.devices.sources import solve

    edges = (np.linspace(0, 1e-6, 6),) * 2
    result, _ = synthetic_result(edges)
    monkeypatch.setattr("beamz.devices.modes.plane.solve_grid", lambda **kw: result)
    context = SimpleNamespace(
        center=(0.0, 0.5e-6, 0.5e-6),
        axis="x",
        eps_profile_full=np.ones((5, 5)),
        diagonal_permittivity={},
        diagonal_permeability={},
        crop_slices=(slice(None), slice(None)),
        profile_edges=edges,
        transverse_axes=("z", "y"),
    )
    monkeypatch.setattr(solve, "mode_plane_context", lambda **kw: context)
    data = solve.solve_mode_plane(
        simulation=SimpleNamespace(
            resolution=1e-7, _material_grid=lambda: SimpleNamespace(shape=(5, 5))
        ),
        plane=None,
        mode_spec=ModeSpec(force_symmetry=True),
        freqs=[2e14],
    )
    assert data.diagnostics[0]["force_symmetry_requested"]
    assert data.diagnostics[0]["force_symmetry_applied"]
    assert (
        data.diagnostics[0]["processing_stage"]
        == "solver_fields_before_yee_interpolation_and_refinement"
    )
    with pytest.raises(TypeError):
        data.diagnostics[0]["force_symmetry_applied"] = False
