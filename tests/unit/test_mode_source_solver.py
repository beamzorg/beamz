import numpy as np
import pytest
import xarray as xr

from beamz import LIGHT_SPEED
from beamz.devices.modes.plane import solve_modes
from tests.mode_family_fixture import mode_family


def test_solve_modes_uses_one_native_solve_and_forwards_pml(monkeypatch):
    calls = []
    mode_count = 11
    dims = ("y", "z", "x", "f", "mode_index")
    values = np.ones((3, 4, 1, 1, mode_count), dtype=np.complex128)
    result = mode_family(
        n_complex=xr.DataArray(
            np.linspace(3.0, 1.0, mode_count)[None, :],
            dims=("f", "mode_index"),
        ),
        field_components={
            name: xr.DataArray(values, dims=dims)
            for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
        },
    )

    def fake_solve(**kwargs):
        calls.append(kwargs)
        return result

    monkeypatch.setattr("beamz.devices.modes.plane.solve_grid", fake_solve)

    neffs, electric, magnetic, axis = solve_modes(
        np.ones((4, 3)),
        omega=2 * np.pi,
        dL=1e-7,
        npml=2,
        m=3,
    )

    assert len(calls) == 1
    assert calls[0]["num_modes"] == 11
    assert calls[0]["pml"] == (2, 2)
    np.testing.assert_array_equal(
        calls[0]["material_grid"].eps_tensor[0, 0], np.ones((4, 3)).T
    )
    assert neffs.shape == (3,)
    assert electric.shape == magnetic.shape == (3, 3, 4, 3)
    assert axis == 0


def test_solve_modes_uses_one_signed_axis_for_normal_and_direction(monkeypatch):
    calls = []
    dims = ("x", "z", "y", "f", "mode_index")
    values = np.ones((4, 1, 1, 1, 7), dtype=np.complex128)
    result = mode_family(
        n_complex=xr.DataArray(
            np.linspace(3.0, 1.0, 7)[None, :],
            dims=("f", "mode_index"),
        ),
        field_components={
            name: xr.DataArray(values, dims=dims)
            for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
        },
    )

    def fake_solve(**kwargs):
        calls.append(kwargs)
        return result

    monkeypatch.setattr("beamz.devices.modes.plane.solve_grid", fake_solve)

    _, _, _, axis = solve_modes(
        np.ones(4),
        omega=2 * np.pi,
        dL=1e-7,
        direction="-y",
    )

    assert calls[0]["material_grid"].grid.normal_axis == axis == 1
    assert calls[0]["direction"] == "-"


def test_solve_modes_projects_fields_to_symmetric_material_parity(monkeypatch):
    dims = ("y", "z", "x", "f", "mode_index")
    y, z = np.meshgrid(np.arange(5.0), np.arange(4.0), indexing="ij")
    skewed = (1.0 + y + 2.0 * z)[..., None, None, None].astype(np.complex128)
    result = mode_family(
        n_complex=xr.DataArray([[2.0]], dims=("f", "mode_index")),
        field_components={
            name: xr.DataArray(
                skewed * (index + 1),
                dims=dims,
                coords={
                    "y": (np.arange(5) + 0.5) * 0.1,
                    "z": (np.arange(4) + 0.5) * 0.1,
                },
            )
            for index, name in enumerate(("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"))
        },
    )
    monkeypatch.setattr("beamz.devices.modes.plane.solve_grid", lambda **_: result)

    _, electric, magnetic, _ = solve_modes(
        np.ones((4, 5)),
        omega=2 * np.pi,
        dL=1e-7,
        m=1,
        force_symmetry=True,
    )

    for component in (*electric[0], *magnetic[0]):
        for axis in (0, 1):
            np.testing.assert_allclose(
                np.abs(component),
                np.flip(np.abs(component), axis=axis),
                rtol=0.0,
                atol=0.0,
            )


@pytest.mark.parametrize("force_symmetry", [False, True])
def test_solve_modes_leaves_2d_launch_line_profiles_unprojected(
    monkeypatch, force_symmetry
):
    dims = ("y", "z", "x", "f", "mode_index")
    profile = (1.0 + np.arange(5.0))[:, None, None, None, None]
    result = mode_family(
        n_complex=xr.DataArray([[2.0]], dims=("f", "mode_index")),
        field_components={
            name: xr.DataArray(profile * (index + 1), dims=dims)
            for index, name in enumerate(("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"))
        },
    )
    monkeypatch.setattr("beamz.devices.modes.plane.solve_grid", lambda **_: result)

    _, electric, magnetic, _ = solve_modes(
        np.ones(5),
        omega=2 * np.pi,
        dL=1e-7,
        m=1,
        force_symmetry=force_symmetry,
    )

    for index, component in enumerate((*electric[0], *magnetic[0])):
        np.testing.assert_allclose(component, (index + 1) * (1.0 + np.arange(5.0)))


def test_solve_modes_rejects_unknown_polarization():
    with pytest.raises(ValueError, match="filter_pol"):
        solve_modes(
            np.ones(4),
            omega=2 * np.pi,
            dL=1e-7,
            filter_pol="hybrid",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("direction", ["x", "forward-x", "", "+q"])
def test_solve_modes_rejects_malformed_signed_axis(direction):
    with pytest.raises(ValueError, match="direction must be one of"):
        solve_modes(
            np.ones(4),
            omega=2 * np.pi,
            dL=1e-7,
            direction=direction,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("axis", "xyz")
def test_solve_modes_preserves_beamz_plane_axes_for_polarization(axis):
    permittivity = np.full((40, 6), 1.44**2)
    permittivity[16:24, :] = 2.2**2

    neffs, _, _, solved_axis = solve_modes(
        permittivity,
        omega=2 * np.pi * LIGHT_SPEED / 1.55e-6,
        dL=80e-9,
        m=1,
        direction=f"+{axis}",
        filter_pol="te",
        target_neff=2.1,
    )

    assert solved_axis == "xyz".index(axis)
    assert neffs[0].real > 1.8
