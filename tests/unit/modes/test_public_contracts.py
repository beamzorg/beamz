import numpy as np
import xarray as xr

from beamz.devices.modes import ModeData, ModeSpec, Result, solve_grid
from beamz.devices.modes.models import Grid


def test_native_mode_package_exposes_only_supported_api():
    from beamz.devices import modes

    assert modes.__all__ == [
        "Materials",
        "ModeData",
        "ModeSpec",
        "Result",
        "solve_grid",
    ]
    assert ModeSpec and ModeData and solve_grid


def test_result_is_a_minimal_labeled_solver_value():
    indices = xr.DataArray(
        np.asarray([[2.0 + 0.1j]]),
        dims=("f", "mode_index"),
        coords={"f": [1.0], "mode_index": [0]},
    )
    result = Result(
        np.asarray(indices),
        {},
        Grid((0.0, 1.0), (0.0, 1.0)),
        np.array([1.0]),
        solver_info={"backend": "test"},
    )

    np.testing.assert_allclose(result.n_eff, [[2.0]])
    np.testing.assert_allclose(result.k_eff, [[0.1]])
    assert result.solver_info == {"backend": "test"}


def test_solver_family_allocates_labeled_views_only_on_request(monkeypatch):
    from beamz.devices.modes.models import Materials

    epsilon = np.full((8, 7), 2.1)
    epsilon[2:6, 2:5] = 11.5
    materials = Materials.from_components(
        eps_xx=epsilon,
        x_edges=np.linspace(0, 1.2, 9),
        y_edges=np.linspace(0, 0.9, 8),
    )

    def unexpected_labeled_allocation(*args, **kwargs):
        raise AssertionError("Core mode solving must not allocate labeled field views")

    with monkeypatch.context() as context:
        context.setattr(xr, "DataArray", unexpected_labeled_allocation)
        family = solve_grid(material_grid=materials, freqs=[1.8e14], num_modes=2)
        assert family.grid is materials.grid
        assert family.fields["Ex"].shape == (1, 2, 8, 7)
    # Explicit public views retain labels and share the solved array storage.
    public = family.field_components["Ex"]
    assert public.dims == ("x", "y", "z", "f", "mode_index")
    assert np.shares_memory(public.values, family.fields["Ex"])
    assert family.field_components["Ex"] is public


def test_dense_fallback_keeps_complex_fields_for_propagating_and_evanescent_modes():
    from beamz.devices.modes import Materials

    materials = Materials.from_components(
        eps_xx=np.array([[3.0, 4.0], [5.0, 3.0]]),
        x_edges=[0.0, 0.2, 0.5],
        y_edges=[0.0, 0.3, 0.6],
    )
    family = solve_grid(material_grid=materials, freqs=[1.9e14], num_modes=7)
    assert np.any(family.neffs.imag > 0)
    assert np.any(family.neffs.real > 0)
    for fields in family.fields.values():
        assert fields.dtype == np.complex128
        assert np.all(np.isfinite(fields))
