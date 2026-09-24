"""Interactive plotting preserves the scientific coordinates and display contract."""

import builtins

import matplotlib.pyplot as mpl
import numpy as np
import pytest

import beamz as bz
from beamz.analysis import get_plotting_backend, get_pyplot, plotting_backend
from beamz.analysis.plotting import (
    _add_field_colorbar,
    _add_patch,
    _draw_field_eps_overlay,
    plot_mode_field_components,
)

xy = pytest.importorskip("xy.pyplot")


@pytest.fixture(autouse=True)
def close_figures():
    yield
    mpl.close("all")
    xy.close("all")


def mode_data():
    fields = np.arange(18).reshape(1, 1, 3, 3, 2) * (1 + 2j)
    return bz.ModeData(
        frequencies=np.array([2e14]),
        neffs=np.array([[2.4 + 0j]]),
        e_fields=fields,
        h_fields=fields,
        eps_profiles=np.ones((1, 3, 2)),
        resolution=0.1 * bz.um,
        grid_edges=(
            np.array([-0.6, -0.2, 0, 0.5]) * bz.um,
            np.array([-1, -0.7, 0.1]) * bz.um,
        ),
        transverse_axes=("z", "y"),
    )


def test_backend_selection_restores_context_and_validates_axes():
    assert get_plotting_backend() == "matplotlib"
    with pytest.raises(RuntimeError), plotting_backend("xy"):
        assert get_pyplot() is xy
        raise RuntimeError("plot failed")
    assert get_plotting_backend() == "matplotlib"
    fig, ax = xy.subplots()
    result, _, _ = plot_mode_field_components(
        mode_data(), field_names=("Ey",), ax=ax, show=False
    )
    assert result is fig
    with pytest.raises(ValueError, match="supplied axes"):
        plot_mode_field_components(mode_data(), ax=ax, backend="matplotlib", show=False)
    with pytest.raises(ValueError, match="Unknown plotting backend"):
        plot_mode_field_components(mode_data(), backend="typo", show=False)


def test_optional_dependency_has_actionable_error(monkeypatch):
    original = builtins.__import__

    def without_xy(name, *args, **kwargs):
        if name == "xy.pyplot":
            raise ImportError("missing xy")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_xy)
    with pytest.raises(ImportError, match=r"beamz\[xy\]"):
        get_pyplot("xy")
    assert get_pyplot("matplotlib") is mpl


@pytest.mark.parametrize("val", ["real", "imag", "abs", "abs^2", "phase"])
def test_mode_fields_match_matplotlib_on_nonuniform_grid(val):
    modes = mode_data()
    mf, ma, mn = plot_mode_field_components(modes, val=val, show=False)
    xf, xa, xn = plot_mode_field_components(modes, val=val, backend="xy", show=False)
    np.testing.assert_allclose(mn, xn)
    for m, x in zip(ma.flat, xa.flat, strict=True):
        assert x.get_xlabel() == m.get_xlabel()
        assert x.get_ylabel() == m.get_ylabel()
        assert x.get_title() == m.get_title()
        np.testing.assert_allclose(x.get_xlim(), m.get_xlim())
        np.testing.assert_allclose(x.get_ylim(), m.get_ylim())
        # XY triangulates each physical cell into two constant-color triangles.
        mesh = x.collections[0]._entry
        np.testing.assert_allclose(
            mesh["source_z"], np.repeat(m.collections[0].get_array().ravel(), 2)
        )
        vertices = np.column_stack(mesh["args"]).reshape(-1, 3, 2)
        v = vertices[:, 1] - vertices[:, 0]
        w = vertices[:, 2] - vertices[:, 0]
        area = np.abs(v[:, 0] * w[:, 1] - v[:, 1] * w[:, 0]) / 2
        expected = np.outer(
            np.diff(modes.grid_edges[0] / bz.um), np.diff(modes.grid_edges[1] / bz.um)
        )
        np.testing.assert_allclose(area, np.repeat(expected.ravel() / 2, 2))
    assert "iframe" in xf._repr_html_()


def test_xy_transparent_material_overlay_keeps_physical_cells():
    fig, ax = xy.subplots()
    _draw_field_eps_overlay(
        ax,
        np.array([[1.0, 12.0], [12.0, 1.0]]),
        extent=(0, 3, 0, 4),
        x_edges=np.array([0.0, 1.0, 3.0]),
        y_edges=np.array([0.0, 1.0, 4.0]),
    )
    mesh = ax.collections[0]._entry
    assert len(mesh["source_z"]) == 4  # only the two core cells, two triangles each
    assert mesh["kwargs"]["opacity"] == 0.2
    assert "iframe" in fig._repr_html_()


def test_xy_polygon_hole_is_transparent():
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    fig, ax = xy.subplots()
    outer = Path.unit_rectangle().transformed(
        mpl.matplotlib.transforms.Affine2D().scale(3)
    )
    inner = Path.unit_rectangle().transformed(
        mpl.matplotlib.transforms.Affine2D().translate(1, 1)
    )
    _add_patch(ax, PathPatch(Path.make_compound_path(outer, inner), facecolor="red"))
    area = 0.0
    for entry in ax._entries:
        if entry.get("factory") == "triangle_mesh":
            points = np.column_stack(entry["args"]).reshape(-1, 3, 2)
            v, w = points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]
            area += np.sum(np.abs(v[:, 0] * w[:, 1] - v[:, 1] * w[:, 0])) / 2
    assert area == pytest.approx(8.0)
    assert "iframe" in fig._repr_html_()


@pytest.mark.parametrize(
    "extent, orientation", [((0, 1, 0, 3), "vertical"), ((0, 3, 0, 1), "horizontal")]
)
@pytest.mark.parametrize("extend", ["max", "both"])
def test_xy_field_colorbar_renders_with_units_and_clipping(extent, orientation, extend):
    fig, ax = xy.subplots()
    image = ax.imshow(
        np.array([[-2.0, 0.0], [1.0, 2.0]]),
        extent=extent,
        cmap="RdBu",
        vmin=-1.0,
        vmax=1.0,
    )
    _add_field_colorbar(fig, ax, image, label="Re(Ey) (V/um)", extend=extend)
    fig.tight_layout(pad=0.5)
    assert ax._colorbar["orientation"] == orientation
    assert ax._colorbar["label"] == "Re(Ey) (V/um)"
    assert ax._colorbar["domain"] == [-1.0, 1.0]
    assert ax._colorbar["extend"] == extend
    assert "iframe" in fig._repr_html_()
