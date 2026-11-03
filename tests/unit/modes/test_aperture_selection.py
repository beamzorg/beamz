"""Spatial selection must distinguish a port guide from an outside structure."""

from dataclasses import replace

import numpy as np
import pytest

from beamz import ModeSpec
from beamz._cache_tokens import cache_token
from beamz.devices.modes import discrete
from beamz.devices.modes.discrete import ModePlaneSpec, _aperture_fraction, _sort_modes


def spec():
    return ModePlaneSpec(
        scalar_permittivity=np.ones((8, 8)),
        frequency=2e14,
        resolution=1e-7,
        dt=None,
        axis="x",
        direction="+x",
        transverse_axes=("z", "y"),
        grid_shape=(8, 8, 4),
        center=(2e-7, 4e-7, 4e-7),
        width=4e-7,
        height=4e-7,
        plane_index=1,
        offset_index=0,
        polarization="te",
        min_aperture_fraction=0.5,
    )


def candidate(neff, row, column, component="Ey"):
    fields = {"E" + a: np.zeros((8, 8), dtype=complex) for a in "xyz"}
    fields[component][row, column] = 1
    return {"neff": complex(neff), "fields": fields}


def test_confined_polarization_family_precedes_outside_higher_index_mode():
    wire = candidate(2.4, 4, 4)
    substrate = candidate(3.0, 0, 4)
    tm = candidate(3.2, 4, 4, "Ez")
    assert _aperture_fraction(wire, spec()) == 1
    assert _aperture_fraction(substrate, spec()) == 0
    assert _sort_modes([substrate, tm, wire], spec()) == [wire, substrate, tm]
    assert (
        _sort_modes([substrate, wire], replace(spec(), min_aperture_fraction=None))[0]
        is substrate
    )


def test_confined_modes_preserve_effective_index_order():
    modes = [candidate(2.0, 3, 3), candidate(2.5, 4, 4)]
    assert _sort_modes(modes, spec())[0] is modes[1]


@pytest.mark.parametrize(
    "options",
    [
        {"min_aperture_fraction": -0.1},
        {"min_aperture_fraction": 1.1},
        {"min_aperture_fraction": np.nan},
    ],
)
def test_selection_options_reject_invalid_values(options):
    with pytest.raises(ValueError):
        ModeSpec(**options)


def test_selection_options_change_cache_identity():
    assert cache_token(ModeSpec()) != cache_token(ModeSpec(min_aperture_fraction=0.5))


@pytest.mark.parametrize("find_wire", [False, True])
def test_candidate_search_expands_or_fails_explicitly(monkeypatch, find_wire):
    request = replace(
        spec(), scalar_permittivity=np.where(np.indices((8, 8))[0] == 4, 4, 1)
    )
    counts = []
    outside, wire = candidate(3.0, 0, 4), candidate(2.4, 4, 4)

    def solve(_cache, **options):
        counts.append(options["num_modes"])
        return object()

    monkeypatch.setattr(discrete, "_solve_candidate_pool", solve)
    monkeypatch.setattr(
        discrete,
        "_candidate_modes",
        lambda *_: [outside, wire] if find_wire and len(counts) > 1 else [outside],
    )

    def selected(_axis, _polarization, fields):
        assert fields is wire["fields"]
        raise RuntimeError("selected wire")

    monkeypatch.setattr(discrete, "_select_phase_reference_component", selected)
    if find_wire:
        with pytest.raises(RuntimeError, match="selected wire"):
            discrete.solve_beamz_mode(request)
        assert counts == [7, 14]
    else:
        with pytest.raises(
            ValueError, match="No requested mode meets min_aperture_fraction"
        ):
            discrete.solve_beamz_mode(request)
        assert counts == [7, 14, 28, 56, 64]


@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
@pytest.mark.parametrize("cache_kind", [None, "preview", "analysis"])
def test_expansion_assigns_unique_modes_from_one_final_family(
    monkeypatch, order, cache_kind
):
    request = replace(
        spec(),
        scalar_permittivity=np.where(np.indices((8, 8))[0] == 4, 4, 1),
        num_modes=2,
        target_neff=2.4,
    )
    a, b = candidate(2.4, 4, 4), candidate(2.8, 3, 3)
    counts = []

    def solve(**kwargs):
        counts.append(kwargs["num_modes"])
        return kwargs["num_modes"]

    monkeypatch.setattr(discrete, "solve_grid", solve)
    monkeypatch.setattr(
        discrete, "_candidate_modes", lambda count, _: [a] if count < 18 else [b, a]
    )
    monkeypatch.setattr(
        discrete,
        "_prepare_discrete_mode",
        lambda request, basis: basis.candidates[request.mode_index]["neff"],
    )
    discrete._CANDIDATE_CACHE.clear()
    try:
        options = (
            {"basis_cache": {}}
            if cache_kind == "preview"
            else {"candidate_cache": discrete._CANDIDATE_CACHE}
            if cache_kind == "analysis"
            else {}
        )
        selected = {
            i: discrete._solve_beamz_mode(replace(request, mode_index=i), **options)
            for i in order
        }
        assert selected == {0: 2.8, 1: 2.4}
        assert counts == ([9, 18] * (2 if cache_kind is None else 1))
    finally:
        discrete._CANDIDATE_CACHE.clear()


def test_missing_unused_member_does_not_invalidate_available_mode(monkeypatch):
    request = replace(
        spec(),
        scalar_permittivity=np.where(np.indices((8, 8))[0] == 4, 4, 1),
        num_modes=2,
    )
    monkeypatch.setattr(
        discrete, "_solve_candidate_pool", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(discrete, "_candidate_modes", lambda *_: [candidate(2.4, 4, 4)])

    def selected(*_args):
        raise RuntimeError("available mode")

    monkeypatch.setattr(discrete, "_select_phase_reference_component", selected)
    with pytest.raises(RuntimeError, match="available mode"):
        discrete.solve_beamz_mode(request)
    with pytest.raises(ValueError, match="No requested mode meets"):
        discrete.solve_beamz_mode(replace(request, mode_index=1))
