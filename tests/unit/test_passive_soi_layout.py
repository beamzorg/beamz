"""Passive-SOI layout generation must not change the caller's active PDK."""

import pytest

from tests.differential.passive_soi.common import (
    _gdsfactory,
    _generate_layout,
    generate_layout,
    load_passive_soi_case,
)


@pytest.fixture(autouse=True)
def require_gdsfactory():
    try:
        _gdsfactory()
    except ModuleNotFoundError as exc:
        # Skip only the absent optional dependency, not a broken installation.
        if exc.name != "gdsfactory":
            raise
        pytest.skip("Requires GDSFactory; install BeamZ with the gds extra.")


@pytest.mark.parametrize(
    "device",
    [
        "crossing",
        "directional_coupler",
        "mode_converter",
        "polarization_splitter_rotator",
        "ring_resonator",
    ],
)
@pytest.mark.parametrize("active", [False, True])
def test_layout_generation_restores_active_pdk(device, active):
    gf = _gdsfactory()
    previous = gf.pdk._ACTIVE_PDK
    caller_pdk = gf.Pdk(name="passive_soi_test_caller") if active else None
    gf.pdk._ACTIVE_PDK = caller_pdk
    _generate_layout.cache_clear()
    try:
        case = load_passive_soi_case(device)
        generate_layout(case)
        assert gf.pdk._ACTIVE_PDK is caller_pdk
        # A cached layout must preserve the caller's PDK too.
        generate_layout(case)
        assert gf.pdk._ACTIVE_PDK is caller_pdk
    finally:
        gf.pdk._ACTIVE_PDK = previous
        _generate_layout.cache_clear()


@pytest.mark.parametrize("failure", ["lookup", "construction"])
def test_layout_generation_restores_active_pdk_after_error(monkeypatch, failure):
    gf = _gdsfactory()
    previous = gf.pdk._ACTIVE_PDK
    caller_pdk = gf.Pdk(name="passive_soi_test_caller")
    caller_pdk.activate()
    _generate_layout.cache_clear()

    def broken_factory(**kwargs):
        raise RuntimeError("layout construction failed")

    try:
        if failure == "construction":
            monkeypatch.setattr(gf.components, "crossing", broken_factory)
            with pytest.raises(RuntimeError, match="layout construction failed"):
                _generate_layout("crossing", "{}")
        else:
            with pytest.raises(ValueError, match="has no generic component"):
                _generate_layout("missing_passive_soi_component", "{}")
        assert gf.get_active_pdk() is caller_pdk
    finally:
        gf.pdk._ACTIVE_PDK = previous
        _generate_layout.cache_clear()
