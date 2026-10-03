"""Public material-library contracts and numerical regression checks."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import beamz as bz
from beamz import material_library
from beamz.material_library import MaterialItem


def test_library_defaults_are_native_immutable_media():
    for item in material_library.values():
        assert item.medium is item[item.default]
        for variant in item.variants.values():
            assert isinstance(variant.medium, bz.PoleResidue)
            assert variant.source.strip()
            assert variant.license
            lo, hi = variant.medium.frequency_range
            assert 0 < lo < hi
            eps = variant.medium.eps_model(np.geomspace(lo, hi, 200))
            assert np.isfinite(eps).all()
            # Native PoleResidue serialization remains usable by simulations.
            restored = bz.PoleResidue.from_spec(variant.medium.to_spec())
            assert restored == variant.medium
    with pytest.raises(TypeError):
        material_library["Al"] = material_library["SiO2"]
    with pytest.raises(TypeError):
        material_library["Al"].variants["new"] = material_library["Al"].variants[
            "Rakic1995"
        ]
    with pytest.raises(FrozenInstanceError):
        material_library["Al"].medium.epsilon_inf = 2


def test_missing_names_fail_instead_of_substituting_a_fit():
    with pytest.raises(KeyError):
        material_library["unknown"]
    with pytest.raises(KeyError):
        material_library["Al"]["unknown"]
    with pytest.raises(ValueError, match="Unknown default"):
        MaterialItem(name="invalid", variants={}, default="missing")


def test_aluminum_default_is_corrected_fit_and_cmos_is_explicit():
    # 550 nm values independently evaluated from the pinned upstream models.
    freq = bz.LIGHT_SPEED / 550e-9
    aluminum = material_library["Al"]
    assert aluminum.default == "Rakic1995"
    assert len(aluminum.medium.poles) == 5
    assert len(aluminum["Rakic1995_CMOS"].poles) == 4
    np.testing.assert_allclose(
        aluminum.medium.eps_model(freq), -42.715340017056114 + 13.341513067741175j
    )
    np.testing.assert_allclose(
        aluminum["Rakic1995_CMOS"].eps_model(freq),
        -21.901043344449196 + 74.8586471102467j,
    )


@pytest.mark.parametrize("channel", ["red", "green", "blue"])
def test_filter_data_use_si_units_and_include_fitting_baseline(
    channel, monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)  # Resources cannot depend on an examples directory.
    variant = material_library["CMOS_RGB"].variants[channel]
    samples = variant.nk_data
    assert samples is not None
    assert samples.shape == (200, 3)
    assert np.isfinite(samples).all()
    np.testing.assert_allclose(samples[:, 0].min(), 400e-9)
    np.testing.assert_allclose(samples[:, 0].max(), 700e-9)
    np.testing.assert_allclose(samples[:, 1], 1.45)
    np.testing.assert_allclose(samples[:, 2].min(), variant.k_offset)
    np.testing.assert_allclose(samples[:, 2].max(), 0.45 + variant.k_offset)
    samples[:] = 0
    fresh = variant.nk_data
    assert fresh is not None
    assert fresh[:, 0].min() > 0
    assert material_library["Al"].variants["Rakic1995"].nk_data is None
