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


@pytest.mark.parametrize(
    "key,tolerance", [("SiO2", 1e-12), ("SiN", 1e-12), ("aSi", 0.1), ("Al", 0.055)]
)
def test_physical_models_match_bundled_cc0_samples(
    key, tolerance, monkeypatch, tmp_path
):
    import hashlib
    import json
    from importlib.resources import files

    monkeypatch.chdir(tmp_path)
    item = material_library[key]
    variant = item.variants[item.default]
    samples = variant.nk_data
    assert samples is not None
    wl, n, k = samples.T
    np.testing.assert_allclose([wl.min(), wl.max()], [400e-9, 700e-9], rtol=1e-12)
    fitted = np.sqrt(variant.medium.eps_model(bz.LIGHT_SPEED / wl))
    np.testing.assert_allclose(fitted.real, n, atol=tolerance, rtol=0)
    np.testing.assert_allclose(fitted.imag, k, atol=tolerance, rtol=0)
    assert variant.license == "CC0-1.0"
    assert variant.references
    assert variant.fit is not None
    assert variant.fit["max_abs_n"] == pytest.approx(max(abs(fitted.real - n)))
    assert variant.fit["max_abs_k"] == pytest.approx(max(abs(fitted.imag - k)))
    data = files("beamz.material_library").joinpath("data")
    record = json.loads(data.joinpath("catalog.json").read_text())[key]["variants"][
        item.default
    ]
    assert (
        hashlib.sha256(data.joinpath(record["source_file"]).read_bytes()).hexdigest()
        == record["source_sha256"]
    )
    samples[:] = 0
    assert variant.nk_data[0, 0] > 0


def test_sellmeier_conversion_matches_independent_source_formula():
    # Formula coefficients from the bundled CC0 source; evaluate without poles.
    wavelengths_um = np.linspace(0.4, 0.7, 37)
    coefficients = {
        "SiO2": [(0.6961663, 0.0684043), (0.4079426, 0.1162414), (0.8974794, 9.896161)],
        "SiN": [(2.8939, 0.13967)],
    }
    for key, terms in coefficients.items():
        exact = 1 + sum(
            b * wavelengths_um**2 / (wavelengths_um**2 - c**2) for b, c in terms
        )
        actual = material_library[key].medium.eps_model(
            bz.LIGHT_SPEED / (wavelengths_um * 1e-6)
        )
        np.testing.assert_allclose(actual, exact, rtol=2e-15)


@pytest.mark.parametrize("channel", ["red", "green", "blue"])
def test_vector_filter_diagnostics_and_passivity(channel):
    from scripts.build_material_library import filter_target

    variant = material_library["CMOS_RGB"].variants[channel]
    wavelengths = np.linspace(400e-9, 700e-9, 3001)
    n, k = filter_target(wavelengths, channel)
    nk = np.sqrt(variant.medium.eps_model(bz.LIGHT_SPEED / wavelengths))
    passband = k < 0.01000001
    transmission = np.exp(-4 * np.pi * nk.imag * 1e-6 / wavelengths)
    target = np.exp(-4 * np.pi * k * 1e-6 / wavelengths)
    error = max(abs(transmission[passband] - target[passband]))
    assert variant.fit["max_transmission_error"] == pytest.approx(
        np.max(abs(transmission - target))
    )
    assert np.max(transmission[k > 0.45999]) < 0.01
    assert variant.fit["optimizer_success"]
    assert variant.fit["transmission_target_met"] == bool(
        np.max(abs(transmission - target)) <= 0.02
    )
    assert variant.fit["pole_entries"] <= 9
    assert variant.fit["weighted_rms_epsilon"] < 0.1
    assert variant.fit["weights"] == [0.1, 1.9]
    assert variant.fit["tolerance_met"] == bool(
        variant.fit["weighted_rms_epsilon"] <= 0.02
    )
    assert variant.fit["max_passband_transmission_error"] == pytest.approx(error)
    index_error = max(abs(nk.real[passband] - n[passband]))
    assert variant.fit["max_passband_index_error"] == pytest.approx(index_error)
    assert variant.fit["index_target_met"] == bool(index_error <= 0.01)
    # Include a dense grid outside the use band to detect narrow gain regions
    # that a logarithmic grid alone can miss.
    frequency = (
        bz.LIGHT_SPEED
        / 550e-9
        * np.unique(np.r_[np.geomspace(1e-7, 1e7, 50003), np.linspace(0.2, 4.5, 80021)])
    )
    epsilon = variant.medium.eps_model(frequency)
    assert np.isfinite(epsilon).all()
    assert np.min(epsilon.imag / np.maximum(1, abs(epsilon))) >= -1e-9
    assert variant.license == "Apache-2.0"
    samples = variant.nk_data
    assert samples is not None
    tn, tk = filter_target(samples[:, 0], channel)
    np.testing.assert_allclose(samples[:, 1], tn)
    np.testing.assert_allclose(samples[:, 2], tk)


def test_filter_specification_passbands_and_transitions():
    from scripts.build_material_library import filter_target

    wavelength = np.array([450, 485, 550, 615, 650]) * 1e-9
    expected = {
        "red": [0.46, 0.46, 0.46, 0.235, 0.01],
        "green": [0.46, 0.235, 0.01, 0.235, 0.46],
        "blue": [0.01, 0.235, 0.46, 0.46, 0.46],
    }
    for channel, target in expected.items():
        n, k = filter_target(wavelength, channel)
        np.testing.assert_allclose(n, 1.45)
        np.testing.assert_allclose(k, target)


@pytest.mark.parametrize(
    "key,params",
    [
        ("SiN", (2.320, 3.585, 6.495, 0.398)),
        ("aSi", (3.109, 17.68, 3.93, 1.92)),
    ],
)
def test_published_lorentz_variants_match_photon_energy_equation(key, params):
    # Direct evaluation of TN08 equation 9, independent of the pole conversion.
    epsilon_inf, epsilon_static, resonance, damping = params
    frequency = bz.LIGHT_SPEED / np.linspace(400e-9, 700e-9, 37)
    energy = 6.582119569e-16 * 2 * np.pi * frequency
    expected = epsilon_inf + (epsilon_static - epsilon_inf) * resonance**2 / (
        resonance**2 - energy**2 - 1j * damping * energy
    )
    variant = material_library[key].variants["Horiba2006"]
    np.testing.assert_allclose(
        variant.medium.eps_model(frequency), expected, rtol=2e-15
    )
    assert "Technical Note 08" in variant.references
    assert "not the source publication" in variant.conditions
