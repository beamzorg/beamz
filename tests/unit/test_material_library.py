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


def _source_formula(text):
    """Formula type and coefficients from a refractiveindex.info YAML page."""
    import re

    kind = re.search(r"type:\s*(formula \d)", text).group(1)
    values = np.array(
        re.search(r"coefficients:\s*([^\n]+)", text).group(1).split(), float
    )
    return kind, values


def _sellmeier(kind, values, wavelength_um):
    eps = np.full(wavelength_um.shape, 1.0 + values[0])
    for strength, resonance in values[1:].reshape(-1, 2):
        squared = resonance**2 if kind == "formula 1" else resonance
        eps += strength * wavelength_um**2 / (wavelength_um**2 - squared)
    return eps


def test_anisotropic_library_media_are_passive_and_serializable():
    from beamz.material_library import anisotropic_library

    assert set(anisotropic_library) == {"Al2O3", "LiNbO3", "hBN", "MoO3"}
    for item in anisotropic_library.values():
        assert item.medium is item[item.default]
        for variant in item.variants.values():
            medium = variant.medium
            assert isinstance(medium, bz.AnisotropicMedium)
            assert variant.source.strip() and variant.license
            restored = bz.AnisotropicMedium.from_spec(medium.to_spec())
            assert restored.cache_spec() == medium.cache_spec()
            lo, hi = medium.frequency_range
            tensor = medium.permittivity_tensor(np.geomspace(lo, hi, 300))
            # Passive: the anti-Hermitian part is positive semidefinite.
            loss = (tensor - np.conj(np.swapaxes(tensor, -1, -2))) / 2j
            assert np.linalg.eigvalsh(loss).min() > -1e-12


@pytest.mark.parametrize(
    "key,stems", [("Al2O3", "Al2O3-Malitson"), ("LiNbO3", "LiNbO3-Zelmon")]
)
def test_uniaxial_crystals_match_bundled_sellmeier_sources(key, stems):
    import hashlib
    import json
    from importlib.resources import files

    from beamz.material_library import anisotropic_library

    data = files("beamz.material_library").joinpath("data")
    catalog = json.loads(data.joinpath("anisotropic_catalog.json").read_text())
    record = next(iter(catalog[key]["variants"].values()))
    medium = anisotropic_library[key].medium
    lo, hi = medium.frequency_range
    wavelength = bz.LIGHT_SPEED / np.geomspace(lo, hi, 200)
    tensor = medium.permittivity_tensor(bz.LIGHT_SPEED / wavelength).real
    for ray, index in (("o", (0, 0)), ("e", (2, 2))):
        name = f"{stems}-{ray}.yml"
        raw = data.joinpath(name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() in record["source_sha256"]
        expected = _sellmeier(*_source_formula(raw.decode()), wavelength * 1e6)
        np.testing.assert_allclose(tensor[(slice(None), *index)], expected, rtol=1e-12)
    np.testing.assert_allclose(tensor[:, 0, 0], tensor[:, 1, 1])


def test_hbn_follows_caldwell_model_and_is_hyperbolic_in_both_bands():
    from beamz.material_library import anisotropic_library

    medium = anisotropic_library["hBN"].medium

    def caldwell(wavenumber, eps_inf, to, lo, gamma):
        return eps_inf * (
            1 + (lo**2 - to**2) / (to**2 - wavenumber**2 - 1j * gamma * wavenumber)
        )

    wavenumber = np.linspace(700, 1700, 501)
    tensor = medium.permittivity_tensor(wavenumber * bz.LIGHT_SPEED * 100)
    np.testing.assert_allclose(
        tensor[:, 0, 0], caldwell(wavenumber, 4.90, 1360, 1614, 7), rtol=1e-10
    )
    np.testing.assert_allclose(
        tensor[:, 2, 2], caldwell(wavenumber, 2.95, 760, 825, 2), rtol=1e-10
    )
    # Main text: Re eps_t < 0 < Re eps_z in the upper band, reversed in the lower.
    upper = medium.permittivity_tensor(1450 * bz.LIGHT_SPEED * 100).real
    lower = medium.permittivity_tensor(790 * bz.LIGHT_SPEED * 100).real
    assert upper[0, 0] < 0 < upper[2, 2]
    assert lower[2, 2] < 0 < lower[0, 0]


def test_moo3_tracks_factorized_alvarez_perez_model():
    from beamz.material_library import anisotropic_library

    medium = anisotropic_library["MoO3"].medium
    model = {
        0: (5.78, ((506.7, 534.3, 49.1), (821.4, 963.0, 6.0), (998.7, 999.2, 0.35))),
        1: (6.07, ((544.6, 850.1, 9.5),)),
        2: (4.47, ((956.7, 1006.9, 1.5),)),
    }
    wavenumber = np.linspace(450, 1100, 2001)
    tensor = medium.permittivity_tensor(wavenumber * bz.LIGHT_SPEED * 100)
    for axis, (eps_inf, oscillators) in model.items():
        reference = np.full(wavenumber.shape, eps_inf, dtype=complex)
        for to, lo, gamma in oscillators:
            reference *= (lo**2 - wavenumber**2 - 1j * gamma * wavenumber) / (
                to**2 - wavenumber**2 - 1j * gamma * wavenumber
            )
        error = np.abs(tensor[:, axis, axis] - reference)
        if len(oscillators) == 1:
            np.testing.assert_allclose(tensor[:, axis, axis], reference, rtol=1e-10)
        else:
            assert np.median(error / np.abs(reference)) < 0.01
            assert error.max() < 0.01 * np.abs(reference).max()
