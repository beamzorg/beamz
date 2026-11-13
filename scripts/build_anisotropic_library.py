"""Rebuild the anisotropic crystal catalog from bundled sources and published models.

Run with the development dependencies installed:

    python scripts/build_anisotropic_library.py

Writes ``beamz/material_library/data/anisotropic_catalog.json``. Crystal frames:
uniaxial crystals put the optic axis on z; alpha-MoO3 uses x = [100],
y = [001], z = [010] as in its source. Rotate media with
``AnisotropicMedium.rotated`` to place them in a device frame.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from beamz.const import LIGHT_SPEED
from beamz.design.anisotropic import AnisotropicMedium, TensorPole

DATA = Path(__file__).resolve().parents[1] / "beamz/material_library/data"
REVISION = "c5c2f188e848453def5970e347399d653df2ffc2"
BASE = (
    "https://raw.githubusercontent.com/polyanskiy/refractiveindex.info-database/"
    f"{REVISION}/database/data/main/"
)
CM = 2.0 * np.pi * LIGHT_SPEED * 100.0  # rad/s per 1/cm

# key, name, variant, (ordinary stem, extraordinary stem), remote folder
SELLMEIER_SOURCES = (
    (
        "Al2O3",
        "Sapphire",
        "Malitson1972",
        ("Al2O3-Malitson-o", "Al2O3-Malitson-e"),
        "Al2O3/nk/Malitson-{}.yml",
    ),
    (
        "LiNbO3",
        "Lithium niobate (congruent)",
        "Zelmon1997",
        ("LiNbO3-Zelmon-o", "LiNbO3-Zelmon-e"),
        "LiNbO3/nk/Zelmon-{}.yml",
    ),
)


def sellmeier_terms(record):
    """(strength, resonance wavelength in m) pairs and constant of formula 1/2."""
    coefficients = np.fromstring(record["coefficients"], sep=" ")
    pairs = coefficients[1:].reshape(-1, 2)
    if record["type"] == "formula 1":
        wavelengths = pairs[:, 1] * 1e-6
    elif record["type"] == "formula 2":
        wavelengths = np.sqrt(pairs[:, 1]) * 1e-6
    else:
        raise ValueError(f"Unsupported source format: {record['type']}")
    return float(coefficients[0]), list(zip(pairs[:, 0], wavelengths, strict=True))


def sellmeier_eps(record, wavelengths):
    constant, terms = sellmeier_terms(record)
    eps = np.full(wavelengths.shape, 1.0 + constant)
    for strength, resonance in terms:
        eps += strength * wavelengths**2 / (wavelengths**2 - resonance**2)
    return eps


def uniaxial_sellmeier(key, name, variant, stems, remote):
    documents = [yaml.safe_load((DATA / f"{stem}.yml").read_text()) for stem in stems]
    records = [document["DATA"][0] for document in documents]
    poles = []
    static = []
    for axes, record in zip(((1.0, 1.0, 0.0), (0.0, 0.0, 1.0)), records, strict=True):
        constant, terms = sellmeier_terms(record)
        static.append(1.0 + constant)
        for strength, resonance in terms:
            omega = 2.0 * np.pi * LIGHT_SPEED / resonance
            poles.append(
                TensorPole.lorentz(
                    np.asarray(axes) * strength, resonance=omega, damping=0.0
                )
            )
    limits = [np.fromstring(r["wavelength_range"], sep=" ") * 1e-6 for r in records]
    shortest = max(limit[0] for limit in limits)
    longest = min(limit[1] for limit in limits)
    band = (LIGHT_SPEED / longest, LIGHT_SPEED / shortest)
    medium = AnisotropicMedium(
        (static[0], static[0], static[1]), electric_poles=poles, frequency_range=band
    )
    wavelengths = np.geomspace(shortest, longest, 401)
    tensor = medium.permittivity_tensor(LIGHT_SPEED / wavelengths).real
    error = max(
        float(np.max(np.abs(tensor[:, 0, 0] - sellmeier_eps(records[0], wavelengths)))),
        float(np.max(np.abs(tensor[:, 2, 2] - sellmeier_eps(records[1], wavelengths)))),
    )
    variant_record = dict(
        anisotropic_medium=medium.to_spec(),
        description=f"{name}, uniaxial with the optic axis on z "
        f"({shortest * 1e6:g}-{longest * 1e6:g} um).",
        source=[BASE + remote.format(ray) for ray in ("o", "e")],
        license="CC0-1.0",
        source_file=[f"{stem}.yml" for stem in stems],
        source_sha256=[
            hashlib.sha256((DATA / f"{stem}.yml").read_bytes()).hexdigest()
            for stem in stems
        ],
        references=documents[0]["REFERENCES"],
        conditions=" / ".join(d.get("COMMENTS", "") for d in documents),
        fit={
            "method": "Exact Sellmeier conversion to lossless tensor Lorentz poles",
            "max_abs_epsilon": error,
        },
    )
    return key, name, variant, variant_record


def to_lorentz(epsilon_inf, omega_to, omega_lo, gamma, axes):
    """Single TO-LO oscillator eps_inf (1 + (LO^2 - TO^2) / (TO^2 - w^2 - i g w))."""
    strength = epsilon_inf * (omega_lo**2 - omega_to**2) / omega_to**2
    return TensorPole.lorentz(
        np.asarray(axes) * strength, resonance=omega_to * CM, damping=gamma * CM
    )


def factorized_to_lorentz(epsilon_inf, oscillators, axes):
    """Exact lossless partial fractions of a factorized TO-LO product.

    eps_inf prod_j (LO_j^2 - w^2) / (TO_j^2 - w^2)
      = eps_inf [1 + sum_j A_j / (TO_j^2 - w^2)],
    A_j = (LO_j^2 - TO_j^2) prod_{k != j} (LO_k^2 - TO_j^2) / (TO_k^2 - TO_j^2).
    Each oscillator keeps its own damping.
    """
    poles = []
    for j, (to_j, lo_j, gamma_j) in enumerate(oscillators):
        residue = lo_j**2 - to_j**2
        for k, (to_k, lo_k, _) in enumerate(oscillators):
            if k != j:
                residue *= (lo_k**2 - to_j**2) / (to_k**2 - to_j**2)
        if residue <= 0:
            raise ValueError("Non-interleaved TO/LO pairs give a non-passive residue.")
        poles.append(
            TensorPole.lorentz(
                np.asarray(axes) * epsilon_inf * residue / to_j**2,
                resonance=to_j * CM,
                damping=gamma_j * CM,
            )
        )
    return poles


def factorized(epsilon_inf, oscillators, wavenumbers):
    value = np.full(wavenumbers.shape, epsilon_inf, dtype=complex)
    for to, lo, gamma in oscillators:
        value *= (lo**2 - wavenumbers**2 - 1j * gamma * wavenumbers) / (
            to**2 - wavenumbers**2 - 1j * gamma * wavenumbers
        )
    return value


# Caldwell et al., Nat. Commun. 5, 5221 (2014), Supplementary Table 1 and Methods
# Eq. (5). The printed table swaps its optical-axis labels relative to the main
# text and Supplementary Fig. 2; the physical assignment (in-plane upper band
# 6.2-7.3 um, c-axis lower band 12.1-13.2 um) is used here.
HBN = {"in_plane": (4.90, 1360.0, 1614.0, 7.0), "c_axis": (2.95, 760.0, 825.0, 2.0)}

# Alvarez-Perez et al., Adv. Mater. 32, 1908176 (2020), Eq. (2) and Table 1,
# correlative far/near-field (experimental) columns.
MOO3 = {
    "x": (5.78, ((506.7, 534.3, 49.1), (821.4, 963.0, 6.0), (998.7, 999.2, 0.35))),
    "y": (6.07, ((544.6, 850.1, 9.5),)),
    "z": (4.47, ((956.7, 1006.9, 1.5),)),
}


def hbn():
    e_t, to_t, lo_t, g_t = HBN["in_plane"]
    e_c, to_c, lo_c, g_c = HBN["c_axis"]
    band = (700.0 * LIGHT_SPEED * 100.0, 1700.0 * LIGHT_SPEED * 100.0)
    medium = AnisotropicMedium(
        (e_t, e_t, e_c),
        electric_poles=[
            to_lorentz(e_t, to_t, lo_t, g_t, (1.0, 1.0, 0.0)),
            to_lorentz(e_c, to_c, lo_c, g_c, (0.0, 0.0, 1.0)),
        ],
        frequency_range=band,
    )
    return dict(
        anisotropic_medium=medium.to_spec(),
        description="Hexagonal boron nitride, mid-IR phonon model (700-1700 1/cm); "
        "c axis on z. Hyperbolic in both Reststrahlen bands.",
        source="https://doi.org/10.1038/ncomms6221",
        license="Apache-2.0",
        references="J. D. Caldwell et al., Sub-diffractional volume-confined "
        "polaritons in the natural hyperbolic material hexagonal boron nitride, "
        "Nat. Commun. 5, 5221 (2014); Methods Eq. (5), Supplementary Table 1.",
        conditions="Published single-oscillator TO-LO model, independently "
        "implemented. Supplementary Table 1 prints the optical-axis labels "
        "swapped relative to the main text; the physical assignment is used "
        "(in-plane TO/LO 1360/1614 1/cm, c axis 760/825 1/cm). Apache-2.0 covers "
        "this implementation, not the publication.",
        fit={
            "method": "Exact TO-LO to Lorentz conversion; no numerical fitting",
            "in_plane": dict(
                zip(("eps_inf", "TO", "LO", "gamma"), HBN["in_plane"], strict=True)
            ),
            "c_axis": dict(
                zip(("eps_inf", "TO", "LO", "gamma"), HBN["c_axis"], strict=True)
            ),
            "units": "1/cm",
        },
    )


def moo3():
    axes = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}
    poles = []
    for name, (epsilon_inf, oscillators) in MOO3.items():
        poles += factorized_to_lorentz(epsilon_inf, oscillators, axes[name])
    band_cm = (450.0, 1100.0)
    medium = AnisotropicMedium(
        tuple(MOO3[a][0] for a in "xyz"),
        electric_poles=poles,
        frequency_range=tuple(v * LIGHT_SPEED * 100.0 for v in band_cm),
    )
    wavenumbers = np.linspace(*band_cm, 4001)
    tensor = medium.permittivity_tensor(wavenumbers * LIGHT_SPEED * 100.0)
    deviation = {}
    for i, name in enumerate("xyz"):
        reference = factorized(*MOO3[name], wavenumbers)
        error = np.abs(tensor[:, i, i] - reference)
        deviation[name] = {
            "max_abs": float(error.max()),
            "max_abs_over_peak": float(error.max() / np.abs(reference).max()),
            "median_relative": float(np.median(error / np.abs(reference))),
        }
    return dict(
        anisotropic_medium=medium.to_spec(),
        description="alpha-MoO3, biaxial mid/far-IR phonon model (450-1100 1/cm); "
        "x = [100], y = [001], z = [010]. In-plane hyperbolic.",
        source="https://doi.org/10.1002/adma.201908176",
        license="Apache-2.0",
        references="G. Alvarez-Perez et al., Infrared permittivity of the biaxial "
        "van der Waals semiconductor alpha-MoO3 from near- and far-field "
        "correlative studies, Adv. Mater. 32, 1908176 (2020); Eq. (2), Table 1 "
        "(experimental columns).",
        conditions="Published factorized TO-LO model, independently implemented. "
        "The x axis has three coupled oscillators with unequal damping in a "
        "factorized form, which a sum of Lorentz poles cannot reproduce exactly; "
        "lossless partial fractions with per-oscillator damping keep eps_x within "
        "about 1% (median) and 0.6% of its peak, with larger relative error only "
        "at its near-zero crossing by 999 1/cm. y and z are exact. Apache-2.0 "
        "covers this implementation, not the publication.",
        fit={
            "method": "Lossless partial-fraction conversion of the factorized model",
            "epsilon_deviation_vs_factorized": deviation,
        },
    )


def main():
    catalog = {}
    for spec in SELLMEIER_SOURCES:
        key, name, variant, record = uniaxial_sellmeier(*spec)
        catalog[key] = dict(name=name, default=variant, variants={variant: record})
        print(key, record["fit"])
    catalog["hBN"] = dict(
        name="Hexagonal boron nitride",
        default="Caldwell2014",
        variants={"Caldwell2014": hbn()},
    )
    catalog["MoO3"] = dict(
        name="alpha-molybdenum trioxide",
        default="AlvarezPerez2020",
        variants={"AlvarezPerez2020": moo3()},
    )
    print("MoO3", catalog["MoO3"]["variants"]["AlvarezPerez2020"]["fit"])
    (DATA / "anisotropic_catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")


if __name__ == "__main__":
    main()
