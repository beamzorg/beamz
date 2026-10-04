"""Rebuild the visible-band catalog offline from bundled CC0 source snapshots.

Run with the development dependencies installed: python scripts/build_material_library.py
Only source formula 1 and tabulated nk are supported; other formats fail explicitly.
"""

from __future__ import annotations

import hashlib
import json
from io import StringIO
from pathlib import Path

import numpy as np

from beamz.const import LIGHT_SPEED
from beamz.design.dispersion import PoleResidue, fit_nk
from beamz.design.vector_fit import fit_nk_vector

DATA = Path(__file__).resolve().parents[1] / "beamz/material_library/data"
REVISION = "c5c2f188e848453def5970e347399d653df2ffc2"
BASE = f"https://raw.githubusercontent.com/polyanskiy/refractiveindex.info-database/{REVISION}/"
WAVELENGTHS = np.linspace(400e-9, 700e-9, 151)
BAND = (LIGHT_SPEED / 700e-9, LIGHT_SPEED / 400e-9)
SOURCES = (
    ("SiO2", "Silica", "Malitson1965", "SiO2-Malitson", "SiO2/nk/Malitson.yml", 0),
    (
        "SiN",
        "Silicon nitride",
        "Philipp1973",
        "Si3N4-Philipp",
        "Si3N4/nk/Philipp.yml",
        0,
    ),
    ("aSi", "Amorphous silicon", "Pierce1972", "Si-Pierce", "Si/nk/Pierce.yml", 3),
    ("Al", "Aluminum", "Rakic1995", "Al-Rakic", "Al/nk/Rakic.yml", 4),
)


def source_model(document, pole_count):
    (record,) = document["DATA"]
    if record["type"] == "formula 1":
        limits = np.fromstring(record["wavelength_range"], sep=" ") * 1e-6
        if limits[0] > WAVELENGTHS[0] or limits[1] < WAVELENGTHS[-1]:
            raise ValueError("Formula does not cover the requested wavelength band.")
        coefficients = np.fromstring(record["coefficients"], sep=" ")
        poles = []
        eps = np.full(WAVELENGTHS.shape, 1 + coefficients[0])
        for strength, resonance_um in coefficients[1:].reshape(-1, 2):
            resonance = 2 * np.pi * LIGHT_SPEED / (resonance_um * 1e-6)
            poles.extend(
                PoleResidue.lorentz(
                    1,
                    strength=strength,
                    resonance=resonance,
                    damping=0,
                    frequency_range=BAND,
                ).poles
            )
            eps += (
                strength
                * WAVELENGTHS**2
                / (WAVELENGTHS**2 - (resonance_um * 1e-6) ** 2)
            )
        samples = np.c_[WAVELENGTHS, np.sqrt(eps), np.zeros_like(eps)]
        return (
            PoleResidue(1 + coefficients[0], poles, frequency_range=BAND),
            samples,
            {
                "method": "Exact Sellmeier conversion",
                "rms_n": 0.0,
                "rms_k": 0.0,
            },
        )
    if record["type"] != "tabulated nk":
        raise ValueError(f"Unsupported source format: {record['type']}")
    raw = np.loadtxt(StringIO(record["data"]))
    raw[:, 0] *= 1e-6
    if (
        np.any(np.diff(raw[:, 0]) <= 0)
        or raw[0, 0] > WAVELENGTHS[0]
        or raw[-1, 0] < WAVELENGTHS[-1]
    ):
        raise ValueError(
            "Source must be ordered and cover the requested wavelength band."
        )
    n, k = (np.interp(WAVELENGTHS, raw[:, 0], raw[:, i]) for i in (1, 2))
    model, report = fit_nk(WAVELENGTHS, n, k, num_poles=pole_count, max_nfev=2000)
    if not report["optimizer_success"]:
        raise RuntimeError(f"Material fit did not converge: {report}")
    # Preserve measured samples within the band, plus interpolated endpoints.
    inside = raw[(raw[:, 0] > WAVELENGTHS[0]) & (raw[:, 0] < WAVELENGTHS[-1])]
    samples = np.vstack(
        ([WAVELENGTHS[0], n[0], k[0]], inside, [WAVELENGTHS[-1], n[-1], k[-1]])
    )
    report.update(
        method="Positive Lorentz fit to linearly interpolated n,k",
        interpolation_samples=len(WAVELENGTHS),
        oscillators=pole_count,
    )
    return model, samples, report


def record_variant(model, samples, filename, **metadata):
    np.savetxt(DATA / filename, samples, delimiter=",", fmt="%.16e")
    return dict(medium=model.to_spec(), nk_file=filename, **metadata)


# Synthetic design specifications, not measurements. Extinction transitions
# smoothly between 0.01 in the passband and 0.46 in the stopband over 30 nm.
FILTER_THICKNESS = 1e-6
FILTER_INDEX_TARGET = 1.45
FILTER_TRANSMISSION_TOLERANCE = 0.02  # absolute transmitted-power fraction
FILTER_INDEX_TOLERANCE = 0.01
FILTER_POLE_COUNT = 9


def filter_target(wavelengths, channel):
    """Independent raised-cosine RGB targets; wavelengths are in metres."""
    wavelengths = np.asarray(wavelengths)

    def transition(start, end):
        x = np.clip((wavelengths - start) / (end - start), 0, 1)
        return 0.5 - 0.5 * np.cos(np.pi * x)

    short = transition(470e-9, 500e-9)
    long = transition(600e-9, 630e-9)
    stop = {"red": 1 - long, "green": 1 - short + long, "blue": short}[channel]
    return np.full_like(wavelengths, FILTER_INDEX_TARGET), 0.01 + 0.45 * stop


def filter_model(channel):
    """Fit independently defined targets with passive vector fitting."""
    wavelengths = np.linspace(400e-9, 700e-9, 200)
    n, k = filter_target(wavelengths, channel)
    model, diagnostics = fit_nk_vector(
        wavelengths,
        n,
        k,
        max_poles=FILTER_POLE_COUNT,
        epsilon_inf=1.0,
        weights=(0.1, 1.9),
        num_iters=200,
        tolerance_rms=0.02,
    )
    # Assess a denser holdout grid, rather than accepting training residuals.
    holdout = np.linspace(400e-9, 700e-9, 3001)
    hn, hk = filter_target(holdout, channel)
    fitted = np.sqrt(model.eps_model(LIGHT_SPEED / holdout))
    passband = hk < 0.01000001
    delta_transmission = abs(
        np.exp(-4 * np.pi * fitted.imag * FILTER_THICKNESS / holdout)
        - np.exp(-4 * np.pi * hk * FILTER_THICKNESS / holdout)
    )
    max_transmission = float(delta_transmission[passband].max())
    max_full_transmission = float(delta_transmission.max())
    max_index = float(abs(fitted.real[passband] - hn[passband]).max())
    diagnostics.update(
        {
            "fitting_samples": len(wavelengths),
            "validation_samples": len(holdout),
            "thickness_m": FILTER_THICKNESS,
            "rms_n": float(np.sqrt(np.mean((fitted.real - hn) ** 2))),
            "rms_k": float(np.sqrt(np.mean((fitted.imag - hk) ** 2))),
            "max_passband_index_error": max_index,
            "max_passband_transmission_error": max_transmission,
            "max_transmission_error": max_full_transmission,
            "transmission_target_met": max_full_transmission
            <= FILTER_TRANSMISSION_TOLERANCE,
            "index_target_met": max_index <= FILTER_INDEX_TOLERANCE,
        }
    )
    return model, np.c_[wavelengths, n, k], diagnostics


# HORIBA Jobin Yvon, Technical Note 08 (September 2006), equation 9 and
# page 4. Only the published scalar parameters and mathematical equation are
# implemented here; no document, measured table or third-party code is bundled.
LORENTZ_SOURCE = "https://www.horiba.com/fileadmin/uploads/Scientific/Downloads/OpticalSchool_CN/TN/ellipsometer/Lorentz_Dispersion_Model.pdf"
LORENTZ_PARAMETERS = {
    "SiN": (2.320, 3.585, 6.495, 0.398),
    "aSi": (3.109, 17.68, 3.93, 1.92),
}


def published_lorentz_model(key):
    """Convert the published photon-energy equation directly into SI poles."""
    epsilon_inf, epsilon_static, energy_ev, damping_ev = LORENTZ_PARAMETERS[key]
    hbar_ev_s = 6.582119569e-16
    return PoleResidue.lorentz(
        epsilon_inf,
        strength=epsilon_static - epsilon_inf,
        resonance=energy_ev / hbar_ev_s,
        damping=damping_ev / hbar_ev_s,
        frequency_range=BAND,
    )


def main():
    import yaml

    catalog = {}
    for key, name, variant, stem, remote_path, count in SOURCES:
        path = DATA / f"{stem}.yml"
        document = yaml.safe_load(path.read_text())
        model, samples, report = source_model(document, count)
        prediction = np.sqrt(model.eps_model(LIGHT_SPEED / samples[:, 0]))
        report.update(
            max_abs_n=float(np.max(abs(prediction.real - samples[:, 1]))),
            max_abs_k=float(np.max(abs(prediction.imag - samples[:, 2]))),
        )
        record = record_variant(
            model,
            samples,
            f"{stem}.csv",
            description=f"{name}, visible band (400–700 nm).",
            source=BASE + "database/data/main/" + remote_path,
            license="CC0-1.0",
            source_file=path.name,
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            references=document["REFERENCES"],
            conditions=document.get("COMMENTS", ""),
            fit=report,
        )
        catalog[key] = dict(name=name, default=variant, variants={variant: record})
        print(key, report)
    for key in LORENTZ_PARAMETERS:
        model = published_lorentz_model(key)
        nk = np.sqrt(model.eps_model(LIGHT_SPEED / WAVELENGTHS))
        catalog[key]["variants"]["Horiba2006"] = record_variant(
            model,
            np.c_[WAVELENGTHS, nk.real, nk.imag],
            f"{key}-Horiba2006.csv",
            description="Published single-Lorentz model, independently implemented.",
            source=LORENTZ_SOURCE,
            license="Apache-2.0",
            references="HORIBA Jobin Yvon, Lorentz Dispersion Model, Technical Note 08, September 2006; equation 9, page 4 parameter table.",
            conditions="Analytic bulk model; sampled values are calculated, not measured. Apache-2.0 covers this implementation, not the source publication.",
            fit={
                "method": "Exact Lorentz conversion; no numerical fitting",
                "epsilon_inf": LORENTZ_PARAMETERS[key][0],
                "epsilon_static": LORENTZ_PARAMETERS[key][1],
                "resonance_energy_eV": LORENTZ_PARAMETERS[key][2],
                "damping_energy_eV": LORENTZ_PARAMETERS[key][3],
            },
        )
    filters = {}
    for channel in ("red", "green", "blue"):
        model, samples, report = filter_model(channel)
        filters[channel] = record_variant(
            model,
            samples,
            f"{channel}.csv",
            description=f"Synthetic {channel} filter fitted to smooth absorption targets.",
            source="Analytic definition in scripts/build_material_library.py",
            license="Apache-2.0",
            fit=report,
        )
        print(channel, report, flush=True)
    catalog["CMOS_RGB"] = dict(
        name="Synthetic RGB filters", default="green", variants=filters
    )
    (DATA / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")


if __name__ == "__main__":
    main()
