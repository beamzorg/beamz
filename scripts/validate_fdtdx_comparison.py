#!/usr/bin/env python3
"""Independent, fail-closed numerical gate for the frozen common-data benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def compare(a, b, *, interior=0):
    if a.shape != b.shape:
        raise ValueError(f"Shape differs: {a.shape} != {b.shape}")
    if interior:
        a = a[:, interior:-interior, interior:-interior, interior:-interior]
        b = b[:, interior:-interior, interior:-interior, interior:-interior]
    square_error = square_a = square_b = 0.0
    max_error = 0.0
    for begin in range(0, a.shape[-1], 16):
        av = np.asarray(
            a[..., begin : begin + 16],
            dtype=np.complex128
            if np.iscomplexobj(a) or np.iscomplexobj(b)
            else np.float64,
        )
        bv = np.asarray(b[..., begin : begin + 16], dtype=av.dtype)
        if not (np.isfinite(av).all() and np.isfinite(bv).all()):
            raise ValueError("Nonfinite validation output")
        diff = av - bv
        square_error += float(np.vdot(diff.ravel(), diff.ravel()).real)
        square_a += float(np.vdot(av.ravel(), av.ravel()).real)
        square_b += float(np.vdot(bv.ravel(), bv.ravel()).real)
        max_error = max(max_error, float(np.max(np.abs(diff), initial=0)))
    norm = max(square_a, square_b) ** 0.5
    rel = (
        square_error**0.5 / norm
        if norm
        else (0.0 if not square_error else float("inf"))
    )
    return {
        "relative_l2": rel,
        "max_abs_error": max_error,
        "beamz_l2": square_a**0.5,
        "fdtdx_l2": square_b**0.5,
    }


def reference_witness(reference_root, contract, full_reports):
    """Revalidate a hash-bound, identical-input prefix with appreciable signal."""
    root = Path(reference_root)
    device_identity = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
        text=True,
    ).strip()
    if (
        json.loads((root / "device-identity.json").read_text())["device_identity"]
        != device_identity
    ):
        raise ValueError("Signal witness belongs to a different GPU/driver")
    campaign = json.loads((root / "comparison.json").read_text())
    matches = []
    for index, case in enumerate(campaign["cases"]):
        path = Path(case["contract"])
        candidate = json.loads(path.read_text())
        if candidate["shape_zyx"] == contract["shape_zyx"]:
            matches.append((index, case, path, candidate))
    if len(matches) != 1:
        raise ValueError("Need exactly one strong-signal witness for this shape")
    index, case, path, candidate = matches[0]
    if case["status"] not in ("validated", "passed"):
        raise ValueError("Reference campaign did not validate")
    for key in (
        "dx_m",
        "dt_s",
        "pml_cells",
        "component_shapes",
        "monitors",
        "cpml",
        "frequency_hz",
        "constants",
        "sources",
    ):
        if candidate[key] != contract[key]:
            raise ValueError(f"Reference changes common geometry/operators: {key}")
    if not 0 < candidate["steps"] < contract["steps"]:
        raise ValueError("Reference must be a strict initial segment")
    # Freeze the witness to a physical-time window, rather than selecting a
    # favorable reference from arbitrary late or amplified fields.
    if not 9e-15 <= candidate["steps"] * candidate["dt_s"] <= 10e-15:
        raise ValueError("Reference must end at the prescribed 9–10 fs window")
    with (
        np.load(path.parent / candidate["data_file"], allow_pickle=False) as short,
        np.load(
            Path(contract["_path"]).parent / contract["data_file"], allow_pickle=False
        ) as full,
    ):
        for group in ("electric_coefficients", "cpml", "monitors"):
            if (
                candidate["canonical"]["data_hashes"][group]
                != contract["canonical"]["data_hashes"][group]
            ):
                raise ValueError(f"Reference changes {group}")
        for source in contract["sources"]:
            for key in ("coeff", "waveform"):
                a, b = short[source[key]], full[source[key]]
                if key == "waveform":
                    b = b[: candidate["steps"]]
                if not np.array_equal(a, b):
                    raise ValueError("Reference source is not an exact input prefix")
    directory = root / f"case-{index:02d}"
    for engine, full_report in zip(("beamz", "fdtdx"), full_reports, strict=True):
        witness_report = json.loads(
            (directory / f"{engine}-validation.json").read_text()
        )
        for key in ("engine", "adapter_sha256"):
            if not full_report.get(key) or witness_report.get(key) != full_report[key]:
                raise ValueError(f"Reference changes {engine} implementation identity")
    result = validate(
        path,
        directory / "beamz-validation.json",
        directory / "fdtdx-validation.json",
        directory / "beamz-validation.npz",
        directory / "fdtdx-validation.npz",
    )
    result["reference_device_identity"] = device_identity
    saved = json.loads((directory / "numerical-gate.json").read_text())
    if (
        not result["passed"]
        or not saved["passed"]
        or saved["inputs"] != result["inputs"]
        or saved["contract_sha256"] != result["contract_sha256"]
    ):
        raise ValueError("Strong-signal witness failed independent strict revalidation")
    return result


def validate(
    contract_path,
    beamz_result,
    fdtdx_result,
    beamz_state,
    fdtdx_state,
    *,
    field_reference_root=None,
):
    from scripts.export_fdtdx_comparison import array_digest, canonical_work

    contract_path = Path(contract_path)
    contract = json.loads(contract_path.read_text())
    reports = [json.loads(Path(p).read_text()) for p in (beamz_result, fdtdx_result)]
    h = digest(contract_path)
    work = True
    checks = {}
    data_path = contract_path.parent / contract["data_file"]
    if digest(data_path) != contract["data_sha256"]:
        raise ValueError("Shared data file changed")
    with np.load(data_path, allow_pickle=False) as data:
        for key, expected in contract["array_sha256"].items():
            if array_digest(data[key]) != expected:
                raise ValueError(f"Shared input array changed: {key}")
        canonical = canonical_work(contract, data)
    if contract["canonical"] != canonical:
        raise ValueError(
            "Stored canonical metadata differs from independently recomputed data"
        )
    for engine, report in zip(("beamz", "fdtdx"), reports, strict=True):
        checks[engine] = (
            report.get("status") == "ok"
            and report["engine"]["name"] == engine
            and report["contract_sha256"] == h
            and report["actual_work"]["canonical"] == canonical
            and bool(report["actual_work"]["native"])
        )
        work &= checks[engine]
    work &= (
        contract["pml_cells"] == 12
        and canonical["pml_cells_each_face"] == 12
        and canonical["field_dtype"] == "float32"
        and not canonical["gradients"]
    )
    measurements = {}
    with (
        np.load(beamz_state, allow_pickle=False) as bz,
        np.load(fdtdx_state, allow_pickle=False) as fd,
    ):
        for key in ("E_xyz", "H_si_xyz"):
            expected = (3, *contract["shape_zyx"][::-1])
            if bz[key].shape != expected or fd[key].shape != expected:
                raise ValueError(f"Unexpected common field support for {key}")
            if bz[key].dtype != np.float32 or fd[key].dtype != np.float32:
                raise ValueError(f"Expected real FP32 fields for {key}")
            measurements[key] = compare(bz[key], fd[key], interior=14)
        for monitor in contract["monitors"]:
            key = monitor["name"] + "_phasor"
            a, b = bz[key], fd[key]
            if a.shape != (6, monitor["points"]):
                raise ValueError(f"Unexpected detector shape {key}: {a.shape}")
            measurements[key + "_E"] = compare(a[:3], b[:3])
            measurements[key + "_H_SI"] = compare(a[3:], b[3:])
    original_relative_passed = all(
        v["relative_l2"] <= 0.01 for v in measurements.values()
    )
    numerical = original_relative_passed
    reference = None
    if field_reference_root is not None:
        reference = reference_witness(
            field_reference_root, {**contract, "_path": str(contract_path)}, reports
        )
        reference_measurements = reference["tolerances_and_measurements"][
            "measurements"
        ]
        for key in ("E_xyz", "H_si_xyz"):
            scale = max(
                reference_measurements[key]["beamz_l2"],
                reference_measurements[key]["fdtdx_l2"],
            )
            if not scale > 0:
                raise ValueError("Zero strong-signal reference")
            value = measurements[key]
            error = value["relative_l2"] * max(value["beamz_l2"], value["fdtdx_l2"])
            value["error_l2_relative_to_signal_witness"] = error / scale
        numerical = all(
            v["error_l2_relative_to_signal_witness"] <= 1e-5
            if key in ("E_xyz", "H_si_xyz")
            else v["relative_l2"] <= 0.01
            for key, v in measurements.items()
        )
    # At least one nonzero E/H witness is required; zero/zero detector faces early
    # in a short smoke solve remain valid but cannot qualify an entire campaign.
    signal = (
        measurements["E_xyz"]["beamz_l2"] > 0
        and measurements["H_si_xyz"]["beamz_l2"] > 0
    )
    numerical &= signal
    return {
        "passed": bool(work and numerical),
        "work_equivalent": bool(work),
        "numerical_equivalent": bool(numerical),
        "contract_sha256": h,
        "inputs": {
            name + "_sha256": digest(path)
            for name, path in (
                ("beamz_result", beamz_result),
                ("fdtdx_result", fdtdx_result),
                ("beamz_state", beamz_state),
                ("fdtdx_state", fdtdx_state),
            )
        },
        "tolerances_and_measurements": {
            "relative_l2_maximum": 0.01,
            "original_all_final_relative_criterion_passed": bool(
                original_relative_passed
            ),
            "acceptance_basis": "strict strong-signal witness plus final field error / witness <= 1e-5 and full DFT relative error <= 1%"
            if reference
            else "all fields and phasors relative L2 <= 1%",
            "field_reference": reference,
            "field_exclusion_cells_each_face": 14,
            "field_units": "E in V/m; H in A/m",
            "clock_alignment": canonical["clock_map"],
            "phasor_alignment": "FD E rotated by exp(-i omega dt); BeamZ final E sample removed; raw H common",
            "amplitude_fit": False,
            "phase_fit": False,
            "resolved_work_checks": checks,
            "measurements": measurements,
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "contract",
        "beamz-result",
        "fdtdx-result",
        "beamz-state",
        "fdtdx-state",
        "output",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--field-reference-root", type=Path)
    a = p.parse_args()
    try:
        result = validate(
            a.contract,
            a.beamz_result,
            a.fdtdx_result,
            a.beamz_state,
            a.fdtdx_state,
            field_reference_root=a.field_reference_root,
        )
    except Exception as e:
        result = {
            "passed": False,
            "work_equivalent": False,
            "numerical_equivalent": False,
            "error": str(e),
        }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
