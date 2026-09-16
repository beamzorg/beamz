"""Replay the pinned upstream FWHM function without its solver-file loaders.

Pass a local checkout's projects/FDTD_solvers/ring/find_FWHM.py. The exact
reviewed source hash is required; only its find_FWHM function is executed.
"""

import argparse
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

HERE = Path(__file__).resolve().parent
REFERENCE_SHA256 = "0bcf1d77ac5fa966707ede8ea37afe835cceeac1d57bc3837031fe818b9000fc"
REFERENCE_URL = (
    "https://github.com/JPPhotonics/fdtd-pipeline/blob/"
    "622e0a9b7429eaf2335b1000b39e283544a198c4/"
    "projects/FDTD_solvers/ring/find_FWHM.py"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_script", type=Path)
    args = parser.parse_args()
    source = args.reference_script.read_bytes()
    if hashlib.sha256(source).hexdigest() != REFERENCE_SHA256:
        parser.error("reference source differs from the reviewed pinned script")
    function = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "find_FWHM"
    )
    namespace = {"np": np, "CubicSpline": CubicSpline, "plt": plt}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), REFERENCE_URL, "exec"),
        namespace,
    )
    rows = [
        r
        for r in json.loads((HERE / "reprojections.json").read_text())
        if r["device"] == "ring_resonator"
    ]
    report = {
        "reference_source": REFERENCE_URL,
        "reference_sha256": REFERENCE_SHA256,
        "warning": "Literal upstream window guards can pair crossings from different resonances; neither extraction establishes time convergence.",
        "runs": [],
    }
    for row in sorted(rows, key=lambda r: r["run_time_ps"]):
        wl = np.asarray(row["wavelengths_um"])
        power = np.asarray(row["powers"]["o2"])
        order = np.argsort(wl)
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            width, center = namespace["find_FWHM"](wl[order], power[order] / max(power))
        indices = json.loads(capture.getvalue().strip().splitlines()[-1])
        dense = np.linspace(min(wl), max(wl), 1000)
        left, right = dense[indices]
        enclosed = [r for r in row["ring"]["resonances_um"] if left < r < right]
        report["runs"].append(
            {
                "run": row["run"],
                "duration_ps": row["run_time_ps"],
                "literal_upstream_fwhm_nm": float(width * 1000),
                "literal_upstream_q": float(center / width),
                "literal_upstream_crossings_nm": [
                    float(left * 1000),
                    float(right * 1000),
                ],
                "dominant_resonances_between_crossings": len(enclosed),
                "beamz_first_complete_dip_fwhm_nm": row["ring"]["fwhm_nm"],
                "beamz_first_complete_dip_q": row["ring"]["q"],
            }
        )
        plt.title(
            f"Literal pinned extraction · {row['run_time_ps']:g} ps\n"
            f"Shaded interval contains {len(enclosed)} detected dominant resonances"
        )
        plt.xlabel("Wavelength (µm)")
        plt.ylabel("Through power / spectral maximum")
        plt.tight_layout()
        plt.savefig(HERE / f"ring_extraction_{row['run_time_ps']:g}ps.png", dpi=150)
        plt.close()
    (HERE / "ring_extraction_audit.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
