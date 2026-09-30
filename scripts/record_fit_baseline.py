"""Record existing-engine tests and FDTD/FIT timings on explicit JAX hardware.

Run from the repository root with its installed Python environment. No packages
are installed and no lockfiles are changed. Timings are observational, not CI
thresholds. Use --junit to import a completed baseline instead of rerunning it.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELECTOR = "(unit or component or optimization) and not slow and not characterization and not pdk"


def _test_summary(path):
    root = ET.parse(path).getroot()
    cases = list(root.iter("testcase"))
    failures = [
        {
            "test": case.attrib.get("classname", "") + "::" + case.attrib["name"],
            "reason": failure.attrib.get("message", ""),
        }
        for case in cases
        for failure in list(case.findall("failure")) + list(case.findall("error"))
    ]
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    return {
        "tests": len(cases),
        "passed": sum(
            not any(
                case.find(tag) is not None for tag in ("failure", "error", "skipped")
            )
            for case in cases
        ),
        "skipped": sum(case.find("skipped") is not None for case in cases),
        "failures": failures,
        "seconds": sum(float(suite.attrib.get("time", 0)) for suite in suites),
    }


def _benchmarks():
    import jax
    import numpy as np

    from beamz import PEC, Design, FITSimulation, Material, Simulation, UniformFITMesh

    results = []
    for shape in [(48, 64), (16, 20, 24)]:
        dx = 1e-6
        mesh = UniformFITMesh(shape, dx)
        fit = FITSimulation(mesh, permittivity=2.25, courant=0.7)
        sizes = tuple(n * dx for n in shape)
        design = Design(
            width=sizes[-1],
            height=sizes[-2],
            depth=sizes[0] if len(shape) == 3 else 0,
            material=Material(2.25),
        )
        fdtd = Simulation(
            design=design,
            resolution=dx,
            time=np.arange(128) * fit.dt,
            boundaries=[PEC(edges="all")],
        )
        for name, sim in [("fdtd", fdtd), ("fit", fit)]:
            if name == "fit":
                coords = mesh.coordinates("Ez")
                seed = np.ones(mesh.electric_shapes["Ez"])
                for axis, coordinate in enumerate(coords):
                    broadcast = [1] * len(shape)
                    broadcast[axis] = coordinate.size
                    seed *= np.sin(np.pi * coordinate.reshape(broadcast) / sizes[axis])
                sim.set_fields(Ez=seed)

                def run():
                    return sim.run(64)

                def sync():
                    return jax.block_until_ready(sim.state)
            else:
                seed = np.ones(sim.fields.Ez.shape)
                for axis, n in enumerate(seed.shape):
                    broadcast = [1] * len(shape)
                    broadcast[axis] = n
                    seed *= np.sin(
                        np.pi * np.arange(n).reshape(broadcast) / max(1, n - 1)
                    )
                sim.fields.Ez = jax.numpy.asarray(seed, dtype=jax.numpy.float32)

                def run():
                    return sim.run_compiled(num_steps=64, progress=False)

                def sync():
                    return jax.block_until_ready(
                        (
                            sim.fields.Ex,
                            sim.fields.Ey,
                            sim.fields.Ez,
                            sim.fields.Hx,
                            sim.fields.Hy,
                            sim.fields.Hz,
                        )
                    )

            sync()
            first_start = time.perf_counter()
            run()
            sync()
            first_seconds = time.perf_counter() - first_start
            warm_start = time.perf_counter()
            run()
            sync()
            warm_seconds = time.perf_counter() - warm_start
            results.append(
                {
                    "backend": name,
                    "cell_shape": shape,
                    "spacing_m": dx,
                    "dt_s": fit.dt,
                    "steps_per_run": 64,
                    "permittivity": 2.25,
                    "boundary": "all PEC",
                    "precision": "float32",
                    "record_fields": False,
                    "first_run_seconds": first_seconds,
                    "warm_run_seconds": warm_seconds,
                    "cell_steps_per_second": int(np.prod(shape)) * 64 / warm_seconds,
                    "comparison": "Same cells/material/timestep; native field storage and seed sampling differ. Timing only, not an accuracy comparison.",
                }
            )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--platform", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--junit", type=Path, help="Import completed baseline test results"
    )
    args = parser.parse_args()
    os.environ["JAX_PLATFORMS"] = args.platform
    if args.output is None:
        args.output = (
            ROOT
            / "docs"
            / (
                "fit_baseline_gpu.json"
                if args.platform == "cuda"
                else "fit_baseline.json"
            )
        )
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(ROOT))
    command = [
        sys.executable,
        "-m",
        "pytest",
        "tests/",
        "--ignore=tests/test_fit.py",
        "--ignore=tests/test_fit_interfaces.py",
        "-m",
        SELECTOR,
        "--tb=short",
    ]
    with tempfile.TemporaryDirectory(prefix="beamz-fit-baseline-") as temporary:
        junit = args.junit or Path(temporary) / "baseline.xml"
        returncode = 0
        if args.junit is None:
            with (Path(temporary) / "pytest.log").open("w") as log:
                completed = subprocess.run(
                    command + [f"--junitxml={junit}"],
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            returncode = completed.returncode
        summary = _test_summary(junit)
    import jax

    jax.devices("gpu" if args.platform == "cuda" else "cpu")

    versions = {}
    for package in (
        "beamz",
        "jax",
        "jaxlib",
        "numpy",
        "scipy",
        "micromode",
        "pytest",
        "pytest-cov",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    report = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_note": "Existing FDTD physics code is unchanged. FIT and import additions are working-tree changes, not included in source_revision. Initial checkout already had dependency changes.",
        "dependency_file_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ("pyproject.toml", "uv.lock")
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "versions": versions,
            "jax_backend": jax.default_backend(),
            "devices": [str(d) for d in jax.devices()],
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
        },
        "tests": {
            **summary,
            "selector": SELECTOR,
            "fit_tests_excluded": True,
            "imported_junit": args.junit is not None,
            "reproduce": f".venv/bin/python scripts/record_fit_baseline.py --platform {args.platform}",
        },
        "benchmarks": _benchmarks(),
        "limits": [
            f"Results measured on {jax.default_backend()}; the selected backend is required and cannot silently fall back.",
            "One first run and one warm run per case; timings are descriptive, not statistical performance claims.",
            "Full integration/characterization suite and coverage were not run.",
            "The pre-FIT gate selected 177 tests and deselected 383; subsequent collection counts can change.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"Saved {args.output}: {summary['passed']} passed; {len(summary['failures'])} failures"
    )
    return returncode or bool(summary["failures"])


if __name__ == "__main__":
    raise SystemExit(main())
