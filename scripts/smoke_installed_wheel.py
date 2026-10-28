"""Install one BeamZ wheel in an isolated environment and load its Rust extension."""

from __future__ import annotations

import argparse
import os
import subprocess
import venv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel_directory", type=Path)
    return parser.parse_args()


def _venv_python(environment: Path) -> Path:
    if os.name == "nt":
        return environment / "Scripts" / "python.exe"
    return environment / "bin" / "python"


def main() -> None:
    args = parse_args()
    wheels = sorted(args.wheel_directory.glob("beamz-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(
            f"Expected one BeamZ wheel in {args.wheel_directory}, found {len(wheels)}"
        )

    # The child process runs outside the checkout to prove it imports the
    # installed wheel.  Make the interpreter path absolute before changing
    # its working directory, otherwise a relative wheel directory is resolved
    # a second time (for example, ``dist/dist/.wheel-smoke/bin/python``).
    environment = (args.wheel_directory / ".wheel-smoke").resolve()
    venv.EnvBuilder(with_pip=True).create(environment)
    python = _venv_python(environment)
    subprocess.run(
        [python, "-m", "pip", "install", "--disable-pip-version-check", wheels[0]],
        check=True,
    )
    smoke = """
import json
import beamz
from beamz.design.raster import _native

# ENGINE_VERSION also includes the raster cache compatibility revision.
assert beamz.__version__ == _native.ENGINE_VERSION.partition("-geometry-")[0]
report = json.loads(
    _native.inspect_mesh(
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        [(0, 1, 2)],
    )
)
assert report["triangles"] == 1
print(f"Loaded Rust extension {_native.__file__}")
"""
    subprocess.run([python, "-c", smoke], check=True, cwd=environment)


if __name__ == "__main__":
    main()
