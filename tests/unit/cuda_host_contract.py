"""Build the real sharded FFI with its portable cell kernel for CPU evidence."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import jax

_LIBRARIES = []


def register_host_sharded_ffi():
    from beamz.simulation._cuda_abi import CUDA_SHARDED_TARGET

    if _LIBRARIES:
        return
    compiler = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++")
    if not compiler:
        raise RuntimeError("The native CPU contract test requires a C++17 compiler")
    root = Path(__file__).resolve().parents[2]
    sanitizer_flags = (
        ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
        if os.environ.get("BEAMZ_CUDA_CPU_SANITIZE") == "1"
        else []
    )
    with tempfile.TemporaryDirectory(prefix="beamz-sharded-host-") as tmp:
        library = Path(tmp) / "sharded_host.so"
        subprocess.run(
            [
                compiler,
                "-std=c++17",
                "-O1",
                "-shared",
                "-fPIC",
                *sanitizer_flags,
                "-DBEAMZ_CUDA_CPU_CONTRACT",
                f"-I{jax.ffi.include_dir()}",
                f"-I{root / 'cuda/src'}",
                str(root / "cuda/src/ffi_handler.cc"),
                str(root / "tests/unit/cuda_sharded_host.cc"),
                "-o",
                str(library),
            ],
            check=True,
            timeout=90,
        )
        loaded = ctypes.CDLL(str(library))
        jax.ffi.register_ffi_target(
            CUDA_SHARDED_TARGET,
            jax.ffi.pycapsule(loaded.beamz_cuda_sharded),
            platform="cpu",
            api_version=1,
        )
        _LIBRARIES.append(loaded)
