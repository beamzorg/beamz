"""Provenance and uncertainty for the matched CUDA benchmark runners."""

import hashlib
import os
import platform
from pathlib import Path

import jaxlib
import numpy as np


def provenance():
    import beamz._cuda as extension

    root = Path(__file__).resolve().parents[1]
    paths = (
        "beamz/simulation/cuda/runtime.py",
        "beamz/simulation/cuda/differentiation.py",
        "beamz/simulation/api/differentiation.py",
        "beamz/simulation/execute/__init__.py",
        "beamz/simulation/results/observe.py",
        "cuda/src/adjoint.cu",
        "cuda/src/update.cu",
        "cuda/src/ffi_handler.cc",
    )
    return dict(
        python_version=platform.python_version(),
        jaxlib_version=jaxlib.__version__,
        cuda_component_version=extension.__version__,
        cuda_abi=extension.__abi_version__,
        native_binary_sha256=hashlib.sha256(
            Path(extension.__file__).read_bytes()
        ).hexdigest(),
        source_sha256={
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in paths
        },
        xla_flags=os.environ.get("XLA_FLAGS", ""),
        cpml_precision=os.environ.get("BEAMZ_CUDA_CPML_PSI_PRECISION", "fp32"),
        preallocate=os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", ""),
    )


def speedup_interval(native_samples, reference_samples):
    """Bootstrap 95% interval for the ratio of warm medians, not a best run."""
    rng = np.random.default_rng(651)
    native, reference = np.asarray(native_samples), np.asarray(reference_samples)
    ratios = np.median(
        rng.choice(reference, (4000, reference.size)), axis=1
    ) / np.median(rng.choice(native, (4000, native.size)), axis=1)
    return np.percentile(ratios, (2.5, 97.5)).tolist()
