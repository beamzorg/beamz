"""Private typed-FFI runtime for BeamZ's optional Futhark-generated library."""

from .runtime import (
    FutharkBackendUnavailable,
    futhark_backend_status,
    run_program,
    source_table,
)

__all__ = [
    "FutharkBackendUnavailable",
    "futhark_backend_status",
    "run_program",
    "source_table",
]
