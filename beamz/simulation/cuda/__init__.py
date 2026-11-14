"""Private typed-FFI runtime for BeamZ's optional CUDA wheel."""

from importlib import import_module

__all__ = [
    "pack_dft_monitors",
    "run_program_steps",
    "run_source_group_steps",
    "run_steps",
    "update_e",
    "update_h",
]


def __getattr__(name):
    if name in __all__:
        return getattr(import_module("beamz.simulation.cuda.runtime"), name)
    raise AttributeError(name)
