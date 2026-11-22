"""One CUDA backend with native execution and typed interoperability bindings."""

# Import aliases preserve existing module paths and object identity during migration.
import sys as _sys

from .mstep import mstep_config, schedule, tuning
from .runtime import (
    capabilities,
    pack_dft_monitors,
    run_program_steps,
    run_source_group_steps,
    run_steps,
    sharding,
    update_e,
    update_h,
)
from .storage import material_blocks, tensor

for _name in (
    "capabilities",
    "sharding",
    "material_blocks",
    "tensor",
    "mstep_config",
    "schedule",
    "tuning",
):
    _sys.modules[f"{__name__}.{_name}"] = globals()[_name]


__all__ = [
    "pack_dft_monitors",
    "run_program_steps",
    "run_source_group_steps",
    "run_steps",
    "update_e",
    "update_h",
]
