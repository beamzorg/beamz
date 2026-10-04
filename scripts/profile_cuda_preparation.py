"""Stage-level memory trace around the fixed-physics cube benchmark.

Accepts benchmark_hopper_cube's arguments. Writes a sibling .stages.jsonl file.
Run each trial in a fresh process. Peak JAX bytes are cumulative; stage entry/exit
and a separate NVML trace distinguish transient allocations from retained pools.
"""

import functools
import importlib
import json
import resource
import sys
import time
from pathlib import Path

import jax


def main():
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    output.parent.mkdir(parents=True, exist_ok=True)
    trace = output.with_suffix(".stages.jsonl")
    device = jax.devices()[0]

    def record(name, event):
        jax.effects_barrier()
        with trace.open("a") as stream:
            stream.write(
                json.dumps(
                    dict(
                        stage=name,
                        event=event,
                        unix_time=time.time(),
                        memory_stats=device.memory_stats(),
                        host_peak_rss_bytes=resource.getrusage(
                            resource.RUSAGE_SELF
                        ).ru_maxrss
                        * 1024,
                    )
                )
                + "\n"
            )

    def instrument(module, name):
        original = getattr(module, name)

        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            record(name, "enter")
            try:
                result = original(*args, **kwargs)
                record(name, "exit")
                return result
            except Exception:
                record(name, "error")
                raise

        setattr(module, name, wrapper)

    modules = {
        "beamz.simulation.compile": (
            "lower_boundaries",
            "_compile_grid",
            "_prepare_compilation",
            "compile_source_specs",
            "compile_monitor_specs",
            "_pack_cuda_lossless_e_coefficients",
            "lower_compiled_arrays",
        ),
        "beamz.devices.sources.mode_launch": (
            "_launch_power_diagnostics_3d",
            "_reconstructed_3d_launch_phasor_state",
            "_yee_plane_power_3d",
        ),
        "beamz.devices.sources.planar_tfsf": (
            "advance_incident_h_3d",
            "advance_incident_e_3d",
        ),
    }
    for module_name, names in modules.items():
        module = importlib.import_module(module_name)
        for name in names:
            instrument(module, name)
    from scripts.benchmark_hopper_cube import main as benchmark

    benchmark()


if __name__ == "__main__":
    main()
