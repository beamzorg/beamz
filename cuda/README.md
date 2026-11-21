# BeamZ CUDA backend

The optional `beamz._cuda` extension provides native CUDA execution for 3D
FDTD. Select it with `sim.run(backend="cuda")`. Its planner selects ordinary
Yee updates or temporal blocking based on the grid, workload, devices, and
available memory. Use `schedule="streamed"` or `schedule="mstep"` to request
a specific CUDA schedule. `results.cuda_plan` reports the selected binding,
schedule, allocation estimates, and selection reason.

```python
results = simulation.run(
    backend="cuda",
    sharding={"axis": "auto", "num_devices": 2},
)
print(dict(results.cuda_plan))
```

Temporal sharding supports uniform and rectilinear metrics, configured CPML
profiles, adaptive termination, and time-domain field recorders. CPML12 has a
specialized rolling kernel for eligible uniform grids. `simulation.advance()`
can return a detached checkpoint for resuming or branching a simulation.
`simulation.clear_compiled_cache()` releases resident CUDA sessions.

Set `BEAMZ_CUDA_MEMORY_BUDGET_BYTES` to constrain the native plan or
`BEAMZ_CUDA_MEMORY_POLICY=capacity` to select the ordinary schedule. Use
`backend="jax"` for 2D, periodic or Bloch, and dispersive simulations, and for
full-tensor gradients.

## Build from source

Build the private component against the supported JAX version in a CUDA
development environment.

```sh
python -m pip wheel ./cuda --no-deps
python -m pip install beamz_cuda_component-*.whl
```

CUDA component ABI 31 stores CPML auxiliary fields in FP32. Rebuild and reinstall
the optional component when updating from an earlier ABI.

For a direct CMake build, set the architecture for the target GPU and configure
the toolkit, Python, and nanobind paths for your environment.

```sh
cmake -S cuda -B build/cuda -DBEAMZ_CUDA_ARCHITECTURES=86
cmake --build build/cuda -j2
cmake --install build/cuda --prefix /path/to/site-packages
```

CMake uses [compiler/physics.py](compiler/physics.py) to generate the shared Yee
operations and checks them against the sources in `src/physics`. To update the
Python and C++ ABI mirrors after changing [abi_layout.json](abi_layout.json),
run `python scripts/generate_cuda_abi.py` from the repository root.
