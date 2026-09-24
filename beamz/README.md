## Module Structure

### `analysis/` - Modal Analysis, S-Parameters, and Plotting
Contains modal projection, port/S-parameter extraction, compact plotting helpers, and small result adapters used by examples and notebooks.

### `design/` - Parametric Design and Geometry
Defines the physical structure of the device through parametric geometry and materials.

### `devices/` - Field Sources and Monitors
Handles electromagnetic field injection (sources) and detection (monitors) that interact with the simulation fields.

### `simulation/` - FDTD Engine
Orchestrates the finite-difference time-domain (FDTD) simulation and field evolution.

### `optimization/` - Inverse Design Helpers
Contains topology optimization, autodiff utilities, adjoint field-history storage, and density polygonization.

### `const.py` - Physical Constants
Defines fundamental physical constants (light speed, vacuum permittivity/permeability) and unit conversions (µm, nm).

### Root foundations

`lattice.py` is intentionally separate from `const.py`: constants are dependency-free
scalars, while the lattice module owns the shared NumPy/JAX Yee geometry and material
sampling used by design, devices, simulation, and analysis. Merging them would make a
constant import load the numerical stack and would erase that dependency boundary.

The two private root helpers are cross-package foundations rather than package-owned
behavior: `_cache_tokens.py` provides canonical value hashing for immutable specs and
`_helpers.py` contains the small validation, unit-display, FDTD-step, logging, and
progress utilities shared by otherwise independent packages. Keep package-specific
helpers beside their owner instead of adding more root utility files.

## Code Architecture

The codebase uses immutable specifications with explicit runtime state:

- **Design geometry**: `Design` owns the background material and ordered, immutable structures. It does not own sources, monitors, or evolving fields.
- **Simulation orchestration**: `Simulation` combines a `Design` with source, monitor, boundary, time, and grid specifications. It lowers them into an immutable compiled program.
- **Runtime and results**: `SimulationState` contains the evolving Yee fields; `SimulationRun` keeps that continuation value separate from detached, immutable `SimulationResults`.
- **Device abstraction**: Sources, monitors, and boundaries are immutable device specifications compiled into grid-aware runtime data.
- **Separation of concerns**: Design geometry, devices, solver execution, analysis, and optimization remain separate packages; caches live outside immutable specifications and plans.

## Interactive plots with XY

Matplotlib remains the default. Install `pip install 'beamz[xy]'` on Python 3.11
or newer to enable the optional [XY](https://github.com/reflex-dev/xy) backend.
Select it for an individual plot:

```python
fig, axes = sim.plot(z=0.0, y=0.0, backend="xy", show=False)
fig  # interactive notebook output
```

Or select it for a notebook, including its standalone power curves:

```python
import xy.pyplot as plt
from beamz.analysis import set_plotting_backend

set_plotting_backend("xy")
fig, ax = results.plot_field("field", "Ey", val="real")
plt.show()
```

`plotting_backend("xy")` is a context manager for temporary selection;
`get_plotting_backend()` and `get_pyplot()` expose the active choice. Passing
native axes selects their backend automatically, and an explicit conflicting
backend raises an error. The plotting functions return native figure/axes
objects with the same tuple shapes as before, including modal effective indices.

Both backends use the same field normalization, physical nonuniform cell edges,
axis limits, geometry, colors, and labels. XY renders interactive HTML with pan,
zoom, and hover. Its browser fonts, colorbar thickness, and hatch spacing can
differ slightly from Matplotlib. Use `fig.savefig("plot.html")` to export an
interactive figure. The executed `examples/notebooks/modal_sources_monitors_xy.ipynb`
is a copy of the Matplotlib walkthrough with the same simulation parameters.
