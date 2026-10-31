"""Optimize a guided 90-degree bend with the common inverse-design workflow.

All lengths are metres. The arbitrary mask retains fixed waveguides as filter
context; the modal transmission objective uses the actual input monitor power.
Re-simulate thresholded geometry separately from the continuous-density objective.
"""

from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from beamz import (
    LIGHT_SPEED,
    PML,
    Design,
    GaussianPulse,
    Material,
    ModeMonitor,
    ModeSource,
    ModeSpec,
    Rectangle,
    Simulation,
    um,
)
from beamz.optimization import (
    InverseDesign,
    LinearSchedule,
    MaskedDesignRegion,
    ModePower,
    OptimizationSchedule,
    Optimizer,
)
from beamz.optimization.topology import create_optimization_mask

width, dx, wavelength, guide_width = 7 * um, 0.1 * um, 1.55 * um, 0.55 * um
frequency, steps = LIGHT_SPEED / wavelength, 50
cladding, core = Material(1.444**2), Material(2.25**2)
design = Design(width=width, height=width, material=cladding)
design += Rectangle(
    position=(0, width / 2 - guide_width / 2),
    width=width / 2,
    height=guide_width,
    material=core,
)
design += Rectangle(
    position=(width / 2 - guide_width / 2, 0),
    width=guide_width,
    height=width / 2,
    material=core,
)
footprint = Rectangle(
    position=(2 * um, 2 * um), width=3 * um, height=3 * um, material=core
)
mask = create_optimization_mask(design.rasterize(dx), footprint)
source = ModeSource(
    center=(1 * um, width / 2, 0),
    size=(0, 4 * guide_width, guide_width),
    direction="+",
    mode_spec=ModeSpec(polarization="tm"),
    source_time=GaussianPulse(freq0=frequency, fwidth=0.2 * frequency, offset=2),
)
incoming = ModeMonitor(
    center=(1.4 * um, width / 2, 0),
    size=source.size,
    mode_spec=source.mode_spec,
    freqs=[frequency],
    name="input",
)
output = incoming.updated_copy(
    center=(width / 2, 1.2 * um, 0),
    size=(4 * guide_width, 0, guide_width),
    name="output",
)
simulation = Simulation(
    design=design,
    resolution=dx,
    run_time=180e-15,
    sources=[source],
    monitors=[incoming, output],
    boundaries=[PML(thickness=0.7 * um, formulation="cpml")],
)


def material_penalty(density, pixel_size):
    del pixel_size
    return 0.1 * (jnp.mean(density[mask]) - 0.3) ** 2


region = MaskedDesignRegion(
    region_mask=mask,
    pixel_size=dx,
    eps_bounds=(cladding.permittivity, core.permittivity),
    filter_radius=0.3 * um,
    penalties=(material_penalty,),
)
problem = InverseDesign(
    simulation,
    region,
    objective=ModePower("output", direction="-", reference_monitor="input"),
    gradient_backend="autodiff",
)
optimizer = Optimizer(
    problem,
    learning_rate=0.01,
    schedule=OptimizationSchedule(beta=LinearSchedule(1, 20, steps)),
    evaluate_after=True,
)
result_dir = Path("results/ceviche-bend")
result_dir.mkdir(parents=True, exist_ok=True)
result = optimizer.run(steps=steps, checkpoint=result_dir / "optimization.npz")
result.plot_optimization()
plt.savefig(result_dir / "objective.png", dpi=150, bbox_inches="tight")
plt.close()
# The last update is included in these final physical parameters and geometry.
continuous = result.to_simulation().run(backend="jax", performance=False)
exported = result.export_design(mode="pixel_exact")
verified = simulation.updated_copy(design=exported).run(
    backend="jax", performance=False
)
for label, data in (("continuous", continuous), ("thresholded", verified)):
    power = abs(data.mode("output").amps.sel(direction="-", mode_index=0).item()) ** 2
    power /= abs(data.mode("input").amps.sel(direction="+", mode_index=0).item()) ** 2
    print(f"{label} transmission: {power:.3%}")
np.save(result_dir / "density.npy", result.final_params)
