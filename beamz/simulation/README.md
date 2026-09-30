# Simulation

Modules that orchestrate how the EM fields evolve depending on the design and devices.

+ core.py       / Main module to orchestrate the simulation.
+ fields.py     / Contains the Field class which owns the field data and defines the field update.
+ ops.py        / Contains the operations used by the field updates.
+ compiled.py   / v0.3 packed-data compiled engine (`run_compiled` / `compile_simulation`).
+ fit/          / Experimental uniform-grid FIT engine (`FITSimulation`), initially with PEC boundaries.

See the [FIT backend development plan](../../docs/fit_backend_plan.md) for scope,
validation and the path toward nonuniform meshes and interface-aware materials.
