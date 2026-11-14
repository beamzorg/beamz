"""Internal simulation execution operations."""

from __future__ import annotations

from typing import Any

import numpy as np

from beamz.simulation.compile import simulation_memory_estimate as memory_estimate
from beamz.simulation.execute import (
    compiled_xla_memory_analysis as analyze_compiled_xla_memory,
)
from beamz.simulation.execute import (
    execute_step,
)
from beamz.simulation.model import (
    CompiledProgram,
    DomainSpec,
    RunConfig,
    SimulationRequest,
    SimulationState,
)
from beamz.simulation.sharding import normalize_sharding_config


class _SimulationExecution:
    def initial_state(self: Any) -> SimulationState:
        """Create a fresh runtime state at the beginning of the time grid.

        Returns
        -------
        SimulationState
            Zero-initialized fields, boundary memory, monitor accumulators, time,
            and timestep index compatible with this simulation.

        Examples
        --------
        >>> state = sim.initial_state()
        >>> int(state.current_step)
        0

        Notes
        -----
        Normal applications should call :meth:`run`. Create state explicitly only
        for :meth:`step`, branching, debugging, or custom continuation workflows.
        """
        # Allocate disabled features as empty fixed-rank arrays to keep runtime state structurally stable.
        from beamz.simulation.dispersion import initial_polarization

        program = self.compile()
        state = SimulationState.initial(program.grid, t=float(self.time[0]))
        return state._replace(polarization=initial_polarization(program.dispersion))

    def step(
        self: Any,
        state=None,
        *,
        donate_state=False,
        backend="auto",
        schedule=None,
        expected_runs=1,
    ) -> SimulationState:
        """Advance runtime state by exactly one timestep.

        Parameters
        ----------
        state : SimulationState, optional
            State to advance. A fresh state is created when omitted.
        donate_state : bool, default=False
            Allow JAX to recycle buffers owned by ``state``. After a donating call,
            the input state must not be read or reused.
            CUDA selects its execution schedule separately from buffer ownership.
        backend : {"auto", "jax", "cuda", "cuda_streamed", "cuda_mstep"}, default="auto"
            Execution policy. ``auto`` uses the optional CUDA extension when it is
            compatible and otherwise retains the JAX implementation. ``cuda_mstep``
            explicitly selects the experimental two-step CUDA scheduler.
        schedule : {"auto", "streamed", "mstep"}, optional
            CUDA execution schedule. Omitting it preserves established streamed
            selection. ``schedule="auto"`` opts into selection by time to result;
            unmeasured cases retain streamed. CUDA backend names remain explicit
            schedule overrides.
        expected_runs : int, default=1
            Expected executions reusing this program. Automatic CUDA selection
            uses this hint to amortize compilation; it does not execute extra runs.

        Returns
        -------
        SimulationState
            State after one timestep, or the unchanged input when it is already at
            the end of the time grid.

        Examples
        --------
        >>> state = sim.initial_state()
        >>> state = sim.step(state)
        >>> int(state.current_step)
        1

        Notes
        -----
        ``step`` returns state only and does not detach :class:`SimulationResults`.
        Use :meth:`run` for normal simulations and :meth:`advance` for multi-step
        continuation with monitor results.
        """
        # Advance explicit state without mutating the immutable Simulation specification.
        current_step = 0 if state is None else int(state.current_step)
        if current_step >= self.num_steps:
            assert state is not None
            return state
        program = self.compile(
            num_steps=1, backend=backend, schedule=schedule, expected_runs=expected_runs
        )
        if state is None:
            state = SimulationState.initial(program.grid, t=float(self.time[0]))
        return execute_step(
            program,
            state,
            monitor_steps=max(1, self.num_steps - current_step),
            donate_state=bool(donate_state),
        )

    def to_request(
        self: Any,
        *,
        num_steps: int | None = None,
        loop_kind: str = "scan",
        source_single_slab_dense: bool = False,
        backend: str = "jax",
        sharding=None,
        progress: bool = False,
        schedule: str | None = None,
        expected_runs: int = 1,
    ) -> SimulationRequest:
        """Build the immutable compiler request for this simulation.

        Parameters
        ----------
        num_steps : int, optional
            Number of timesteps represented by the request. The full time grid is
            used when omitted.
        loop_kind : {"scan", "fori_loop"}, default="scan"
            JAX control-flow lowering selected for the timestep loop.
        source_single_slab_dense : bool, default=False
            Use dense lowering for a single slab source. This is an advanced
            compilation experiment and does not change source semantics.
        backend : str, default="jax"
            Resolved execution backend embedded in the compiler request.
        sharding : ShardingConfig or compatible value, optional
            Device-sharding policy, normalized once into the request.
        progress : bool, default=False
            Display setup progress while rasterizing the material grid.
        schedule : {"auto", "streamed", "mstep"}, optional
            CUDA execution schedule. Omitting it preserves established streamed
            selection. ``schedule="auto"`` opts into selection by time to result;
            unmeasured cases retain streamed. CUDA backend names remain explicit
            schedule overrides.
        expected_runs : int, default=1
            Expected executions reusing this program. Automatic CUDA selection
            uses this hint to amortize compilation; it does not execute extra runs.

        Returns
        -------
        SimulationRequest
            Immutable, data-only input to the BeamZ compiler.

        Notes
        -----
        Most users should call :meth:`run` or :meth:`advance`; both construct and
        cache requests automatically. This method is exposed for diagnostics,
        compiler development, and reproducibility tooling.
        """
        # 1. Resolve Design's material raster; numerical boundary lowering belongs to compile.py.
        from beamz.simulation.cuda.schedule import normalize_schedule_request

        cuda_schedule, expected_runs = normalize_schedule_request(
            backend, schedule, expected_runs
        )
        sharding = normalize_sharding_config(sharding)
        material_grid = self._material_grid(progress=progress)
        requested_steps = int(self.num_steps if num_steps is None else num_steps)
        # 2. Snapshot grid, run, and domain controls into immutable value objects, then
        # attach normalized sources, monitors, boundaries, and sharding configuration.
        return SimulationRequest(
            RunConfig(
                dt=float(self.dt),
                num_steps=requested_steps,
                total_steps=int(self.num_steps),
                t0=float(np.ravel(np.asarray(self.time, dtype=float))[0]),
                loop_kind=str(loop_kind),
                source_single_slab_dense=bool(source_single_slab_dense),
                backend=str(backend),
                cuda_schedule=cuda_schedule,
                cuda_expected_runs=expected_runs,
                sharding=sharding,
                resolution=self.resolution,
                plane_2d=self.plane_2d,
                is_3d=self.is_3d,
                polarization_2d=self.polarization,
                metric_kind=self.grid.metric_kind_for(
                    ("x", "y", "z") if self.is_3d else ("x", "y")
                ),
            ),
            DomainSpec(
                tuple(float(v) for v in self.size),  # type: ignore[arg-type]
                bool(self.is_3d),
                str(self.plane_2d),
                self.coordinate_offset,
                self.polarization,
            ),
            material_grid,
            tuple(self.sources),
            tuple(self.monitors),
            tuple(self.boundaries),
        )

    def compile(
        self: Any,
        num_steps=None,
        sharding=None,
        backend="auto",
        progress: bool = False,
        schedule: str | None = None,
        expected_runs: int = 1,
    ) -> CompiledProgram:
        """Prepare or retrieve a reusable compiled execution plan.

        Parameters
        ----------
        num_steps : int, optional
            Static segment length compiled into the JAX loop. The complete time-grid
            length is used when omitted.
        sharding : ShardingConfig or compatible value, optional
            Runtime device-sharding policy. ``None`` selects the default placement.
        backend : {"auto", "jax", "cuda", "cuda_streamed", "cuda_mstep"}, default="auto"
            Execution policy. Explicit CUDA variants fail if their required typed
            FFI target or GPU architecture is unavailable.
        progress : bool, default=False
            Display setup progress while rasterizing and lowering the simulation.
        schedule : {"auto", "streamed", "mstep"}, optional
            CUDA execution schedule. Omitting it preserves established streamed
            selection. ``schedule="auto"`` opts into selection by time to result;
            unmeasured cases retain streamed. CUDA backend names remain explicit
            schedule overrides.
        expected_runs : int, default=1
            Expected executions reusing this program. Automatic CUDA selection
            uses this hint to amortize compilation; it does not execute extra runs.

        Returns
        -------
        CompiledProgram
            Immutable numerical plan. Executables are normally JIT-compiled lazily.
            Eligible CUDA workloads select storage from domain geometry during
            this call, without calibration runs.

        Examples
        --------
        >>> program = sim.compile(num_steps=50)
        >>> program.config.num_steps
        50

        Notes
        -----
        Calling :meth:`compile` is optional. :meth:`run`, :meth:`advance`, and
        :meth:`step` compile and reuse the appropriate plan automatically.
        On RTX3090, large lossless CPML12 programs of at least 32 steps estimate
        a storage layout from domain geometry, without executing timing trials.
        Set ``BEAMZ_CUDA_AUTOTUNE=off`` to retain canonical storage, or
        ``BEAMZ_CUDA_AUTOTUNE=calibrate`` to explicitly measure candidates and
        cache the result. Explicit CUDA layout or kernel overrides take
        precedence. Opt-in calibration probes at most 256 steps for long runs.
        """
        from beamz.simulation import api as api_runtime

        # Lower an immutable request and cache by every value that changes generated code or storage.
        return api_runtime.compile_program(
            self,
            num_steps=num_steps,
            sharding=sharding,
            backend=backend,
            progress=progress,
            schedule=schedule,
            expected_runs=expected_runs,
            setup_context_factory=api_runtime._resolved_setup_device_context,
            compile_factory=api_runtime.compile_simulation,
        )

    def differentiable(
        self: Any, *, num_steps=None, backend="jax", checkpoint_interval=None
    ):
        """Build a reusable solver for ``jax.grad`` and inverse design.

        Construct once outside JAX transformations, then call with dynamic cell
        permittivity or Ex/Ey/Ez arrays. The result is a ``SimulationState`` with
        differentiable fields and raw monitor accumulators. ``backend='cuda'``
        uses native float32 forward and reverse-mode Yee/CPML kernels on one GPU.
        ``checkpoint_interval=None`` selects sqrt(T) chunking; zero disables it.
        Geometry, source profiles, boundaries, and monitor definitions stay fixed.

        Parameters
        ----------
        num_steps : int, optional
            Number of fixed timesteps to execute; defaults to this time grid.
        backend : str, default "jax"
            Execution backend. "cuda" selects first-order FP32 native autodiff.
        checkpoint_interval : int or None, optional
            Steps per retained checkpoint. None selects sqrt(T); zero stores
            the complete trajectory instead of rematerializing chunks.

        Returns
        -------
        DifferentiableSimulation
            Callable solver with dynamic material/source parameters and raw,
            differentiable field and monitor state.
        """
        from beamz.simulation.differentiation import DifferentiableSimulation

        program = self.compile(num_steps=num_steps, backend=backend)
        return DifferentiableSimulation(
            program, t0=float(self.time[0]), checkpoint_interval=checkpoint_interval
        )

    def clear_compiled_cache(self: Any) -> None:
        """Clear BeamZ's in-process simulation compilation caches.

        Returns
        -------
        None

        Notes
        -----
        This clears cached material grids, immutable compiled plans, and JIT function
        references for all simulations in the current process, not only this object.
        It does not modify the simulation or any existing results and does not remove
        JAX's optional persistent on-disk compilation cache.
        """
        from beamz.simulation import api as api_runtime

        api_runtime._MATERIAL_GRID_CACHE.clear()
        api_runtime.clear_program_cache()
        api_runtime.clear_execution_cache()

    def memory_estimate(
        self: Any,
        *,
        include_compiled: bool = True,
        num_steps: int | None = None,
        sharding=None,
    ) -> dict:
        """Estimate deterministic array storage for a simulation plan.

        Parameters
        ----------
        include_compiled : bool, default=True
            Include update coefficients, boundary terms, sources, monitors, and
            initialized monitor-state buffers in addition to the discretized grid.
        num_steps : int, optional
            Segment length used when sizing step-dependent monitor buffers.
        sharding : ShardingConfig or compatible value, optional
            Device-sharding policy used to report padded and per-device storage.

        Returns
        -------
        dict
            Allocation report containing total bytes, GiB totals, category and
            residency summaries, individual entries, grid shape, and—when requested—
            a nested ``compiled`` report.

        Examples
        --------
        >>> report = sim.memory_estimate(num_steps=100)
        >>> report["total_bytes"] > 0
        True

        Notes
        -----
        The report counts known array payloads. Backend temporary allocations and
        compiler-dependent workspace are available, when supported, through
        :meth:`compiled_xla_memory_analysis`.
        """
        return memory_estimate(
            self,
            include_compiled=include_compiled,
            num_steps=num_steps,
            sharding=sharding,
        )

    def compiled_xla_memory_analysis(
        self: Any, *, num_steps: int | None = None, sharding=None
    ) -> dict:
        """Return backend memory analysis for the lowered JAX executable.

        Parameters
        ----------
        num_steps : int, optional
            Static segment length to lower. The full time grid is used when omitted.
        sharding : ShardingConfig or compatible value, optional
            Runtime placement used during lowering.

        Returns
        -------
        dict
            Scalar fields reported by the backend's compiled memory analysis. The
            mapping contains ``available=False`` when the backend provides no report.

        Notes
        -----
        Unlike :meth:`memory_estimate`, this method lowers and compiles an executable
        and may therefore be expensive. It does not advance simulation state.
        """
        # Lower an immutable request and cache by every value that changes generated code or storage.
        program = self.compile(num_steps=num_steps, sharding=sharding, backend="jax")
        state = SimulationState.initial(program.grid, t=float(self.time[0]))
        return analyze_compiled_xla_memory(
            program,
            state,
            monitor_steps=max(1, int(program.config.num_steps)),
        )
