"""Internal simulation presentation operations."""

from __future__ import annotations

from typing import Any

from beamz.simulation.execute import (
    run_simulation_program,
    run_until_terminated,
)
from beamz.simulation.model import (
    AutoTermination,
)
from beamz.simulation.results import SimulationResults, SimulationRun

from .preparation import (
    _analysis_function,
)


class _SimulationPresentation:
    def advance(
        self: Any,
        *,
        state=None,
        num_steps=None,
        progress=False,
        store_full_materials=False,
        sharding=None,
        donate_state=False,
        backend="auto",
        schedule=None,
        expected_runs=1,
        performance=True,
    ) -> SimulationRun:
        """Execute a fresh or continued segment and return results plus next state.

        Parameters
        ----------
        state : SimulationState, optional
            Continuation state. Omitting it starts from zero fields at the beginning
            of the simulation time grid.
        num_steps : int, optional
            Number of timesteps to execute. The remaining time grid is used when
            omitted.
        progress : bool, default=False
            Display JIT and execution progress in an interactive terminal.
        store_full_materials : bool, default=False
            Retain the full material grid in result metadata. By default BeamZ stores
            only material regions required by configured analysis monitors.
        sharding : ShardingConfig or compatible value, optional
            Runtime device-sharding policy.
        donate_state : bool, default=False
            Transfer ownership of the input state's device buffers to JAX. This can
            reduce peak memory, but the input state must never be used afterward.
            CUDA keeps its fast schedule when estimated workspace fits; otherwise
            it selects in-place execution. Set BEAMZ_CUDA_MEMORY_POLICY to
            ``speed`` or ``capacity`` to override the default ``auto`` policy.
        backend : {"auto", "jax", "cuda", "cuda_streamed", "cuda_mstep"}, default="auto"
            Execution policy. ``auto`` preserves JAX as the fallback when the
            optional CUDA runtime is not installed.
        performance : bool, default=True
            Print JIT- and setup-free runtime and GCUPS after this segment finishes.
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
        SimulationRun
            A small owner containing durable ``results`` and the separate
            continuation ``state`` produced at the end of the segment.

        Raises
        ------
        ValueError
            If ``state.current_step`` is outside the time grid or ``num_steps`` is
            not positive and within the remaining number of timesteps.

        Examples
        --------
        Start a segmented run, then continue it:

        >>> first = sim.advance(num_steps=40)
        >>> second = sim.advance(state=first.state, num_steps=20)
        >>> int(second.state.current_step)
        60

        Branch safely from the same state; preservation is the default:

        >>> branch_a = sim.advance(state=first.state, num_steps=10)
        >>> branch_b = sim.advance(state=first.state, num_steps=10)

        Opt into buffer donation only for a state that will not be reused:

        >>> last = sim.advance(
        ...     state=second.state,
        ...     num_steps=10,
        ...     donate_state=True,
        ... )

        Notes
        -----
        Use :meth:`run` when final analysis data is all that is needed. ``run``
        executes the complete grid, returns :class:`SimulationResults` directly,
        and safely donates its private initial state. ``advance`` exists specifically
        for checkpointing, branching, streaming, and other continuation workflows.
        ``SimulationResults`` never contains continuation state.
        """
        current_step = 0 if state is None else int(state.current_step)
        if not 0 <= current_step <= self.num_steps:
            raise ValueError("state.current_step is outside the simulation time grid.")
        remaining = self.num_steps - current_step
        steps = remaining if num_steps is None else int(num_steps)
        if not 0 < steps <= remaining:
            raise ValueError(f"num_steps must be in [1, {remaining}], got {steps}.")
        return run_simulation_program(
            self,
            state,
            num_steps=steps,
            sharding=sharding,
            backend=backend,
            schedule=schedule,
            expected_runs=expected_runs,
            store_full_materials=bool(store_full_materials),
            monitor_steps=remaining,
            donate_state=bool(donate_state),
            performance=bool(performance),
            progress=bool(progress),
        )

    def run(
        self: Any,
        *,
        progress=False,
        store_full_materials=False,
        sharding=None,
        backend="auto",
        schedule=None,
        expected_runs=1,
        termination: AutoTermination | None = None,
        performance: bool = True,
    ) -> SimulationResults:
        """Execute the complete simulation and return immutable analysis results.

        Parameters
        ----------
        progress : bool, default=False
            Display JIT and execution progress in an interactive terminal.
        store_full_materials : bool, default=False
            Retain the full material grid in result metadata. The default stores only
            material regions needed by configured analysis monitors.
        sharding : ShardingConfig or compatible value, optional
            Runtime device-sharding policy.
        backend : {"auto", "jax", "cuda", "cuda_streamed", "cuda_mstep"}, default="auto"
            Execution policy. ``cuda`` chooses the best compatible CUDA target.
        termination : AutoTermination, optional
            Bounded convergence policy. When supplied, BeamZ executes reusable
            chunks and may stop before the end of the time grid after sources are
            inactive and the requested field and monitor residuals stabilize.
        performance : bool, default=True
            Print JIT- and setup-free runtime and GCUPS after completion. Set to
            ``False`` to suppress reporting and timing instrumentation.
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
        SimulationResults
            Detached monitor acquisitions, immutable simulation metadata, and source
            normalization information. No continuation state is attached.

        Examples
        --------
        >>> results = sim.run(progress=True)
        >>> field_data = results["fields"]

        Enable bounded automatic termination:

        >>> results = sim.run(termination=AutoTermination(chunk_steps=200))
        >>> results.termination.reason in {"converged", "time_limit"}
        True

        Limit elapsed execution time while retaining partial measurements:

        >>> results = sim.run(termination=AutoTermination(max_wall_time=600.0))

        The elapsed-time budget includes compilation and is checked between
        chunks. A time-limited run returns its acquired results with termination
        reason ``"wall_time_limit"``; finishing a chunk and constructing the result
        may take execution beyond the budget.

        Notes
        -----
        This is the normal user-facing execution method. Compilation is automatic.
        Since the initial runtime state belongs only to this call and is not returned,
        BeamZ donates its buffers internally to reduce peak device memory.

        Use :meth:`advance` instead when execution must be split into segments or the
        final :class:`SimulationState` is required. Use :meth:`step` only for
        state-only single-timestep debugging.
        """
        if termination is not None:
            return run_until_terminated(
                self,
                termination,
                progress=bool(progress),
                store_full_materials=bool(store_full_materials),
                sharding=sharding,
                backend=backend,
                schedule=schedule,
                expected_runs=expected_runs,
                performance=bool(performance),
            )
        # The initial state is private to this call, so donating it is always safe and
        # avoids retaining a second full set of device buffers during execution.
        return self.advance(
            progress=bool(progress),
            store_full_materials=bool(store_full_materials),
            sharding=sharding,
            backend=backend,
            schedule=schedule,
            expected_runs=expected_runs,
            donate_state=True,
            performance=bool(performance),
        ).results

    def plot(self: Any, **kwargs):
        """Create a Matplotlib view of the simulation layout.

        Parameters
        ----------
        **kwargs : object
            Plot options forwarded to the analysis plotting backend, including
            ``ax``, ``figsize``, ``z``, ``y``, axis limits, and marker controls.
            ``show`` defaults to ``False`` for this method.

            Two-dimensional layouts share the 3D material palette, shaded PML
            bands, and source/monitor glyphs. Display coordinates and axis limits
            are in micrometres; slice positions and simulation inputs use metres.

            For a native three-dimensional :class:`~beamz.Design`, passing ``z``
            or ``y`` creates antialiased geometry cross sections. These setup
            plots preserve polygon boundaries and do not compile or show the FDTD
            grid. Pass ``categorical_cross_sections=False`` to inspect the
            rasterized material field instead.

        Returns
        -------
        tuple
            Matplotlib ``(figure, axes)`` objects. A 3D cross-section layout may
            return an array of axes.

        Examples
        --------
        >>> fig, ax = sim.plot()
        """
        kwargs.setdefault("show", False)
        return _analysis_function("plotting", "plot_simulation")(self, **kwargs)

    def plot_eps(self: Any, **kwargs):
        """Plot the simulation's actual relative-permittivity raster.

        Uses the supplied MaterialGrid when present, including optimized density
        designs. Accepts one ``x``, ``y``, or ``z`` slice for 3D, source/monitor marker
        controls, and Matplotlib axes/color limits.
        Slice positions are metres; display axes are micrometres.

        Returns
        -------
        tuple
            Matplotlib ``(figure, axes)`` containing the permittivity plot.
        """
        kwargs.setdefault("show", False)
        return _analysis_function("plotting", "plot_simulation_permittivity")(
            self, **kwargs
        )

    def show(self: Any, **kwargs):
        """Display the simulation layout using Matplotlib.

        Parameters
        ----------
        **kwargs : object
            Plot options accepted by :meth:`plot`. ``show`` defaults to ``True``.

        Returns
        -------
        tuple
            Matplotlib ``(figure, axes)`` objects.
        """
        # Route display through the plotting API while changing only the presentation flag.
        kwargs.setdefault("show", True)
        return self.plot(**kwargs)

    def view3d(self, **kwargs):
        """Create a static, 3D-oriented cross-section view.

        Parameters
        ----------
        **kwargs : object
            Cross-section and Matplotlib options forwarded to the plotting backend.
            ``show`` defaults to ``False``.

        Returns
        -------
        tuple
            Matplotlib figure and axes for the selected cross sections.

        Notes
        -----
        This method creates static notebook-friendly, antialiased geometry cross
        sections; it does not launch an interactive browser viewer. Pass
        ``categorical_cross_sections=False`` to inspect the rasterized material
        field instead of the design layout.
        """
        return _analysis_function("plotting", "view_simulation_3d")(self, **kwargs)

    def show3d(self, **kwargs):
        """Return the same static 3D-oriented view as :meth:`view3d`.

        Parameters
        ----------
        **kwargs : object
            Cross-section and Matplotlib options forwarded to :meth:`view3d`.

        Returns
        -------
        tuple
            Matplotlib figure and axes for the selected cross sections.

        Notes
        -----
        ``show3d`` is currently an alias and does not force interactive display.
        """
        # Route display through the plotting API while changing only the presentation flag.
        return self.view3d(**kwargs)
