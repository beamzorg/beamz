"""Detached raw outputs produced by simulation execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType

import numpy as np

from beamz.devices._immutable import immutable_snapshot
from beamz.devices.monitors.geometry import retained_sample_region
from beamz.simulation.model import CompiledGrid
from beamz.simulation.results.metadata import (
    RunTermination,
    SimulationMetadata,
    SimulationPerformance,
    _analysis_function,
    material_region_for_monitor,
)
from beamz.simulation.results.monitor import MonitorResults
from beamz.simulation.results.observe import renormalize_result


@dataclass(frozen=True)
class SimulationResults:
    """Store durable, immutable analysis outputs from simulation execution.

    ``SimulationResults`` contains monitor acquisitions and the metadata required
    to interpret them after the solver has finished. It deliberately does not own
    electromagnetic continuation state. A segmented :meth:`Simulation.advance`
    call returns a separate :class:`SimulationRun` whose ``state`` may be consumed
    independently while these results remain valid.

    Attributes
    ----------
    metadata : SimulationMetadata
        Frozen grid, time, dimensionality, coordinate, and optional material data.
    monitors : mapping of str to MonitorResults
        Immutable monitor outputs keyed by canonical monitor name.
    sources : tuple
        Immutable source specifications needed for normalization and analysis.
    normalization_source : int or None
        Source index currently used to normalize frequency-domain acquisitions.
        ``None`` means raw monitor data.
    source_launch_powers : tuple of float or None
        Internally calibrated launched powers corresponding to ``sources``.
    performance : SimulationPerformance or None
        JIT- and setup-free execution statistics for the completed run.

    Examples
    --------
    Normal execution returns this type directly:

    >>> results = sim.run()
    >>> output = results["output"]
    >>> flux = output.flux

    Segmented execution keeps results and state separate:

    >>> segment = sim.advance(num_steps=50)
    >>> results = segment.results
    >>> state = segment.state
    >>> not hasattr(results, "state")
    True

    Notes
    -----
    NumPy arrays are copied from runtime buffers and made read-only. Result mappings
    are also read-only, making a retained result safe across later continuations,
    buffer donation, cache clearing, and deletion of the originating simulation.
    """

    metadata: SimulationMetadata
    monitors: Mapping[str, MonitorResults] = field(default_factory=dict)
    sources: tuple[object, ...] = field(default_factory=tuple, repr=False)
    normalization_source: int | None = None
    source_launch_powers: tuple[float | None, ...] = field(
        default_factory=tuple, repr=False
    )
    termination: RunTermination | None = None
    performance: SimulationPerformance | None = None

    def __post_init__(self):
        if not isinstance(self.metadata, SimulationMetadata):
            raise TypeError("SimulationResults.metadata must be SimulationMetadata.")
        if not isinstance(self.monitors, Mapping):
            raise TypeError("SimulationResults.monitors must be a mapping.")
        for name, result in self.monitors.items():
            if not isinstance(name, str):
                raise TypeError("SimulationResults.monitors keys must be strings.")
            if not isinstance(result, MonitorResults):
                raise TypeError(
                    "SimulationResults.monitors values must be "
                    f"MonitorResults; got {type(result).__name__}."
                )
        monitors = {}
        for name, result in self.monitors.items():
            region = retained_sample_region(
                self.metadata, result.monitor, result.sample_region
            )
            monitors[name] = (
                result
                if region is result.sample_region
                else replace(result, sample_region=region)
            )
        object.__setattr__(self, "monitors", MappingProxyType(monitors))
        object.__setattr__(self, "sources", immutable_snapshot(tuple(self.sources)))
        launch_powers = tuple(self.source_launch_powers)
        if not launch_powers:
            launch_powers = (None,) * len(self.sources)
        if len(launch_powers) != len(self.sources):
            raise ValueError(
                "SimulationResults.source_launch_powers must match the number of sources."
            )
        object.__setattr__(
            self,
            "source_launch_powers",
            tuple(None if value is None else float(value) for value in launch_powers),
        )
        source = self.normalization_source
        if source is not None:
            source = int(source)
            if source < 0 or source >= len(self.sources):
                raise ValueError(
                    f"normalization source {source} is invalid for {len(self.sources)} sources."
                )
        object.__setattr__(self, "normalization_source", source)
        if self.termination is not None and not isinstance(
            self.termination, RunTermination
        ):
            raise TypeError(
                "SimulationResults.termination must be RunTermination or None."
            )
        if self.performance is not None and not isinstance(
            self.performance, SimulationPerformance
        ):
            raise TypeError(
                "SimulationResults.performance must be SimulationPerformance or None."
            )

    def __getitem__(self, name):
        """Return a named monitor result.

        Parameters
        ----------
        name : str
            Canonical monitor name.

        Returns
        -------
        MonitorResults
            Immutable acquisition associated with ``name``.

        Raises
        ------
        KeyError
            If no monitor with that name exists.

        Examples
        --------
        >>> fields = results["fields"]
        """
        return self.monitor(name)

    def monitor(self, name: str) -> MonitorResults:
        """Return the result-owned acquisition for a named monitor.

        Parameters
        ----------
        name : str
            Canonical monitor name.

        Returns
        -------
        MonitorResults
            Immutable monitor acquisition.

        Raises
        ------
        KeyError
            If ``name`` is absent from :attr:`monitors`.
        """
        return self.monitors[str(name)]

    def save(self, path):
        """Save detached results, DFTs, and full port material context to NPZ.

        The versioned format does not use pickle. Standard frequency monitors
        and source specifications are supported; unsupported custom values
        raise TypeError instead of being silently omitted.

        Parameters
        ----------
        path : str or os.PathLike
            Destination archive path, written without adding a suffix.

        Returns
        -------
        None
            Writes the archive to ``path``.

        Raises
        ------
        TypeError
            If a result contains an unsupported custom value or object array.
        """
        from beamz.simulation.results.analysis import save_results

        save_results(self, path)

    @classmethod
    def load(cls, path):
        """Load a versioned result archive without requiring a live simulation.

        Parameters
        ----------
        path : str or os.PathLike
            Path to an archive written by :meth:`save`.

        Returns
        -------
        SimulationResults
            Detached results with their monitor and port material context.

        Raises
        ------
        ValueError
            If the archive version or result representation is unsupported.
        """
        from beamz.simulation.results.analysis import load_results

        return load_results(path)

    def launched_power(self, source: int = 0) -> float:
        """Return the internally calibrated net power leaving a source.

        Mode sources use the discrete source-plane Poynting quadrature,
        including normal Yee interpolation and the launch-amplitude correction.
        Modal analysis uses the same measured-flux convention. Different plane
        offsets, finite apertures and discretization can affect downstream flux.
        Analytic Gaussian-beam sources
        report their requested power after discrete source-plane normalization.
        Neither path requires an additional runtime monitor or reference
        simulation.

        Parameters
        ----------
        source : int, default=0
            Index into the simulation's source sequence.

        Returns
        -------
        float
            Positive net launched power in watts.

        Raises
        ------
        ValueError
            If ``source`` is outside the configured source range.
        RuntimeError
            If BeamZ could not calibrate a finite positive launch power for the
            selected source type.

        Examples
        --------
        >>> input_power = results.launched_power(0)
        >>> transmission = results["output"].flux / input_power
        """
        source = int(source)
        if source < 0 or source >= len(self.sources):
            raise ValueError(
                f"source={source} is invalid for {len(self.sources)} sources."
            )
        power = self.source_launch_powers[source]
        if power is None or not np.isfinite(power) or power <= 0.0:
            raise RuntimeError(
                f"An internally calibrated launched power is unavailable for source {source}."
            )
        return float(power)

    def renormalize(self, source: int | None):
        """Return an immutable raw or source-normalized result view.

        Parameters
        ----------
        source : int or None
            Source index used for field-amplitude and power normalization. Pass
            ``None`` to recover raw frequency-domain acquisitions.

        Returns
        -------
        SimulationResults
            New immutable result sharing semantic metadata while owning the selected
            normalized monitor views.

        Raises
        ------
        ValueError
            If ``source`` is outside the configured source range.

        Examples
        --------
        >>> raw = results.renormalize(None)
        >>> source_1 = results.renormalize(1)

        Notes
        -----
        Renormalization always derives from retained raw acquisitions, so repeatedly
        switching source indices does not compound scaling or lose information.
        """
        return renormalize_result(self, source)

    def to_xarray(self):
        """Convert stored acquisitions to labeled xarray data.

        Returns
        -------
        xarray.Dataset
            Dataset containing recorded fields and coordinates derived from result
            metadata.

        Notes
        -----
        xarray integration is imported lazily. Conversion uses detached result data
        and does not require the original simulation or runtime state.
        """
        return _analysis_function("adapters", "to_xarray")(self)

    def mode(self, name: str):
        """Return labeled modal data for a named mode monitor.

        Parameters
        ----------
        name : str
            Name of a ``ModeMonitor`` acquisition.

        Returns
        -------
        ModeMonitorData
            Modal amplitudes, frequencies, mode metadata, and coordinates.

        Raises
        ------
        KeyError
            If no monitor named ``name`` exists.
        TypeError or ValueError
            If the named acquisition cannot be interpreted as modal data.
        """
        return _analysis_function("adapters", "mode_data")(self, name)

    def plot_field(self, *args, **kwargs):
        """Plot a recorded field frame or frequency-domain monitor field.

        Parameters
        ----------
        *args : object
            Optional positional monitor and component selectors. Prefer explicit
            keywords in reusable code.
        **kwargs : object
            Field-selection and Matplotlib options such as ``monitor_name``,
            ``field``, ``frequency``, ``frame``, ``val``, ``ax``, and
            ``figsize``. ``show`` defaults to ``False``.

        Returns
        -------
        tuple
            Matplotlib ``(figure, axes)`` objects.

        Examples
        --------
        >>> fig, ax = results.plot_field(
        ...     monitor_name="frequency_fields",
        ...     field="Ez",
        ...     val="abs",
        ... )
        """
        kwargs.setdefault("show", False)
        return _analysis_function("plotting", "plot_result_field")(
            self, *args, **kwargs
        )

    def plot(self, **kwargs):
        """Plot stored simulation data without forcing interactive display.

        Parameters
        ----------
        **kwargs : object
            Options accepted by :meth:`plot_field`. ``show`` defaults to ``False``.

        Returns
        -------
        tuple
            Matplotlib figure and axes.
        """
        # Import plotting lazily so numerical simulation does not require
        # visualization setup.
        kwargs.setdefault("show", False)
        return self.plot_field(**kwargs)

    def show(self, **kwargs):
        """Plot stored simulation data and request interactive display.

        Parameters
        ----------
        **kwargs : object
            Options accepted by :meth:`plot_field`. ``show`` defaults to ``True``.

        Returns
        -------
        tuple
            Matplotlib figure and axes.
        """
        # Route display through the plotting API while changing only the presentation
        # flag.
        kwargs.setdefault("show", True)
        return self.plot_field(**kwargs)

    @classmethod
    def from_run(
        cls,
        simulation,
        *,
        runtime_fields: CompiledGrid,
        monitor_results: Mapping[str, MonitorResults],
        store_full_materials: bool = False,
        source_launch_powers: tuple[float | None, ...] = (),
        performance: SimulationPerformance | None = None,
        completed_steps: int | None = None,
    ) -> "SimulationResults":
        """Detach canonical analysis outputs from completed runtime buffers.

        Parameters
        ----------
        simulation : Simulation
            Immutable specification that produced the run.
        runtime_fields : CompiledGrid
            Compiled lattice used for field shapes and material metadata.
        monitor_results : mapping of str to MonitorResults
            Already-decoded monitor acquisitions.
        store_full_materials : bool, default=False
            Retain the complete material grid instead of monitor-local regions only.
        source_launch_powers : tuple of float or None, optional
            Calibrated launched powers corresponding to simulation sources.
        performance : SimulationPerformance, optional
            Execution statistics measured around the compiled simulation executable.
        completed_steps : int, optional
            Number of acquired steps, including continuation history. Source DFT
            normalization uses this elapsed window rather than the future run horizon.

        Returns
        -------
        SimulationResults
            Detached, normalized, immutable analysis result.

        Raises
        ------
        TypeError
            If ``simulation`` or ``runtime_fields`` has the wrong type.

        Notes
        -----
        This is the internal construction boundary used by execution. Application
        code should obtain results from ``Simulation.run()`` or from the ``results``
        member of ``Simulation.advance()``.
        """
        from beamz.simulation.api import Simulation

        if not isinstance(simulation, Simulation):
            raise TypeError("SimulationResults.from_run requires Simulation.")
        if not isinstance(runtime_fields, CompiledGrid):
            raise TypeError("SimulationResults.from_run requires CompiledGrid.")
        metadata = SimulationMetadata.from_simulation(
            simulation,
            runtime_fields=runtime_fields,
            store_full_materials=store_full_materials,
        )
        if completed_steps is not None:
            metadata = replace(
                metadata,
                time=simulation.time[: int(completed_steps)],
            )
        monitor_results_dict = {
            name: replace(
                result,
                material_region=material_region_for_monitor(
                    result, runtime_fields=runtime_fields
                ),
            )
            for name, result in monitor_results.items()
        }
        result = cls(
            metadata=metadata,
            monitors=monitor_results_dict,
            sources=simulation.sources,
            source_launch_powers=source_launch_powers,
            performance=performance,
        )
        source = simulation.normalize_source
        return result.renormalize(None if not result.sources else source)
