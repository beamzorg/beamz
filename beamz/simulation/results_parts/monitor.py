"""Detached raw outputs produced by simulation execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

from beamz.devices._immutable import immutable_snapshot, readonly_array
from beamz.devices._placement import SnappedRegion
from beamz.devices.monitors.compiler import CompiledMonitorSpec
from beamz.devices.monitors.monitors import _Monitor
from beamz.simulation.model import RunConfig, SimulationState
from beamz.simulation.observe import (
    monitor_dft_component,
    monitor_dft_flux,
    monitor_flux,
    monitor_frequencies,
)

from .metadata import MaterialRegion


@dataclass(frozen=True)
class MonitorResults:
    """Store the immutable acquisitions produced by one monitor.

    Instances are created during execution and owned by :class:`SimulationResults`.
    Arrays and mappings are detached from JAX runtime buffers and made read-only, so
    retaining or analyzing monitor data never affects continuation state.

    Attributes
    ----------
    monitor : monitor specification
        Immutable monitor geometry and acquisition settings used for this result.
    fields : mapping of str to numpy.ndarray
        Recorded time-domain field frames keyed by component name.
    power_history, power_timestamps : numpy.ndarray
        Time-domain power samples and their physical times.
    power_spectrum : numpy.ndarray
        Complex frequency-domain flux samples after the selected source
        normalization is applied.
    dft_fields : mapping of str to numpy.ndarray
        Complex frequency-domain field samples keyed by component name.
    dft_frequencies : numpy.ndarray
        Frequencies in hertz corresponding to the first DFT axis.
    dft_weight_sum : numpy.ndarray
        Temporal-window normalization accumulated at each DFT frequency.
    dft_base_dt : float
        Base timestep in seconds used for DFT accumulation.
    resolution : float
        Spatial cell size in metres used for flux integration.
    normal_axis, normal_sign, power_scale
        Orientation and scaling metadata for flux monitors.
    field_times, field_steps : numpy.ndarray
        Physical times and completed-step indices for recorded field frames.
    objective_value : float or None
        Optional scalar objective associated with this acquisition.
    material_region : MaterialRegion or None
        Material snapshot colocated with the monitor for downstream modal analysis.
    sample_region : SnappedRegion or None
        Exact grid-aware spatial region retained from monitor compilation.

    Examples
    --------
    >>> results = sim.run()
    >>> monitor = results["fields"]
    >>> frames = monitor.fields["Ez"]
    >>> steps = monitor.field_steps

    Notes
    -----
    Different monitor types populate different fields. For example, a
    ``FieldRecorder`` fills ``fields`` and frame coordinates, while a frequency
    ``FieldMonitor`` fills ``dft_fields`` and ``dft_frequencies``. Empty arrays or
    mappings indicate that an acquisition was not configured.
    """

    monitor: _Monitor
    fields: Mapping[str, np.ndarray]
    power_history: np.ndarray
    power_timestamps: np.ndarray
    power_spectrum: np.ndarray
    dft_fields: Mapping[str, np.ndarray] = field(default_factory=dict)
    dft_frequencies: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )
    dft_weight_sum: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    dft_base_dt: float = 0.0
    # A real carrier contains half its phasor at positive frequency; a complex
    # exp(-i omega t) carrier contains the full phasor.
    dft_amplitude_scale: float = 2.0
    resolution: float = 0.0
    normal_axis: int = -1
    normal_sign: float = 1.0
    power_scale: float = 0.0
    integration_weights: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )
    _raw_dft_fields: Mapping[str, np.ndarray] | None = field(
        default=None, compare=False, repr=False
    )
    _raw_power_spectrum: np.ndarray | None = field(
        default=None, compare=False, repr=False
    )
    field_times: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    field_steps: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    objective_value: float | None = None
    material_region: MaterialRegion | None = field(
        default=None, compare=False, repr=False
    )
    sample_region: SnappedRegion | None = field(default=None, compare=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.monitor, _Monitor):
            raise TypeError("MonitorResults.monitor must be a canonical monitor.")
        if self.material_region is not None and not isinstance(
            self.material_region, MaterialRegion
        ):
            raise TypeError(
                "MonitorResults.material_region must be a MaterialRegion or None."
            )
        if self.sample_region is not None and not isinstance(
            self.sample_region, SnappedRegion
        ):
            raise TypeError(
                "MonitorResults.sample_region must be a SnappedRegion or None."
            )
        if not isinstance(self.fields, Mapping) or not isinstance(
            self.dft_fields, Mapping
        ):
            raise TypeError("MonitorResults field collections must be mappings.")
        object.__setattr__(self, "fields", immutable_snapshot(self.fields))
        object.__setattr__(self, "dft_fields", immutable_snapshot(self.dft_fields))
        for name in (
            "power_history",
            "power_timestamps",
            "power_spectrum",
            "field_times",
            "field_steps",
            "dft_frequencies",
            "dft_weight_sum",
            "integration_weights",
        ):
            object.__setattr__(self, name, readonly_array(getattr(self, name)))
        if self.field_times.shape != self.field_steps.shape:
            raise ValueError(
                "Recorded field times and steps must have matching shapes."
            )
        if self.objective_value is not None:
            object.__setattr__(self, "objective_value", float(self.objective_value))
        for name in ("dft_base_dt", "resolution", "normal_sign", "power_scale"):
            object.__setattr__(self, name, float(getattr(self, name)))
        object.__setattr__(self, "normal_axis", int(self.normal_axis))
        for name, current, snapshot in (
            ("_raw_dft_fields", self.dft_fields, immutable_snapshot),
            ("_raw_power_spectrum", self.power_spectrum, readonly_array),
        ):
            raw = getattr(self, name)
            object.__setattr__(self, name, current if raw is None else snapshot(raw))

    def get_dft_frequencies(self):
        """Return frequencies associated with frequency-domain acquisitions.

        Returns
        -------
        numpy.ndarray
            One-dimensional frequency coordinates in hertz.

        Notes
        -----
        This compatibility accessor applies the same frequency interpretation used
        by BeamZ analysis adapters. The returned data is derived from this immutable
        result and does not access simulation state.
        """
        return monitor_frequencies(self)

    def get_dft_component(self, component: str):
        """Return one complex frequency-domain field component.

        Parameters
        ----------
        component : str
            Public field component name, such as ``"Ez"`` or ``"Hy"``.

        Returns
        -------
        numpy.ndarray
            Complex DFT samples with frequency as the leading axis.

        Raises
        ------
        ValueError
            If the requested component was not acquired by this monitor.

        Examples
        --------
        >>> ez = results["frequency_fields"].get_dft_component("Ez")
        """
        return monitor_dft_component(self, component)

    def get_dft_flux(self):
        """Return the monitor's complex frequency-domain flux.

        Returns
        -------
        numpy.ndarray
            Complex flux indexed by :meth:`get_dft_frequencies`.

        Notes
        -----
        Orientation, integration scale, and source normalization have already been
        applied to this result view.
        """
        return monitor_dft_flux(self)

    @property
    def flux(self):
        """Return the most appropriate flux acquisition for this monitor.

        Returns
        -------
        numpy.ndarray
            Frequency-domain flux when configured, otherwise the recorded
            time-domain power history.

        Examples
        --------
        >>> transmitted = results["output"].flux
        """
        return monitor_flux(self)

    def snapshot(self, component: str, index: int = -1) -> Mapping[str, Any]:
        """Return one immutable view of a recorded field frame.

        Parameters
        ----------
        component : str
            Recorded field component to select.
        index : int, default=-1
            Frame index. Standard Python negative indexing is supported.

        Returns
        -------
        mapping
            Read-only mapping with ``field``, ``field_name``, physical time ``t``,
            and completed ``step``.

        Raises
        ------
        ValueError
            If the monitor did not record ``component``.
        IndexError
            If ``index`` is outside the recorded frame range.

        Examples
        --------
        >>> final = results["fields"].snapshot("Ez")
        >>> final["step"] == results["fields"].field_steps[-1]
        True
        """
        try:
            frames = self.fields[str(component)]
        except KeyError as exc:
            raise ValueError(
                f"Monitor did not record {component!r}; available: {tuple(self.fields)}"
            ) from exc
        return MappingProxyType(
            {
                "field": frames[index],
                "field_name": str(component),
                "t": float(self.field_times[index]),
                "step": int(self.field_steps[index]),
            }
        )

    @classmethod
    def from_compiled_state(
        cls,
        monitor: _Monitor,
        spec: CompiledMonitorSpec,
        state: SimulationState,
        config: RunConfig,
    ) -> "MonitorResults":
        """Detach one monitor's immutable data from packed runtime state.

        Parameters
        ----------
        monitor : monitor specification
            Public monitor associated with the compiled entry.
        spec : CompiledMonitorSpec
            Grid-aware packed acquisition plan.
        state : monitor-buffer object
            Completed runtime carry containing packed monitor arrays.
        config : run-config object
            Time-step, resolution, dimensionality, and plane metadata.

        Returns
        -------
        MonitorResults
            NumPy-backed, read-only monitor acquisitions independent of runtime state.

        Notes
        -----
        This constructor is part of BeamZ's execution-to-results boundary. Typical
        application code receives instances from ``Simulation.run()`` or
        ``Simulation.advance()`` rather than calling it directly.
        """
        fc = int(spec.freq_count)
        pc = int(spec.dft_point_count)
        if spec.recorder_index >= 0:
            recorder_count = int(np.asarray(state.recorded_counts[spec.recorder_index]))
            fields = {
                component: state.recorded_fields[buffer_index][:recorder_count]
                for component, buffer_index in zip(
                    spec.components, spec.field_buffer_indices, strict=True
                )
            }
            return cls(
                monitor=monitor,
                fields=fields,
                power_history=np.empty(0, dtype=float),
                power_timestamps=np.empty(0, dtype=float),
                power_spectrum=np.empty(0, dtype=np.complex64),
                field_times=np.asarray(
                    state.recorded_times[spec.recorder_index][:recorder_count],
                    dtype=float,
                ),
                field_steps=np.asarray(
                    state.recorded_steps[spec.recorder_index][:recorder_count],
                    dtype=int,
                ),
                sample_region=spec.sample_region,
            )
        dft_fields: dict[str, np.ndarray] = {}
        if spec.dft_enabled and fc > 0 and pc > 0:
            # Host conversion precedes complex construction to retain float64 decode.
            packed = spec.unpack_dft(
                np.asarray(state.dft_vec_re, dtype=np.float64),
                np.asarray(state.dft_vec_im, dtype=np.float64),
            )
            ordered = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
            dft_fields = {
                public: sign * packed[ordered.index(canonical)]
                for public, canonical, sign in zip(
                    spec.components,
                    spec.canonical_components,
                    spec.component_signs,
                    strict=True,
                )
            }
        configured_frequencies = np.asarray(monitor.freqs, dtype=float).reshape(-1)
        dft_frequencies = (
            configured_frequencies[:fc]
            if configured_frequencies.size >= fc
            else np.asarray(spec.freq_hz[:fc], dtype=float)
        )
        return cls(
            monitor=monitor,
            fields={},
            power_history=np.empty(0, dtype=float),
            power_timestamps=np.empty(0, dtype=float),
            power_spectrum=np.empty(0, dtype=np.complex64),
            dft_fields=dft_fields,
            dft_frequencies=dft_frequencies,
            dft_weight_sum=np.asarray(
                state.dft_weight_sum[spec.dft_slices()[1]],
                dtype=np.float64,
            ),
            dft_base_dt=float(config.dt),
            dft_amplitude_scale=1.0
            if np.iscomplexobj(getattr(state, "ex", 0.0))
            else 2.0,
            resolution=float(config.resolution),
            normal_axis=int(spec.normal_axis),
            normal_sign=float(spec.normal_sign),
            power_scale=float(spec.power_scale),
            integration_weights=np.asarray(spec.integration_weights),
            objective_value=None,
            sample_region=spec.sample_region,
        )
