"""Detached raw outputs produced by simulation execution."""

from __future__ import annotations

# Retain the names available from this module before its implementation was split.
from collections.abc import Mapping as Mapping
from dataclasses import dataclass, field
from dataclasses import replace as replace
from importlib import import_module as import_module
from types import MappingProxyType as MappingProxyType
from typing import Any as Any

import numpy as np  # noqa: F401

from beamz._cache_tokens import cache_token as cache_token
from beamz.design.grid import RectilinearGrid as RectilinearGrid
from beamz.devices._immutable import immutable_snapshot as immutable_snapshot
from beamz.devices._immutable import readonly_array as readonly_array
from beamz.devices._placement import SnappedRegion as SnappedRegion
from beamz.devices.monitors.compiler import CompiledMonitorSpec as CompiledMonitorSpec
from beamz.devices.monitors.geometry import (
    retained_sample_region as retained_sample_region,
)
from beamz.lattice import canonical_component_2d as canonical_component_2d
from beamz.simulation.model import SimulationState

from .model import CompiledGrid as CompiledGrid
from .model import RunConfig as RunConfig
from .observe import monitor_dft_component as monitor_dft_component
from .observe import monitor_dft_flux as monitor_dft_flux
from .observe import monitor_flux as monitor_flux
from .observe import monitor_frequencies as monitor_frequencies
from .observe import renormalize_result as renormalize_result
from .results_parts.metadata import (  # noqa: F401
    FieldMetadata,
    MaterialRegion,
    RunTermination,
    SimulationMetadata,
    SimulationPerformance,
    _analysis_function,
    _snapshot_material_region,
    material_region_for_monitor,
)
from .results_parts.monitor import MonitorResults as MonitorResults
from .results_parts.result import SimulationResults as SimulationResults


@dataclass(frozen=True, slots=True)
class SimulationRun:
    """Own the two outputs of a segmented simulation execution.

    ``SimulationRun`` is returned by :meth:`Simulation.advance`. It groups values
    for convenience without merging their lifecycles: ``results`` is immutable,
    detached analysis data, while ``state`` is an explicitly owned continuation
    value that may later be preserved, branched, or donated.

    Attributes
    ----------
    results : SimulationResults
        Durable monitor acquisitions and metadata for all samples accumulated up to
        the end of this segment.
    state : SimulationState
        Complete runtime carry positioned at the next timestep after this segment.

    Examples
    --------
    Continue a segmented execution:

    >>> first = sim.advance(num_steps=50)
    >>> results = first.results
    >>> second = sim.advance(state=first.state, num_steps=25)

    Branch from the same preserved state:

    >>> branch_a = sim.advance(state=first.state, num_steps=10)
    >>> branch_b = sim.advance(state=first.state, num_steps=10)

    Transfer state ownership for the lower-memory path:

    >>> final = sim.advance(
    ...     state=second.state,
    ...     num_steps=25,
    ...     donate_state=True,
    ... )
    >>> # Do not access second.state after the donating call.

    Notes
    -----
    ``Simulation.run()`` intentionally returns ``SimulationResults`` directly, not
    ``SimulationRun``, because a complete normal run has no public continuation
    requirement. Use this container only when working with ``advance``.

    Donation affects only the state passed into the next execution. It never
    invalidates ``results`` from this or any earlier segment.
    """

    results: SimulationResults
    state: SimulationState = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.results, SimulationResults):
            raise TypeError("SimulationRun.results must be SimulationResults.")
        if not isinstance(self.state, SimulationState):
            raise TypeError("SimulationRun.state must be SimulationState.")
