"""Transparent GDSFactory-to-BeamZ simulation preparation.

This module is deliberately part of :mod:`beamz.design`: it converts a
GDSFactory component into ordinary BeamZ geometry and devices without adding a
solver-integration dependency.  The returned setup exposes the native
:class:`~beamz.design.core.Design`, :class:`~beamz.devices.ports.Port`, and
:class:`~beamz.simulation.api.Simulation` values so applications can inspect or
modify every numerical choice before execution.

Install the optional layout dependency with ``pip install 'beamz[gds]'``.
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from beamz.analysis import SParameterResult, s_parameters
from beamz.const import LIGHT_SPEED, µm
from beamz.design.gds import ImportedComponent, import_component
from beamz.design.grid_spec import GridSpec
from beamz.design.materials import Material
from beamz.devices.boundaries import PML
from beamz.devices.modes import ModeSpec
from beamz.devices.ports import Port
from beamz.simulation import AutoTermination, Simulation, SimulationResults

Layer = tuple[int, int]


def _positive(value: float, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


@dataclass(frozen=True, slots=True)
class Settings:
    """Numerical policy for a GDSFactory component simulation.

    All lengths are in metres and wavelengths are vacuum wavelengths in metres.
    The policy intentionally exposes domain padding, PML, grid, and run duration;
    :func:`prepare` does not silently select these values from a PDK.
    """

    layer: Layer = (1, 0)
    wavelengths: tuple[float, float] = (1.50 * µm, 1.60 * µm)
    wavelength_points: int = 21
    n_core: float = 3.47
    n_clad: float = 1.44
    core_thickness: float = 0.22 * µm
    clad_below: float = 1.0 * µm
    clad_above: float = 1.0 * µm
    xy_padding: float = 3.0 * µm
    z_padding: float = 1.0 * µm
    pml_thickness: float = 1.0 * µm
    port_overlap: float = 0.0
    run_time: float = 2.0e-12
    grid_spec: GridSpec | None = None
    mode_spec: ModeSpec = field(
        default_factory=lambda: ModeSpec(num_modes=1, mode_index=0, polarization="te")
    )

    def __post_init__(self) -> None:
        layer = tuple(int(value) for value in self.layer)
        if len(layer) != 2:
            raise ValueError("layer must be a (layer, datatype) pair.")
        wavelengths = tuple(float(value) for value in self.wavelengths)
        if len(wavelengths) != 2:
            raise ValueError("wavelengths must contain exactly (start, stop).")
        if not all(np.isfinite(value) and value > 0.0 for value in wavelengths):
            raise ValueError("wavelengths must be finite and positive.")
        if wavelengths[0] >= wavelengths[1]:
            raise ValueError("wavelengths must be strictly increasing.")
        if int(self.wavelength_points) < 1:
            raise ValueError("wavelength_points must be at least one.")
        for name in (
            "n_core",
            "n_clad",
            "core_thickness",
            "clad_below",
            "clad_above",
            "xy_padding",
            "z_padding",
            "pml_thickness",
            "run_time",
        ):
            _positive(getattr(self, name), name=name)
        if float(self.port_overlap) < 0.0 or not np.isfinite(float(self.port_overlap)):
            raise ValueError("port_overlap must be finite and non-negative.")
        if self.grid_spec is not None and not isinstance(self.grid_spec, GridSpec):
            raise TypeError("grid_spec must be a GridSpec or None.")
        if not isinstance(self.mode_spec, ModeSpec):
            raise TypeError("mode_spec must be a ModeSpec.")
        object.__setattr__(self, "layer", layer)
        object.__setattr__(self, "wavelengths", wavelengths)
        object.__setattr__(self, "wavelength_points", int(self.wavelength_points))

    @property
    def frequencies(self) -> np.ndarray:
        """Return ascending frequency samples shared by mode monitors."""
        wl = np.linspace(*self.wavelengths, self.wavelength_points, dtype=float)
        return np.sort(LIGHT_SPEED / wl)

    @property
    def center_frequency(self) -> float:
        """Return the arithmetic center frequency in hertz."""
        return float(np.mean(self.frequencies))

    @property
    def frequency_width(self) -> float:
        """Return a Gaussian bandwidth that covers the requested samples."""
        freqs = self.frequencies
        return max(float(freqs[-1] - freqs[0]), 0.05 * self.center_frequency)

    def resolved_grid_spec(self) -> GridSpec:
        """Return the requested grid or a wavelength-aware default policy."""
        return self.grid_spec or GridSpec.auto(
            wavelength=float(np.mean(self.wavelengths)), min_steps_per_wvl=12.0
        )

    def updated_copy(self, **changes: Any) -> Settings:
        """Return a new validated settings value with selected fields replaced."""
        return replace(self, **changes)


@dataclass(frozen=True, slots=True)
class PortMetadata:
    """Trace one native BeamZ port back to a GDSFactory port datum.

    ``outward_direction`` is the direction from the component to the external
    circuit; ``port.direction`` is the opposite, inward direction used to launch
    a BeamZ mode.  Sources and monitors share ``datum_center`` in the first
    release, while the appended waveguide segment keeps that plane clear of PML.
    """

    name: str
    datum_center: tuple[float, float, float]
    outward_direction: str
    inward_direction: str
    axis: str
    size: tuple[float, float, float]


def _opposite(direction: str) -> str:
    return ("-" if direction.startswith("+") else "+") + direction[1:]


def _port_metadata(ports: Sequence[Port]) -> tuple[PortMetadata, ...]:
    return tuple(
        PortMetadata(
            name=port.name,
            datum_center=port.center,
            outward_direction=_opposite(port.signed_direction),
            inward_direction=port.signed_direction,
            axis=port.axis,
            size=port.size,
        )
        for port in ports
    )


@dataclass(frozen=True, slots=True)
class PreparedComponent:
    """An inspectable GDSFactory component translated to native BeamZ values.

    The object owns no runtime state.  :meth:`simulation_for` returns a new
    immutable :class:`~beamz.Simulation` for a selected source port and leaves
    this setup unchanged.
    """

    imported: ImportedComponent
    settings: Settings
    component: Any = field(repr=False, compare=False)
    layer_stack: Any | None = field(default=None, repr=False, compare=False)
    material_map: Mapping[str, Material | float] | None = field(
        default=None, repr=False, compare=False
    )

    @property
    def design(self):
        """Return the native BeamZ geometry used by every derived simulation."""
        return self.imported.design

    @property
    def ports(self) -> tuple[Port, ...]:
        """Return native modal ports at the named GDSFactory port datums."""
        return self.imported.ports

    @property
    def port_metadata(self) -> tuple[PortMetadata, ...]:
        """Return the documented source/monitor and port-direction convention."""
        return _port_metadata(self.ports)

    @property
    def component_name(self) -> str:
        """Return the resolved GDSFactory component name."""
        return self.imported.component_name

    def port(self, name: str) -> Port:
        """Return a native port by its stable GDSFactory name."""
        return self.imported.port(name)

    def simulation_template(self) -> Simulation:
        """Build a source-free native BeamZ simulation with all modal monitors."""
        monitors = tuple(
            port.to_monitor(self.settings.frequencies) for port in self.ports
        )
        return Simulation(
            design=self.design,
            monitors=monitors,
            boundaries=(
                PML(thickness=self.settings.pml_thickness, formulation="cpml"),
            ),
            grid_spec=self.settings.resolved_grid_spec(),
            run_time=self.settings.run_time,
            normalize_source=0,
        )

    def simulation_for(self, source_port: str) -> Simulation:
        """Return a native BeamZ simulation excited through ``source_port``.

        The source and every modal monitor are ordinary BeamZ device objects.
        Call ``updated_copy`` on the returned simulation to alter any numerical
        control before execution.
        """
        port = self.port(source_port)
        source = port.to_source(
            self.settings.center_frequency,
            self.settings.frequency_width,
            source_time=None,
        )
        return self.simulation_template().updated_copy(sources=(source,))

    def estimate_resources(self, source_port: str | None = None) -> dict[str, Any]:
        """Return a pre-run cell and memory estimate.

        The lightweight estimate is available before native raster/compile support
        is loaded.  ``compiled_estimate`` is included when the optional native
        raster extension is available and offers BeamZ's more detailed allocation
        report.
        """
        port = self.ports[0].name if source_port is None else source_port
        simulation = self.simulation_for(port)
        shape = tuple(int(value) for value in simulation.grid.shape)
        cells = int(np.prod(shape))
        # Six E/H fields plus common material and boundary work arrays.  This is a
        # deliberately conservative planning figure, not a backend allocation claim.
        estimate: dict[str, Any] = {
            "grid_shape": shape,
            "grid_cells": cells,
            "estimated_memory_bytes": cells * 12 * np.dtype(np.float32).itemsize,
            "estimated_memory_gb": cells * 12 * np.dtype(np.float32).itemsize / 1e9,
            "n_simulations": len(self.ports),
            "compiled_estimate": None,
        }
        # A source checkout without the maturin extension can still display a
        # useful setup preview; built wheels provide the detailed estimate.
        with suppress(ImportError):
            estimate["compiled_estimate"] = simulation.memory_estimate()
        return estimate

    def preview(self, source_port: str | None = None, **kwargs: Any):
        """Plot geometry, PML, source, and modal-monitor placement before a run."""
        port = self.ports[0].name if source_port is None else source_port
        return self.simulation_for(port).plot(**kwargs)

    def run_sparameters(
        self,
        excitations: str | Sequence[str] = "all",
        *,
        termination: AutoTermination | None = None,
        progress: bool = False,
        sharding: Any = None,
        min_incident_db: float = -40.0,
        mode_strategy: str = "per_frequency",
    ) -> ComponentSimulationResults:
        """Run one local BeamZ simulation per requested input port.

        This is a convenience over :meth:`simulation_for`, not a distinct solver
        API: every run uses ordinary BeamZ sources, monitors, and detached
        ``SimulationResults``.  ``excitations='all'`` returns the full named
        S-matrix; a sequence, or one port name, computes selected columns.
        """
        if excitations == "all":
            source_names = tuple(port.name for port in self.ports)
        elif isinstance(excitations, str):
            source_names = (excitations,)
        else:
            source_names = tuple(str(name) for name in excitations)
        if not source_names:
            raise ValueError("excitations must select at least one source port.")
        if len(set(source_names)) != len(source_names):
            raise ValueError("excitations must not contain duplicate port names.")
        unknown = sorted(set(source_names).difference(port.name for port in self.ports))
        if unknown:
            raise ValueError(f"excitations contains unknown port(s): {unknown}.")

        native_results: dict[str, SimulationResults] = {}
        extracted: dict[str, SParameterResult] = {}
        combined: dict[tuple[str, str], np.ndarray] = {}
        diagnostics: dict[str, Any] = {}
        for source_name in source_names:
            result = self.simulation_for(source_name).run(
                termination=termination,
                progress=progress,
                sharding=sharding,
            )
            column = s_parameters(
                result,
                source_port=source_name,
                ports=self.ports,
                frequencies=self.settings.frequencies,
                min_incident_db=min_incident_db,
                mode_strategy=mode_strategy,
            )
            native_results[source_name] = result
            extracted[source_name] = column
            combined.update(column.s_matrix)
            diagnostics[source_name] = column.diagnostics

        s_matrix = SParameterResult(
            s_matrix=combined,
            frequencies=self.settings.frequencies,
            diagnostics={
                "source_columns": diagnostics,
                "port_order": tuple(port.name for port in self.ports),
                "normalization": "modal incident amplitude at source port datum",
                "mode_strategy": str(mode_strategy).lower(),
            },
        )
        return ComponentSimulationResults(
            setup=self,
            sparameters=s_matrix,
            native_results=native_results,
            source_columns=extracted,
            provenance=_provenance(self, source_names),
        )


def _provenance(
    setup: PreparedComponent, source_names: Sequence[str]
) -> Mapping[str, object]:
    """Build serializable, result-owned setup provenance."""
    import beamz

    return MappingProxyType(
        {
            "beamz_version": beamz.__version__,
            "component_name": setup.component_name,
            "source_ports": tuple(source_names),
            "port_order": tuple(port.name for port in setup.ports),
            "frequencies_hz": tuple(
                float(value) for value in setup.settings.frequencies
            ),
            "wavelengths_m": setup.settings.wavelengths,
            "layer": setup.settings.layer,
            "pml_thickness_m": setup.settings.pml_thickness,
            "run_time_s": setup.settings.run_time,
            "normalization": "S[out, in] = outgoing/outgoing modal amplitude at port datums",
        }
    )


@dataclass(frozen=True, slots=True)
class ComponentSimulationResults:
    """Detached GDSFactory-component results with native BeamZ provenance.

    The result owns named S-parameters plus the native BeamZ results for each
    excitation.  It never retains mutable simulation state.
    """

    setup: PreparedComponent
    sparameters: SParameterResult
    native_results: Mapping[str, SimulationResults]
    source_columns: Mapping[str, SParameterResult]
    provenance: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.setup, PreparedComponent):
            raise TypeError("setup must be a PreparedComponent.")
        if not isinstance(self.sparameters, SParameterResult):
            raise TypeError("sparameters must be an SParameterResult.")
        object.__setattr__(
            self, "native_results", MappingProxyType(dict(self.native_results))
        )
        object.__setattr__(
            self, "source_columns", MappingProxyType(dict(self.source_columns))
        )
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    def plot_sparameters(self, *, ax: Any = None, db: bool = True):
        """Plot the named S-matrix columns against vacuum wavelength."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        wavelength_um = LIGHT_SPEED / self.sparameters.frequencies / µm
        for (output, source), values in self.sparameters.s_matrix.items():
            magnitude = np.abs(np.asarray(values))
            data = 20.0 * np.log10(np.maximum(magnitude, 1e-15)) if db else magnitude
            ax.plot(wavelength_um, data, label=f"S{output},{source}")
        ax.set_xlabel("Wavelength (um)")
        ax.set_ylabel("Magnitude (dB)" if db else "Magnitude")
        ax.legend()
        return ax.figure, ax

    def plot_field(self, excitation: str, *args: Any, **kwargs: Any):
        """Plot a field from one result-owned native BeamZ execution."""
        return self.native_results[str(excitation)].plot_field(*args, **kwargs)

    def check_reciprocity(self) -> Mapping[str, object]:
        """Return pairwise reciprocal-column residuals for available S entries."""
        residuals: dict[tuple[str, str], float] = {}
        matrix = self.sparameters.s_matrix
        for (output, source), values in matrix.items():
            reverse = matrix.get((source, output))
            if reverse is not None and source <= output:
                residuals[(output, source)] = float(
                    np.max(np.abs(np.asarray(values) - np.asarray(reverse)))
                )
        return MappingProxyType(
            {
                "pairwise_max_abs_error": MappingProxyType(residuals),
                "max_abs_error": max(residuals.values(), default=float("nan")),
                "complete": len(residuals) > 0,
            }
        )

    def check_passivity(self) -> Mapping[str, np.ndarray]:
        """Return guided-output power sums for each available source column."""
        frequencies = self.sparameters.frequencies
        values: dict[str, np.ndarray] = {}
        for source in self.source_columns:
            total = np.zeros(frequencies.shape, dtype=float)
            for port in self.setup.ports:
                coefficient = self.sparameters.s_matrix.get((port.name, source))
                if coefficient is not None:
                    total += np.abs(np.asarray(coefficient)) ** 2
            total.setflags(write=False)
            values[source] = total
        return MappingProxyType(values)


def prepare(
    component: Any = "mmi1x2",
    *,
    layer_stack: Any | None = None,
    settings: Settings | None = None,
    material_map: Mapping[str, Material | float] | None = None,
    component_settings: Mapping[str, Any] | None = None,
) -> PreparedComponent:
    """Prepare a GDSFactory component for explicit, local BeamZ simulation.

    The function converts component geometry, adds port extensions to the domain
    boundary, and returns native BeamZ geometry and ports.  It does not compile or
    execute a simulation.  Pass an explicit ``layer_stack`` and ``material_map``
    for PDK-aware vertical geometry; otherwise the settings' simple core/cladding
    model is used.
    """
    resolved = Settings() if settings is None else settings
    if not isinstance(resolved, Settings):
        raise TypeError("settings must be beamz.design.gdsfactory.Settings or None.")
    imported = import_component(
        component,
        layer=resolved.layer,
        n_core=resolved.n_core,
        n_clad=resolved.n_clad,
        core_thickness=resolved.core_thickness,
        clad_below=resolved.clad_below,
        clad_above=resolved.clad_above,
        xy_padding=resolved.xy_padding,
        z_padding=resolved.z_padding,
        extend_ports=True,
        port_overlap=resolved.port_overlap,
        settings=component_settings,
        layer_stack=layer_stack,
        material_map=material_map,
    )
    if not imported.ports:
        raise ValueError("GDSFactory component has no optical ports to simulate.")
    ports = tuple(
        port.updated_copy(mode_spec=resolved.mode_spec) for port in imported.ports
    )
    imported = ImportedComponent(
        design=imported.design,
        ports=ports,
        component_name=imported.component_name,
        world_origin=imported.world_origin,
    )
    return PreparedComponent(
        imported=imported,
        settings=resolved,
        component=component,
        layer_stack=layer_stack,
        material_map=material_map,
    )


prepare_component = prepare


__all__ = [
    "ComponentSimulationResults",
    "PortMetadata",
    "PreparedComponent",
    "Settings",
    "prepare",
    "prepare_component",
]
