"""Labeled-data and modal-analysis adapters for raw simulation results."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from beamz.analysis._coordinates import field_coordinates
from beamz.analysis.data import AnalysisData, analysis_data, analysis_inputs
from beamz.const import LIGHT_SPEED
from beamz.devices._immutable import readonly_array
from beamz.devices.monitors.monitors import FieldRecorder, ModeMonitor
from beamz.lattice import plane_axes_3d


def mode_data_to_dataframe(modes):
    """Return effective index, loss, and mode area as a pandas DataFrame."""
    import pandas as pd

    rows = []
    index = []
    dx_um = float(modes.resolution) * 1e6
    for frequency_index, frequency in enumerate(modes.frequencies):
        wavelength_um = LIGHT_SPEED / float(frequency) * 1e6
        wavelength_cm = LIGHT_SPEED / float(frequency) * 100.0
        for mode_index, neff in enumerate(np.atleast_1d(modes.neffs[frequency_index])):
            electric = np.asarray(modes.e_fields[frequency_index, mode_index])
            intensity = np.sum(np.abs(electric) ** 2, axis=0)
            numerator = (float(np.sum(intensity)) * dx_um**2) ** 2
            denominator = max(float(np.sum(intensity**2)) * dx_um**2, 1e-30)
            loss = float(max(np.imag(neff), 0.0))
            rows.append(
                {
                    "wavelength": wavelength_um,
                    "n eff": float(np.real(neff)),
                    "k eff": loss,
                    "loss (dB/cm)": 0.0
                    if loss == 0.0
                    else 4.0 * np.pi * loss / wavelength_cm * (10.0 / np.log(10.0)),
                    "mode area": numerator / denominator,
                }
            )
            index.append((float(frequency), int(mode_index)))
    return pd.DataFrame(
        rows,
        index=pd.MultiIndex.from_tuples(index, names=("f", "mode_index")),
    )


_FIELD_UNITS = {
    "Ex": "V/m",
    "Ey": "V/m",
    "Ez": "V/m",
    "Hx": "A/m",
    "Hy": "A/m",
    "Hz": "A/m",
    "permittivity": "relative",
}


def _xarray():
    import xarray as xr

    return xr


def _spatial_dims(ndim, plane_2d="xy"):
    return (
        ("z", "y", "x")
        if ndim == 3
        else {"xy": ("y", "x"), "xz": ("z", "x"), "yz": ("z", "y")}[plane_2d]
    )


def _time_coords(length, *, times=None, steps=None):
    if times is not None and len(times) == length:
        coords = {"t": ("t", np.asarray(times, dtype=float), {"units": "s"})}
        if steps is not None and len(steps) == length:
            coords["step"] = ("t", np.asarray(steps, dtype=int), {})
        return "t", coords
    coords = {"frame": ("frame", np.arange(length, dtype=int), {})}
    if steps is not None and len(steps) == length:
        coords["step"] = ("frame", np.asarray(steps, dtype=int), {})
    return "frame", coords


def _field_data_array(values, *, name: str, data: AnalysisData):
    arr = np.asarray(values)
    metadata = data.coordinates
    if arr.ndim == 0:
        raise ValueError(f"Field {name!r} must have at least one dimension.")
    if data.frequencies.size and name not in {"power", "permittivity"}:
        dims = ("f", *tuple(f"sample_{i}" for i in range(1, arr.ndim)))
        coords = {"f": ("f", data.frequencies, {"units": "Hz"})}
        frame = "sample"
    else:
        monitor = data.monitor_geometry
        sliced = isinstance(monitor, FieldRecorder) and monitor.region == "slice"
        timed = name != "permittivity" and (sliced or arr.ndim in {1, 3, 4})
        if arr.ndim == 1:
            arr = arr[:, None]
        shape = arr.shape[1:] if timed else arr.shape
        dims, coords = (
            _time_coords(
                arr.shape[0], times=metadata.time, steps=data.fields.get("step")
            )
            if timed
            else (None, {})
        )
        if isinstance(monitor, FieldRecorder) and sliced and metadata.is_3d:
            assert data.sample_region is not None
            axes = plane_axes_3d(monitor.plane_normal)
            coordinates = {
                axis: values - metadata.coordinate_offset["xyz".index(axis)]
                for axis, values in zip(
                    axes, data.sample_region.sample_coordinates, strict=True
                )
            }
        elif not sliced and name in _FIELD_UNITS:
            axes = _spatial_dims(len(shape), metadata.plane_2d)
            coordinates = field_coordinates(metadata, name, shape)
        else:
            axes = (
                ("s",)
                if len(shape) == 1
                else tuple(f"sample_{i}" for i in range(len(shape)))
            )
            coordinates = None
        frame = "public" if coordinates is not None else "sample"
        coords.update(
            {
                axis: (axis, coordinates[axis], {"units": "m"})
                if coordinates is not None
                else (axis, np.arange(length), {})
                for axis, length in zip(axes, shape, strict=True)
            }
        )
        dims = (dims, *axes) if timed else axes
    return _xarray().DataArray(
        arr,
        dims=dims,
        coords=coords,
        name=name,
        attrs={
            "component": str(name),
            "coordinate_frame": frame,
            "units": _FIELD_UNITS.get(str(name), ""),
            "design_width": metadata.width,
            "design_height": metadata.height,
            "design_depth": metadata.depth,
        },
    )


def to_xarray(results):
    """Return stored simulation fields as an xarray Dataset."""
    xr = _xarray()
    if isinstance(results, xr.Dataset):
        return results
    inputs = analysis_inputs(results)
    data = next(
        (
            item
            for item in inputs.values()
            if not item.frequencies.size
            and any(name not in {"power", "flux", "step"} for name in item.fields)
        ),
        next(iter(inputs.values())),
    )
    data_vars = {
        name: _field_data_array(values, name=name, data=data)
        for name, values in data.fields.items()
        if name not in {"power", "flux", "step"}
    }
    return xr.Dataset(
        data_vars=data_vars,
        attrs={
            "beamz_kind": "SimulationResults",
            "coordinate_frame": "public",
            "resolution": data.resolution,
            "plane_2d": data.plane_2d,
            "design_width": data.coordinates.width,
            "design_height": data.coordinates.height,
            "design_depth": data.coordinates.depth,
        },
    )


def monitor_to_xarray(result):
    """Return one canonical analysis contract as an xarray Dataset."""
    data = analysis_data(result)
    return _xarray().Dataset(
        {
            name: _field_data_array(values, name=name, data=data)
            for name, values in data.fields.items()
            if name not in {"t", "step"} and np.asarray(values).size
        },
        attrs={"monitor_name": data.name},
    )


@dataclass(frozen=True)
class ModeMonitorData:
    """Store labeled modal amplitudes and projection diagnostics.

    Parameters
    ----------
    monitor : object
        Detached mode-monitor configuration.
    amps : array-like or xarray.DataArray
        Complex forward/backward modal amplitudes in sqrt(W). In 3D, power is
        defined by the measured basis's discrete Poynting flux, matching the
        source calibration and flux monitor at equivalent sampling planes.
    flux : array-like or xarray.DataArray, optional
        Total measured flux in watts.
    modal_flux : array-like or xarray.DataArray, optional
        Sampled flux reconstructed from projected modes, including overlap cross
        terms. Compare this to ``flux``, not a sum of squared modal amplitudes.
    projection_residual : array-like or xarray.DataArray, optional
        Relative field reconstruction residual.
    condition_number : array-like or xarray.DataArray, optional
        Conditioning of the modal overlap system.
    """

    monitor: object
    amps: Any
    flux: Any = None
    modal_flux: Any = None
    projection_residual: Any = None
    condition_number: Any = None

    def __post_init__(self):
        for name in (
            "amps",
            "flux",
            "modal_flux",
            "projection_residual",
            "condition_number",
        ):
            value = getattr(self, name)
            if value is not None:
                if type(value).__module__.startswith("xarray"):
                    value = value.copy(deep=True)
                    value.data = readonly_array(value.data)
                else:
                    value = readonly_array(value)
                object.__setattr__(self, name, value)

    def to_xarray(self):
        """Return amplitudes and diagnostics as an xarray Dataset."""
        return _xarray().Dataset(
            data_vars={
                name: value
                for name, value in vars(self).items()
                if name != "monitor" and value is not None
            }
        )

    @property
    def power(self):
        """Directional modal contributions in discrete Poynting-flux watts.

        These are individual mode contributions. In a coherent, nonorthogonal
        basis their sum is not the reconstructed flux; use ``modal_flux`` for
        measured-field power accounting.
        """
        value = np.abs(self.amps) ** 2
        if hasattr(value, "rename"):
            value = value.rename("power")
            value.attrs = {"units": "W", "power_convention": "modal"}
        return value

    @property
    def unresolved_flux(self):
        """Measured flux minus its sampled modal reconstruction, in W."""
        if self.flux is None or self.modal_flux is None:
            raise ValueError("Flux accounting requires flux and modal_flux.")
        value = self.flux - self.modal_flux
        if hasattr(value, "rename"):
            value = value.rename("unresolved_flux")
            value.attrs = {"units": "W", "power_convention": "sampled"}
        return value

    def normalized_to(self, reference, *, mode_index=0, direction="+", power=1.0):
        """Return a view referenced to one incident mode's power spectrum.

        ``reference`` is modal data from a matched incident monitor or reference
        simulation, in the same raw/source-normalized convention as this data.
        Frequency coordinates must match exactly. Both flux quantities use the
        same scale as the amplitudes; their sampled-power convention is retained.
        Residuals and condition numbers are dimensionless and remain unchanged.
        """
        if not isinstance(reference, ModeMonitorData):
            raise TypeError("A modal power reference must be ModeMonitorData.")
        if direction not in ("+", "-"):
            raise ValueError("Reference direction must be '+' or '-'.")
        if not np.isfinite(power) or power <= 0:
            raise ValueError("Reference power must be finite and positive.")
        if not hasattr(self.amps, "coords") or not hasattr(reference.amps, "coords"):
            raise TypeError(
                "Modal reference normalization requires labeled amplitudes."
            )
        if not np.array_equal(self.amps.coords["f"], reference.amps.coords["f"]):
            raise ValueError("Modal reference frequencies must match exactly.")
        incident = reference.power.sel(
            direction=direction, mode_index=mode_index, drop=True
        )
        if np.any(~np.isfinite(incident)) or np.any(incident <= 0):
            raise ValueError("Incident modal power must be finite and positive.")
        scale = power / incident
        return replace(
            self,
            amps=self.amps * np.sqrt(scale),
            flux=None if self.flux is None else self.flux * scale,
            modal_flux=None if self.modal_flux is None else self.modal_flux * scale,
        )


def _data_array_1d(values, *, name, freqs):
    return _xarray().DataArray(
        np.asarray(values),
        dims=("f",),
        coords={"f": ("f", np.asarray(freqs, dtype=float), {"units": "Hz"})},
        name=name,
    )


def mode_data(results, name):
    """Project one raw mode-monitor result into labeled modal data.

    Parameters
    ----------
    results : SimulationResults
        Detached simulation results containing the named mode monitor.
    name : str
        Monitor result name.

    Returns
    -------
    ModeMonitorData
        Forward/backward amplitudes, flux, and projection diagnostics labeled by
        frequency and mode.

    Raises
    ------
    KeyError
        If no monitor result exists under ``name``.
    """
    from beamz.analysis import sparameters as _sp
    from beamz.analysis.mode_projection import _monitor_port

    data = analysis_data(results, name)
    monitor = data.monitor_geometry
    if not isinstance(monitor, ModeMonitor):
        raise TypeError(f"Monitor {name!r} is not a mode monitor.")
    freqs = data.frequencies
    num_modes = int(monitor.mode_spec.num_modes or 1)
    monitor_name = str(monitor.name or name)
    ports = [
        _monitor_port(
            monitor, idx, None if data.is_3d else data.coordinates.polarization_2d
        )
        for idx in range(num_modes)
    ]
    waves = _sp._extract_port_waves_dft({monitor_name: data}, ports, freqs)
    amps = np.zeros((freqs.size, 2, num_modes), dtype=np.complex128)
    for idx, port in enumerate(ports):
        wave = waves[port.name]
        positive_selector, negative_selector = _sp._wave_selectors(
            port, is_3d=data.is_3d
        )
        amps[:, 0, idx] = wave[f"a_{positive_selector}"]
        amps[:, 1, idx] = wave[f"a_{negative_selector}"]
    diagnostics = waves[ports[0].name]
    default = np.full(freqs.size, np.nan)
    modal_flux = np.asarray(
        diagnostics.get("projected_signed_power", default), dtype=float
    )
    xr = _xarray()
    amp_data = xr.DataArray(
        amps,
        dims=("f", "direction", "mode_index"),
        coords={
            "f": ("f", freqs, {"units": "Hz"}),
            "direction": ("direction", np.asarray(["+", "-"], dtype=object)),
            "mode_index": ("mode_index", np.arange(num_modes)),
        },
        name="amps",
        attrs={"monitor_name": monitor_name},
    )
    flux = np.asarray(data.fields.get("flux", np.full(freqs.size, np.nan)))
    return ModeMonitorData(
        monitor=monitor,
        amps=amp_data,
        flux=_data_array_1d(flux, name="flux", freqs=freqs),
        modal_flux=_data_array_1d(modal_flux, name="modal_flux", freqs=freqs),
        projection_residual=_data_array_1d(
            diagnostics.get("projection_residual", default),
            name="projection_residual",
            freqs=freqs,
        ),
        condition_number=_data_array_1d(
            diagnostics.get("condition_number", default),
            name="condition_number",
            freqs=freqs,
        ),
    )
