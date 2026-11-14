"""Source normalization and recorded monitor analysis."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from beamz.devices._immutable import readonly_array
from beamz.devices.monitors.monitors import _line_integral_scale_2d, _line_normal_2d
from beamz.devices.sources.specs import GaussianBeamSource, GaussianSource, ModeSource


@dataclass(frozen=True)
class SourceNormalization:
    """Store source-spectrum and launch-power normalization data."""

    waveform_spectrum: np.ndarray
    launch_power_ratio: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "waveform_spectrum",
            readonly_array(self.waveform_spectrum, dtype=np.complex128),
        )
        if self.launch_power_ratio is not None:
            object.__setattr__(
                self,
                "launch_power_ratio",
                readonly_array(self.launch_power_ratio, dtype=float),
            )

    @property
    def field_amplitude_norm(self) -> np.ndarray:
        """Return complex field-amplitude normalization by frequency."""
        spectrum = np.asarray(self.waveform_spectrum, dtype=np.complex128).reshape(-1)
        power = self.launch_power_ratio
        if power is None:
            return spectrum
        power = np.asarray(power, dtype=float).reshape(-1)
        if power.shape != spectrum.shape:
            return spectrum
        scale = np.ones_like(power)
        valid = np.isfinite(power) & (power > 1e-24)
        scale[valid] = np.sqrt(power[valid])
        return spectrum * scale

    @property
    def power_norm(self) -> np.ndarray:
        """Return real power normalization by frequency."""
        spectrum = np.asarray(self.waveform_spectrum, dtype=np.complex128).reshape(-1)
        scale = np.abs(spectrum) ** 2
        power = self.launch_power_ratio
        if power is None:
            return scale
        power = np.asarray(power, dtype=float).reshape(-1)
        if power.shape == spectrum.shape:
            valid = np.isfinite(power) & (power > 1e-24)
            scale[valid] *= power[valid]
        return scale


def _source_launch_power_ratio(source, freqs, *, fields=None, dt=None):
    if not isinstance(source, ModeSource):
        return None
    power = source.launch_power_normalization_spectrum(freqs, fields=fields, dt=dt)
    if power is None:
        return None
    power = np.asarray(power, dtype=float).reshape(-1)
    if power.shape != np.asarray(freqs).reshape(-1).shape:
        return None
    valid = np.isfinite(power) & (power > 1e-24)
    if not np.any(valid):
        return None
    return np.where(valid, power, 1.0)


def _signal_dft_spectrum(signal, sample_times, freqs):
    from beamz.devices.sources.time import _discrete_spectrum

    freqs = np.asarray(freqs, dtype=float).reshape(-1)
    times = np.asarray(sample_times, dtype=float).reshape(-1)
    signal = np.asarray(signal, dtype=float).reshape(-1)
    n = min(times.size, signal.size)
    if freqs.size == 0 or n == 0:
        return None
    times, signal = times[:n], signal[:n]
    spectrum = (2.0 / float(n)) * _discrete_spectrum(signal, times, freqs)
    return spectrum if np.any(np.abs(spectrum) > 1e-12) else None


def _sampled_source_spectrum(source, freqs, *, time):
    time = np.asarray(time, dtype=float).reshape(-1)
    if time.size == 0:
        return None
    dt = float(np.median(np.diff(time))) if time.size > 1 else 0.0
    if isinstance(source, GaussianSource):
        signal = np.asarray(source.signal, dtype=float).reshape(-1)
        sample_times = time[: signal.size]
    elif isinstance(source, (ModeSource, GaussianBeamSource)) and dt > 0.0:
        from beamz.devices.sources.time import sample_source_waveforms

        signal, _ = sample_source_waveforms(
            source.source_time,
            t0=float(time[0]),
            dt=dt,
            num_steps=time.size,
            total_steps=time.size,
        )
        signal = np.asarray(signal, dtype=float).reshape(-1)
        sample_times = time[: signal.size]
    else:
        return None
    return _signal_dft_spectrum(signal, sample_times, freqs)


def source_normalization(sources, freqs, *, time=None, fields=None, dt=None):
    """Interpret source configuration as a frequency-domain normalization."""
    return _source_normalization(
        sources, freqs, time=time, fields=fields, dt=dt, spectra={}
    )


def _source_normalization(sources, freqs, *, time, fields, dt, spectra):
    from beamz.simulation import observe as observation_runtime

    freqs = np.asarray(freqs, dtype=float).reshape(-1)
    if freqs.size == 0:
        return None
    for source in sources or ():
        key = (id(source), tuple(freqs))
        if key not in spectra:
            spectrum = (
                observation_runtime._sampled_source_spectrum(source, freqs, time=time)
                if time is not None
                else None
            )
            if spectrum is None and isinstance(
                source, (ModeSource, GaussianBeamSource)
            ):
                spectrum = source.source_spectrum(freqs, normalize=True)
            spectra[key] = spectrum
        spectrum = spectra[key]
        if spectrum is None:
            continue
        spectrum = np.asarray(spectrum, dtype=np.complex128).reshape(-1)
        if spectrum.shape == freqs.shape and np.any(np.abs(spectrum) > 1e-12):
            return SourceNormalization(
                spectrum,
                _source_launch_power_ratio(source, freqs, fields=fields, dt=dt),
            )
    return None


def source_field_amplitude_normalization(normalization):
    if normalization is None:
        return None
    value = (
        normalization.field_amplitude_norm
        if isinstance(normalization, SourceNormalization)
        else normalization
    )
    return np.asarray(value, dtype=np.complex128).reshape(-1)


def monitor_frequencies(monitor):
    recorded = np.asarray(monitor.dft_frequencies, dtype=float).reshape(-1)
    return recorded if recorded.size else np.asarray(monitor.monitor.freqs, dtype=float)


def monitor_dft_component(monitor, component):
    component = str(component)
    fields = monitor.dft_fields
    if component not in fields:
        raise ValueError(f"No DFT data recorded for component '{component}'.")
    values = np.asarray(fields[component], dtype=np.complex128)
    nfreq = monitor_frequencies(monitor).size
    if nfreq == 0:
        raise ValueError(f"Monitor '{monitor.name}' has no configured DFT frequencies.")
    values = values.reshape(nfreq, -1)
    weights = np.maximum(
        np.asarray(monitor.dft_weight_sum, dtype=float), 1e-18
    ).reshape(nfreq, 1)
    return (getattr(monitor, "dft_amplitude_scale", 2.0) / weights) * values


def _flux_component(monitor, component):
    values = monitor_dft_component(monitor, component)
    if component.startswith("H") and monitor.dft_base_dt:
        phase = np.exp(-1j * np.pi * monitor_frequencies(monitor) * monitor.dft_base_dt)
        values = values * phase[:, None]
    return values


def _optional_flux_component(monitor, component):
    if component in monitor.dft_fields:
        return _flux_component(monitor, component)
    for available in monitor.dft_fields:
        return np.zeros_like(_flux_component(monitor, available))
    raise ValueError(f"Monitor '{monitor.name}' has no recorded DFT fields.")


def monitor_dft_flux(monitor):
    dx = float(monitor.resolution or 1.0)
    config = monitor.monitor
    if 0 <= monitor.normal_axis <= 2:
        axis = ("x", "y", "z")[monitor.normal_axis]
        sign = monitor.normal_sign
    else:
        normal = _line_normal_2d(config.start, config.end)
        axis, sign = normal or (config.plane_normal, 1.0)

    def field(name):
        return _optional_flux_component(monitor, name)

    if axis == "x":
        component = field("Ey") * np.conjugate(field("Hz")) - field(
            "Ez"
        ) * np.conjugate(field("Hy"))
    elif axis == "y":
        component = field("Ez") * np.conjugate(field("Hx")) - field(
            "Ex"
        ) * np.conjugate(field("Hz"))
    else:
        component = field("Ex") * np.conjugate(field("Hy")) - field(
            "Ey"
        ) * np.conjugate(field("Hx"))

    integration_weights = np.asarray(
        getattr(monitor, "integration_weights", ()), dtype=float
    ).reshape(-1)
    if integration_weights.size == component.shape[1]:
        return 0.5 * np.real(
            np.sum(sign * component * integration_weights[None, :], axis=1)
        )
    measure = float(monitor.power_scale)
    if not measure:
        measure = dx * dx if config.is_3d else _line_integral_scale_2d(axis, dx, dx)
    return 0.5 * np.real(np.sum(sign * component, axis=1)) * measure


def normalization_from_result(results, result, *, source=None, spectra=None):
    """Derive source normalization from a raw run and monitor result."""
    materials = results.metadata.fields.materials or result.material_region
    sources = results.sources if source is None else (results.sources[int(source)],)
    return _source_normalization(
        sources,
        monitor_frequencies(result),
        time=results.metadata.time,
        fields=materials,
        dt=results.metadata.dt,
        spectra={} if spectra is None else spectra,
    )


def monitor_flux(result, results=None):
    del results
    if result.dft_fields:
        return monitor_dft_flux(result)
    return np.asarray(result.power_spectrum)


def _scale_frequency_axis(values, scale):
    array = np.asarray(values)
    scale = np.asarray(scale).reshape(-1)
    if not array.size or array.shape[0] != scale.size:
        return array
    divisor = scale.reshape((scale.size,) + (1,) * (array.ndim - 1))
    return np.divide(
        array,
        divisor,
        out=np.zeros_like(array, dtype=np.result_type(array, divisor)),
        where=np.abs(divisor) > 1e-24,
    )


def renormalize_result(result, source: int | None):
    """Return an immutable raw or source-normalized view of a run result."""
    if source is not None:
        source = int(source)
        if source < 0 or source >= len(result.sources):
            raise ValueError(
                f"normalization source {source} is invalid for "
                f"{len(result.sources)} sources."
            )

    raw_monitors = {
        name: replace(
            monitor,
            dft_fields=monitor._raw_dft_fields or {},
            power_spectrum=monitor._raw_power_spectrum,
        )
        for name, monitor in result.monitors.items()
    }
    raw_result = replace(result, monitors=raw_monitors, normalization_source=None)
    if source is None:
        return raw_result

    normalized = {}
    # Waveforms share one acquisition window; spatial launch calibration remains
    # monitor-specific. Keep this reuse local to one normalization operation.
    spectra: dict = {}
    for name, monitor in raw_monitors.items():
        normalization = normalization_from_result(
            raw_result, monitor, source=source, spectra=spectra
        )
        if normalization is None:
            normalized[name] = monitor
            continue
        field_scale = np.asarray(
            normalization.field_amplitude_norm, dtype=np.complex128
        )
        normalized[name] = replace(
            monitor,
            dft_fields={
                component: _scale_frequency_axis(values, field_scale)
                for component, values in monitor.dft_fields.items()
            },
            power_spectrum=_scale_frequency_axis(
                monitor.power_spectrum, normalization.power_norm
            ),
        )
    return replace(raw_result, monitors=normalized, normalization_source=source)
