"""Differentiable objectives sharing BeamZ's ordinary modal-analysis conventions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Real

import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from beamz.analysis.data import analysis_data
from beamz.analysis.modal_projection.geometry import (
    _modal_projection_spatial_phase,
    _monitor_projection_phase,
)
from beamz.analysis.mode_projection import _build_port_projection
from beamz.devices.modes.specs import ModeSpec
from beamz.devices.monitors import ModeMonitor
from beamz.devices.ports import Port
from beamz.simulation.observe import normalization_from_result


class ModalObjective:
    """Composable scalar modal objective; all optimization objectives are maximized.

    Combine objectives with ``+``, ``-``, and multiplication by finite real weights.
    For example, ``transmission - 0.2 * reflection`` rewards transmission while
    penalizing reflection. A weighted combination uses one forward simulation.
    Its derivative uses one autodiff reverse pass or frequency-grouped adjoint
    simulations, according to the selected gradient backend.
    """

    def __add__(self, other):
        if not isinstance(other, ModalObjective):
            return NotImplemented
        return WeightedObjective(
            (*self.terms, *other.terms), (*self.weights, *other.weights)
        )

    def __radd__(self, other):
        if isinstance(other, Real) and other == 0:
            return self
        return NotImplemented

    def __mul__(self, weight):
        if not isinstance(weight, Real):
            return NotImplemented
        if not np.isfinite(float(weight)):
            raise ValueError("Objective weights must be finite.")
        return WeightedObjective(
            self.terms, tuple(float(weight) * w for w in self.weights)
        )

    __rmul__ = __mul__

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        if not isinstance(other, ModalObjective):
            return NotImplemented
        return self + (-other)

    @property
    def terms(self) -> tuple[ModePower, ...]:
        raise NotImplementedError

    @property
    def weights(self) -> tuple[float, ...]:
        raise NotImplementedError

    def bind(self, results, program, cache=None):
        raise NotImplementedError


@dataclass(frozen=True)
class ModePower(ModalObjective):
    """Mean modal power or transmission over selected monitor frequencies.

    ``frequencies=None`` uses every target-monitor frequency with equal weight.
    Otherwise supply a nonempty sequence in Hz; every entry must match a recorded
    frequency (no interpolation). To assign different spectral weights, combine
    multiple ``ModePower`` terms. One frequency retains the original API behavior.

    ``direction`` matches ``results.mode(name).amps``. With ``reference_monitor``,
    divide each power by that monitor's power at the *same frequency*, then average
    the ratios. The reference defaults to the positive-direction fundamental mode.
    Without a reference, divide by nominal source power. Both mode bases stay fixed;
    the derivative includes the measured reference amplitude's dependence on fields.
    """

    monitor: str
    mode_index: int = 0
    direction: str = "+"
    reference_monitor: str | None = None
    frequencies: tuple[float, ...] | None = None
    reference_mode_index: int = 0
    reference_direction: str = "+"

    def __post_init__(self):
        if not isinstance(self.monitor, str) or not self.monitor:
            raise ValueError("ModePower.monitor must be a nonempty name.")
        if self.reference_monitor is not None and (
            not isinstance(self.reference_monitor, str) or not self.reference_monitor
        ):
            raise ValueError("ModePower.reference_monitor must be a nonempty name.")
        for name in ("direction", "reference_direction"):
            if getattr(self, name) not in {"+", "-"}:
                raise ValueError(f"ModePower.{name} must be '+' or '-'.")
        for name in ("mode_index", "reference_mode_index"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not np.isfinite(float(value))
                or int(float(value)) != value
                or value < 0
            ):
                raise ValueError(f"ModePower.{name} must be a nonnegative integer.")
            object.__setattr__(self, name, int(float(value)))
        if self.frequencies is not None:
            values = np.asarray(self.frequencies, dtype=float)
            if (
                values.ndim != 1
                or not values.size
                or not np.isfinite(values).all()
                or np.any(values <= 0)
                or np.unique(values).size != values.size
            ):
                raise ValueError(
                    "frequencies must be a nonempty sequence of distinct positive finite values in Hz."
                )
            object.__setattr__(self, "frequencies", tuple(float(f) for f in values))

    @property
    def terms(self):
        return (self,)

    @property
    def weights(self):
        return (1.0,)

    def bind(self, results, program, cache=None):
        spectrum = self.bind_spectrum(results, program, cache)
        return lambda state: jnp.mean(spectrum(state))

    def bind_spectrum(self, results, program, cache=None):
        """Bind per-frequency powers; retain requested order when selecting a band."""
        cache = {} if cache is None else cache
        target = analysis_data(results, self.monitor)
        frequencies = (
            tuple(target.frequencies) if self.frequencies is None else self.frequencies
        )
        power = self._bind_power(results, program, frequencies, cache)
        if self.reference_monitor is None:
            return power
        reference = replace(
            self,
            monitor=self.reference_monitor,
            mode_index=self.reference_mode_index,
            direction=self.reference_direction,
            reference_monitor=None,
        )._bind_power(results, program, frequencies, cache)

        def transmission(state):
            incident = reference(state)
            return power(state) / jnp.where(incident > 1e-12, incident, jnp.nan)

        return transmission

    def _bind_power(self, results, program, frequencies, cache):
        amplitude = self._bind_amplitude(results, program, frequencies, cache)
        incident_power = float(results.sources[0].power)

        def power_spectrum(state):
            value = amplitude(state)
            return jnp.real(value * jnp.conj(value)) / incident_power

        return power_spectrum

    def _bind_amplitude(self, results, program, frequencies, cache):
        """Bind complex amplitudes using ordinary BeamZ modal normalization."""
        key = ("amplitude", self.monitor, self.mode_index, self.direction, frequencies)
        if key in cache:
            return cache[key]
        data = analysis_data(results, self.monitor)
        monitor = data.monitor_geometry
        if not isinstance(monitor, ModeMonitor):
            raise ValueError("ModePower requires a ModeMonitor.")
        if self.mode_index >= monitor.mode_spec.num_modes:
            raise ValueError("ModePower.mode_index exceeds the monitor's num_modes.")
        selected = []
        for frequency in frequencies:
            matches = np.flatnonzero(
                np.isclose(data.frequencies, frequency, rtol=1e-10, atol=0)
            )
            if matches.size != 1:
                raise ValueError(
                    f"Monitor '{self.monitor}' must record frequency {frequency:g} Hz exactly once."
                )
            selected.append(int(matches[0]))
        if not selected:
            raise ValueError("ModePower requires at least one recorded frequency.")
        port = Port(
            center=monitor.center,
            size=monitor.size,
            name=self.monitor,
            mode_spec=ModeSpec(
                num_modes=monitor.mode_spec.num_modes,
                mode_index=self.mode_index,
                polarization=monitor.mode_spec.polarization
                or program.config.polarization_2d,
            ),
            direction="+",
        )
        spec = next(
            item
            for item in program.monitors
            if item.monitor_index == list(results.monitors).index(self.monitor)
        )
        normalization = normalization_from_result(
            results, results[self.monitor], source=0
        )
        if normalization is None:
            raise ValueError(
                "The source has no usable spectrum at the objective frequency."
            )
        norm = np.asarray(normalization.field_amplitude_norm)[selected]
        if not np.isfinite(norm).all() or np.any(abs(norm) < 1e-12):
            raise ValueError(
                "The source spectrum is too small at an objective frequency."
            )
        incident_power = float(results.sources[0].power)
        if not np.isfinite(incident_power) or incident_power <= 0:
            raise ValueError("The incident source power must be positive.")
        rows, phase_rows = [], []
        mode_cache = {}
        for frequency in np.asarray(data.frequencies)[selected]:
            projection = _build_port_projection(
                data, port, monitor, float(frequency), mode_cache
            )
            components = projection["components"]
            pinv = projection["pinv"][0 if self.direction == "+" else 1]
            rows.append(pinv * projection.get("projection_weights", np.ones(pinv.size)))
            phase_rows.append(
                [
                    _monitor_projection_phase(c, [frequency], data.dt)[0]
                    * _modal_projection_spatial_phase(
                        c, [frequency], projection.get("modal_plane_delay_s", 0.0)
                    )[0]
                    for c in components
                ]
            )
        indices = jnp.asarray(
            [("Ex", "Ey", "Ez", "Hx", "Hy", "Hz").index(c) for c in components]
        )
        rows = jnp.asarray(np.stack(rows))
        phases = jnp.asarray(phase_rows)[:, :, None]
        norm = jnp.asarray(norm)[:, None, None]
        selected = jnp.asarray(selected)
        start = spec.dft_value_offset
        count = 6 * spec.freq_count * spec.dft_point_count

        def amplitude_spectrum(state):
            fields = (
                state.dft_vec_re[start : start + count]
                + 1j * state.dft_vec_im[start : start + count]
            ).reshape(6, spec.freq_count, spec.dft_point_count)
            window_sum = jnp.maximum(
                state.dft_weight_sum[spec.dft_weight_offset + selected], 1e-18
            )
            fields = fields[indices][:, selected, :].transpose(1, 0, 2)
            vector = (fields * phases * (2 / window_sum[:, None, None]) / norm).reshape(
                len(frequencies), -1
            )
            return jnp.sum(rows * vector, axis=1)

        cache[key] = amplitude_spectrum
        return amplitude_spectrum


@dataclass(frozen=True)
class SoftMinModePower(ModePower):
    """Smooth lower bound on the weakest selected modal transmission.

    Reduce the normalized spectrum ``p`` as
    ``-temperature * logsumexp(-p / temperature)``. For N frequencies this
    lies between ``min(p) - temperature * log(N)`` and ``min(p)``. There is
    no division by N inside the logarithm. A smaller temperature focuses the
    derivative more strongly on the weakest frequencies. ``spectra`` still
    returns the unreduced physical powers.
    """

    temperature: float = 0.03

    def __post_init__(self):
        super().__post_init__()
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, Real)
            or not np.isfinite(self.temperature)
            or self.temperature <= 0
        ):
            raise ValueError("temperature must be a finite positive scalar.")
        object.__setattr__(self, "temperature", float(self.temperature))

    def bind(self, results, program, cache=None):
        spectrum = self.bind_spectrum(results, program, cache)
        return lambda state: (
            -self.temperature * logsumexp(-spectrum(state) / self.temperature)
        )


@dataclass(frozen=True, init=False)
class WeightedObjective(ModalObjective):
    """A weighted sum of reduced modal powers, evaluated in one FDTD solve.

    Negative weights penalize unwanted powers. Weights are not automatically
    normalized, and no power values are clipped. Usually construct this through
    objective arithmetic, e.g. ``0.5 * (short_to_top + long_to_bottom) - leakage``.
    """

    _terms: tuple[ModePower, ...]
    _weights: tuple[float, ...]

    def __init__(self, terms, weights):
        terms, weights = tuple(terms), tuple(float(w) for w in weights)
        if not terms or not all(isinstance(t, ModePower) for t in terms):
            raise ValueError("WeightedObjective requires nonempty ModePower terms.")
        if len(terms) != len(weights) or not np.isfinite(weights).all():
            raise ValueError("Objective weights must be finite and match the terms.")
        object.__setattr__(self, "_terms", terms)
        object.__setattr__(self, "_weights", weights)

    @property
    def terms(self):
        return self._terms

    @property
    def weights(self):
        return self._weights

    def canonical_spec(self):
        return self.terms, self.weights

    def bind(self, results, program, cache=None):
        cache = {} if cache is None else cache
        objectives = [term.bind(results, program, cache) for term in self.terms]
        weights = jnp.asarray(self.weights)
        return lambda state: jnp.sum(
            weights * jnp.stack([objective(state) for objective in objectives])
        )
