from __future__ import annotations

import math
import time
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from beamz import (
    LIGHT_SPEED,
    PEC,
    PML,
    Design,
    Material,
    ModeMonitor,
    ModeSource,
    ModeSpec,
    Port,
    Rectangle,
    SampledSignal,
    Simulation,
    calc_optimal_fdtd_params,
    µm,
)
from beamz.analysis import s_parameters


@dataclass(frozen=True)
class StraightWaveguideSParamConfig:
    axis: str = "x"
    polarization: str = "tm"
    monitor_behind_source: bool = False
    mirror_on_grid: bool = False
    reflecting_termination: bool = False
    wavelength_um: float = 1.55
    num_freqs: int = 5
    waveguide_width_um: float = 0.60
    n_core: float = 2.04
    n_clad: float = 1.444
    guide_length_um: float = 8.0
    vertical_clearance_um: float = 1.6
    source_offset_um: float = 0.45
    input_monitor_offset_um: float = 1.10
    output_monitor_offsets_um: tuple[float, float] = (1.60, 0.90)
    port_margin_um: float = 0.50
    pml_thickness_wl: float = 1.50
    courant_safety: float = 0.95
    pulse_sigma_periods: float = 6.0
    pulse_center_sigmas: float = 4.0
    settle_transit_multiples: float = 6.0
    decay_ratio: float = 1e-3
    lookback_records: int = 12
    pml_formulation: str = "sponge"
    cpml_kappa_max: float = 8.0
    cpml_alpha_max: float | None = None

    @property
    def wavelength_m(self) -> float:
        return float(self.wavelength_um) * µm

    @property
    def frequency_hz(self) -> float:
        return LIGHT_SPEED / self.wavelength_m

    @property
    def dft_frequencies_hz(self) -> np.ndarray:
        if int(self.num_freqs) <= 1:
            return np.asarray([self.frequency_hz], dtype=float)
        wavelengths = np.linspace(
            0.99 * self.wavelength_m,
            1.01 * self.wavelength_m,
            int(self.num_freqs),
            dtype=float,
        )
        return LIGHT_SPEED / wavelengths

    @property
    def pml_m(self) -> float:
        return float(self.pml_thickness_wl) * self.wavelength_m

    @property
    def waveguide_width_m(self) -> float:
        return float(self.waveguide_width_um) * µm

    @property
    def guide_length_m(self) -> float:
        return float(self.guide_length_um) * µm

    @property
    def vertical_clearance_m(self) -> float:
        return float(self.vertical_clearance_um) * µm

    @property
    def port_margin_m(self) -> float:
        return float(self.port_margin_um) * µm


@dataclass
class StraightWaveguideSParamResult:
    resolution_ppw: int
    dx_nm: float
    dt_fs: float
    steps_to_decay: int
    runtime_s: float
    frequencies_hz: np.ndarray
    wavelengths_um: np.ndarray
    monitor_x_um: dict[str, float]
    s11: np.ndarray
    s21_by_monitor: dict[str, np.ndarray]
    power_sum_by_monitor: dict[str, np.ndarray]
    phase_residual_rad_by_monitor: dict[str, float]
    phase_slope_s_by_monitor: dict[str, float]
    condition_numbers: dict[str, np.ndarray]
    backward_launch: np.ndarray | None


def _build_case(
    cfg: StraightWaveguideSParamConfig,
    *,
    resolution_ppw: int,
    launch_direction: str = "+",
):
    if launch_direction not in {"+", "-"}:
        raise ValueError("launch_direction must be '+' or '-'")
    if cfg.axis not in {"x", "y"}:
        raise ValueError("axis must be 'x' or 'y'")

    def xy(longitudinal, transverse):
        return (
            (longitudinal, transverse)
            if cfg.axis == "x"
            else (transverse, longitudinal)
        )

    pml_m = cfg.pml_m
    width_m = cfg.guide_length_m + 2.0 * pml_m
    height_m = cfg.waveguide_width_m + 2.0 * (cfg.vertical_clearance_m + pml_m)
    dx, dt = calc_optimal_fdtd_params(
        cfg.wavelength_m,
        max(cfg.n_core, cfg.n_clad),
        dims=2,
        safety_factor=cfg.courant_safety,
        points_per_wavelength=int(resolution_ppw),
        width=xy(width_m, height_m)[0],
        height=xy(width_m, height_m)[1],
    )

    design = Design(
        width=xy(width_m, height_m)[0],
        height=xy(width_m, height_m)[1],
        material=Material(cfg.n_clad**2),
    )
    design += Rectangle(
        position=xy(0.0, 0.5 * (height_m - cfg.waveguide_width_m)),
        width=xy(width_m, cfg.waveguide_width_m)[0],
        height=xy(width_m, cfg.waveguide_width_m)[1],
        material=Material(cfg.n_core**2),
    )
    freqs = np.asarray(cfg.dft_frequencies_hz, dtype=float)
    period = 1.0 / cfg.frequency_hz
    sigma_t = float(cfg.pulse_sigma_periods) * period
    t0 = float(cfg.pulse_center_sigmas) * sigma_t
    n_eff_guess = 0.5 * (cfg.n_core + cfg.n_clad)
    transit_time = n_eff_guess * width_m / LIGHT_SPEED
    min_time_s = t0 + 4.0 * sigma_t + float(cfg.settle_transit_multiples) * transit_time
    total_time_s = min_time_s + 2.0 * transit_time
    num_steps = max(64, int(np.ceil(total_time_s / dt)) + 1)
    time_axis = np.arange(num_steps, dtype=float) * dt
    signal = np.exp(-0.5 * ((time_axis - t0) / max(sigma_t, 1e-30)) ** 2) * np.cos(
        2.0 * np.pi * cfg.frequency_hz * (time_axis - t0)
    )
    pulse = SimpleNamespace(
        signal=np.asarray(signal, dtype=float),
        time=np.asarray(time_axis, dtype=float),
        source_end_time=float(t0 + 4.0 * sigma_t),
        tail_time=float(cfg.settle_transit_multiples) * transit_time,
    )

    y_center = 0.5 * height_m
    source_span = max(
        cfg.waveguide_width_m + 2.0 * cfg.port_margin_m,
        3.0 * cfg.waveguide_width_m,
    )
    source_x = (
        pml_m + float(cfg.source_offset_um) * µm
        if launch_direction == "+"
        else width_m - pml_m - float(cfg.source_offset_um) * µm
    )
    normal_cells = int(np.ceil(width_m / dx))
    if cfg.mirror_on_grid:
        # The raster domain is rounded up to whole cells. Mirror the H-plane
        # source reference about that domain, not the requested float extent.
        forward_center = (
            round((pml_m + cfg.source_offset_um * µm) / dx - 0.5) + 0.5
        ) * dx
        source_x = (
            forward_center
            if launch_direction == "+"
            else normal_cells * dx - forward_center
        )
    source = ModeSource(
        center=(*xy(source_x, y_center), 0.0),
        size=(*xy(0.0, source_span), cfg.waveguide_width_m),
        source_time=SampledSignal(pulse.signal, dt=dt, freq0=cfg.frequency_hz),
        direction=launch_direction,
        mode_spec=ModeSpec(polarization=cfg.polarization),
    )

    if launch_direction == "+":
        monitor_x = {
            "o1": pml_m + float(cfg.input_monitor_offset_um) * µm,
            "mid": width_m - pml_m - float(cfg.output_monitor_offsets_um[0]) * µm,
            "far": width_m - pml_m - float(cfg.output_monitor_offsets_um[1]) * µm,
        }
    else:
        monitor_x = {
            "o1": width_m - pml_m - float(cfg.input_monitor_offset_um) * µm,
            "mid": pml_m + float(cfg.output_monitor_offsets_um[0]) * µm,
            "far": pml_m + float(cfg.output_monitor_offsets_um[1]) * µm,
        }
    if cfg.monitor_behind_source:
        behind_offset = 0.5 * float(cfg.source_offset_um) * µm
        monitor_x["behind"] = (
            source_x - behind_offset
            if launch_direction == "+"
            else source_x + behind_offset
        )
    if cfg.mirror_on_grid:
        forward_positions = {
            "o1": pml_m + cfg.input_monitor_offset_um * µm,
            "mid": width_m - pml_m - cfg.output_monitor_offsets_um[0] * µm,
            "far": width_m - pml_m - cfg.output_monitor_offsets_um[1] * µm,
            "behind": pml_m + 0.5 * cfg.source_offset_um * µm,
        }
        # Raw 2D monitor index i references H at (i+1/2)*dx; its mirror
        # therefore has index N-1-i. All desired port separations match exactly.
        monitor_x = {
            name: (
                round(forward_positions[name] / dx)
                if launch_direction == "+"
                else normal_cells - 1 - round(forward_positions[name] / dx)
            )
            * dx
            for name in monitor_x
        }
    monitors = {
        name: ModeMonitor(
            center=(*xy(x_m, y_center), 0.0),
            size=(*xy(0.0, source_span), cfg.waveguide_width_m),
            freqs=freqs,
            name=name,
            mode_spec=ModeSpec(polarization=cfg.polarization),
        )
        for name, x_m in monitor_x.items()
    }

    exit_edge = (
        ("right" if cfg.axis == "x" else "top")
        if launch_direction == "+"
        else ("left" if cfg.axis == "x" else "bottom")
    )
    absorber_edges = (
        tuple(edge for edge in ("left", "right", "bottom", "top") if edge != exit_edge)
        if cfg.reflecting_termination
        else "all"
    )
    boundaries = [
        PML(
            edges=absorber_edges,
            thickness=pml_m,
            formulation=cfg.pml_formulation,
            kappa_max=float(cfg.cpml_kappa_max),
            alpha_max=cfg.cpml_alpha_max,
        )
    ]
    if cfg.reflecting_termination:
        boundaries.append(PEC(edges=exit_edge))
    sim = Simulation(
        design=design,
        sources=[source],
        monitors=list(monitors.values()),
        boundaries=boundaries,
        time=pulse.time,
        resolution=dx,
        polarization=cfg.polarization,
    )
    return sim, monitors, monitor_x, freqs, pulse, dx, dt


def _phase_linearity_residual(
    phase: np.ndarray, freqs: np.ndarray
) -> tuple[float, float]:
    ph = np.unwrap(np.asarray(phase, dtype=float))
    f = np.asarray(freqs, dtype=float)
    if ph.size < 3:
        return 0.0, 0.0
    slope, intercept = np.polyfit(f - float(np.mean(f)), ph, 1)
    residual = ph - (slope * (f - float(np.mean(f))) + intercept)
    return float(np.max(np.abs(residual))), float(slope / (2.0 * np.pi))


def run_straight_waveguide_sparam_case(
    *,
    resolution_ppw: int,
    cfg: StraightWaveguideSParamConfig | None = None,
    launch_direction: str = "+",
) -> StraightWaveguideSParamResult:
    cfg = cfg or StraightWaveguideSParamConfig()
    sim, monitors, monitor_x, freqs, pulse, dx, dt = _build_case(
        cfg,
        resolution_ppw=int(resolution_ppw),
        launch_direction=launch_direction,
    )
    t0 = time.perf_counter()
    run_results = sim.advance(progress=False)
    steps = run_results.state.current_step
    runtime_s = max(time.perf_counter() - t0, 1e-12)

    result = s_parameters(
        run_results.results,
        source_port="o1",
        ports=[
            Port(
                center=monitor.center,
                size=monitor.size,
                name=name,
                direction=(
                    launch_direction
                    if name in {"o1", "behind"}
                    else "-"
                    if launch_direction == "+"
                    else "+"
                ),
                mode_spec=ModeSpec(polarization=cfg.polarization),
            )
            for name, monitor in monitors.items()
        ],
        output_ports=list(monitors),
        frequencies=freqs,
        min_incident_db=-60.0,
    )
    s_matrix = result.s_matrix
    s11 = np.asarray(s_matrix[("o1", "o1")], dtype=np.complex128)
    s21_by_monitor = {
        name: np.asarray(s_matrix[(name, "o1")], dtype=np.complex128)
        for name in ("mid", "far")
    }
    power_sum_by_monitor = {
        name: np.abs(s11) ** 2 + np.abs(s21) ** 2
        for name, s21 in s21_by_monitor.items()
    }

    phase_residual = {}
    phase_slope = {}
    ref_s21 = s21_by_monitor["mid"]
    for name, s21 in s21_by_monitor.items():
        denom = np.where(np.abs(ref_s21) > 1e-30, ref_s21, 1e-30 + 0.0j)
        ratio = np.ones_like(s21) if name == "mid" else s21 / denom
        residual, slope = _phase_linearity_residual(np.angle(ratio), freqs)
        phase_residual[name] = residual
        phase_slope[name] = slope

    diagnostics = result.diagnostics
    cond = {
        name: np.asarray(
            diagnostics["condition_numbers"][name]["monitor"],
            dtype=float,
        )
        for name in ("o1", "mid", "far")
    }
    return StraightWaveguideSParamResult(
        resolution_ppw=int(resolution_ppw),
        dx_nm=float(dx / 1e-9),
        dt_fs=float(dt / 1e-15),
        steps_to_decay=int(steps),
        runtime_s=float(runtime_s),
        frequencies_hz=np.asarray(freqs, dtype=float),
        wavelengths_um=np.asarray(LIGHT_SPEED / freqs / µm, dtype=float),
        monitor_x_um={name: float(x / µm) for name, x in monitor_x.items()},
        s11=s11,
        s21_by_monitor=s21_by_monitor,
        power_sum_by_monitor=power_sum_by_monitor,
        phase_residual_rad_by_monitor=phase_residual,
        phase_slope_s_by_monitor=phase_slope,
        condition_numbers=cond,
        backward_launch=(
            np.asarray(s_matrix[("behind", "o1")], dtype=np.complex128)
            if cfg.monitor_behind_source
            else None
        ),
    )


def summarize_center_metrics(result: StraightWaveguideSParamResult) -> dict[str, float]:
    idx = int(np.argmin(np.abs(result.wavelengths_um - 1.55)))
    metrics = {
        "s11_db": 20.0 * math.log10(max(float(abs(result.s11[idx])), 1e-12)),
    }
    for name, s21 in result.s21_by_monitor.items():
        metrics[f"s21_{name}_db"] = 20.0 * math.log10(max(float(abs(s21[idx])), 1e-12))
        metrics[f"power_sum_{name}"] = float(result.power_sum_by_monitor[name][idx])
    return metrics
