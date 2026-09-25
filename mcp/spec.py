"""Declarative 2D and 3D scene spec for BeamZ.
Supports Rect, Box, Circle, Ring, Polygon, ModeSource, GaussianSource, FluxMonitor, ModeMonitor, and TimeMonitor (FieldRecorder).
"""
from __future__ import annotations
import math
from typing import Literal, Optional, List, Tuple
import numpy as np
from pydantic import BaseModel, Field

C = 299_792_458.0
Name = Field(pattern=r"^[A-Za-z0-9_]+$", max_length=40)


class Domain(BaseModel):
    width_um: float = Field(gt=0, description="x extent (propagation or lateral)")
    height_um: float = Field(gt=0, description="y extent")
    depth_um: Optional[float] = Field(default=None, gt=0, description="z extent. None = 2D simulation, float = 3D simulation.")
    cladding_index: float = Field(default=1.44, ge=1.0, description="Background refractive index")
    resolution_um: float = Field(default=0.03, gt=0, description="Grid step in micrometers")
    pml_um: float = Field(default=1.0, ge=0, description="PML boundary thickness in micrometers")


class Spectrum(BaseModel):
    center_wavelength_um: float = Field(default=1.55, gt=0)
    bandwidth_fraction: float = Field(default=0.1, gt=0, le=0.5, description="fwidth / f0; monitors span f0 +/- fwidth")
    n_freqs: int = Field(default=11, ge=1, le=101)


# --- Geometries ---

class Rect(BaseModel):
    """Axis-aligned box/rectangle. (x_um, y_um, z_um) is the LOWER-LEFT corner."""
    name: str = Name
    x_um: float
    y_um: float
    z_um: float = Field(default=0.0, description="z lower coordinate (used in 3D)")
    width_um: float = Field(gt=0, description="x extent")
    height_um: float = Field(gt=0, description="y extent")
    depth_um: Optional[float] = Field(default=None, gt=0, description="z extent (3D only; defaults to domain.depth_um)")
    index: float = Field(gt=0, description="Refractive index (eps = n^2)")


class Circle(BaseModel):
    """Circular cylinder / disc. (x_um, y_um) is center in xy plane."""
    name: str = Name
    x_um: float
    y_um: float
    z_um: float = Field(default=0.0, description="z lower coordinate (3D)")
    radius_um: float = Field(gt=0, description="Radius in micrometers")
    depth_um: Optional[float] = Field(default=None, gt=0, description="z extent (3D only)")
    index: float = Field(gt=0, description="Refractive index")


class Ring(BaseModel):
    """Concentric circular ring (microring resonator). (x_um, y_um) is center."""
    name: str = Name
    x_um: float
    y_um: float
    z_um: float = Field(default=0.0, description="z lower coordinate (3D)")
    inner_radius_um: float = Field(gt=0, description="Inner radius in micrometers")
    outer_radius_um: float = Field(gt=0, description="Outer radius in micrometers")
    depth_um: Optional[float] = Field(default=None, gt=0, description="z extent (3D only)")
    index: float = Field(gt=0, description="Refractive index")


class Polygon(BaseModel):
    """Arbitrary polygon in the xy plane specified by 2D vertices."""
    name: str = Name
    vertices: List[Tuple[float, float]] = Field(description="List of (x, y) coordinates defining polygon vertices in um")
    z_um: float = Field(default=0.0, description="z lower coordinate (3D)")
    depth_um: Optional[float] = Field(default=None, gt=0, description="z extent (3D only)")
    index: float = Field(gt=0, description="Refractive index")


# --- Sources ---

class ModeSpec(BaseModel):
    num_modes: int = Field(default=1, ge=1, le=10, description="Number of modes to solve/project")
    polarization: Optional[Literal["te", "tm"]] = Field(default=None, description="Polarization filter ('te', 'tm', or None for all)")
    target_neff: Optional[float] = Field(default=None, gt=0, description="Target effective index for eigensolver")


class ModeSource(BaseModel):
    """Mode source on a plane or line."""
    name: str = Name
    x_um: float
    y_um: float
    z_um: float = Field(default=0.0, description="z center coordinate (used in 3D)")
    span_um: float = Field(gt=0, description="Transverse span across y (or z); ~3-4x waveguide width")
    span_z_um: Optional[float] = Field(default=None, gt=0, description="Transverse span along z in 3D")
    normal: Literal["x", "y", "z"] = "x"
    direction: Literal["+", "-"] = "+"
    mode_spec: ModeSpec = ModeSpec()


class GaussianSourceSpec(BaseModel):
    """Spatial Gaussian beam or point source."""
    name: str = Name
    x_um: float
    y_um: float
    z_um: float = Field(default=0.0, description="z center coordinate (3D)")
    waist_radius_um: float = Field(default=1.0, gt=0, description="Gaussian waist radius / beam width in um")
    direction: Literal["+x", "-x", "+y", "-y", "+z", "-z"] = "+x"
    span_um: Optional[float] = Field(default=None, gt=0, description="Aperture span (defaults to ~3x waist_radius)")


# --- Monitors ---

class FluxMonitor(BaseModel):
    """Frequency-domain power-flux monitor across a plane or line."""
    name: str = Name
    x_um: float
    y_um: float
    z_um: float = Field(default=0.0, description="z center coordinate")
    span_um: float = Field(gt=0, description="Transverse span")
    span_z_um: Optional[float] = Field(default=None, gt=0, description="Transverse span along z (3D)")
    normal: Literal["x", "y", "z"] = "x"


class ModeMonitor(BaseModel):
    """Modal decomposition monitor projecting fields onto guided eigenmodes."""
    name: str = Name
    x_um: float
    y_um: float
    z_um: float = Field(default=0.0, description="z center coordinate")
    span_um: float = Field(gt=0, description="Transverse span")
    span_z_um: Optional[float] = Field(default=None, gt=0, description="Transverse span along z (3D)")
    normal: Literal["x", "y", "z"] = "x"
    mode_spec: ModeSpec = ModeSpec()


class TimeMonitorSpec(BaseModel):
    """Time-domain field recorder (snapshots of field components over time)."""
    name: str = Name
    components: List[Literal["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]] = Field(default=["Ez"], description="Field components to record")
    interval: int = Field(default=10, ge=1, description="Interval in simulation time steps between snapshots")


class Scene(BaseModel):
    id: str
    domain: Domain
    spectrum: Spectrum = Spectrum()
    structures: list[Rect] = []
    circles: list[Circle] = []
    rings: list[Ring] = []
    polygons: list[Polygon] = []
    sources: list[ModeSource] = []
    gaussian_sources: list[GaussianSourceSpec] = []
    monitors: list[FluxMonitor] = []
    mode_monitors: list[ModeMonitor] = []
    time_monitors: list[TimeMonitorSpec] = []
    run_time_fs: Optional[float] = Field(default=None, gt=0, description="None = auto-pick from lint heuristic")


def _all_indices(s: Scene) -> list[float]:
    indices = [s.domain.cladding_index]
    for r in s.structures: indices.append(r.index)
    for c in s.circles: indices.append(c.index)
    for rg in s.rings: indices.append(rg.index)
    for p in s.polygons: indices.append(p.index)
    return indices


def _n_max(s: Scene) -> float:
    return max(_all_indices(s))


def recommended_run_time_fs(s: Scene) -> float:
    """Pulse offset (~4/fwidth, plus tail) + transit across the domain diagonal at n_max, with margin."""
    f0 = C / (s.spectrum.center_wavelength_um * 1e-6)
    fw = f0 * s.spectrum.bandwidth_fraction
    d = s.domain
    if d.depth_um is None:
        diag = math.hypot(d.width_um, d.height_um) * 1e-6
    else:
        diag = math.sqrt(d.width_um**2 + d.height_um**2 + d.depth_um**2) * 1e-6
    return (8.0 / fw + 2.0 * _n_max(s) * diag / C) * 1e15


def estimate_cost(s: Scene) -> dict:
    """Estimate grid cell count, time steps, and CPU compute time."""
    d = s.domain
    is_3d = d.depth_um is not None
    dx = d.resolution_um
    nx = int(math.ceil(d.width_um / dx))
    ny = int(math.ceil(d.height_um / dx))
    nz = int(math.ceil(d.depth_um / dx)) if d.depth_um is not None else 1
    total_cells = nx * ny * nz
    dim = 3 if is_3d else 2
    dt = (dx * 1e-6 * 0.99) / (C * math.sqrt(dim))
    rt_fs = s.run_time_fs if s.run_time_fs is not None else recommended_run_time_fs(s)
    n_steps = int(math.ceil((rt_fs * 1e-15) / dt))
    cell_updates = total_cells * n_steps
    est_cpu_s = round(cell_updates / 1.5e8, 1)  # ~150 MCUPS on modern CPU
    return {
        "dims": 3 if is_3d else 2,
        "grid_shape": [nx, ny, nz] if is_3d else [nx, ny],
        "total_cells": total_cells,
        "time_steps": n_steps,
        "recommended_run_time_fs": round(recommended_run_time_fs(s), 1),
        "run_time_fs": round(rt_fs, 1),
        "estimated_cpu_seconds": est_cpu_s,
    }


def validate_scene(s: Scene) -> list[dict]:
    out = []
    d, wl = s.domain, s.spectrum.center_wavelength_um
    wl_min = wl * (1 - s.spectrum.bandwidth_fraction)
    n_max = _n_max(s)
    ppw = wl_min / n_max / d.resolution_um
    crit_ppw = 6.0 if d.depth_um is not None else 8.0
    rec_ppw = 10.0
    if ppw < crit_ppw:
        out.append(dict(level="error", code="LOW_RESOLUTION",
            message=f"Only {ppw:.1f} points/wavelength in highest-index material ({n_max}).",
            fix=f"Set resolution_um <= {wl_min / n_max / rec_ppw:.4f}."))
    elif ppw < rec_ppw:
        out.append(dict(level="warning", code="COARSE_RESOLUTION",
            message=f"Resolution is {ppw:.1f} points/wavelength (coarse, fast exploration).",
            fix=f"Use resolution_um <= {wl_min / n_max / rec_ppw:.4f} for production accuracy."))

    if d.pml_um < 0.5 * wl:
        out.append(dict(level="warning", code="THIN_PML", message=f"PML {d.pml_um} um < wavelength/2.",
            fix=f"Use pml_um >= {wl:.2f}."))
    if not (0.05 <= wl <= 20):
        out.append(dict(level="warning", code="UNIT_SANITY",
            message=f"Center wavelength {wl} um is unusual.", fix="Units are micrometers (1.55 not 1550)."))

    has_sources = bool(s.sources or s.gaussian_sources)
    if not has_sources:
        out.append(dict(level="error", code="NO_SOURCE", message="No source.", fix="Call add_source or add_gaussian_source."))

    has_monitors = bool(s.monitors or s.mode_monitors or s.time_monitors)
    if not has_monitors:
        out.append(dict(level="warning", code="NO_MONITOR", message="No monitors.", fix="Call add_monitor, add_mode_monitor, or add_time_monitor."))

    for rg in s.rings:
        if rg.inner_radius_um >= rg.outer_radius_um:
            out.append(dict(level="error", code="INVALID_RING",
                message=f"Ring '{rg.name}' inner_radius ({rg.inner_radius_um}) >= outer_radius ({rg.outer_radius_um}).",
                fix="inner_radius_um must be smaller than outer_radius_um."))

    for poly in s.polygons:
        if len(poly.vertices) < 3:
            out.append(dict(level="error", code="INVALID_POLYGON",
                message=f"Polygon '{poly.name}' must have at least 3 vertices.",
                fix="Provide at least 3 (x, y) coordinates."))

    need = recommended_run_time_fs(s)
    if s.run_time_fs is not None and s.run_time_fs < need:
        out.append(dict(level="error", code="SHORT_RUN",
            message=f"run_time_fs={s.run_time_fs:.0f} is shorter than pulse delay + transit (~{need:.0f} fs); "
                    "fields will not reach the monitors.",
            fix="Set run_time_fs to None (auto) or >= %.0f." % need))

    est = estimate_cost(s)
    if est["estimated_cpu_seconds"] > 90:
        out.append(dict(level="warning", code="HIGH_COMPUTE_COST",
            message=f"Estimated CPU run time is ~{est['estimated_cpu_seconds']}s ({est['total_cells']:,} cells).",
            fix=f"Consider increasing resolution_um to >= {d.resolution_um * 1.5:.3f} or reducing domain size."))

    out.append(dict(level="info", code="COST_ESTIMATE",
        message=f"{'3D' if est['dims'] == 3 else '2D'} grid: {'x'.join(str(v) for v in est['grid_shape'])} "
                f"({est['total_cells']:,} cells), ~{est['time_steps']} steps, estimated CPU: ~{est['estimated_cpu_seconds']}s."))
    return out


def export_python(s: Scene) -> str:
    """Deterministic BeamZ script. The server exec()s this same text to run, so export == run."""
    d, sp = s.domain, s.spectrum
    is_3d = d.depth_um is not None
    rt = s.run_time_fs if s.run_time_fs is not None else round(recommended_run_time_fs(s), 1)

    L = [
        "import numpy as np",
        "import beamz as bz",
        "",
        "um = bz.um",
        f"f0 = bz.LIGHT_SPEED / ({sp.center_wavelength_um!r} * um)",
        f"fwidth = f0 * {sp.bandwidth_fraction!r}",
        f"freqs = np.linspace(f0 - fwidth, f0 + fwidth, {sp.n_freqs})",
        f"wavelengths_um = bz.LIGHT_SPEED / freqs / um",
        "",
    ]

    if is_3d:
        L.append(f"design = bz.Design(width={d.width_um!r} * um, height={d.height_um!r} * um, depth={d.depth_um!r} * um,")
    else:
        L.append(f"design = bz.Design(width={d.width_um!r} * um, height={d.height_um!r} * um,")
    L.append(f"                   material=bz.Material(permittivity={d.cladding_index**2!r}))")

    # Rectangles
    for r in s.structures:
        if is_3d:
            rz = r.depth_um if r.depth_um is not None else d.depth_um
            L += [
                f"design += bz.Rectangle(position=({r.x_um!r} * um, {r.y_um!r} * um, {r.z_um!r} * um), width={r.width_um!r} * um,",
                f"                       height={r.height_um!r} * um, depth={rz!r} * um, material=bz.Material(permittivity={r.index**2!r}))  # {r.name}"
            ]
        else:
            L += [
                f"design += bz.Rectangle(position=({r.x_um!r} * um, {r.y_um!r} * um), width={r.width_um!r} * um,",
                f"                       height={r.height_um!r} * um, material=bz.Material(permittivity={r.index**2!r}))  # {r.name}"
            ]

    # Circles
    for c in s.circles:
        depth_code = f", depth={c.depth_um!r} * um, z={c.z_um!r} * um" if (is_3d and c.depth_um is not None) else ""
        L.append(f"design += bz.Circle(position=({c.x_um!r} * um, {c.y_um!r} * um), radius={c.radius_um!r} * um{depth_code}, material=bz.Material(permittivity={c.index**2!r}))  # {c.name}")

    # Rings
    for rg in s.rings:
        depth_code = f", depth={rg.depth_um!r} * um, z={rg.z_um!r} * um" if (is_3d and rg.depth_um is not None) else ""
        L.append(f"design += bz.Ring(position=({rg.x_um!r} * um, {rg.y_um!r} * um), inner_radius={rg.inner_radius_um!r} * um, outer_radius={rg.outer_radius_um!r} * um{depth_code}, material=bz.Material(permittivity={rg.index**2!r}))  # {rg.name}")

    # Polygons
    for p in s.polygons:
        v_str = ", ".join(f"({x!r} * um, {y!r} * um)" for x, y in p.vertices)
        depth_code = f", depth={p.depth_um!r} * um, z={p.z_um!r} * um" if (is_3d and p.depth_um is not None) else ""
        L.append(f"design += bz.Polygon(vertices=[{v_str}]{depth_code}, material=bz.Material(permittivity={p.index**2!r}))  # {p.name}")

    def _size_str(normal, span_y, span_z):
        sz = span_z if span_z is not None else (span_y if is_3d else 0.5)
        if normal == "x":
            return f"(0.0, {span_y!r} * um, {sz!r} * um)"
        elif normal == "y":
            return f"({span_y!r} * um, 0.0, {sz!r} * um)"
        else:
            return f"({span_y!r} * um, {sz!r} * um, 0.0)"

    pre_source_lines = []
    source_lines = []

    for it in s.sources:
        cz = it.z_um if is_3d else 0.0
        sz_str = _size_str(it.normal, it.span_um, it.span_z_um)
        ms_args = [f"num_modes={it.mode_spec.num_modes}"]
        if it.mode_spec.polarization:
            ms_args.append(f"polarization={it.mode_spec.polarization!r}")
        if it.mode_spec.target_neff:
            ms_args.append(f"target_neff={it.mode_spec.target_neff!r}")
        ms_str = ", ".join(ms_args)
        source_lines.append(
            f"    bz.ModeSource(center=({it.x_um!r} * um, {it.y_um!r} * um, {cz!r} * um), size={sz_str}, direction={it.direction!r},"
        )
        source_lines.append(
            f"                  source_time=bz.GaussianPulse(freq0=f0, fwidth=fwidth), mode_spec=bz.ModeSpec({ms_str})),  # {it.name}"
        )

    for gs in s.gaussian_sources:
        if is_3d:
            aperture = gs.span_um if gs.span_um is not None else (gs.waist_radius_um * 3.0)
            sz = f"(0.0, {aperture!r} * um, {aperture!r} * um)" if "x" in gs.direction else f"({aperture!r} * um, 0.0, {aperture!r} * um)"
            source_lines.append(
                f"    bz.GaussianBeamSource(center=({gs.x_um!r} * um, {gs.y_um!r} * um, {gs.z_um!r} * um), size={sz},"
                f" direction={gs.direction!r}, waist_radius={gs.waist_radius_um!r} * um, source_time=bz.GaussianPulse(freq0=f0, fwidth=fwidth)),  # {gs.name}"
            )
        else:
            p_name = f"_pulse_{gs.name}"
            dt_name = f"_dt_{gs.name}"
            t_name = f"_t_{gs.name}"
            v_name = f"_vals_{gs.name}"
            pre_source_lines += [
                f"{p_name} = bz.GaussianPulse(freq0=f0, fwidth=fwidth)",
                f"{dt_name} = {d.resolution_um!r} * um / (bz.LIGHT_SPEED * np.sqrt(2.0))",
                f"{t_name} = np.arange(0, {rt!r} * 1e-15, {dt_name})",
                f"{v_name}, _ = {p_name}.sample({t_name})",
            ]
            source_lines.append(f"    bz.GaussianSource(position=({gs.x_um!r} * um, {gs.y_um!r} * um), width={gs.waist_radius_um!r} * um, signal={v_name}),  # {gs.name}")

    if pre_source_lines:
        L += [""] + pre_source_lines
    L.append("")
    L.append("sources = [")
    L.extend(source_lines)
    L.append("]")

    L.append("monitors = [")
    for it in s.monitors:
        cz = it.z_um if is_3d else 0.0
        sz_str = _size_str(it.normal, it.span_um, it.span_z_um)
        L.append(f"    bz.FluxMonitor(center=({it.x_um!r} * um, {it.y_um!r} * um, {cz!r} * um), size={sz_str}, freqs=freqs, name={it.name!r}),")
    for it in s.mode_monitors:
        cz = it.z_um if is_3d else 0.0
        sz_str = _size_str(it.normal, it.span_um, it.span_z_um)
        ms_args = [f"num_modes={it.mode_spec.num_modes}"]
        if it.mode_spec.polarization:
            ms_args.append(f"polarization={it.mode_spec.polarization!r}")
        if it.mode_spec.target_neff:
            ms_args.append(f"target_neff={it.mode_spec.target_neff!r}")
        ms_str = ", ".join(ms_args)
        L.append(
            f"    bz.ModeMonitor(center=({it.x_um!r} * um, {it.y_um!r} * um, {cz!r} * um), size={sz_str}, freqs=freqs, name={it.name!r}, mode_spec=bz.ModeSpec({ms_str})),"
        )
    for tm in s.time_monitors:
        comp_tuple = tuple(tm.components)
        L.append(f"    bz.FieldRecorder(components={comp_tuple!r}, interval={tm.interval!r}, name={tm.name!r}),")
    L.append("]")

    L += [
        "",
        f"sim = bz.Simulation(design=design, sources=sources, monitors=monitors,",
        f"                    boundaries=[bz.PML(edges='all', thickness={d.pml_um!r} * um)],",
        f"                    resolution={d.resolution_um!r} * um, run_time={rt!r} * 1e-15)",
        "results = sim.run()",
    ]
    return "\n".join(L)


def solve_scene_modes(s: Scene, source_name: str | None = None) -> tuple[list[dict], np.ndarray | None, tuple | None]:
    """Pre-solve guided eigenmodes on the source plane before launching FDTD."""
    import beamz as bz

    um = bz.um
    d, sp = s.domain, s.spectrum
    is_3d = d.depth_um is not None
    depth_um = d.depth_um if is_3d else 3.0

    f0 = bz.LIGHT_SPEED / (sp.center_wavelength_um * um)
    fwidth = f0 * sp.bandwidth_fraction

    design = bz.Design(
        width=d.width_um * um,
        height=d.height_um * um,
        depth=depth_um * um, # pyright: ignore[reportOptionalOperand]
        material=bz.Material(permittivity=d.cladding_index**2),
    )

    for r in s.structures:
        rz = r.depth_um if (is_3d and r.depth_um is not None) else depth_um
        rz_pos = r.z_um if is_3d else 0.0
        design += bz.Rectangle(
            position=(r.x_um * um, r.y_um * um, rz_pos * um),
            width=r.width_um * um,
            height=r.height_um * um,
            depth=rz * um,  # pyright: ignore[reportOptionalOperand]
            material=bz.Material(permittivity=r.index**2),
        )

    for c in s.circles:
        cdepth = c.depth_um if (is_3d and c.depth_um is not None) else depth_um
        cz_pos = c.z_um if is_3d else 0.0
        design += bz.Circle(
            position=(c.x_um * um, c.y_um * um),
            radius=c.radius_um * um,
            depth=cdepth * um, # pyright: ignore[reportOptionalOperand]
            z=cz_pos * um,
            material=bz.Material(permittivity=c.index**2),
        )

    for rg in s.rings:
        rdepth = rg.depth_um if (is_3d and rg.depth_um is not None) else depth_um
        rz_pos = rg.z_um if is_3d else 0.0
        design += bz.Ring(
            position=(rg.x_um * um, rg.y_um * um),
            inner_radius=rg.inner_radius_um * um,
            outer_radius=rg.outer_radius_um * um,
            depth=rdepth * um,  # pyright: ignore[reportOptionalOperand]
            z=rz_pos * um,
            material=bz.Material(permittivity=rg.index**2),
        )

    for p in s.polygons:
        pdepth = p.depth_um if (is_3d and p.depth_um is not None) else depth_um
        pz_pos = p.z_um if is_3d else 0.0
        design += bz.Polygon(
            vertices=[(x * um, y * um) for x, y in p.vertices],
            depth=pdepth * um,  # pyright: ignore[reportOptionalOperand]
            z=pz_pos * um,
            material=bz.Material(permittivity=p.index**2),
        )

    target_src = None
    if source_name:
        for src in s.sources:
            if src.name == source_name:
                target_src = src
                break
    elif s.sources:
        target_src = s.sources[0]

    if target_src is None:
        raise ValueError("No ModeSource found to solve modes.")

    cz = target_src.z_um if is_3d else (depth_um / 2.0) # pyright: ignore[reportOptionalOperand]
    span_z = target_src.span_z_um if (is_3d and target_src.span_z_um is not None) else (target_src.span_um if is_3d else 2.5)

    if target_src.normal == "x":
        size = (0.0, target_src.span_um * um, span_z * um)
    elif target_src.normal == "y":
        size = (target_src.span_um * um, 0.0, span_z * um)
    else:
        size = (target_src.span_um * um, span_z * um, 0.0)

    ms_spec = bz.ModeSpec(
        num_modes=target_src.mode_spec.num_modes,
        polarization=target_src.mode_spec.polarization,
        target_neff=target_src.mode_spec.target_neff,
    )

    bz_source = bz.ModeSource(
        center=(target_src.x_um * um, target_src.y_um * um, cz * um),
        size=size,
        direction=target_src.direction,
        source_time=bz.GaussianPulse(freq0=f0, fwidth=fwidth),
        mode_spec=ms_spec,
    )

    sim = bz.Simulation(
        design=design,
        sources=[bz_source],
        boundaries=[bz.PML(edges="all", thickness=d.pml_um * um)],
        resolution=d.resolution_um * um,
        run_time=10e-15,
    )

    modes = bz_source.solve_modes(sim, freqs=[f0])
    neff_arr = np.asarray(modes.neffs[0])
    results = []
    clad_n = d.cladding_index
    for idx, neff in enumerate(neff_arr):
        results.append({
            "mode_index": idx,
            "neff_real": round(float(neff.real), 5),
            "neff_imag": round(float(neff.imag), 5),
            "is_guided": bool(neff.real > clad_n),
            "wavelength_um": sp.center_wavelength_um,
            "cladding_index": clad_n,
        })

    e_fields = np.asarray(modes.e_fields) if hasattr(modes, "e_fields") else None
    y_span = target_src.span_um
    z_span = span_z
    coords = (
        np.linspace(-y_span / 2, y_span / 2, e_fields.shape[3]) if e_fields is not None else None,
        np.linspace(-z_span / 2, z_span / 2, e_fields.shape[4]) if e_fields is not None else None,
    )
    return results, e_fields, coords
