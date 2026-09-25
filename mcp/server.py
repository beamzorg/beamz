"""BeamZ MCP server (v0.7).
Full support for Rect, Circle, Ring, Polygon, ModeSource, GaussianSource, FluxMonitor, ModeMonitor, and TimeMonitor.
"""
from __future__ import annotations
import os, sys, contextlib, io, time, traceback, uuid
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from mcp.server.mcpserver import MCPServer, Image
from spec import (Scene, Domain, Spectrum, Rect, Circle, Ring, Polygon,
                  ModeSource, GaussianSourceSpec, FluxMonitor, ModeMonitor, TimeMonitorSpec, ModeSpec,
                  validate_scene, export_python, recommended_run_time_fs, estimate_cost, solve_scene_modes)

INSTRUCTIONS = """You are an expert computational nanophotonics engineer with access to the BeamZ FDTD simulation suite.

COORDINATE CONVENTIONS (CRITICAL):
- All spatial dimensions, spans, and coordinates are in MICROMETERS (um).
- Rect / add_structure: (x_um, y_um, z_um) is the LOWER-LEFT-FRONT corner! To center a waveguide of width W and height H at (xc, yc), place it at (xc - W/2, yc - H/2).
- Circle, Ring, Polygon: (x_um, y_um) is the xy-CENTER.
- ModeSource, GaussianSource, FluxMonitor, ModeMonitor, TimeMonitor: (x_um, y_um, z_um) is the CENTER.

PML BOUNDARIES & CONTINUITY:
- Continuous input and output waveguides MUST extend completely through the PML boundaries (e.g., from x=0 to x=domain.width_um) to eliminate artificial facet reflections.
- Sources and monitors MUST be placed inside the active domain, away from PML borders (separated by at least 0.5 um).

RECOMMENDED WORKFLOWS:
Option A (Fast Single-Shot):
- Call 'simulate_device' with the full device layout in a single tool call. It validates, runs FDTD, and returns transmission spectra and summaries in one step without polling.

Option B (Granular Step-by-Step):
1. 'create_scene(domain, spectrum)': Initialize 2D or 3D domain.
2. Add geometries: 'add_structure', 'add_circle', 'add_ring', 'add_polygon'.
3. Add excitation and monitors: 'add_source', 'add_monitor', 'add_mode_monitor', 'add_time_monitor'.
4. Check design: 'validate(scene_id)' and/or 'solve_modes(scene_id)'.
5. Execute: 'run_simulation(scene_id, wait=True)'.
6. Analyze: 'get_transmission(job_id)', 'get_mode_data(job_id)', or 'plot_field_snapshot(job_id)'.
"""

mcp = MCPServer(
    name="beamz",
    title="BeamZ Photonics FDTD Simulator",
    description="Electromagnetic FDTD simulation and eigenmode analysis for 2D and 3D photonic integrated circuits.",
    instructions=INSTRUCTIONS,
)
SCENES: dict[str, Scene] = {}
JOBS: dict[str, dict] = {}
POOL = ThreadPoolExecutor(max_workers=1)


def _scene(sid: str) -> Scene:
    if sid not in SCENES:
        raise ValueError(f"Unknown scene_id '{sid}'. Call create_scene first.")
    return SCENES[sid]


def _job(jid: str) -> dict:
    if jid not in JOBS:
        raise ValueError(f"Unknown job_id '{jid}'.")
    return JOBS[jid]


def _done(job_id: str) -> dict:
    j = _job(job_id)
    if j["status"] != "done":
        raise ValueError(f"Job is '{j['status']}', results not available.")
    return j


# ---------------- Scene & Geometry Tools ----------------

@mcp.tool()
def simulate_device(
    domain: Domain,
    spectrum: Spectrum = Spectrum(),
    structures: list[Rect] = [],
    circles: list[Circle] = [],
    rings: list[Ring] = [],
    polygons: list[Polygon] = [],
    sources: list[ModeSource] = [],
    gaussian_sources: list[GaussianSourceSpec] = [],
    monitors: list[FluxMonitor] = [],
    mode_monitors: list[ModeMonitor] = [],
    time_monitors: list[TimeMonitorSpec] = [],
    run_time_fs: float | None = None,
    timeout_s: float = 60.0,
) -> dict:
    """All-in-one tool: create, validate, simulate, and analyze a complete photonic device in ONE single call.
    Recommended for AI chatbots to avoid multi-turn polling.
    Returns transmission summary, status, and job_id."""
    sid = uuid.uuid4().hex[:8]
    scene = Scene(
        id=sid,
        domain=domain,
        spectrum=spectrum,
        structures=structures,
        circles=circles,
        rings=rings,
        polygons=polygons,
        sources=sources,
        gaussian_sources=gaussian_sources,
        monitors=monitors,
        mode_monitors=mode_monitors,
        time_monitors=time_monitors,
        run_time_fs=run_time_fs,
    )
    SCENES[sid] = scene

    errors = [w for w in validate_scene(scene) if w["level"] == "error"]
    if errors:
        return {"success": False, "scene_id": sid, "status": "validation_failed", "errors": errors}

    sim_res = run_simulation(sid, wait=True, timeout_s=timeout_s)
    sim_res["scene_id"] = sid
    sim_res["success"] = (sim_res.get("status") == "done")
    return sim_res


@mcp.tool()
def create_scene(domain: Domain, spectrum: Spectrum = Spectrum(), run_time_fs: float | None = None) -> str:
    """Create a 2D or 3D FDTD scene. Lengths in micrometers.
    Set domain.depth_um for 3D simulation, or omit for 2D.
    Leave run_time_fs unset to auto-pick a sufficient run time."""
    sid = uuid.uuid4().hex[:8]
    SCENES[sid] = Scene(id=sid, domain=domain, spectrum=spectrum, run_time_fs=run_time_fs)
    return sid


@mcp.tool()
def add_structure(scene_id: str, structure: Rect) -> str:
    """Add a rectangle or 3D box. (x_um, y_um, z_um) is its lower-left corner; index is refractive index."""
    _scene(scene_id).structures.append(structure)
    return f"Added rectangle/box '{structure.name}'."


@mcp.tool()
def add_circle(scene_id: str, circle: Circle) -> str:
    """Add a circular cylinder / disc. (x_um, y_um) is center, radius_um is radius."""
    _scene(scene_id).circles.append(circle)
    return f"Added circle '{circle.name}' (radius={circle.radius_um} um, index={circle.index})."


@mcp.tool()
def add_ring(scene_id: str, ring: Ring) -> str:
    """Add a concentric ring (e.g. microring resonator). (x_um, y_um) is center."""
    _scene(scene_id).rings.append(ring)
    return f"Added ring '{ring.name}' (inner_r={ring.inner_radius_um}, outer_r={ring.outer_radius_um} um)."


@mcp.tool()
def add_polygon(scene_id: str, polygon: Polygon) -> str:
    """Add a custom polygon defined by a list of 2D vertices [(x1, y1), (x2, y2), ...]."""
    _scene(scene_id).polygons.append(polygon)
    return f"Added polygon '{polygon.name}' with {len(polygon.vertices)} vertices."


# ---------------- Sources & Monitors ----------------

@mcp.tool()
def add_source(scene_id: str, source: ModeSource) -> str:
    """Add a mode source (launches the fundamental/specified guided mode) on a line or plane."""
    _scene(scene_id).sources.append(source)
    return f"Added mode source '{source.name}'."


@mcp.tool()
def add_gaussian_source(scene_id: str, source: GaussianSourceSpec) -> str:
    """Add a spatial Gaussian beam source (waist radius, propagation direction)."""
    _scene(scene_id).gaussian_sources.append(source)
    return f"Added Gaussian source '{source.name}' (waist={source.waist_radius_um} um, dir={source.direction})."


@mcp.tool()
def add_monitor(scene_id: str, monitor: FluxMonitor) -> str:
    """Add a frequency-domain power-flux monitor on a line or plane across the waveguide."""
    _scene(scene_id).monitors.append(monitor)
    return f"Added flux monitor '{monitor.name}'."


@mcp.tool()
def add_mode_monitor(scene_id: str, monitor: ModeMonitor) -> str:
    """Add a modal decomposition monitor that projects fields into forward/backward guided mode amplitudes."""
    _scene(scene_id).mode_monitors.append(monitor)
    return f"Added mode monitor '{monitor.name}'."


@mcp.tool()
def add_time_monitor(scene_id: str, monitor: TimeMonitorSpec) -> str:
    """Add a time-domain field recorder (takes 2D/3D field snapshots at intervals)."""
    _scene(scene_id).time_monitors.append(monitor)
    return f"Added time monitor '{monitor.name}' (components={monitor.components}, interval={monitor.interval})."


# ---------------- Inspection & Linting ----------------

@mcp.tool()
def get_scene(scene_id: str) -> dict:
    """Return scene spec as JSON, plus auto run time in fs and compute cost estimates."""
    s = _scene(scene_id)
    return {
        **s.model_dump(),
        "recommended_run_time_fs": round(recommended_run_time_fs(s), 1),
        "estimates": estimate_cost(s),
    }


@mcp.tool()
def validate(scene_id: str) -> list[dict]:
    """Physics lint (resolution, PML, placement, run time, compute cost). Empty list = OK."""
    return validate_scene(_scene(scene_id))


@mcp.tool()
def estimate_compute_cost(scene_id: str) -> dict:
    """Estimate grid cell count, FDTD time steps, and CPU compute time for the scene."""
    return estimate_cost(_scene(scene_id))


@mcp.tool()
def solve_modes(scene_id: str, source_name: str | None = None) -> list[dict]:
    """Pre-solve waveguide eigenmodes before running FDTD. Returns n_eff and guiding check."""
    s = _scene(scene_id)
    info, _, _ = solve_scene_modes(s, source_name)
    return info


@mcp.tool()
def export_code(scene_id: str) -> str:
    """Export the scene as the exact BeamZ Python script that run_simulation executes."""
    return export_python(_scene(scene_id))


# ---------------- Simulation Execution ----------------

def _run(jid: str, code: str):
    job = JOBS[jid]
    job.update(status="running", started=time.time())
    try:
        ns: dict = {}
        with contextlib.redirect_stdout(sys.stderr):
            exec(compile(code, f"<beamz-scene {job['scene_id']}>", "exec"), ns)
        res = ns["results"]
        job["data"] = {}
        job["modal_data"] = {}
        job["time_data"] = {}
        job["_raw_snapshots"] = {}

        # Flux monitors
        for m in job.get("monitors", []):
            if hasattr(res, "monitors") and m in res.monitors:
                job["data"][m] = np.asarray(res.monitors[m].flux, float)
            elif hasattr(res, m):
                job["data"][m] = np.asarray(getattr(res, m).flux, float)

        # Mode monitors
        for m in job.get("mode_monitors", []):
            try:
                m_data = res.mode(m)
                amps_fwd = m_data.amps.sel(direction="+").values if hasattr(m_data.amps, "sel") else m_data.amps
                amps_bwd = m_data.amps.sel(direction="-").values if hasattr(m_data.amps, "sel") else m_data.amps

                def _ser_amps(arr):
                    out = []
                    for val in np.asarray(arr).flatten():
                        c = complex(val)
                        out.append({
                            "real": round(float(c.real), 5),
                            "imag": round(float(c.imag), 5),
                            "abs": round(float(abs(c)), 5),
                            "phase_deg": round(float(np.degrees(np.angle(c))), 2),
                        })
                    return out

                flx = np.asarray(m_data.flux, float).flatten().tolist()
                mflx = np.asarray(m_data.modal_flux, float).flatten().tolist()
                purity = [round(float(m / f * 100), 2) if f > 0 else 0.0 for m, f in zip(mflx, flx)]

                job["modal_data"][m] = {
                    "amps_forward": _ser_amps(amps_fwd),
                    "amps_backward": _ser_amps(amps_bwd),
                    "flux": flx,
                    "modal_flux": mflx,
                    "modal_purity_percent": purity,
                }
                job["data"][m] = np.asarray(m_data.flux, float)
            except Exception:
                pass

        # Time monitors (FieldRecorder)
        for tm in job.get("time_monitors", []):
            try:
                rec = res.monitor(tm) if hasattr(res, "monitor") else res.monitors[tm]
                fields_dict = rec.fields
                job["_raw_snapshots"][tm] = fields_dict
                job["time_data"][tm] = {
                    "components": list(fields_dict.keys()),
                    "num_snapshots": len(next(iter(fields_dict.values()))),
                    "snapshot_shape": list(next(iter(fields_dict.values()))[0].shape),
                }
            except Exception:
                pass

        job["wavelengths_um"] = np.asarray(ns["wavelengths_um"], float)
        job["status"] = "done"
    except Exception as e:
        job.update(status="failed", error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-1500:])
    finally:
        job["finished"] = time.time()


@mcp.tool()
def run_simulation(scene_id: str, wait: bool = True, timeout_s: float = 60.0) -> dict:
    """Validate and start simulation.
    If wait=True (default), blocks until done or timeout_s.
    If wait=False, starts in background and returns job_id immediately."""
    s = _scene(scene_id)
    errors = [w for w in validate_scene(s) if w["level"] == "error"]
    if errors:
        return {"started": False, "status": "failed", "errors": errors}

    jid = uuid.uuid4().hex[:8]
    mon_names = [m.name for m in s.monitors]
    mode_mon_names = [m.name for m in s.mode_monitors]
    time_mon_names = [tm.name for tm in s.time_monitors]

    JOBS[jid] = dict(
        status="queued",
        scene_id=scene_id,
        monitors=mon_names,
        mode_monitors=mode_mon_names,
        time_monitors=time_mon_names,
        created=time.time(),
    )
    POOL.submit(_run, jid, export_python(s))

    if not wait:
        return {"started": True, "job_id": jid, "status": "queued"}

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if JOBS[jid]["status"] in ("done", "failed"):
            break
        time.sleep(0.25)

    status_info = get_status(jid)
    out = {"started": True, "job_id": jid, **status_info}

    if JOBS[jid]["status"] == "done":
        all_mons = mon_names + mode_mon_names
        if len(all_mons) >= 2:
            try:
                trans = get_transmission(jid, monitor=all_mons[-1], reference=all_mons[0])
                out["transmission_summary"] = {
                    "monitor": all_mons[-1],
                    "reference": all_mons[0],
                    "at_center": trans["at_center"],
                    "mean": trans["mean"],
                }
            except Exception:
                pass
        if mode_mon_names:
            out["mode_monitors"] = {mm: get_mode_data(jid, monitor=mm) for mm in mode_mon_names}
        if time_mon_names:
            out["time_monitors"] = {tm: get_time_data(jid, monitor=tm) for tm in time_mon_names}
    return out


@mcp.tool()
def get_status(job_id: str) -> dict:
    """Job status: queued | running | done | failed."""
    j = _job(job_id)
    out = {"status": j["status"], "elapsed_s": round((j.get("finished") or time.time()) - j.get("started", j["created"]), 1)}
    if j["status"] == "failed":
        out.update(error=j["error"], trace=j.get("trace"))
    return out


# ---------------- Results & Visualization ----------------

@mcp.tool()
def get_transmission(job_id: str, monitor: str, reference: str) -> dict:
    """Transmission spectrum = flux(monitor) / flux(reference), plus scalar summary."""
    j = _done(job_id)
    for m in (monitor, reference):
        if m not in j["data"]:
            raise ValueError(f"Unknown monitor '{m}'. Available: {list(j['data'])}")
    T = j["data"][monitor] / j["data"][reference]
    wl = j["wavelengths_um"]
    return {
        "wavelength_um": [round(float(x), 4) for x in wl],
        "transmission": [round(float(x), 4) for x in T],
        "mean": round(float(T.mean()), 4),
        "min": round(float(T.min()), 4),
        "max": round(float(T.max()), 4),
        "at_center": round(float(T[len(T) // 2]), 4),
    }


@mcp.tool()
def get_mode_data(job_id: str, monitor: str) -> dict:
    """Return modal decomposition results for a ModeMonitor (amplitudes, modal purity %, flux)."""
    j = _done(job_id)
    modal = j.get("modal_data", {})
    if monitor not in modal:
        raise ValueError(f"Unknown mode monitor '{monitor}'. Available: {list(modal)}")
    return modal[monitor]


@mcp.tool()
def get_time_data(job_id: str, monitor: str) -> dict:
    """Return summary metadata for a TimeMonitor (FieldRecorder snapshots)."""
    j = _done(job_id)
    tdata = j.get("time_data", {})
    if monitor not in tdata:
        raise ValueError(f"Unknown time monitor '{monitor}'. Available: {list(tdata)}")
    return tdata[monitor]


@mcp.tool()
def plot_transmission(job_id: str, monitor: str, reference: str) -> Image:
    """PNG plot of transmission vs wavelength."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    j = _done(job_id)
    T = j["data"][monitor] / j["data"][reference]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(j["wavelengths_um"], T, "o-")
    ax.set_xlabel("Wavelength (um)"); ax.set_ylabel(f"{monitor} / {reference}"); ax.grid(alpha=.3)
    buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png", dpi=110); plt.close(fig)
    return Image(data=buf.getvalue(), format="png")


@mcp.tool()
def plot_mode_profile(scene_id: str, source_name: str | None = None, mode_index: int = 0) -> Image:
    """PNG plot of transverse electric field intensity |E|^2 for a solved waveguide mode."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = _scene(scene_id)
    mode_info, e_fields, coords = solve_scene_modes(s, source_name)
    if e_fields is None or mode_index >= e_fields.shape[1]:
        raise ValueError(f"Mode index {mode_index} not available (solved {len(mode_info)} modes).")
    e_mode = e_fields[0, mode_index]
    intensity = np.sum(np.abs(e_mode)**2, axis=0)
    if coords is None:
        raise ValueError("Mode coordinates are not available for plotting.")
    y_coords, z_coords = coords

    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    im = ax.pcolormesh(y_coords, z_coords, intensity.T, cmap="inferno", shading="auto")
    neff = mode_info[mode_index]["neff_real"]
    ax.set_title(f"Mode #{mode_index} ($n_{{eff}} = {neff:.4f}$)")
    ax.set_xlabel("Transverse y (um)")
    ax.set_ylabel("Transverse z (um)")
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, label="Intensity |E|² (a.u.)", shrink=0.8)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return Image(data=buf.getvalue(), format="png")


@mcp.tool()
def plot_field_snapshot(job_id: str, monitor: str = "fields", step_index: int = -1, component: str = "Ez") -> Image:
    """PNG plot of a time-domain field snapshot recorded by a TimeMonitor (FieldRecorder)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    j = _done(job_id)
    raw = j.get("_raw_snapshots", {}).get(monitor)
    if not raw or component not in raw:
        raise ValueError(f"No snapshot for component '{component}' in monitor '{monitor}'.")
    snaps = raw[component]
    if len(snaps) == 0:
        raise ValueError(f"Monitor '{monitor}' has no recorded snapshots.")
    field_arr = snaps[step_index]
    # If 3D, take middle slice
    if field_arr.ndim == 3:
        mid_z = field_arr.shape[0] // 2
        field_2d = field_arr[mid_z]
    else:
        field_2d = field_arr

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    vmax = float(np.max(np.abs(field_2d))) or 1.0
    im = ax.imshow(field_2d.T, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
    step_label = len(snaps) + step_index if step_index < 0 else step_index
    ax.set_title(f"Field Snapshot: {component} (frame {step_label}/{len(snaps)})")
    ax.set_xlabel("Grid X")
    ax.set_ylabel("Grid Y")
    fig.colorbar(im, ax=ax, label=f"{component} field", shrink=0.8)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return Image(data=buf.getvalue(), format="png")


# ---------------- Convenience Macro Tool ----------------

@mcp.tool()
def simulate_straight_waveguide(
    width_um: float = 1.0,
    height_um: float | None = 1.0,
    length_um: float = 10.0,
    core_index: float = 3.48,
    cladding_index: float = 1.44,
    wavelength_um: float = 1.55,
    resolution_um: float = 0.05,
    wait: bool = True,
    timeout_s: float = 90.0,
) -> dict:
    """One-shot helper: builds, validates, solves modes, runs FDTD, and returns modal transmission and purity.
    If height_um is provided, simulates in 3D. If height_um is None, simulates in 2D."""
    is_3d = height_um is not None
    pml_um = 0.8
    dom_width = length_um + 3.5
    dom_height = width_um + 2.2
    dom_depth = (height_um + 2.2) if is_3d else None

    dom = Domain(
        width_um=dom_width,
        height_um=dom_height,
        depth_um=dom_depth,
        cladding_index=cladding_index,
        resolution_um=resolution_um,
        pml_um=pml_um,
    )
    sid = create_scene(domain=dom, spectrum=Spectrum(center_wavelength_um=wavelength_um))

    cy = dom_height / 2.0
    cz = (dom_depth / 2.0) if dom_depth is not None else 0.0

    add_structure(sid, Rect(
        name="wg",
        x_um=0.0,
        y_um=cy - width_um / 2.0,
        z_um=(cz - height_um / 2.0) if is_3d else 0.0,
        width_um=dom_width,
        height_um=width_um,
        depth_um=height_um if is_3d else None,
        index=core_index,
    ))

    span = max(width_um * 2.5, 2.0)
    add_source(sid, ModeSource(
        name="src",
        x_um=1.5,
        y_um=cy,
        z_um=cz,
        span_um=span,
        span_z_um=span if is_3d else None,
        normal="x",
        direction="+",
        mode_spec=ModeSpec(num_modes=1),
    ))

    mon_in_x = 2.0
    mon_out_x = mon_in_x + length_um

    add_mode_monitor(sid, ModeMonitor(
        name="mon_in",
        x_um=mon_in_x,
        y_um=cy,
        z_um=cz,
        span_um=span,
        span_z_um=span if is_3d else None,
        normal="x",
        mode_spec=ModeSpec(num_modes=1),
    ))
    add_mode_monitor(sid, ModeMonitor(
        name="mon_out",
        x_um=mon_out_x,
        y_um=cy,
        z_um=cz,
        span_um=span,
        span_z_um=span if is_3d else None,
        normal="x",
        mode_spec=ModeSpec(num_modes=1),
    ))

    modes = solve_modes(sid)
    sim_res = run_simulation(sid, wait=wait, timeout_s=timeout_s)

    return {
        "scene_id": sid,
        "is_3d": is_3d,
        "solved_modes": modes,
        "simulation": sim_res,
    }


# ---------------- Pre-Configured Prompts ----------------

@mcp.prompt(name="design_straight_waveguide", description="Template to simulate a straight dielectric waveguide.")
def prompt_straight_wg(wavelength_um: float = 1.55, width_um: float = 0.5, length_um: float = 10.0) -> str:
    return (
        f"Design and simulate a straight waveguide of width {width_um} um and length {length_um} um "
        f"at lambda = {wavelength_um} um. Run modal analysis and compute transmission across the waveguide."
    )


@mcp.prompt(name="design_y_splitter", description="Template to design a 50/50 Y-junction power splitter.")
def prompt_y_splitter(wavelength_um: float = 1.55, arm_pitch_um: float = 2.0) -> str:
    return (
        f"Design and simulate a 50/50 Y-beam splitter operating at {wavelength_um} um with an output arm "
        f"pitch of {arm_pitch_um} um. Optimize taper and branch dimensions to minimize excess loss."
    )


@mcp.prompt(name="design_microring_resonator", description="Template to design a bus-coupled microring resonator.")
def prompt_ring(wavelength_um: float = 1.55, ring_radius_um: float = 5.0, gap_nm: float = 150.0) -> str:
    return (
        f"Design and simulate a bus-coupled microring resonator with ring radius {ring_radius_um} um and "
        f"coupling gap {gap_nm} nm at center wavelength {wavelength_um} um. Extract the resonance dip and Q-factor."
    )


@mcp.prompt(name="design_wdm", description="Template to design a multi-channel wavelength division multiplexer.")
def prompt_wdm(channels_nm: str = "450,532,650") -> str:
    return (
        f"Design and simulate a multi-channel WDM demultiplexer separating target wavelengths [{channels_nm}] nm "
        f"into dedicated single-mode output waveguides using foundry PDK dimensions."
    )


if __name__ == "__main__":
    mcp.run()
