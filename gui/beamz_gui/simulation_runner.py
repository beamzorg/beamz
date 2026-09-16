"""
Assembles a live beamz.Simulation from the connector's current scene, and
runs it off the Qt main thread (Simulation.run() is a long blocking call
with no progress-callback mechanism — confirmed by reading its
docstring — so the best available UX is an indeterminate progress
indicator, not a percentage bar).
"""
from __future__ import annotations

import math
from typing import Any, Optional

import beamz as bz
from PySide6.QtCore import QObject, QThread, Signal

from . import geometry_adapters as ga
from .model import BeamzConnector

DEFAULT_RESOLUTION = 2e-8  # beamz's own default (20 nm) — matches what
# memory_estimate()/compile() will use unless the user overrides it.

# GUI face names (x-/x+/y-/y+/z-/z+, matching the property editor) ->
# beamz's own PML/PEC/Absorber `edges` naming — confirmed directly by
# constructing `PML(edges=[...])` with all six and checking it accepts
# them, since the docstring alone doesn't spell out the exact strings.
_FACE_TO_EDGE = {"x-": "left", "x+": "right", "y-": "bottom", "y+": "top", "z-": "front", "z+": "back"}
_BOUNDARY_CLASSES = {"PML": bz.PML, "PEC": bz.PEC, "Absorber": bz.Absorber}


def default_run_time(span_x: float, span_y: float, span_z: float, sources: tuple = ()) -> float:
    """Heuristic default: long enough for BOTH (a) light to cross the
    domain's diagonal a few times over, and (b) every source's own pulse
    to actually finish ramping up in the first place.

    (b) matters more than it sounds: a GaussianPulse's peak arrives at
    t = offset/fwidth (default offset=4.0), with the bulk of its energy
    spread over roughly +/-2/fwidth around that — for a narrowband pulse
    (small fwidth), that can EASILY exceed a small domain's propagation
    time. Confirmed directly: a default 20 THz-wide pulse peaks at 200 fs,
    while a 2x2x2 micron domain's propagation-only heuristic gave 69 fs —
    the simulation was stopping before the source had even ramped up,
    let alone before anything reached a monitor. That is what an earlier
    version's "empty monitor results" turned out to be, not a solver bug.
    """
    propagation_time = 6 * math.sqrt(span_x**2 + span_y**2 + span_z**2) / bz.LIGHT_SPEED
    pulse_time = 0.0
    for src in sources:
        st = getattr(src, "source_time", None)
        offset = getattr(st, "offset", None)
        fwidth = getattr(st, "fwidth", None)
        if offset is not None and fwidth:
            pulse_time = max(pulse_time, 2 * offset / fwidth)
    return pulse_time + propagation_time


def _build_boundaries(face_settings: dict[str, str]) -> list:
    """Groups per-face settings (e.g. {'x-': 'PML', 'x+': 'PML', 'y-':
    'PEC', ...}) into one boundary-condition object per TYPE, each
    covering the right subset of edges — beamz takes a list of PML/PEC/
    Absorber objects, each itself covering one or more edges, not a
    per-face mapping directly.
    """
    groups: dict[str, list[str]] = {}
    for face, kind in face_settings.items():
        edge = _FACE_TO_EDGE.get(face)
        if edge is None:
            continue
        groups.setdefault(kind, []).append(edge)
    return [_BOUNDARY_CLASSES[kind](edges=edges) for kind, edges in groups.items() if kind in _BOUNDARY_CLASSES]


def _shift_recipe(recipe: dict[str, Any], adapter: ga.GeometryAdapter, offset: tuple[float, float, float]) -> dict:
    """A pure-translation copy of `recipe`'s position field(s) by
    `-offset`. Applied to the CANONICAL (unrotated) recipe — translation
    commutes with rotation, so shift-then-rotate (what _shifted_object
    below does) gives the same placement as rotate-then-shift, without
    needing to reverse-engineer a shift on an already-rotated Polygon's
    raw vertex list.
    """
    ox, oy, oz = offset
    pos = list(recipe.get(adapter.position_field, (0.0,) * adapter.position_len))
    pos[0] -= ox
    if adapter.position_len >= 2:
        pos[1] -= oy
    if adapter.position_len == 3:
        pos[2] -= oz
    changes: dict[str, Any] = {adapter.position_field: tuple(pos)}
    if adapter.position_len == 2 and adapter.z_field:
        changes[adapter.z_field] = recipe.get(adapter.z_field, 0.0) - oz
    if adapter.sync_z_field:
        changes["z"] = pos[2] if adapter.position_len == 3 else changes.get(adapter.z_field, -oz)
    return {**recipe, **changes}


def _shifted_object(so, offset: tuple[float, float, float]) -> Any:
    """A copy of `so`'s beamz instance translated by `-offset` — used to
    re-anchor the whole scene onto beamz's own origin-corner-anchored
    Design when a SimulationRegion places the FDTD region somewhere other
    than the origin. Classes with no geometry adapter (e.g. Polygon) have
    no known position field to shift and are passed through unchanged —
    a documented v1 limitation, not a silent wrong answer.
    """
    if offset == (0.0, 0.0, 0.0):
        return so.obj
    adapter = ga.get_adapter(so.class_name)
    if adapter is None:
        return so.obj
    cls = getattr(bz, so.class_name)
    shifted_recipe = _shift_recipe(so.recipe, adapter, offset)
    obj = cls(**shifted_recipe)
    if so.category == "structure" and so.rotation and hasattr(obj, "rotate"):
        obj = obj.rotate(so.rotation, axis=so.rotation_axis)
    return obj


def compute_region_offset(connector: BeamzConnector) -> tuple[float, float, float]:
    """The (x,y,z) shift applied to every structure/source/monitor when
    assembling a Simulation — beamz's Design has no offset field, so a
    region centered anywhere other than the origin requires re-anchoring
    the whole scene onto it (see build_simulation's docstring). Exposed
    separately so callers that need to show beamz's own shifted-frame
    output (like the mesh-view plot) can shift labels back to the user's
    own world coordinates instead of just showing beamz's numbers as-is
    — the two frames are legitimately different, which is confusing if
    unexplained, not a bug in the shift itself.
    """
    regions = connector.by_category("region")
    if not regions:
        return (0.0, 0.0, 0.0)
    r = regions[0].recipe
    return (r["x"] - r["x_span"] / 2, r["y"] - r["y_span"] / 2, r["z"] - r["z_span"] / 2)


def build_simulation(
    connector: BeamzConnector,
    *,
    run_time: Optional[float] = None,
    resolution: Optional[float] = None,
) -> "bz.Simulation":
    """Assemble a Simulation from the connector's current scene.

    If a SimulationRegion object exists (model.REGION_KIND — added via
    the "+ Region" toolbar action), its x/y/z/x_span/y_span/z_span define
    the domain SIZE, and every structure/source/monitor is re-anchored
    (shifted) onto it, since beamz's own `Design` has no position/offset
    field at all — it's always corner-anchored at the origin (confirmed
    via its docstring). Without a region object, falls back to the
    connector's plain design_width/height/depth (legacy default, always
    at the origin) and PML on every face, for backward compatibility.

    The region's per-face boundary settings and dynamically-introspected
    solver settings (resolution, run_time, plane_2d, polarization, ...)
    are applied here too — see property_editor.py's _render_region for
    where those get edited.

    Validates that every structure has a material assigned FIRST — see
    the inline comment below for why that check exists at all.
    """
    missing_material = [so.name for so in connector.by_category("structure") if so.obj.material is None]
    if missing_material:
        raise ValueError(
            "These structures have no material assigned yet: "
            + ", ".join(missing_material)
            + ". Set a material for each one in the Properties panel before running, "
            "viewing the mesh, or estimating memory."
        )

    regions = connector.by_category("region")
    region = regions[0] if regions else None

    if region is not None:
        r = region.recipe
        sx, sy, sz = r["x_span"], r["y_span"], r["z_span"]
        offset = compute_region_offset(connector)
        design_w, design_h, design_d = sx, sy, sz
        boundaries = _build_boundaries(r.get("boundaries", {})) or [bz.PML()]
        settings = dict(r.get("settings", {}))
    else:
        offset = (0.0, 0.0, 0.0)
        design_w, design_h, design_d = connector.design_width, connector.design_height, connector.design_depth
        boundaries = [bz.PML()]
        settings = {}

    structures = tuple(_shifted_object(so, offset) for so in connector.by_category("structure"))
    design = bz.Design(
        width=design_w, height=design_h, depth=design_d,
        background=connector.background_material, structures=structures,
    )
    sources = [_shifted_object(so, offset) for so in connector.by_category("source")]
    monitors = [_shifted_object(so, offset) for so in connector.by_category("monitor")]

    kwargs: dict[str, Any] = {k: v for k, v in settings.items() if v is not None}
    kwargs["design"] = design
    kwargs["sources"] = sources
    kwargs["monitors"] = monitors
    kwargs["boundaries"] = boundaries
    kwargs["resolution"] = resolution if resolution is not None else kwargs.get("resolution", DEFAULT_RESOLUTION)
    default_spans = (sx, sy, sz) if region else (design_w, design_h, design_d)
    kwargs["run_time"] = run_time if run_time is not None else kwargs.get("run_time") or default_run_time(
        *default_spans, sources=tuple(sources)
    )

    return bz.Simulation(**kwargs)


def device_info() -> dict[str, Any]:
    """What backend/device beamz (via JAX) will actually run on — CPU,
    and GPU if one is visible to JAX in this environment.
    """
    try:
        import jax

        devices = jax.devices()
        return {
            "backend": jax.default_backend(),
            "devices": [str(d) for d in devices],
            "device_count": len(devices),
        }
    except Exception as exc:  # pragma: no cover - defensive only
        return {"backend": "unknown", "devices": [], "device_count": 0, "error": str(exc)}


def format_memory_report(report: dict[str, Any]) -> str:
    """Human-readable summary of Simulation.memory_estimate()'s dict."""
    lines = [f"Estimated memory: {report.get('total_gib', 0):.3f} GiB"]
    compiled_gib = report.get("total_with_compiled_gib")
    if compiled_gib is not None:
        lines.append(f"  (including compiled/runtime overhead: {compiled_gib:.3f} GiB)")

    grid_shape = report.get("grid_shape_zyx")
    if grid_shape:
        lines.append(f"Grid shape (z, y, x): {tuple(grid_shape)}")

    by_category = report.get("totals_by_category") or {}
    if by_category:
        lines.append("")
        lines.append("By category:")
        for name, nbytes in sorted(by_category.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {name}: {nbytes / 1e9:.3f} GB")

    info = device_info()
    lines.append("")
    lines.append(f"Backend: {info['backend']} ({info['device_count']} device(s): {', '.join(info['devices'])})")
    return "\n".join(lines)


class SimulationWorker(QObject):
    """Runs Simulation.run() on a background QThread. `run()` itself is
    synchronous/blocking with no incremental progress callback (verified
    by reading beamz's own docs), so the only progress UI that's honest
    to offer is "running" vs. "done"/"failed" — not a percentage.
    """

    finished = Signal(object)  # SimulationResults
    failed = Signal(str)

    def __init__(self, simulation: "bz.Simulation") -> None:
        super().__init__()
        self._simulation = simulation

    def run(self) -> None:
        try:
            results = self._simulation.run(progress=False)
        except Exception as exc:  # noqa: BLE001 — surfaced to the UI, not swallowed
            self.failed.emit(str(exc))
            return
        self.finished.emit(results)


def run_simulation_async(simulation: "bz.Simulation", on_finished, on_failed) -> tuple[QThread, SimulationWorker]:
    """Wires up a SimulationWorker on its own QThread. Caller must keep a
    reference to the returned (thread, worker) pair alive until finished/
    failed fires — Qt does not keep a QThread alive on your behalf just
    because it's running.
    """
    thread = QThread()
    worker = SimulationWorker(simulation)
    worker.moveToThread(thread)

    thread.started.connect(worker.run)
    worker.finished.connect(on_finished)
    worker.failed.connect(on_failed)
    worker.finished.connect(thread.quit)
    worker.failed.connect(thread.quit)
    # `thread.quit()` merely REQUESTS the thread's event loop to stop —
    # it does not block until the underlying OS thread has actually
    # finished. Connecting deleteLater directly off quit() (an earlier
    # version did exactly that) let Qt schedule the QThread object for
    # deletion before the thread had actually joined, which reproduced as
    # a real crash on process/app exit: "QThread: Destroyed while thread
    # '' is still running" followed by a hard abort — confirmed by
    # actually running a full build -> thread -> run -> plot cycle end to
    # end, not just reasoning about it. `thread.wait()` blocks (briefly —
    # the thread is already finishing) until the OS thread has genuinely
    # exited, so deleteLater only runs once that's true.
    thread.finished.connect(thread.wait)
    worker.finished.connect(worker.deleteLater)
    worker.failed.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)

    thread.start()
    return thread, worker


class _CallableWorker(QObject):
    """Generic version of SimulationWorker for any slow, blocking call —
    used for memory_estimate()/plot(), which were previously called
    directly on the GUI thread and could freeze (or, for a large grid,
    OOM-crash) the entire app with no way to even see a progress
    indicator. Confirmed directly: building this app's own memory
    estimate for a 25M-cell grid was slow enough to be indistinguishable
    from a hang, and OOM-killed the process outright in a
    memory-constrained environment — exactly the "massive lag... crashes
    entire UI" symptom. Same fix as Run Solver: off the GUI thread, with
    only "running"/"done"/"failed" progress (no real percentage is
    available for these calls either).
    """

    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, fn) -> None:
        super().__init__()
        self._fn = fn

    def run(self) -> None:
        try:
            result = self._fn()
        except Exception as exc:  # noqa: BLE001 — surfaced to the UI, not swallowed
            self.failed.emit(str(exc))
            return
        self.finished.emit(result)


def run_off_thread(fn, on_finished, on_failed) -> tuple[QThread, _CallableWorker]:
    """Runs any zero-argument callable on a background QThread — same
    proven wiring as run_simulation_async (including the thread.wait()
    fix for the "destroyed while still running" crash on cleanup), just
    generalized to an arbitrary function instead of Simulation.run()
    specifically. Caller must keep the returned (thread, worker) pair
    alive until finished/failed fires.
    """
    thread = QThread()
    worker = _CallableWorker(fn)
    worker.moveToThread(thread)

    thread.started.connect(worker.run)
    worker.finished.connect(on_finished)
    worker.failed.connect(on_failed)
    worker.finished.connect(thread.quit)
    worker.failed.connect(thread.quit)
    thread.finished.connect(thread.wait)
    worker.finished.connect(worker.deleteLater)
    worker.failed.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)

    thread.start()
    return thread, worker