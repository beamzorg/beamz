"""
BeamzConnector — the single source of truth bridging the GUI and the live
beamz session.

Two facts about beamz's dataclasses drive most of the design here:

1. They are FROZEN. Every edit is a `replace`, never a mutation. So the
   GUI can't use Python object identity as a stable handle — every
   structure/source/monitor is wrapped in a `SceneObject` with a GUI-owned
   stable `id`, and the connector is the only thing allowed to swap the
   underlying beamz instance out from under that id.

2. Rotation is NOT a stored field. `rectangle.rotate(angle)` doesn't set
   an attribute — it returns a completely different type (a `Polygon`)
   with the rotation baked into its vertices. A naive "rotation" spinbox
   editing the live object would work exactly once, after which the
   object silently becomes a Polygon and width/height are gone for good.

   So a SceneObject does not track "the current beamz object" as its
   ground truth. It tracks a *recipe*: the canonical (unrotated)
   constructor kwargs, plus a separate `rotation` angle. The actual beamz
   instance (`SceneObject.obj`) is always a *derived, regenerated* value —
   `cls(**recipe).rotate(rotation, axis)` — rebuilt from scratch on every
   edit. This keeps every field editable forever, independent of rotation
   state, at the cost of never trusting `so.obj` as anything other than a
   cache of the last regeneration.

Sources and monitors are placeable objects too (not just structures), so
`SceneObject` carries a `category` ("structure" | "source" | "monitor")
used purely for UI grouping/rendering — the recipe/regenerate machinery is
identical for all three, though rotation only makes sense for structures.
"""
from __future__ import annotations

import dataclasses
import uuid
from typing import Any, Optional

import beamz as bz
import numpy as np
from PySide6.QtCore import QObject, Signal

from . import geometry_adapters as ga
from .undo import UndoManager

# Explicit whitelists rather than duck-typing on shared methods like
# `updated_copy` (Material/Design/Simulation have that too, and are not
# placeable scene objects).
STRUCTURE_KINDS = (
    "Rectangle",
    "Circle",
    "Ring",
    "Taper",
    "CircularBend",
    "Polygon",
    "Box",
    "Sphere",
)
SOURCE_KINDS = (
    "GaussianBeamSource",
    "GaussianSource",
    "ModeSource",
    "CustomSource",
)
MONITOR_KINDS = (
    "FieldMonitor",
    "FluxMonitor",
    "ModeMonitor",
    # A real beamz monitor class (subclasses beamz's internal _Monitor,
    # confirmed directly — `Simulation(monitors=...)` accepts it in the
    # exact same list as the three above, no separate kwarg needed) that
    # had simply never been wired into the GUI before now: records raw
    # time-domain field snapshots (Ex/Ey/Ez/Hx/Hy/Hz) on a center/size
    # plane, as opposed to FieldMonitor's frequency-domain DFT fields.
    "FieldRecorder",
)
# Not a real beamz class — a GUI-only placeable representing the FDTD
# simulation region itself (Lumerical calls this the "FDTD" object).
# beamz's own `Design` has no position/offset field at all (always
# corner-anchored at the origin; confirmed via its docstring/signature),
# so a user-movable, center-based region requires a coordinate-shift step
# when actually assembling a Simulation — see simulation_runner.py.
REGION_KIND = "SimulationRegion"

_CATEGORY_BY_KIND = {
    **{k: "structure" for k in STRUCTURE_KINDS},
    **{k: "source" for k in SOURCE_KINDS},
    **{k: "monitor" for k in MONITOR_KINDS},
}


def classify_category(value: Any) -> Optional[str]:
    """Which placeable category (if any) a namespace value belongs to,
    used by both toolbar-driven creation and script-namespace syncing.
    """
    return _CATEGORY_BY_KIND.get(type(value).__name__)


def has_native_rotation_field(class_name: str) -> bool:
    """Some beamz classes (e.g. `CircularBend`) define their OWN
    `rotation` constructor field, distinct from the post-hoc `.rotate()`
    transform every structure supports. For those, rotation should be
    edited as an ordinary recipe field (`update_param(rotation=...)`),
    NOT via `update_rotation()`'s recipe+rotate()-fresh mechanism —
    calling `.rotate()` on top of a class that already has its own
    rotation field would rotate it twice, in two different ways.
    """
    cls = getattr(bz, class_name, None)
    if cls is None or not dataclasses.is_dataclass(cls):
        return False
    return any(f.name == "rotation" and f.init for f in dataclasses.fields(cls))


def _recipe_from_instance(obj: Any) -> dict[str, Any]:
    """Capture an existing beamz dataclass instance's constructor kwargs.
    Only `init=True` fields are included — derived/computed fields like
    `vertices`/`interiors` (which the class fills in via `__post_init__`,
    not accepted as constructor args) are excluded automatically.
    """
    return {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj) if f.init}


def _values_equal(a: Any, b: Any) -> bool:
    """Safe equality for recipe values — needed by restore()'s diff below
    because a plain `a == b` on some recipe values doesn't return a bool
    at all: a numpy array's `==` (e.g. a FieldMonitor's `freqs`) returns
    an ELEMENTWISE array, and `bool()`-ing that to decide "are these
    equal" raises "the truth value of an array... is ambiguous" instead
    of ever getting an answer. Recurses into dicts/lists/tuples and
    beamz dataclass instances (Material, GaussianPulse, ...) field-by-
    field rather than comparing by identity/repr, so two SEPARATELY
    CONSTRUCTED but equal-content values (e.g. the same Material made
    twice) still compare equal.
    """
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_values_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_values_equal(x, y) for x, y in zip(a, b))
    if (
        dataclasses.is_dataclass(a)
        and dataclasses.is_dataclass(b)
        and not isinstance(a, type)
        and not isinstance(b, type)
    ):
        if type(a) is not type(b):
            return False
        return all(_values_equal(getattr(a, f.name), getattr(b, f.name)) for f in dataclasses.fields(a) if f.init)
    return a == b


def _recipe_equal(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return _values_equal(a, b)


@dataclasses.dataclass
class SceneObject:
    """Stable-identity wrapper around a *derived* beamz instance.

    `obj` is always regenerated from `recipe` (+ `rotation` for
    structures) — never edited or trusted as ground truth on its own.
    """

    id: str
    category: str  # "structure" | "source" | "monitor"
    class_name: str  # beamz class name, e.g. "Rectangle", "FieldMonitor"
    recipe: dict[str, Any]  # canonical (unrotated) constructor kwargs
    name: str = ""
    rotation: float = 0.0  # DEGREES; meaningful for category == "structure"
    rotation_axis: str = "z"
    obj: Any = None  # cached beamz instance, regenerated on every edit
    enabled: bool = True  # if False, excluded from build_design/sources/
    # monitors (simulation_runner.py) but still visible/editable in the
    # GUI — "suppress from a run without deleting it" (e.g. A/B testing
    # a structure's presence).


class BeamzConnector(QObject):
    """Owns the current scene (structures + sources + monitors) and design
    settings, and exposes CRUD via stable ids. Emits Qt signals so any
    number of views (object tree, 2D canvas, 3D preview, property editor)
    stay in sync without knowing about each other.

    Two front doors mutate the same state: GUI widgets call `add_object` /
    `update_param` / `update_rotation` / `remove_structure` directly; the
    Script Console runs arbitrary code against `self.namespace` and then
    calls `sync_from_script()` to reconcile whatever the script did back
    into tracked SceneObjects. Either path ends up emitting the same
    signals, so every view is agnostic to which one happened.
    """

    structure_added = Signal(str)  # sid
    structure_removed = Signal(str)  # sid
    structure_changed = Signal(str)  # sid
    selection_changed = Signal(object)  # sid or None
    design_changed = Signal()  # anything about the overall design changed

    def __init__(self) -> None:
        super().__init__()
        self._structures: dict[str, SceneObject] = {}
        self._order: list[str] = []  # preserves insertion/z-order
        self._name_to_sid: dict[str, str] = {}  # namespace var name -> sid

        # Design-level (simulation domain) settings — minimal for v1.
        self.design_width: float = 10e-6
        self.design_height: float = 10e-6
        self.design_depth: float = 1e-6
        self.background_material: bz.Material = bz.Material(permittivity=1.0)

        # Shared live namespace — the Script Console executes user code
        # against this exact dict, so GUI edits and typed script commands
        # are two views onto one truth rather than two things to keep in
        # sync. `sync_from_script()` is what reconciles it back.
        self.namespace: dict[str, Any] = {"bz": bz}

        # Whole-scene-snapshot undo/redo — see undo.py. Coarse-grained on
        # purpose (checkpoints are taken by GUI code right before a
        # discrete action like Apply/Delete/Add, not per keystroke).
        self.undo_manager = UndoManager(self)

    # ------------------------------------------------------------------ #
    # Placeable-object CRUD (GUI-driven front door)
    # ------------------------------------------------------------------ #
    def add_object(self, category: str, class_name: str, name: Optional[str] = None, **recipe: Any) -> str:
        """Create a structure/source/monitor from its beamz class name +
        constructor kwargs (the canonical, unrotated recipe).
        """
        sid = uuid.uuid4().hex[:8]
        # `recipe` may ALREADY carry its own "name" entry — e.g. a
        # monitor's stored `so.recipe` (see the setdefault() a few lines
        # down) needs "name" baked in so `_regenerate()`'s `cls(**so.recipe)`
        # reproduces a monitor with a name at all, and restore()/
        # duplicate()/paste all reconstruct FROM that stored recipe while
        # ALSO passing an explicit `name=` for the (possibly different,
        # e.g. "_copy"-suffixed) new name. Without this, those two collide
        # — Python raises "got multiple values for keyword argument
        # 'name'" — which is exactly what undo/redo surfaced (restore()
        # is the one call site that always passes both). The explicit
        # `name` parameter is authoritative; strip any duplicate out of
        # the raw recipe before it can conflict.
        recipe.pop("name", None)
        name = name or f"{class_name}_{sid}"

        # If the class has its own `name` constructor field (monitors do
        # — needed later to look results up via
        # `SimulationResults.monitor(name)`), keep it in sync with the
        # GUI's own name automatically, so the two can't silently drift
        # apart without the user having to think about it. Skipped for
        # the region (REGION_KIND isn't a real beamz class at all).
        if class_name != REGION_KIND:
            cls = getattr(bz, class_name)
            if dataclasses.is_dataclass(cls) and any(
                f.name == "name" and f.init for f in dataclasses.fields(cls)
            ):
                recipe.setdefault("name", name)

        so = SceneObject(id=sid, category=category, class_name=class_name, recipe=dict(recipe), name=name)
        self._regenerate(so)
        self._structures[sid] = so
        self._order.append(sid)
        self.namespace[name] = so.obj
        self._name_to_sid[name] = sid

        self.structure_added.emit(sid)
        self.design_changed.emit()
        return sid

    # Thin convenience wrappers — kept because they read better at call
    # sites than `add_object("structure", ...)` everywhere.
    def add_structure(self, class_name: str, name: Optional[str] = None, **recipe: Any) -> str:
        return self.add_object("structure", class_name, name, **recipe)

    def add_source(self, class_name: str, name: Optional[str] = None, **recipe: Any) -> str:
        return self.add_object("source", class_name, name, **recipe)

    def add_monitor(self, class_name: str, name: Optional[str] = None, **recipe: Any) -> str:
        return self.add_object("monitor", class_name, name, **recipe)

    def remove_structure(self, sid: str) -> None:
        so = self._structures.pop(sid, None)
        if so is None:
            return
        self._order.remove(sid)
        self.namespace.pop(so.name, None)
        self._name_to_sid.pop(so.name, None)

        self.structure_removed.emit(sid)
        self.design_changed.emit()

    def set_enabled(self, sid: str, enabled: bool) -> None:
        """Suppress/restore a structure/source/monitor from a run WITHOUT
        deleting it (SceneObject.enabled — see its docstring; already
        respected by build_design/build_sources/build_monitors, just
        nothing in the UI could set it until now). A no-op, on-purpose,
        for the simulation region: it doesn't participate in
        build_design/sources/monitors as an object in its own right, so
        "disabling" it wouldn't mean anything.
        """
        so = self._structures[sid]
        if so.category == "region" or so.enabled == enabled:
            return
        so.enabled = enabled
        self.structure_changed.emit(sid)
        self.design_changed.emit()

    def update_param(self, sid: str, **changes: Any) -> None:
        """Update one or more recipe fields and regenerate. This is the
        one place ordinary (non-rotation) property edits flow through,
        whether they came from the auto-generated property editor or
        elsewhere in the GUI.

        Transactional: builds a CANDIDATE recipe and only commits it (to
        `so.recipe` and `so.obj`) if `cls(**candidate)` actually succeeds.
        An earlier version updated `so.recipe` first and regenerated
        second — if regeneration raised (e.g. beamz's own validation
        rejecting a negative width with `ValueError`), the bad value was
        already sitting in `so.recipe`, so `so.obj` stayed at its last
        good value but every SUBSEQUENT edit — even to a totally
        unrelated field — kept reconstructing from that same poisoned
        recipe and kept failing the same way. From the GUI that reads as
        a permanent freeze: the structure becomes uneditable through any
        field until deleted and recreated. Confirmed by reproducing it
        directly against the connector before fixing.
        """
        so = self._structures[sid]
        candidate_recipe = dict(so.recipe)
        candidate_recipe.update(changes)

        if so.category == "region":
            so.recipe = candidate_recipe
            self._regenerate(so)
            self.structure_changed.emit(sid)
            self.design_changed.emit()
            return

        cls = getattr(bz, so.class_name)
        candidate_obj = cls(**candidate_recipe)  # raises on invalid values; so.recipe untouched if so
        if so.category == "structure" and so.rotation and hasattr(candidate_obj, "rotate"):
            candidate_obj = candidate_obj.rotate(so.rotation, axis=so.rotation_axis)

        so.recipe = candidate_recipe
        so.obj = candidate_obj
        self.namespace[so.name] = candidate_obj

        self.structure_changed.emit(sid)
        self.design_changed.emit()

    def update_rotation(self, sid: str, angle_deg: float, axis: Optional[str] = None) -> None:
        """Set rotation angle (DEGREES — matches beamz's own `.rotate()`
        convention, confirmed by reading its source: it does
        `np.radians(float(angle))` internally, i.e. it expects degrees in.
        An earlier version of this method took radians and passed them
        straight through, silently double-converting and making every
        rotation wrong by a factor of ~57. Fixed by standardizing on
        degrees everywhere rotation is stored or displayed — including for
        classes like `CircularBend` that have their own native `rotation`
        field (its sibling `angle` field defaults to 90.0, which only
        makes sense as degrees, so the same convention is used there too;
        see `has_native_rotation_field`).

        Also transactional, same reasoning as `update_param` above: tries
        the regenerate first, only commits `so.rotation`/`so.obj` if it
        actually succeeds.
        """
        so = self._structures[sid]
        new_axis = axis if axis is not None else so.rotation_axis

        cls = getattr(bz, so.class_name)
        candidate_obj = cls(**so.recipe)
        if angle_deg and hasattr(candidate_obj, "rotate"):
            candidate_obj = candidate_obj.rotate(angle_deg, axis=new_axis)

        so.rotation = angle_deg
        so.rotation_axis = new_axis
        so.obj = candidate_obj
        self.namespace[so.name] = candidate_obj

        self.structure_changed.emit(sid)
        self.design_changed.emit()

    def rename(self, sid: str, new_name: str) -> None:
        so = self._structures[sid]
        self.namespace.pop(so.name, None)
        self._name_to_sid.pop(so.name, None)
        so.name = new_name
        if "name" in so.recipe:  # keep beamz's own `name` field in sync too
            so.recipe["name"] = new_name
            self._regenerate(so)
        self.namespace[new_name] = so.obj
        self._name_to_sid[new_name] = sid
        self.structure_changed.emit(sid)

    def get(self, sid: str) -> SceneObject:
        return self._structures[sid]

    def has(self, sid: str) -> bool:
        return sid in self._structures

    def name_exists(self, name: str) -> bool:
        """Whether `name` is already in use by some OTHER tracked object —
        used by the Rename UI to reject a name collision before it reaches
        `rename()` (which does not check this itself: a script-driven
        rebind onto an existing name is a legitimate, if unusual, thing to
        do, and sync_from_script relies on rename never refusing on its
        own). A GUI-initiated rename should not be allowed to silently
        make two SceneObjects share one namespace key.
        """
        return name in self._name_to_sid

    def all_structures(self) -> list[SceneObject]:
        """All placeable objects (despite the name, kept for backwards
        compatibility) in insertion order. Use `by_category` to filter.
        """
        return [self._structures[sid] for sid in self._order]

    def by_category(self, category: str, enabled_only: bool = False) -> list[SceneObject]:
        items = [so for so in self.all_structures() if so.category == category]
        if enabled_only:
            items = [so for so in items if so.enabled]
        return items

    def add_region(self, name: Optional[str] = None, **recipe: Any) -> str:
        """The FDTD simulation region — a GUI-only concept, not a real
        beamz class (see REGION_KIND). Recipe holds x/y/z/x_span/y_span/
        z_span (plain center+span, no corner conversion needed — this is
        our own concept, always center-based), `boundaries` (dict face ->
        "PML"/"PEC"/"Absorber"), and `settings` (kwargs forwarded to
        Simulation.__init__, e.g. resolution/run_time/polarization).
        """
        defaults = dict(
            x=0.0, y=0.0, z=0.0, x_span=10e-6, y_span=10e-6, z_span=1e-6,
            boundaries={"x-": "PML", "x+": "PML", "y-": "PML", "y+": "PML", "z-": "PML", "z+": "PML"},
            settings={},
        )
        defaults.update(recipe)
        return self.add_object("region", REGION_KIND, name, **defaults)

    def _unique_name(self, base: str) -> str:
        if base not in self._name_to_sid:
            return base
        i = 2
        while f"{base}{i}" in self._name_to_sid:
            i += 1
        return f"{base}{i}"

    def duplicate(self, sid: str) -> str:
        """A copy of the structure/source/monitor at `sid`, nudged by a
        small, visible offset in x/y so it doesn't land exactly on top of
        the original. Regions aren't duplicable — there's only ever one
        meaningful simulation region.
        """
        so = self._structures[sid]
        if so.category == "region":
            raise ValueError("The simulation region can't be duplicated.")

        new_recipe = dict(so.recipe)
        adapter = ga.get_adapter(so.class_name)
        if adapter is not None:
            # Reuses the exact same corner/center-aware math the
            # Property Editor's own x/y edits go through (geometry_
            # adapters.xyz_changes), so the nudge behaves consistently
            # whether this class's position field is a stored corner
            # (Rectangle) or already a center (Circle/Ring/...) — no
            # separate "duplicate offset" logic to keep in sync with that.
            x, y, _z = ga.get_xyz(new_recipe, adapter)
            offset = 0.5e-6
            new_recipe.update(ga.xyz_changes(new_recipe, adapter, x=x + offset, y=y + offset))

        new_name = self._unique_name(f"{so.name}_copy")
        new_sid = self.add_object(so.category, so.class_name, name=new_name, **new_recipe)
        if so.rotation:
            self.update_rotation(new_sid, so.rotation, axis=so.rotation_axis)
        return new_sid

    def export_entries(self, sids: list[str]) -> list[dict[str, Any]]:
        """Same per-object shape `snapshot()` uses (recipe/rotation/etc,
        NOT JSON-safe — see snapshot()'s docstring), but only for the
        given ids, and silently skipping the simulation region (there's
        only ever one; copying it never makes sense). Used by Copy —
        `workspace.py`'s `entries_to_clipboard_text()` wraps this with
        the actual JSON conversion, same layering as snapshot()/
        `_serialize_model()` for full-workspace save.
        """
        out = []
        for sid in sids:
            so = self._structures.get(sid)
            if so is None or so.category == "region":
                continue
            out.append(
                dict(
                    category=so.category,
                    class_name=so.class_name,
                    name=so.name,
                    recipe=dict(so.recipe),
                    rotation=so.rotation,
                    rotation_axis=so.rotation_axis,
                )
            )
        return out

    def import_entries(self, entries: list[dict[str, Any]]) -> list[str]:
        """Create fresh objects from `export_entries()`-shaped entries —
        used by Paste. Each gets a name guaranteed not to collide with
        anything already tracked (via `_unique_name`, same as
        `duplicate()`) rather than the original name verbatim, since the
        whole point of Paste is landing a copy ALONGSIDE existing objects
        (including, for copy/paste BETWEEN two separate running windows,
        one that may already have an object with that exact name) — and
        nudged by the same small, visible x/y offset `duplicate()` uses,
        for the same reason: so it doesn't land exactly on top of
        whatever's already there.
        """
        new_sids = []
        for entry in entries:
            new_recipe = dict(entry["recipe"])
            adapter = ga.get_adapter(entry["class_name"])
            if adapter is not None:
                x, y, _z = ga.get_xyz(new_recipe, adapter)
                offset = 0.5e-6
                new_recipe.update(ga.xyz_changes(new_recipe, adapter, x=x + offset, y=y + offset))

            new_name = self._unique_name(f"{entry['name']}_copy")
            new_sid = self.add_object(entry["category"], entry["class_name"], name=new_name, **new_recipe)
            if entry.get("rotation"):
                self.update_rotation(new_sid, entry["rotation"], axis=entry.get("rotation_axis", "z"))
            new_sids.append(new_sid)
        return new_sids

    def add_from_object(self, category: str, obj: Any, name: Optional[str] = None) -> str:
        """Track an ALREADY-CONSTRUCTED beamz instance (e.g. one produced
        by an external importer like GDS import — see workspace.py-style
        importers in main_window._import_gds) as a new SceneObject,
        capturing its current field values as the canonical recipe — the
        same technique sync_from_script() uses for script-created objects
        (see _recipe_from_instance). `add_object()` then reconstructs an
        equivalent instance from that captured recipe (`cls(**recipe)`),
        same as it does for anything else — nothing here bypasses the
        normal construction path, `obj` itself is only ever read from,
        never stored directly.
        """
        return self.add_object(category, type(obj).__name__, name, **_recipe_from_instance(obj))

    def snapshot(self) -> list[dict[str, Any]]:
        """A lightweight, in-memory snapshot of every placeable's state —
        used by UndoManager (undo.py) for undo/redo. NOT JSON-safe
        (recipe values may be numpy arrays or nested beamz objects like
        Material/GaussianPulse, kept as live Python objects as-is) — for
        a JSON-safe version suitable for writing to a file, see
        workspace.py's own export, which serializes each entry's recipe
        through a to-jsonable pass first.
        """
        return [
            dict(
                category=so.category,
                class_name=so.class_name,
                name=so.name,
                recipe=dict(so.recipe),
                rotation=so.rotation,
                rotation_axis=so.rotation_axis,
                enabled=so.enabled,
            )
            for so in self.all_structures()
        ]

    def restore(self, snapshot: list[dict[str, Any]]) -> None:
        """Bring the scene to match `snapshot` (from an earlier
        `snapshot()` call, or a JSON-decoded equivalent from
        workspace.py) — as a DIFF against the CURRENT scene, matched by
        name, not a blanket remove-everything-then-recreate-everything.
        An object whose name/recipe/rotation/enabled are all already
        exactly what the snapshot says keeps its sid and gets touched
        NOT AT ALL (no signal fires for it); one that differs gets
        updated in place via the normal update_param/update_rotation/
        set_enabled (same sid — it stays the same tracked object, just
        with new content); one absent from the snapshot gets removed;
        one present in the snapshot but not currently tracked gets added
        fresh. Matching is by NAME since a snapshot never records sids —
        so undoing a RENAME specifically still shows as a remove+re-add
        (there is no earlier sid recorded to revert a name onto), but
        every OTHER object in the scene, unaffected by whatever's being
        undone, is genuinely left alone.

        This isn't just an efficiency nicety: an earlier version's
        unconditional remove-everything-then-recreate-everything meant
        undoing ONE change also tore down and rebuilt every OTHER,
        completely unrelated object in the scene as a side effect —
        confirmed as the source of a specific reported bug where
        pressing Undo right after adding a structure appeared to do
        nothing (a visible delete-then-recreate of that exact structure,
        completely unchanged) whenever the scene had other content too:
        the unconditional approach can't tell "already correct, leave it"
        from "needs to change", so it always did the latter to
        everything, and whatever the ACTUAL undo target was could be
        buried in a wall of identical-looking del/re-add noise for
        objects that never needed touching at all.
        """
        target_by_name = {entry["name"]: entry for entry in snapshot}
        current_by_name = {so.name: so for so in self.all_structures()}

        for name, so in list(current_by_name.items()):
            if name not in target_by_name:
                self.remove_structure(so.id)

        for entry in snapshot:
            current = current_by_name.get(entry["name"])
            if current is not None and current.class_name != entry["class_name"]:
                # Same name, but a fundamentally different KIND of object
                # between the two states (e.g. a script rebind that
                # pointed a name at something else entirely) — can't
                # update a Rectangle "in place" into a Circle, so this
                # one case still falls back to remove-old + add-new
                # (a fresh sid), same as if the name simply hadn't
                # existed in the current scene at all.
                self.remove_structure(current.id)
                current = None

            if current is None:
                if entry["category"] == "region":
                    sid = self.add_region(name=entry["name"], **entry["recipe"])
                else:
                    sid = self.add_object(
                        entry["category"], entry["class_name"], name=entry["name"], **entry["recipe"]
                    )
                if entry.get("rotation"):
                    self.update_rotation(sid, entry["rotation"], axis=entry.get("rotation_axis", "z"))
                self._structures[sid].enabled = entry.get("enabled", True)
                continue

            so = current
            target_rotation = entry.get("rotation", 0.0)
            target_axis = entry.get("rotation_axis", "z")
            target_enabled = entry.get("enabled", True)

            if not _recipe_equal(so.recipe, entry["recipe"]):
                self.update_param(so.id, **entry["recipe"])
            if so.rotation != target_rotation or so.rotation_axis != target_axis:
                self.update_rotation(so.id, target_rotation, axis=target_axis)
            if so.enabled != target_enabled:
                self.set_enabled(so.id, target_enabled)

    def _regenerate(self, so: SceneObject) -> None:
        """The one place a beamz instance actually gets constructed from
        a SceneObject's recipe. Called after any recipe or rotation edit.
        Regions have no real beamz instance (see REGION_KIND) — nothing
        to construct.
        """
        if so.category == "region":
            so.obj = None
            self.namespace[so.name] = so.recipe
            return
        cls = getattr(bz, so.class_name)
        obj = cls(**so.recipe)
        if so.category == "structure" and so.rotation and hasattr(obj, "rotate"):
            obj = obj.rotate(so.rotation, axis=so.rotation_axis)
        so.obj = obj
        self.namespace[so.name] = obj

    # ------------------------------------------------------------------ #
    # Script Console front door
    # ------------------------------------------------------------------ #
    def sync_from_script(self) -> list[str]:
        """Reconcile tracked SceneObjects with whatever the last script
        command did to `self.namespace`. Three cases, since every
        placeable is frozen (a script edits by rebinding a name, never by
        mutation):

          1. A namespace name we track now points to a *different* object
             -> treat as an edit (structure_changed). Its recipe is
             re-captured from the new instance's current field values, and
             rotation resets to 0 — if the script itself called
             `.rotate()`, that's already baked into the new instance's
             vertices, so there is nothing left for our separate rotation
             tracking to represent.
          2. A namespace name holds a structure/source/monitor-kind object
             we don't track yet -> treat as script-created
             (structure_added).
          3. A name we track is gone from the namespace, or no longer
             holds a placeable -> treat as script-deleted (remove it).

        Returns the sids that changed, so the console can report on it.
        """
        affected: list[str] = []

        for name, value in list(self.namespace.items()):
            if name == "bz" or name.startswith("_"):
                continue
            category = classify_category(value)
            if category is None:
                continue

            existing_sid = self._name_to_sid.get(name)
            if existing_sid is None:
                sid = uuid.uuid4().hex[:8]
                so = SceneObject(
                    id=sid,
                    category=category,
                    class_name=type(value).__name__,
                    recipe=_recipe_from_instance(value),
                    name=name,
                    obj=value,
                )
                self._structures[sid] = so
                self._order.append(sid)
                self._name_to_sid[name] = sid
                self.structure_added.emit(sid)
                affected.append(sid)
            else:
                so = self._structures[existing_sid]
                if so.obj is not value:
                    so.category = category
                    so.class_name = type(value).__name__
                    so.recipe = _recipe_from_instance(value)
                    so.rotation = 0.0
                    so.obj = value
                    self.structure_changed.emit(existing_sid)
                    affected.append(existing_sid)

        # Case 3: a previously tracked name no longer refers to a placeable.
        # SimulationRegion (model.REGION_KIND) is a real, deliberate
        # exception: it's a GUI-only concept with no beamz class at all,
        # so its namespace value is a plain recipe dict, not something
        # classify_category recognizes — meaning without this guard,
        # EVERY call to sync_from_script() (i.e. every single console
        # command) treated an existing region as "no longer a placeable"
        # and silently deleted it. Confirmed directly: a region added
        # right before calling sync_from_script() (as import_script does)
        # vanished immediately. Regions are skipped here entirely; they
        # have no script-editing story of their own (there's no
        # `bz.SimulationRegion(...)` call to rebind), so there is nothing
        # for the script namespace to legitimately change out from under.
        for name, sid in list(self._name_to_sid.items()):
            so = self._structures.get(sid)
            if so is not None and so.category == "region":
                continue
            current = self.namespace.get(name)
            if current is None or classify_category(current) is None:
                self.remove_structure(sid)
                affected.append(sid)

        if affected:
            self.design_changed.emit()
        return affected

    # ------------------------------------------------------------------ #
    # Design / Simulation assembly
    # ------------------------------------------------------------------ #
    def build_design(self) -> bz.Design:
        """Assemble a live beamz.Design from current structures (sources
        and monitors are excluded — they belong to the Simulation, not the
        Design). Never cached, since it must always reflect current param
        values. Disabled structures (SceneObject.enabled == False) are
        excluded too — that's the entire point of the enabled toggle.
        """
        return bz.Design(
            width=self.design_width,
            height=self.design_height,
            depth=self.design_depth,
            background=self.background_material,
            structures=tuple(so.obj for so in self.by_category("structure", enabled_only=True)),
        )

    def build_sources(self) -> tuple[Any, ...]:
        return tuple(so.obj for so in self.by_category("source", enabled_only=True))

    def build_monitors(self) -> tuple[Any, ...]:
        return tuple(so.obj for so in self.by_category("monitor", enabled_only=True))