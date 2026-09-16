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
from PySide6.QtCore import QObject, Signal

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

    # ------------------------------------------------------------------ #
    # Placeable-object CRUD (GUI-driven front door)
    # ------------------------------------------------------------------ #
    def add_object(self, category: str, class_name: str, name: Optional[str] = None, **recipe: Any) -> str:
        """Create a structure/source/monitor from its beamz class name +
        constructor kwargs (the canonical, unrotated recipe).
        """
        sid = uuid.uuid4().hex[:8]
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

    def all_structures(self) -> list[SceneObject]:
        """All placeable objects (despite the name, kept for backwards
        compatibility) in insertion order. Use `by_category` to filter.
        """
        return [self._structures[sid] for sid in self._order]

    def by_category(self, category: str) -> list[SceneObject]:
        return [so for so in self.all_structures() if so.category == category]

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
        values.
        """
        return bz.Design(
            width=self.design_width,
            height=self.design_height,
            depth=self.design_depth,
            background=self.background_material,
            structures=tuple(so.obj for so in self.by_category("structure")),
        )

    def build_sources(self) -> tuple[Any, ...]:
        return tuple(so.obj for so in self.by_category("source"))

    def build_monitors(self) -> tuple[Any, ...]:
        return tuple(so.obj for so in self.by_category("monitor"))