from __future__ import annotations

from typing import Any, Optional

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QWidget,
)

import inspect

import beamz as bz

from .. import geometry_adapters as ga
from .. import introspection as intro
from .. import materials
from ..model import BeamzConnector, has_native_rotation_field, _recipe_from_instance
from . import dynamic_fields as df
from .spinbox import SmartDoubleSpinBox

# beamz geometry is specified in metres internally (e.g. 1.5e-6 for 1.5
# microns) — that never changes, staged/committed values throughout this
# file stay in metres. Only the canonical geometry block's DISPLAY (what
# you see and type into x/y/z/width/radius/... boxes) is now in microns,
# with the unit spelled out in each row's label, converting at the
# widget boundary only. Kept scoped to just the geometry block (not
# Additional Properties, where a "float" field might be an angle, a
# power, or an index rather than a length, and blindly converting
# everything to microns would be wrong) per an explicit request to keep
# this a minimal, contained change.
GEOM_UNIT = 1e-6
GEOM_UNIT_LABEL = "\u00b5m"
GEOM_DECIMALS = 6  # 1e-6 um = 1 pm resolution, plenty for any photonic use
GEOM_STEP = 0.001  # 1 nm per spin-arrow click
GEOM_RANGE = 1e5  # +/- 100 mm in microns — far beyond any realistic device

AXIS_LABELS = ("x", "y", "z")

# Direction dropdown choices and nested-object type hints used to be
# defined here; both now live in dynamic_fields.py (CHOICE_FIELDS /
# NESTED_FIELD_TYPES) since NestedObjectDialog needs them too — kept in
# one place rather than duplicated across both.


class PropertyEditor(QWidget):
    """Right-hand dock. Every selection gets the same top block regardless
    of its underlying beamz class — x, y, z, x span, y span, z span,
    material, rotation — via `geometry_adapters`. Fields the adapter

    doesn't cover appear below under "Additional Properties".

    STAGED EDITS, NOT LIVE: every widget edits a local working copy
    (`self._working_recipe` / `self._working_rotation`) — nothing reaches
    the connector until "Apply" is clicked. An earlier version committed
    on every keystroke, which had two real problems: (1) beamz validates
    eagerly (e.g. rejects a negative width), and a bad value entered by
    mistake — or even just transiently, while retyping a number — could
    reach the connector; (2) that surfaced as an uncaught exception deep
    in a Qt signal handler, which read as the GUI freezing/crashing rather
    than a clean error message. Apply wraps the actual commit in
    try/except and shows failures inline without losing the user's
    in-progress edits; nothing about beamz's own validation strictness
    changes, only when and how a rejection is surfaced.
    """

    def __init__(self, connector: BeamzConnector, parent=None) -> None:
        super().__init__(parent)
        self.connector = connector
        self.sid: Optional[str] = None
        self._rendered_class_name: Optional[str] = None

        self._form = QFormLayout()
        self._form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self.setLayout(self._form)

        self._widgets: dict[str, QWidget] = {}  # scalar "Additional Properties" fields
        self._tuple_boxes: dict[str, list[QDoubleSpinBox]] = {}  # tuple "Additional Properties" fields
        self._material_combo: Optional[QComboBox] = None
        self._material_custom_box: Optional[QDoubleSpinBox] = None
        self._rotation_angle_box: Optional[QDoubleSpinBox] = None
        self._rotation_axis_combo: Optional[QComboBox] = None
        self._geom_boxes: dict[str, QDoubleSpinBox] = {}  # "x"/"y"/"z"/"x span"/"y span"/"z span"
        self._adapter: Optional[ga.GeometryAdapter] = None
        self._error_label: Optional[QLabel] = None
        self._apply_button: Optional[QPushButton] = None

        # Working copy edited by every widget; only reaches the connector
        # when Apply is clicked. Reset from the connector's authoritative
        # state on every set_selection() call.
        self._working_recipe: dict[str, Any] = {}
        self._working_rotation: float = 0.0
        self._working_rotation_axis: str = "z"

        self._updating = False  # guards against feedback loops while rebuilding

        connector.structure_changed.connect(self._on_external_change)

        self.set_selection(None)

    # ------------------------------------------------------------------ #
    def set_selection(self, sid: Optional[str]) -> None:
        self.sid = sid
        self._clear_form()

        if sid is None:
            self._form.addRow(QLabel("No selection."))
            self._rendered_class_name = None
            return

        so = self.connector.get(sid)
        self._rendered_class_name = so.class_name

        if so.category == "region":
            self._working_recipe = dict(so.recipe)  # plain, always-concrete — no resolve needed
            self._render_region(so)
            return
        # Resolve so.recipe into a FULLY CONCRETE recipe — every field
        # given an actual value, not just the ones explicitly set —
        # before using it as the working copy. This matters because
        # so.recipe can be genuinely sparse (e.g. `{}` when a structure
        # was created via defaults.build_default_recipe() trusting the
        # class's own no-arg defaults) and geometry_adapters reads
        # straight from whatever dict it's given via plain `.get(field,
        # 0.0)` — so a sparse recipe silently displayed 0 for any field
        # not explicitly present, even when the REAL object (built via
        # `cls(**recipe)`, which correctly fills in class defaults) had a
        # nonzero value there. Confirmed directly: an empty recipe made
        # get_xyz/get_span report an all-zero 0x0x0 Rectangle in the UI
        # while the actual constructed object was a full 1x1x1 METRE
        # cube — an oversized structure a user would have no way to
        # notice from the property panel, and exactly the kind of thing
        # that silently overlaps a simulation's PML region.
        resolved_instance = getattr(bz, so.class_name)(**so.recipe)
        self._working_recipe = _recipe_from_instance(resolved_instance)
        self._working_rotation = so.rotation
        self._working_rotation_axis = so.rotation_axis
        self._adapter = ga.get_adapter(so.class_name)

        header = QLabel(f"<b>{so.class_name}</b> — {so.name}  <span style='color:#888'>({so.category})</span>")
        self._form.addRow(header)

        self._error_label = QLabel("")
        self._error_label.setStyleSheet("color: #c0392b;")
        self._error_label.setWordWrap(True)
        self._error_label.setVisible(False)
        self._form.addRow(self._error_label)

        consumed = self._add_geometry_block(so.class_name)

        material_field = self._find_material_field(so.class_name)
        if material_field is not None:
            self._add_material_row(material_field)
            consumed.add(material_field.name)

        if so.category == "structure":
            self._add_rotation_rows(so)
            if has_native_rotation_field(so.class_name):
                consumed.add("rotation")

        self._add_additional_properties(so.class_name, consumed)
        self._add_apply_bar()

    def _on_external_change(self, sid: str) -> None:
        # Something else (script console, or a future collaborator/undo
        # system) changed the object we're showing. Rebuilding discards
        # any not-yet-applied local edits in favor of the new
        # authoritative state — a reasonable simplification for v1, but
        # worth knowing about: if you have pending edits open in this
        # panel and a script touches the same object, your pending edits
        # are lost rather than merged.
        if sid != self.sid or self._updating:
            return
        self.set_selection(sid)

    def _clear_form(self) -> None:
        while self._form.rowCount():
            self._form.removeRow(0)
        self._widgets.clear()
        self._tuple_boxes.clear()
        self._geom_boxes.clear()
        self._material_combo = None
        self._material_custom_box = None
        self._rotation_angle_box = None
        self._rotation_axis_combo = None
        self._error_label = None
        self._apply_button = None

    # ------------------------------------------------------------------ #
    # Canonical geometry block: x / y / z / x span / y span / z span.
    # Reads/writes `self._working_recipe`, not the connector — see class
    # docstring. `geometry_adapters` handles the corner-vs-center
    # conversion (Rectangle's `position` is a corner, not a center — an
    # earlier version treated every class's position as already-centered,
    # which silently moved a structure's center every time a span was
    # edited; see geometry_adapters.py's module docstring for the audit).
    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    # Simulation Region — a GUI-only concept (model.REGION_KIND), not a
    # real beamz class, so it bypasses geometry_adapters/introspection
    # entirely and gets its own small, dedicated rendering path: plain
    # center+span geometry (always center-based — no corner conversion
    # needed, this is our own concept), per-face boundary condition
    # dropdowns, and Simulation's own settings introspected DYNAMICALLY
    # from its constructor signature (so a future beamz version adding/
    # removing a simple settings kwarg picks it up automatically without
    # code changes here — the actual point of doing it this way instead
    # of hand-listing resolution/run_time/etc.).
    # ------------------------------------------------------------------ #
    BOUNDARY_TYPES = ("PML", "PEC", "Absorber")
    BOUNDARY_FACES = ("x-", "x+", "y-", "y+", "z-", "z+")
    # Structural Simulation kwargs assembled by simulation_runner.py
    # itself (design/sources/monitors/boundaries/domain from the
    # connector's scene) or too complex for a simple settings row
    # (grid/raster_options/material_grid/scene/grid_spec/time) — excluded
    # from the dynamically-built settings block, everything else in
    # Simulation.__init__'s signature is shown.
    _SIM_STRUCTURAL_PARAMS = {
        "design", "sources", "monitors", "boundaries", "domain", "size", "background",
        "material_grid", "scene", "raster_grid", "grid_spec", "time", "grid", "raster_options",
    }

    def _render_region(self, so) -> None:
        header = QLabel(f"<b>FDTD Simulation Region</b> — {so.name}")
        self._form.addRow(header)
        self._error_label = QLabel("")
        self._error_label.setStyleSheet("color: #c0392b;")
        self._error_label.setWordWrap(True)
        self._error_label.setVisible(False)
        self._form.addRow(self._error_label)

        for label in ("x", "y", "z", "x_span", "y_span", "z_span"):
            value = self._working_recipe.get(label, 0.0)
            self._add_geom_row(label.replace("_", " "), value, lambda v, n=label: self._stage_region_geom(n, v))

        self._form.addRow(QFrame(frameShape=QFrame.Shape.HLine))
        self._form.addRow(QLabel("<b>Boundary Conditions</b>"))
        boundaries = self._working_recipe.get("boundaries", {})
        for face in self.BOUNDARY_FACES:
            combo = QComboBox()
            combo.addItems(list(self.BOUNDARY_TYPES))
            combo.setCurrentText(boundaries.get(face, "PML"))
            combo.currentTextChanged.connect(lambda v, f=face: self._stage_region_boundary(f, v))
            self._form.addRow(f"boundary {face}", combo)

        self._form.addRow(QFrame(frameShape=QFrame.Shape.HLine))
        self._form.addRow(QLabel("<b>Solver Settings</b>"))
        settings = self._working_recipe.get("settings", {})
        sig = inspect.signature(bz.Simulation.__init__)
        for name, p in sig.parameters.items():
            if name == "self" or name in self._SIM_STRUCTURAL_PARAMS:
                continue
            if p.default is inspect.Parameter.empty:
                continue
            current = settings.get(name, p.default)
            widget = self._build_setting_widget(name, current)
            self._form.addRow(name, widget)

        self._add_apply_bar()

    def _build_setting_widget(self, name: str, current: Any) -> QWidget:
        if isinstance(current, bool):
            w = QCheckBox()
            w.setChecked(current)
            w.toggled.connect(lambda v, n=name: self._stage_region_setting(n, v))
            return w
        if isinstance(current, (int, float)):
            w = SmartDoubleSpinBox()
            w.setRange(-1e20, 1e20)
            w.setDecimals(15)
            w.setValue(float(current))
            w.valueChanged.connect(lambda v, n=name: self._stage_region_setting(n, v))
            return w
        if isinstance(current, str):
            w = QComboBox()
            w.setEditable(True)
            w.addItem(current)
            w.currentTextChanged.connect(lambda v, n=name: self._stage_region_setting(n, v))
            return w
        # None or something complex — a blank optional override, staged
        # as a string the runner can interpret (or leave at None if
        # blank), rather than pretending to know its true type.
        w = QLineEdit("" if current is None else str(current))
        w.setPlaceholderText("(auto)")
        w.editingFinished.connect(lambda n=name, w=w: self._stage_region_setting(n, w.text() or None))
        return w

    def _stage_region_geom(self, field: str, value: float) -> None:
        if self._updating:
            return
        self._working_recipe[field] = value
        self._mark_dirty()

    def _stage_region_boundary(self, face: str, boundary_type: str) -> None:
        if self._updating:
            return
        boundaries = dict(self._working_recipe.get("boundaries", {}))
        boundaries[face] = boundary_type
        self._working_recipe["boundaries"] = boundaries
        self._mark_dirty()

    def _stage_region_setting(self, name: str, value: Any) -> None:
        if self._updating:
            return
        settings = dict(self._working_recipe.get("settings", {}))
        settings[name] = value
        self._working_recipe["settings"] = settings
        self._mark_dirty()

    def _add_geometry_block(self, class_name: str) -> set[str]:
        adapter = ga.get_adapter(class_name)
        if adapter is None:
            return set()

        x, y, z = ga.get_xyz(self._working_recipe, adapter)
        self._add_geom_row("x", x, lambda v: self._stage_xyz(x=v))
        self._add_geom_row("y", y, lambda v: self._stage_xyz(y=v))
        self._add_geom_row("z", z, lambda v: self._stage_xyz(z=v))

        consumed = {adapter.position_field}
        if adapter.z_field:
            consumed.add(adapter.z_field)

        # Render each DISTINCT underlying size field once, labeled by its
        # REAL beamz name — "width"/"height"/"depth" for a Rectangle,
        # "radius" for a Circle, "outer_radius" for a Ring — rather than
        # a generic "x span"/"y span"/"z span". Two reasons: (1) it's what
        # was actually asked for ("add correct attributes like inner
        # radius outer radius radius... display all attributes
        # dynamically") rather than translating them into span-speak, and
        # (2) when two axes are literally the same field (Circle's radius
        # drives both x and y), showing "x span" AND "y span" as two rows
        # was redundant and confusing — now it's one "radius" row.
        # `inner_radius` isn't part of this block at all (Ring has no
        # axis mapped to it) — it flows through to Additional Properties
        # below via the normal dynamic-field listing, same as any other
        # field an adapter doesn't claim.
        # Render each DISTINCT underlying size field once, labeled by its
        # REAL beamz name — "width"/"height"/"depth" for a Rectangle,
        # "radius" for a Circle, "outer_radius" for a Ring — rather than
        # a generic "x span"/"y span"/"z span". When two axes are
        # literally the same SCALAR field (Circle's radius drives both x
        # and y), they collapse into one row.
        #
        # Grouping key is `span.field` itself (name, OR (name, index) for
        # a tuple field), NOT just the field's name — an earlier version
        # grouped by name alone, which was correct for Circle's `radius`
        # but WRONG for a tuple field: GaussianBeamSource/FieldMonitor's
        # x_span/y_span/z_span are `size[0]`/`size[1]`/`size[2]` — three
        # DIFFERENT components of the same-NAMED field `size`. Keying by
        # name alone collapsed all three into one "size" row that only
        # ever read/wrote index 0, leaving y/z span permanently stuck at
        # 0 and uneditable — confirmed by reproducing exactly this against
        # a real GaussianBeamSource.
        groups: dict[Any, list[str]] = {}
        for axis, span in (("x", adapter.x_span), ("y", adapter.y_span), ("z", adapter.z_span)):
            if span is None:
                continue
            groups.setdefault(span.field, []).append(axis)
            field_name = span.field[0] if isinstance(span.field, tuple) else span.field
            consumed.add(field_name)

        for field_key, axes in groups.items():
            span = getattr(adapter, f"{axes[0]}_span")
            value = ga.get_span(self._working_recipe, span)
            if isinstance(field_key, tuple):
                # A specific component of a vector field (e.g. `size[1]`)
                # — always suffix with the axis so distinct components of
                # the SAME-named tuple field get distinct, unambiguous
                # rows/labels instead of colliding (this is what was
                # broken before: three components, one shared label).
                label = f"{field_key[0]}.{axes[0]}"
            else:
                label = field_key
            self._add_geom_row(label, value, lambda v, axs=axes: self._stage_span_group(axs, v))

        return consumed

    def _add_geom_row(self, label: str, value: float, on_change) -> None:
        box = SmartDoubleSpinBox()
        box.setRange(-GEOM_RANGE, GEOM_RANGE)
        box.setDecimals(GEOM_DECIMALS)
        box.setSingleStep(GEOM_STEP)
        box.setValue(value / GEOM_UNIT)  # metres (internal) -> microns (displayed)
        box.valueChanged.connect(lambda v: on_change(v * GEOM_UNIT))  # microns (typed) -> metres (staged)
        self._form.addRow(f"{label} ({GEOM_UNIT_LABEL})", box)
        self._geom_boxes[label] = box

    def _stage_xyz(self, **coords: float) -> None:
        if self._updating or self._adapter is None:
            return
        self._working_recipe.update(ga.xyz_changes(self._working_recipe, self._adapter, **coords))
        self._refresh_geometry_boxes()
        self._mark_dirty()

    def _stage_span_group(self, axes: list[str], value: float) -> None:
        if self._updating or self._adapter is None:
            return
        # A field can drive more than one axis at once (Circle's `radius`
        # is both x_span and y_span). Apply the corner-compensation pass
        # for EVERY axis that field drives — for our current classes only
        # one of a group is ever corner-based at most, but this stays
        # correct even if a future class couples a corner axis with a
        # centered one under the same field.
        for axis in axes:
            self._working_recipe.update(ga.span_changes(self._working_recipe, self._adapter, axis, value))
        self._refresh_geometry_boxes()
        self._mark_dirty()

    def _refresh_geometry_boxes(self) -> None:
        if self._adapter is None:
            return
        x, y, z = ga.get_xyz(self._working_recipe, self._adapter)
        values = {"x": x, "y": y, "z": z}

        # Must use the exact same grouping/labeling as _add_geometry_block
        # (group by `span.field` identity, suffix tuple-component labels
        # with their axis) or this silently updates the wrong box — which
        # is exactly the bug being fixed here in the first place.
        groups: dict[Any, list[str]] = {}
        for axis, span in (("x", self._adapter.x_span), ("y", self._adapter.y_span), ("z", self._adapter.z_span)):
            if span is None:
                continue
            groups.setdefault(span.field, []).append(axis)
        for field_key, axes in groups.items():
            span = getattr(self._adapter, f"{axes[0]}_span")
            label = f"{field_key[0]}.{axes[0]}" if isinstance(field_key, tuple) else field_key
            values[label] = ga.get_span(self._working_recipe, span)

        self._updating = True
        try:
            for label, value in values.items():
                box = self._geom_boxes.get(label)
                if box is not None:
                    box.blockSignals(True)
                    box.setValue(value / GEOM_UNIT)  # metres (internal) -> microns (displayed)
                    box.blockSignals(False)
        finally:
            self._updating = False

    # ------------------------------------------------------------------ #
    # Material — dropdown of constant-index presets + a custom-index
    # fallback. See materials.py for why these are non-dispersive
    # placeholders rather than a real database.
    # ------------------------------------------------------------------ #
    def _find_material_field(self, class_name: str) -> Optional[intro.FieldSpec]:
        for spec in intro.introspect_fields(class_name, self._working_recipe):
            if spec.kind == "material":
                return spec
        return None

    def _add_material_row(self, spec: intro.FieldSpec) -> None:
        current_index = materials.index_from_material(spec.default)
        current_label = materials.label_for_index(current_index)

        combo = QComboBox()
        combo.addItems(list(materials.PRESETS.keys()) + [materials.CUSTOM_LABEL])
        combo.setCurrentText(current_label)

        custom_box = SmartDoubleSpinBox()
        custom_box.setRange(1.0, 20.0)
        custom_box.setDecimals(4)
        custom_box.setSingleStep(0.01)
        custom_box.setValue(current_index)
        custom_box.setEnabled(current_label == materials.CUSTOM_LABEL)

        combo.currentTextChanged.connect(lambda text: self._on_material_combo_changed(text, custom_box))
        custom_box.valueChanged.connect(lambda v: self._stage_material(v))

        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(combo)
        h.addWidget(custom_box)
        self._form.addRow(spec.name, row)

        self._material_combo = combo
        self._material_custom_box = custom_box

    def _on_material_combo_changed(self, label: str, custom_box: QDoubleSpinBox) -> None:
        if label == materials.CUSTOM_LABEL:
            custom_box.setEnabled(True)
            self._stage_material(custom_box.value())
        else:
            custom_box.setEnabled(False)
            n = materials.PRESETS[label]
            custom_box.blockSignals(True)
            custom_box.setValue(n)
            custom_box.blockSignals(False)
            self._stage_material(n)

    def _stage_material(self, index: float) -> None:
        if self._updating:
            return
        self._working_recipe["material"] = materials.material_from_index(index)
        self._mark_dirty()

    # ------------------------------------------------------------------ #
    # Rotation — see model.py: NOT a beamz field for most classes
    # (`.rotate()` converts the object to a Polygon rather than storing an
    # angle), so this normally stages into `self._working_rotation` /
    # `self._working_rotation_axis`, applied via `connector.update_rotation`
    # on Apply. A few classes (CircularBend) DO define their own
    # `rotation` field, in which case it's staged into
    # `self._working_recipe["rotation"]` instead — see
    # has_native_rotation_field. Both paths use DEGREES (confirmed by
    # reading beamz's `.rotate()` source: it does np.radians() internally,
    # i.e. expects degrees in).
    # ------------------------------------------------------------------ #
    def _add_rotation_rows(self, so) -> None:
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        self._form.addRow(divider)

        native = has_native_rotation_field(so.class_name)
        current_deg = self._working_recipe.get("rotation", 0.0) if native else self._working_rotation

        angle_box = SmartDoubleSpinBox()
        angle_box.setRange(-360.0, 360.0)
        angle_box.setDecimals(3)
        angle_box.setSuffix(" deg")
        angle_box.setValue(current_deg)
        angle_box.valueChanged.connect(lambda v: self._stage_rotation(native))

        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(angle_box)

        axis_combo = None
        if not native:
            axis_combo = QComboBox()
            axis_combo.addItems(AXIS_LABELS)
            axis_combo.setCurrentText(self._working_rotation_axis)
            axis_combo.currentTextChanged.connect(lambda: self._stage_rotation(native))
            h.addWidget(QLabel("about axis"))
            h.addWidget(axis_combo)

        self._form.addRow("rotation", row)
        self._rotation_angle_box = angle_box
        self._rotation_axis_combo = axis_combo

    def _stage_rotation(self, native: bool) -> None:
        if self._updating:
            return
        angle_deg = self._rotation_angle_box.value()
        if native:
            self._working_recipe["rotation"] = angle_deg
        else:
            self._working_rotation = angle_deg
            self._working_rotation_axis = self._rotation_axis_combo.currentText()
        self._mark_dirty()

    # ------------------------------------------------------------------ #
    # Additional Properties — every recipe field the geometry/material/
    # rotation blocks above didn't already claim. Widget-building itself
    # is delegated to dynamic_fields.build_field_widget, shared with
    # NestedObjectDialog — this is what makes arrays (freqs, signal) and
    # nested objects (source_time) genuinely editable rather than shown
    # as an opaque repr string, without duplicating that logic in two
    # places.
    # ------------------------------------------------------------------ #
    def _add_additional_properties(self, class_name: str, consumed: set[str]) -> None:
        specs = [s for s in intro.introspect_fields(class_name, self._working_recipe) if s.name not in consumed]
        if not specs:
            return

        self._form.addRow(QLabel("<i>Additional Properties</i>"))

        for spec in specs:
            widget = df.build_field_widget(
                class_name,
                spec,
                stage=self._stage,
                tuple_boxes=self._tuple_boxes,
                parent_for_dialogs=self,
            )
            self._form.addRow(spec.name, widget)
            self._widgets[spec.name] = widget

    def _stage(self, field_name: str, value: Any) -> None:
        if self._updating:
            return
        self._working_recipe[field_name] = value
        self._mark_dirty()

    # ------------------------------------------------------------------ #
    # Apply / Revert
    # ------------------------------------------------------------------ #
    def _add_apply_bar(self) -> None:
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 8, 0, 0)

        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(self._on_apply)
        revert_button = QPushButton("Revert")
        revert_button.clicked.connect(self._on_revert)
        delete_button = QPushButton("Delete")
        delete_button.setStyleSheet("color: #c0392b;")
        delete_button.clicked.connect(self._on_delete)

        h.addWidget(apply_button)
        h.addWidget(revert_button)
        h.addStretch(1)
        h.addWidget(delete_button)
        self._form.addRow(row)
        self._apply_button = apply_button

    def _on_delete(self) -> None:
        if self.sid is not None:
            self.window()._checkpoint()
            self.connector.remove_structure(self.sid)
            self.set_selection(None)

    def _mark_dirty(self) -> None:
        if self._apply_button is not None:
            self._apply_button.setText("Apply*")

    def _on_revert(self) -> None:
        if self.sid is not None:
            self.set_selection(self.sid)  # nothing was committed yet — just reload

    def _on_apply(self) -> None:
        if self.sid is None:
            return
        so = self.connector.get(self.sid)
        # Captured BEFORE attempting the mutation, but only pushed onto
        # the undo stack on success (below) — Apply can fail validation
        # (beamz rejecting a bad value), and a failed Apply shouldn't
        # waste an undo slot on a no-op or clear the redo stack. See
        # undo.py's checkpoint_with() docstring.
        pre_snapshot = self.connector.snapshot()

        try:
            self.connector.update_param(self.sid, **self._working_recipe)
            if so.category != "region" and (
                self._working_rotation != so.rotation or self._working_rotation_axis != so.rotation_axis
            ) and not has_native_rotation_field(so.class_name):
                self.connector.update_rotation(self.sid, self._working_rotation, axis=self._working_rotation_axis)
        except Exception as exc:  # noqa: BLE001 — deliberately broad: beamz's
            # own validation can raise ValueError/TypeError depending on
            # the field, and the point of this try/except is to catch ANY
            # of it and show a message instead of letting it propagate
            # uncaught into Qt's event loop (which is what "GUI freeze/
            # crash on wrong values" turned out to be — an unhandled
            # exception inside a signal-handler call chain).
            if self._error_label is not None:
                self._error_label.setText(f"\u26a0 {exc}")
                self._error_label.setVisible(True)
            return

        self.window()._checkpoint_with_snapshot(pre_snapshot)

        # Success — rebuild fresh from the connector's new (authoritative)
        # state, which also clears the working copy / dirty marker / error.
        self.set_selection(self.sid)


def _short_repr(value: Any, limit: int = 60) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."