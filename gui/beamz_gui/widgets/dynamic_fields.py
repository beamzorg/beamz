"""
Builds an editable widget for a single `FieldSpec`, for whatever
`FieldSpec.kind` `introspection.py` classified it as. Shared by the main
Property Editor's "Additional Properties" section AND
`NestedObjectDialog` (for editing e.g. a source's `source_time` nested
GaussianPulse), so array/nested-object editing exists in exactly one
place rather than being duplicated between them.

Every widget calls `stage(field_name, value)` on change — it never talks
to a connector directly. The caller decides what staging means (the
Property Editor's working-copy-plus-Apply pattern; the nested dialog's
plain in-memory dict).
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QWidget,
)

import beamz as bz

from .. import introspection as intro
from .. import defaults
from .spinbox import SmartDoubleSpinBox

FLOAT_DECIMALS = 12
FLOAT_STEP = 1e-8
FLOAT_RANGE = 1e20
AXIS_LABELS = ("x", "y", "z")

# Known valid values for each class's `direction`-style field, so it gets
# a dropdown instead of free text (typo-prone otherwise — beamz validates
# these strictly, e.g. ModeSource only accepts "+"/"-", not "+x"/"-x").
CHOICE_FIELDS: dict[tuple[str, str], tuple[str, ...]] = {
    ("GaussianBeamSource", "direction"): ("+x", "-x", "+y", "-y", "+z", "-z"),
    ("ModeSource", "direction"): ("+", "-"),
}

# What nested class a field holds when its CURRENT value is None (so
# there's nothing to introspect a type from) but the field is known to
# accept a specific nested beamz class when set — lets "Create ..." offer
# a real, usable starting point instead of just showing "None" forever.
NESTED_FIELD_TYPES: dict[tuple[str, str], str] = {
    ("GaussianBeamSource", "source_time"): "GaussianPulse",
    ("ModeSource", "source_time"): "GaussianPulse",
}


def build_field_widget(
    class_name: str,
    spec: intro.FieldSpec,
    *,
    stage: Callable[[str, Any], None],
    tuple_boxes: dict[str, list[QDoubleSpinBox]],
    parent_for_dialogs: QWidget,
) -> QWidget:
    """Returns a widget editing `spec`, wired to call `stage(spec.name,
    new_value)` on change. `tuple_boxes` accumulates per-field spin-box
    lists for tuple2/tuple3 fields, keyed by field name — the caller owns
    this dict (needed so IT can read combined values back out; building
    the widget alone can't return "3 boxes' worth" through one return
    value).
    """
    choices = CHOICE_FIELDS.get((class_name, spec.name))
    if choices is not None:
        w = QComboBox()
        w.addItems(list(choices))
        if spec.default in choices:
            w.setCurrentText(spec.default)
        w.currentTextChanged.connect(lambda v, n=spec.name: stage(n, v))
        return w

    if spec.kind == "float":
        w = SmartDoubleSpinBox()
        w.setRange(-FLOAT_RANGE, FLOAT_RANGE)
        w.setDecimals(FLOAT_DECIMALS)
        w.setSingleStep(FLOAT_STEP)
        w.setValue(spec.default)
        w.valueChanged.connect(lambda v, n=spec.name: stage(n, v))
        return w

    if spec.kind == "int":
        w = QSpinBox()
        w.setRange(-1_000_000, 1_000_000)
        w.setValue(spec.default)
        w.valueChanged.connect(lambda v, n=spec.name: stage(n, v))
        return w

    if spec.kind == "bool":
        w = QCheckBox()
        w.setChecked(bool(spec.default))
        w.toggled.connect(lambda v, n=spec.name: stage(n, v))
        return w

    if spec.kind == "str":
        w = QLineEdit(str(spec.default))
        w.editingFinished.connect(lambda n=spec.name, w=w: stage(n, w.text()))
        return w

    if spec.kind in ("tuple2", "tuple3"):
        return _build_tuple_widget(spec, stage=stage, tuple_boxes=tuple_boxes)

    if spec.kind == "array":
        return _build_array_widget(spec, stage=stage, parent_for_dialogs=parent_for_dialogs)

    if spec.kind == "nested":
        return _build_nested_widget(spec, stage=stage, parent_for_dialogs=parent_for_dialogs)

    if spec.kind == "readonly" and spec.default is None:
        nested_class_name = NESTED_FIELD_TYPES.get((class_name, spec.name))
        if nested_class_name is not None:
            return _build_create_nested_widget(
                spec, nested_class_name, stage=stage, parent_for_dialogs=parent_for_dialogs
            )

    # Genuine fallback: no editor known for this shape yet (e.g. an
    # Optional field of an unrecognized nested type). Shown plainly
    # rather than silently hidden, so at least its current value is
    # visible even though it can't be changed here.
    return QLabel(_short_repr(spec.default))


# -------------------------------------------------------------------- #
def _build_tuple_widget(
    spec: intro.FieldSpec,
    *,
    stage: Callable[[str, Any], None],
    tuple_boxes: dict[str, list[QDoubleSpinBox]],
) -> QWidget:
    from PySide6.QtWidgets import QHBoxLayout

    n = int(spec.kind[-1])
    container = QWidget()
    h = QHBoxLayout(container)
    h.setContentsMargins(0, 0, 0, 0)
    boxes: list[QDoubleSpinBox] = []

    def commit() -> None:
        stage(spec.name, tuple(b.value() for b in boxes))

    for i in range(n):
        b = SmartDoubleSpinBox()
        b.setRange(-FLOAT_RANGE, FLOAT_RANGE)
        b.setDecimals(FLOAT_DECIMALS)
        b.setSingleStep(FLOAT_STEP)
        b.setValue(spec.default[i])
        b.valueChanged.connect(lambda _v: commit())
        boxes.append(b)
        label = QLabel(AXIS_LABELS[i])
        h.addWidget(label)
        h.addWidget(b)
    tuple_boxes[spec.name] = boxes
    return container


def _array_to_text(value: Any) -> str:
    if isinstance(value, np.ndarray):
        return ", ".join(f"{v:g}" for v in value.tolist())
    return ", ".join(str(v) for v in value)


def _build_array_widget(
    spec: intro.FieldSpec, *, stage: Callable[[str, Any], None], parent_for_dialogs: QWidget
) -> QWidget:
    """A comma-separated-values text editor for a numpy array or a plain
    numeric list/tuple of arbitrary length (`freqs`, `signal`, ...).
    Deliberately simple (no table/spreadsheet widget) — matches "quick to
    set up" for typical short lists (a handful of frequencies); very long
    signals are still editable, just as one long line of text, which is
    an honest v1 tradeoff rather than a hidden limitation.
    """
    was_ndarray = isinstance(spec.default, np.ndarray)
    w = QLineEdit(_array_to_text(spec.default))
    w.setToolTip("Comma-separated numbers")

    def commit() -> None:
        text = w.text().strip()
        try:
            values = [float(x) for x in text.split(",") if x.strip() != ""]
        except ValueError:
            w.setStyleSheet("border: 1px solid #c0392b;")
            w.setToolTip(f"Invalid — expected comma-separated numbers, got: {text!r}")
            return
        w.setStyleSheet("")
        w.setToolTip("Comma-separated numbers")
        stage(spec.name, np.array(values) if was_ndarray else values)

    w.editingFinished.connect(commit)
    return w


def _build_nested_widget(
    spec: intro.FieldSpec, *, stage: Callable[[str, Any], None], parent_for_dialogs: QWidget
) -> QWidget:
    """An existing nested object (e.g. a source's `source_time` already
    holding a GaussianPulse) — "Edit ..." opens a NestedObjectDialog on a
    copy of its own fields, introspected the same recursive way.
    """
    instance = spec.default
    nested_class_name = type(instance).__name__
    button = QPushButton(f"Edit {nested_class_name}...")

    def on_click() -> None:
        from ..model import _recipe_from_instance  # local import: avoids a
        # module-level cycle (model.py doesn't import widgets; widgets
        # import model — importing at call time here sidesteps having to
        # think about import order at module load time).
        from .nested_object_dialog import NestedObjectDialog

        nested_recipe = _recipe_from_instance(instance)
        dialog = NestedObjectDialog(nested_class_name, nested_recipe, parent=parent_for_dialogs)
        if dialog.exec():
            new_instance = getattr(bz, nested_class_name)(**dialog.result_recipe)
            stage(spec.name, new_instance)

    button.clicked.connect(on_click)
    return button


def _build_create_nested_widget(
    spec: intro.FieldSpec,
    nested_class_name: str,
    *,
    stage: Callable[[str, Any], None],
    parent_for_dialogs: QWidget,
) -> QWidget:
    """A currently-None field known to accept a specific nested class
    (see NESTED_FIELD_TYPES) — "Create ..." builds a sensible default via
    the SAME `defaults.build_default_recipe` the toolbar's Add menus use,
    then immediately opens it for editing.
    """
    button = QPushButton(f"Create {nested_class_name}...")

    def on_click() -> None:
        from .nested_object_dialog import NestedObjectDialog

        try:
            recipe = defaults.build_default_recipe(nested_class_name)
        except TypeError:
            recipe = {}
        dialog = NestedObjectDialog(nested_class_name, recipe, parent=parent_for_dialogs)
        if dialog.exec():
            new_instance = getattr(bz, nested_class_name)(**dialog.result_recipe)
            stage(spec.name, new_instance)

    button.clicked.connect(on_click)
    return button


def _short_repr(value: Any, limit: int = 60) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."