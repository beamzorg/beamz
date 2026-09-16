"""
A small dialog for editing a NESTED beamz object's own fields — e.g. the
`GaussianPulse` sitting inside a source's `source_time`. Built with the
exact same introspection machinery as the main Property Editor
(`introspection.introspect_fields`), so it stays fully dynamic: if a
nested class gains/loses fields in a future beamz version, this dialog
adapts automatically, same as the rest of the app.

Deliberately simple compared to the main Property Editor — no geometry
block, no material/rotation special-casing (nested objects like
GaussianPulse aren't placeables and don't have those concepts) — just
every field, generically, plus OK/Cancel.
"""
from __future__ import annotations

from typing import Any, Optional

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

import beamz as bz

from .. import introspection as intro
from . import dynamic_fields as df


class NestedObjectDialog(QDialog):
    """Edits a copy of `recipe` for `class_name`. On OK, validates by
    actually constructing `cls(**working)` — if beamz rejects it, shows
    the error and stays open rather than closing with bad data (same
    validate-before-commit principle as the main Property Editor's Apply).
    `self.result_recipe` is set (and non-None) only after a successful OK.
    """

    def __init__(self, class_name: str, recipe: dict[str, Any], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.class_name = class_name
        self.working: dict[str, Any] = dict(recipe)
        self.result_recipe: Optional[dict[str, Any]] = None

        self.setWindowTitle(f"Edit {class_name}")
        self.resize(380, 300)

        layout = QVBoxLayout(self)
        self._error_label = QLabel("")
        self._error_label.setStyleSheet("color: #c0392b;")
        self._error_label.setWordWrap(True)
        self._error_label.setVisible(False)
        layout.addWidget(self._error_label)

        form = QFormLayout()
        layout.addLayout(form)
        self._tuple_boxes: dict[str, list[QDoubleSpinBox]] = {}

        for spec in intro.introspect_fields(class_name, self.working):
            widget = df.build_field_widget(
                class_name,
                spec,
                stage=self._stage,
                tuple_boxes=self._tuple_boxes,
                parent_for_dialogs=self,
            )
            form.addRow(spec.name, widget)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _stage(self, field_name: str, value: Any) -> None:
        self.working[field_name] = value

    def _on_accept(self) -> None:
        try:
            getattr(bz, self.class_name)(**self.working)
        except Exception as exc:  # noqa: BLE001 — same reasoning as the
            # main Property Editor's Apply: beamz validates eagerly, and
            # the point is to show whatever it says rather than crash or
            # silently accept bad data.
            self._error_label.setText(f"\u26a0 {exc}")
            self._error_label.setVisible(True)
            return
        self.result_recipe = self.working
        self.accept()
