"""
Parameter dialog for File -> Import GDS Layout... — collects the kwargs
`beamz.design.import_gds()` needs to turn a 2D GDS layer into a real 3D
simulation geometry (a vertical layer stack has to come from SOMEWHERE;
GDS itself is purely 2D). Deliberately simple/flat compared to the main
Property Editor or NestedObjectDialog — this is a fixed, small, hand-
picked subset of `import_gds()`'s real keyword arguments (see its
docstring in beamz.design.gds), not a dynamically-introspected beamz
class's full field set, so there's no need for that machinery here.
"""
from __future__ import annotations

from typing import Any, Optional

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class GdsImportDialog(QDialog):
    """On OK, `self.result_kwargs` holds `import_gds()`-ready keyword
    arguments in SI units (the fields below are all shown in more
    GDS/photonics-natural microns, converted here) — None if cancelled.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.result_kwargs: Optional[dict[str, Any]] = None
        self.setWindowTitle("Import GDS Layout")
        self.resize(360, 320)

        layout = QVBoxLayout(self)
        note = QLabel(
            "Requires beamz's optional layout dependency: pip install beamz[gds]"
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #888;")
        layout.addWidget(note)

        form = QFormLayout()
        layout.addLayout(form)

        self.layer_num = QSpinBox()
        self.layer_num.setRange(0, 9999)
        self.layer_num.setValue(1)
        form.addRow("GDS layer number", self.layer_num)

        self.layer_datatype = QSpinBox()
        self.layer_datatype.setRange(0, 9999)
        self.layer_datatype.setValue(0)
        form.addRow("GDS layer datatype", self.layer_datatype)

        self.n_core = self._um_box(2.0, 0.5, 10.0, decimals=3)
        form.addRow("Core index n_core", self.n_core)

        self.n_clad = self._um_box(1.44, 0.5, 10.0, decimals=3)
        form.addRow("Cladding index n_clad", self.n_clad)

        self.core_thickness_um = self._um_box(0.22, 0.001, 100.0)
        form.addRow("Core thickness (\u00b5m)", self.core_thickness_um)

        self.clad_below_um = self._um_box(0.5, 0.0, 100.0)
        form.addRow("Cladding below (\u00b5m)", self.clad_below_um)

        self.clad_above_um = self._um_box(0.5, 0.0, 100.0)
        form.addRow("Cladding above (\u00b5m)", self.clad_above_um)

        self.xy_padding_um = self._um_box(0.0, 0.0, 1000.0)
        form.addRow("XY padding (\u00b5m)", self.xy_padding_um)

        self._error_label = QLabel("")
        self._error_label.setStyleSheet("color: #c0392b;")
        self._error_label.setWordWrap(True)
        self._error_label.setVisible(False)
        layout.addWidget(self._error_label)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @staticmethod
    def _um_box(value: float, minimum: float, maximum: float, *, decimals: int = 4) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setDecimals(decimals)
        box.setRange(minimum, maximum)
        box.setValue(value)
        return box

    def _on_accept(self) -> None:
        if self.core_thickness_um.value() <= 0:
            self._error_label.setText("\u26a0 Core thickness must be positive.")
            self._error_label.setVisible(True)
            return
        self.result_kwargs = dict(
            layer=(self.layer_num.value(), self.layer_datatype.value()),
            n_core=self.n_core.value(),
            n_clad=self.n_clad.value(),
            core_thickness=self.core_thickness_um.value() * 1e-6,
            clad_below=self.clad_below_um.value() * 1e-6,
            clad_above=self.clad_above_um.value() * 1e-6,
            xy_padding=self.xy_padding_um.value() * 1e-6,
        )
        self.accept()