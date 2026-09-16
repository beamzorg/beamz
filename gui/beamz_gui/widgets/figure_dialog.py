"""
A plain dialog that embeds an already-created Matplotlib (figure, axes)
in Qt via FigureCanvasQTAgg — used for both "View Mesh" (Simulation.plot())
and monitor "View Results" (SimulationResults.plot_field()), since both
just need to show a figure beamz itself already knows how to draw. This
deliberately does NOT try to re-implement plotting; it's a thin host.
"""
from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from PySide6.QtWidgets import QDialog, QVBoxLayout


class FigureDialog(QDialog):
    def __init__(self, figure, title: str = "Plot", parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(800, 600)

        layout = QVBoxLayout(self)
        canvas = FigureCanvasQTAgg(figure)
        toolbar = NavigationToolbar2QT(canvas, self)  # pan/zoom/save, matplotlib's own
        layout.addWidget(toolbar)
        layout.addWidget(canvas)
