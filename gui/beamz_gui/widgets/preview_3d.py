"""
3D preview dock — rotate/pan/zoom + click-select, mirroring how Lumerical
actually uses its 3D view (a visualization/inspection aid, not a
direct-manipulation CAD surface; see the architecture discussion this was
scoped from). Precise editing still happens in the Properties panel / 2D
canvas; this is where you go to see whether your extrusion depths, z
positions, and rotations actually look right in 3D.

Mesh construction lives in `preview_mesh.py`, factored out specifically so
it's unit-testable without a working interactive OpenGL context — see that
module's docstring for why: `QtInteractor` needs real GPU/display support
to render correctly, and fails silently (blank output, no exception)
without one, unlike `pv.Plotter(off_screen=True)` which degrades to
software rendering gracefully. That split let mesh generation get proper
automated test coverage even in a headless sandbox; the actual on-screen
result still needs eyes-on confirmation on a real machine.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget
from pyvistaqt import QtInteractor

from .. import preview_mesh
from ..model import BeamzConnector

SELECTED_EDGE_COLOR = "#f1c40f"


class Preview3D(QWidget):
    selected = Signal(object)  # sid: str | None

    def __init__(self, connector: BeamzConnector, parent=None) -> None:
        super().__init__(parent)
        self.connector = connector
        self._selected_sid: Optional[str] = None
        self._suppress_selection_signal = False
        self._picking_enabled = False
        self._has_fit_once = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plotter = QtInteractor(self)
        self.plotter.set_background("black")  # match the rest of the app's
        # dark theme — VTK's own default background otherwise looks
        # inconsistent next to the other (dark-themed) panels.
        # Deliberately only ONE orientation widget, not both: an earlier
        # version kept `add_axes()` (a static, non-interactive labeled
        # X/Y/Z triad) ALONGSIDE `add_camera_orientation_widget()` (the
        # clickable one added below), which just put two overlapping
        # legends in the view for no reason — the clickable one already
        # shows which axis is which via its own coloring, and is strictly
        # more useful since it's also the click-to-snap-camera control.
        layout.addWidget(self.plotter.interactor)

        self._actor_to_sid: dict[int, str] = {}

        connector.structure_added.connect(self._redraw)
        connector.structure_removed.connect(self._redraw)
        connector.structure_changed.connect(self._redraw)

        self._try_enable_picking()
        self._try_enable_camera_orientation_widget()
        self._redraw()

    def _try_enable_picking(self) -> None:
        """`enable_mesh_picking` needs `self.plotter.iren` (the VTK render
        window interactor), which is only initialized once the underlying
        render window has actually been created — not necessarily true
        the instant `QtInteractor.__init__` returns, and never true at all
        in an environment with no working GPU/display context (which is
        how this was actually found: it raised `AttributeError:
        'NoneType' object has no attribute 'picker'` when constructed
        under Qt's offscreen platform in a headless sandbox). Rather than
        assume construction-time is always safe, this degrades gracefully
        and retries shortly after the event loop has had a chance to
        finish setting up the render window.
        """
        if self._picking_enabled:
            return
        try:
            self.plotter.enable_mesh_picking(
                callback=self._on_pick,
                use_actor=True,
                left_clicking=True,
                show_message=False,
            )
            self._picking_enabled = True
        except Exception:
            QTimer.singleShot(200, self._try_enable_picking)

    def _try_enable_camera_orientation_widget(self) -> None:
        """Adds the clickable orientation widget: click a face/axis on it
        and the camera animates to look straight down that axis — e.g.
        clicking its "top" face gives the same view as `view_xy()`. Same
        render-window-not-ready caveat as `_try_enable_picking` above
        (this needs `self.plotter.iren` too), so it gets the same
        retry-shortly-after pattern rather than assuming construction
        time is safe.
        """
        try:
            self.plotter.add_camera_orientation_widget()
        except Exception:
            QTimer.singleShot(200, self._try_enable_camera_orientation_widget)

    # ------------------------------------------------------------------ #
    def _redraw(self, *_args) -> None:
        had_actors = bool(self._actor_to_sid)
        self.plotter.clear()
        self._actor_to_sid.clear()

        for so in self.connector.all_structures():
            result = preview_mesh.mesh_and_style_for(so)
            if result is None:
                continue
            mesh, style = result
            if so.id == self._selected_sid:
                style = dict(style, show_edges=True, edge_color=SELECTED_EDGE_COLOR, line_width=3)
            actor = self.plotter.add_mesh(mesh, **style)
            self._actor_to_sid[id(actor)] = so.id

        # Camera reset only the first time content appears (mirrors
        # canvas_2d.py's identical fix) — otherwise every property edit
        # would snap the camera back to frame the whole scene, discarding
        # whatever the user just rotated/panned/zoomed to. VTK's default
        # trackball-camera interactor style already provides rotate
        # (left-drag) / pan (shift+left-drag or middle-drag) / zoom
        # (scroll or right-drag) for free — nothing extra needed there.
        if not had_actors and self._actor_to_sid and not self._has_fit_once:
            self.plotter.reset_camera()
            self._has_fit_once = True

    def fit_to_content(self) -> None:
        """Public entry point for an explicit "Fit View" toolbar action."""
        self.plotter.reset_camera()

    # ------------------------------------------------------------------ #
    def _on_pick(self, actor) -> None:
        if self._suppress_selection_signal:
            return
        sid = self._actor_to_sid.get(id(actor))
        self._selected_sid = sid
        self._redraw()  # reapply highlight styling
        self.selected.emit(sid)

    def select_external(self, sid: Optional[str]) -> None:
        """Programmatically select `sid` (e.g. the object tree or 2D
        canvas was clicked) without re-emitting `selected`.
        """
        if sid == self._selected_sid:
            return
        self._suppress_selection_signal = True
        try:
            self._selected_sid = sid
            self._redraw()
        finally:
            self._suppress_selection_signal = False