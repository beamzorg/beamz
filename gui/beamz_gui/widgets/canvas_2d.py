from __future__ import annotations

import math
from typing import Optional

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
)

from ..model import BeamzConnector, SceneObject

# beamz coordinates are in metres; a photonic device is typically a
# handful of microns across. Rendering 1:1 would make everything
# sub-pixel, so we scale metres -> pixels for display only — this never
# touches simulation units anywhere, it's purely a view concern.
PIXELS_PER_METRE = 1e6  # i.e. 1 px per micron

SID_KEY = 0  # QGraphicsItem::data() key used to stash the scene-object id

# Above this span (in the same px-per-metre scale as everything else), the
# fit-to-content view is almost certainly dominated by a units mistake —
# e.g. typing position=(1,1,1) meaning "one micron" but getting one metre,
# since beamz takes raw SI values and a bare `1` is one full metre. A real
# on-chip photonic device rarely spans more than a few hundred microns;
# 1e5 px == 100,000 microns == 10 cm is well past "obviously wrong".
SUSPICIOUS_EXTENT_PX = 1e5


class Canvas2D(QGraphicsView):
    """Top-down (XY) layout view, analogous to Lumerical's 2D layout
    editor. Structures are rendered generically from `obj.vertices` (+
    `obj.interiors` for holes) via a painter path — this works uniformly
    for every structure kind, INCLUDING a rotated one, which beamz
    represents as a `Polygon` with different vertices rather than a
    rotated Rectangle (see model.py). No per-shape special-casing needed.

    Sources/monitors don't have `vertices` at all — they're plane-based
    (`center` + `size`, with one axis usually zeroed) — so they get a
    distinct glyph: a line across the plane's in-view extent, dashed for
    monitors, with a small direction tick for sources.
    """

    selected = Signal(object)  # sid: str | None
    status_message = Signal(str)  # empty string means "clear"

    def __init__(self, connector: BeamzConnector, parent=None) -> None:
        super().__init__(parent)
        self.connector = connector

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.scale(1, -1)  # +y points up, matching physical convention

        self._items: dict[str, QGraphicsItem] = {}
        self._suppress_selection_signal = False
        self._panning = False
        self._pan_start = None
        self._has_fit_once = False

        connector.structure_added.connect(self._redraw)
        connector.structure_removed.connect(self._redraw)
        connector.structure_changed.connect(self._redraw)

        self._scene.selectionChanged.connect(self._on_scene_selection_changed)

    # ------------------------------------------------------------------ #
    def _redraw(self, *_args) -> None:
        # `_scene.clear()` fires an intermediate selectionChanged as items
        # are removed; left unsuppressed it looks exactly like "user
        # clicked empty canvas" and drops the property panel's selection
        # a frame before we restore it below. So the whole
        # clear+rebuild+reselect sequence runs under one suppression
        # window (found the hard way — see smoke_test.py's regression
        # guard for this).
        previously_selected = next(
            (sid for sid, item in self._items.items() if item.isSelected()), None
        )
        was_empty = not self._items

        self._suppress_selection_signal = True
        try:
            self._scene.clear()
            self._items.clear()
            for so in self.connector.all_structures():
                item = self._make_item(so)
                if item is None:
                    continue
                item.setData(SID_KEY, so.id)
                item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
                self._scene.addItem(item)
                self._items[so.id] = item

            self._draw_grid()

            if previously_selected is not None and previously_selected in self._items:
                self._items[previously_selected].setSelected(True)
        finally:
            self._suppress_selection_signal = False

        # Auto-fit only the first time content appears (empty -> non-empty)
        # — every subsequent redraw (any property edit, since that's what
        # triggers structure_changed) preserves whatever pan/zoom the user
        # currently has. An earlier version fit-to-content unconditionally
        # on every redraw, which meant editing a single property while
        # zoomed in anywhere would suddenly snap the view back out to fit
        # everything — disorienting during actual editing work. The
        # units-mistake extent warning still needs to be (re)checked every
        # time regardless, so it's split out from the fit action itself.
        if was_empty and self._items:
            self.fit_to_content()
        else:
            self._check_extent_warning()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        # Safety net for the case where content was added before the
        # widget had ever been shown/laid out: `_redraw`'s empty -> non-
        # empty fit can run against a stale (pre-layout) viewport size if
        # structures are added immediately after `.show()` without an
        # intervening event-loop pass, producing a tiny/mispositioned
        # initial view. A first real showEvent guarantees final geometry,
        # so re-fit once here too — `_has_fit_once` still ensures this
        # never overrides a manual pan/zoom after that.
        if self._items and not self._has_fit_once:
            self.fit_to_content()

    def _make_item(self, so: SceneObject) -> Optional[QGraphicsItem]:
        if so.category == "structure":
            return self._make_structure_item(so)
        if so.category == "source":
            return self._make_plane_item(so, stroke="#e67e22", dashed=False, arrow=True)
        if so.category == "monitor":
            return self._make_plane_item(so, stroke="#2980b9", dashed=True, arrow=False)
        return None

    def _make_structure_item(self, so: SceneObject) -> Optional[QGraphicsItem]:
        obj = so.obj
        vertices = getattr(obj, "vertices", None)
        if not vertices:
            # Box/Sphere and similar non-planar primitives have no 2D
            # outline in v1 — they still exist fully in the tree/property
            # editor, just without a canvas glyph yet.
            return None

        s = PIXELS_PER_METRE
        outer = QPolygonF([QPointF(v[0] * s, v[1] * s) for v in vertices])
        path = QPainterPath()
        path.addPolygon(outer)
        for hole in getattr(obj, "interiors", ()) or ():
            path.addPolygon(QPolygonF([QPointF(v[0] * s, v[1] * s) for v in hole]))
        path.setFillRule(Qt.FillRule.OddEvenFill)  # cuts holes (Ring, etc.)

        item = QGraphicsPathItem(path)
        item.setBrush(QBrush(QColor(getattr(obj, "color", "#6699cc"))))
        pen = QPen(Qt.GlobalColor.black)
        pen.setWidth(0)  # cosmetic pen: always 1px regardless of view scale
        item.setPen(pen)
        return item

    def _make_plane_item(self, so: SceneObject, *, stroke: str, dashed: bool, arrow: bool) -> QGraphicsItem:
        s = PIXELS_PER_METRE
        center = so.recipe.get("center", (0.0, 0.0, 0.0))
        size = so.recipe.get("size", (0.0, 0.0, 0.0))
        cx, cy = center[0], center[1]
        sx, sy = size[0], size[1]

        x0, y0 = (cx - sx / 2) * s, (cy - sy / 2) * s
        x1, y1 = (cx + sx / 2) * s, (cy + sy / 2) * s

        path = QPainterPath()
        path.moveTo(x0, y0)
        path.lineTo(x1, y1)

        if arrow:
            # Perpendicular tick at the midpoint standing in for a
            # propagation-direction indicator. Reading the actual
            # +/-axis direction out of the recipe is a nice follow-up;
            # this at least visually distinguishes a source from a
            # monitor at a glance.
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            dx, dy = (x1 - x0), (y1 - y0)
            length = max(math.hypot(dx, dy), 1e-9)
            nx, ny = -dy / length, dx / length
            tick = max(length * 0.15, 2.0)
            path.moveTo(mx, my)
            path.lineTo(mx + nx * tick, my + ny * tick)

        item = QGraphicsPathItem(path)
        pen = QPen(QColor(stroke))
        pen.setWidth(0)
        if dashed:
            pen.setStyle(Qt.PenStyle.DashLine)
        item.setPen(pen)
        return item

    # ------------------------------------------------------------------ #
    # Guardrails against the units-mistake failure mode: a light
    # reference grid with a labeled spacing, and a status-bar warning
    # (routed to MainWindow) when content spans a suspiciously large area.
    # ------------------------------------------------------------------ #
    def _draw_grid(self) -> None:
        bounds = self._scene.itemsBoundingRect()
        if bounds.isEmpty():
            return
        span = max(bounds.width(), bounds.height())
        step = _nice_grid_step(span)

        pen = QPen(QColor("#e0e0e0"))
        pen.setWidth(0)
        margin = step * 2
        x0, x1 = bounds.left() - margin, bounds.right() + margin
        y0, y1 = bounds.bottom() - margin, bounds.top() + margin

        x = _round_down(x0, step)
        while x <= x1:
            self._scene.addLine(x, y0, x, y1, pen)
            x += step
        y = _round_down(y0, step)
        while y <= y1:
            self._scene.addLine(x0, y, x1, y, pen)
            y += step

        step_um = step / PIXELS_PER_METRE * 1e6
        label = QGraphicsSimpleTextItem(f"grid: {step_um:g} \u00b5m")
        label.setBrush(QBrush(QColor("#999")))
        # Keeps constant on-screen size/orientation regardless of the
        # view's zoom AND its y-flip — without this the label would
        # render mirrored (upside down) because of the `scale(1, -1)`
        # applied in __init__ for a physically-conventional +y-up view.
        label.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        label.setPos(x0 + margin * 0.25, y1 - margin * 0.25)
        self._scene.addItem(label)

    def _fit_to_content(self) -> None:
        bounds = self._scene.itemsBoundingRect()
        if bounds.isEmpty():
            self.status_message.emit("")
            return
        margin = max(bounds.width(), bounds.height(), 1.0) * 0.15
        padded = bounds.adjusted(-margin, -margin, margin, margin)
        self.fitInView(padded, Qt.AspectRatioMode.KeepAspectRatio)
        self._check_extent_warning()

    def fit_to_content(self) -> None:
        """Public entry point for an explicit "Fit View" action (toolbar
        button) as well as the automatic first-population/first-show fit.
        """
        self._fit_to_content()
        self._has_fit_once = True

    def _check_extent_warning(self) -> None:
        bounds = self._scene.itemsBoundingRect()
        if bounds.isEmpty():
            self.status_message.emit("")
            return
        span = max(bounds.width(), bounds.height())
        if span > SUSPICIOUS_EXTENT_PX:
            span_m = span / PIXELS_PER_METRE
            self.status_message.emit(
                f"\u26a0 Scene spans ~{span_m:.3g} m \u2014 likely a units mistake "
                f"(bare numbers are metres; use bz.um / bz.nm, e.g. 1.5*bz.um)."
            )
        else:
            self.status_message.emit("")

    # ------------------------------------------------------------------ #
    # Pan (middle-mouse drag) + zoom (scroll wheel, anchored under the
    # cursor via AnchorUnderMouse set in __init__). Left-button drag stays
    # rubber-band selection, matching the object tree / 3D preview's
    # click-to-select convention — the middle button was free and is the
    # common CAD/DCC convention for panning (Blender, Fusion360, KiCad).
    # ------------------------------------------------------------------ #
    def wheelEvent(self, event) -> None:  # noqa: N802
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)
        event.accept()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_start = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._panning and self._pan_start is not None:
            pos = event.position()
            delta = pos - self._pan_start
            self._pan_start = pos
            h_bar = self.horizontalScrollBar()
            v_bar = self.verticalScrollBar()
            h_bar.setValue(h_bar.value() - int(delta.x()))
            v_bar.setValue(v_bar.value() - int(delta.y()))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.MiddleButton and self._panning:
            self._panning = False
            self._pan_start = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    # ------------------------------------------------------------------ #
    def _on_scene_selection_changed(self) -> None:
        if self._suppress_selection_signal:
            return
        selected_items = self._scene.selectedItems()
        sid = selected_items[0].data(SID_KEY) if selected_items else None
        self.selected.emit(sid)

    def select_external(self, sid: Optional[str]) -> None:
        """Programmatically select `sid` (e.g. the object tree was
        clicked) without re-emitting `selected` and causing a signal loop.
        """
        self._suppress_selection_signal = True
        try:
            self._scene.clearSelection()
            item = self._items.get(sid) if sid else None
            if item is not None:
                item.setSelected(True)
        finally:
            self._suppress_selection_signal = False


def _nice_grid_step(span_px: float) -> float:
    """Pick a round-ish grid spacing so the grid has a sensible line count
    regardless of whether the scene is 2 microns or 2 millimetres across.
    """
    target = span_px / 8
    if target <= 0:
        return PIXELS_PER_METRE * 1e-6
    exponent = math.floor(math.log10(target))
    base = 10**exponent
    for mult in (1, 2, 5, 10):
        if base * mult >= target:
            return base * mult
    return base * 10


def _round_down(value: float, step: float) -> float:
    return math.floor(value / step) * step
