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
    QHBoxLayout,
    QToolButton,
    QWidget,
)

from .. import geometry_adapters as ga
from .. import materials
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


class _AxisLegend(QWidget):
    """CAD-style clickable axis legend (X/Y/Z), overlaid as a plain child
    widget in the corner of the viewport rather than drawn into the
    QGraphicsScene — same idea as the 3D preview's own corner axes gizmo:
    it needs to stay fixed on screen regardless of pan/zoom, which a
    scene item (subject to the view's transform) would not do.

    Click an axis to look ALONG it, same convention as any CAD view-cube
    / the 3D preview's own orientation widget: the axis you click is the
    one that collapses to a point ("goes into the screen"), so clicking
    "Z" gives the Top (XY) view, "Y" gives Front (XZ), "X" gives Side
    (YZ) — this is what replaced the separate plane-selector toolbar
    dropdown a first version of this had: the legend already tells you
    which way is which, so it may as well BE the control too, rather
    than duplicating that information in two separate widgets.
    """

    AXES = (("X", "#e74c3c", "yz"), ("Y", "#2ecc71", "xz"), ("Z", "#3498db", "xy"))

    def __init__(self, on_axis_clicked, parent=None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)
        self._buttons: dict[str, QToolButton] = {}
        for label, color, plane in self.AXES:
            btn = QToolButton(self)
            btn.setText(label)
            btn.setAutoRaise(True)
            btn.setCheckable(True)
            btn.setToolTip(f"View along {label} ({plane.upper()} plane)")
            btn.setStyleSheet(
                f"QToolButton {{ color: {color}; font-weight: bold; background: rgba(0,0,0,140);"
                f" border: 1px solid {color}; border-radius: 9px; min-width: 18px; min-height: 18px; }}"
                f"QToolButton:checked {{ background: {color}; color: black; }}"
                f"QToolButton:hover {{ background: {color}; color: black; }}"
            )
            btn.clicked.connect(lambda _checked=False, p=plane: on_axis_clicked(p))
            layout.addWidget(btn)
            self._buttons[plane] = btn
        self.set_active_plane("xy")

    def set_active_plane(self, plane: str) -> None:
        for p, btn in self._buttons.items():
            btn.setChecked(p == plane)


class Canvas2D(QGraphicsView):
    """Layout view, analogous to Lumerical's 2D layout editor — Top (XY)
    by default, with Front (XZ) / Side (YZ) also selectable (see
    `set_plane`) so every side of a device is actually inspectable in 2D,
    not just the top-down footprint. Structures are rendered generically:
    in the XY plane, straight from `obj.vertices` (+ `obj.interiors` for
    holes) via a painter path, which works uniformly for every structure
    kind, INCLUDING a rotated one (beamz represents that as a `Polygon`
    with different vertices rather than a rotated Rectangle — see
    model.py) with no per-shape special-casing needed; in XZ/YZ, as an
    axis-aligned bounding box (see `_make_structure_bbox_item` for why
    that's an approximation, not the true side silhouette).

    Sources/monitors don't have `vertices` at all — they're plane-based
    (`center` + `size`, with one axis usually zeroed) — so they get a
    distinct glyph: a line across the plane's in-view extent, dashed for
    monitors, with a small direction tick for sources. Being a plain
    center+size box already, this one generalizes to any of the three
    view planes exactly, with no approximation needed.
    """

    selected = Signal(object)  # sid: str | None
    status_message = Signal(str)  # empty string means "clear"

    # plane -> (u_axis_index, v_axis_index, u_label, v_label, title)
    _PLANES = {
        "xy": (0, 1, "x", "y", "Top (XY)"),
        "xz": (0, 2, "x", "z", "Front (XZ)"),
        "yz": (1, 2, "y", "z", "Side (YZ)"),
    }

    def __init__(self, connector: BeamzConnector, parent=None) -> None:
        super().__init__(parent)
        self.connector = connector
        self._plane = "xy"

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.scale(1, -1)  # +y points up, matching physical convention
        # QGraphicsView defaults to NoFocus — without this, clicking into
        # the canvas selects an item fine (that's mouse-event routing,
        # unrelated to focus) but arrow-key nudge and Delete never fire
        # at all, since keyPressEvent is never called on a widget that
        # never actually holds keyboard focus.
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._items: dict[str, QGraphicsItem] = {}
        self._suppress_selection_signal = False
        self._panning = False
        self._pan_start = None
        self._has_fit_once = False
        # Real content-only bounds, captured BEFORE _draw_grid() adds its
        # own (much larger) grid lines to the scene — see _redraw()'s
        # comment on why fit-to-content must use THIS, not
        # `self._scene.itemsBoundingRect()`.
        self._content_bounds = None

        connector.structure_added.connect(self._redraw)
        connector.structure_removed.connect(self._redraw)
        connector.structure_changed.connect(self._redraw)

        self._scene.selectionChanged.connect(self._on_scene_selection_changed)

        # Corner-anchored, positioned in resizeEvent (not layout-managed —
        # QGraphicsView has no layout of its own to speak of; this floats
        # as a plain child widget over the viewport, top-left corner).
        self._axis_legend = _AxisLegend(self.set_plane, self)
        self._axis_legend.adjustSize()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._axis_legend.move(8, 8)

    def set_plane(self, plane: str) -> None:
        """Switch the view to "xy" (top, default) / "xz" (front) / "yz"
        (side). A no-op if already on that plane. Always re-fits after
        switching — whatever pan/zoom made sense for the old projection
        has no reason to make sense for a completely different one.
        """
        if plane not in self._PLANES or plane == self._plane:
            return
        self._plane = plane
        self._axis_legend.set_active_plane(plane)
        self._redraw()
        self.fit_to_content()

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

            # Captured BEFORE _draw_grid() adds its own grid lines (which
            # deliberately extend past the content on every side — see
            # _draw_grid()) to the scene. A previous version fit to
            # `self._scene.itemsBoundingRect()` AFTER _draw_grid() ran,
            # which meant "fit to content" was actually fitting to the
            # GRID's extent instead — several grid-steps wider/taller
            # than the real content on every side, which is exactly why
            # the initial view looked zoomed out far more than the
            # actual structures warranted. Confirmed directly: removing
            # _draw_grid()'s lines from the equation shrank the "fit"
            # view to a sensible size.
            self._content_bounds = self._scene.itemsBoundingRect()
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
        item = self._make_structure_outline_item(so) if self._plane == "xy" else self._make_structure_bbox_item(so)
        if item is not None and not so.enabled:
            item.setOpacity(0.25)  # suppressed-from-a-run — see SceneObject.enabled
        return item

    def _make_structure_outline_item(self, so: SceneObject) -> Optional[QGraphicsItem]:
        """The TRUE footprint (XY plane only) straight from `obj.vertices`
        — exact for every structure kind, rotated or not, unlike the
        XZ/YZ bounding-box approximation below.
        """
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
        # Colored by MATERIAL index, not beamz's own `.color` field —
        # that field turned out to be a fixed default identical across
        # every structure regardless of `material` (see
        # materials.color_for_material's docstring), which is why two
        # structures with deliberately different materials used to
        # render as the exact same shade of blue.
        item.setBrush(QBrush(QColor(materials.color_for_material(so.recipe.get("material")))))
        pen = QPen(Qt.GlobalColor.black)
        pen.setWidth(0)  # cosmetic pen: always 1px regardless of view scale
        item.setPen(pen)
        return item

    def _make_structure_bbox_item(self, so: SceneObject) -> Optional[QGraphicsItem]:
        """XZ/YZ side-view APPROXIMATION: an axis-aligned rectangle built
        from the structure's center +/- span on the two displayed axes,
        via the exact same geometry_adapters math the Property Editor's
        x/y/z fields use. This is NOT the true side silhouette of an
        extruded arbitrary polygon — a Ring's real XZ cross-section, say,
        is two separate bars (inner/outer wall), not one solid box — that
        would need actual 3D projection, which this 2D canvas doesn't do.
        Good enough to see where something sits and how tall/deep it is;
        the 3D preview is where you go for the real shape. Returns None
        for classes with no position/span mapping at all (Polygon,
        CustomSource — same limitation the XY view already accepts for
        anything with no usable geometry description).
        """
        adapter = ga.get_adapter(so.class_name)
        if adapter is None:
            return None
        ui, vi, *_ = self._PLANES[self._plane]

        center = ga.get_xyz(so.recipe, adapter)
        spans = (
            ga.get_span(so.recipe, adapter.x_span) or 0.0,
            ga.get_span(so.recipe, adapter.y_span) or 0.0,
            ga.get_span(so.recipe, adapter.z_span) or 0.0,
        )
        u, v = center[ui], center[vi]
        su, sv = spans[ui], spans[vi]

        s = PIXELS_PER_METRE
        rect_path = QPainterPath()
        rect_path.addRect((u - su / 2) * s, (v - sv / 2) * s, su * s, sv * s)

        item = QGraphicsPathItem(rect_path)
        item.setBrush(QBrush(QColor(materials.color_for_material(so.recipe.get("material")))))
        pen = QPen(Qt.GlobalColor.black)
        pen.setWidth(0)
        item.setPen(pen)
        return item

    def _make_plane_item(self, so: SceneObject, *, stroke: str, dashed: bool, arrow: bool) -> QGraphicsItem:
        s = PIXELS_PER_METRE
        center = so.recipe.get("center", (0.0, 0.0, 0.0))
        size = so.recipe.get("size", (0.0, 0.0, 0.0))
        ui, vi, *_ = self._PLANES[self._plane]
        cu, cv = center[ui], center[vi]
        su, sv = size[ui], size[vi]

        x0, y0 = (cu - su / 2) * s, (cv - sv / 2) * s
        x1, y1 = (cu + su / 2) * s, (cv + sv / 2) * s

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
        if not so.enabled:
            item.setOpacity(0.25)  # suppressed-from-a-run — see SceneObject.enabled
        return item

    # ------------------------------------------------------------------ #
    # Guardrails against the units-mistake failure mode: a light
    # reference grid with a labeled spacing, and a status-bar warning
    # (routed to MainWindow) when content spans a suspiciously large area.
    # ------------------------------------------------------------------ #
    def _draw_grid(self) -> None:
        bounds = self._content_bounds
        if bounds is None or bounds.isEmpty():
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
        _u, _v, u_label, v_label, title = self._PLANES[self._plane]
        label = QGraphicsSimpleTextItem(f"{title}  ({u_label}\u2192, {v_label}\u2191)  \u2014  grid: {step_um:g} \u00b5m")
        label.setBrush(QBrush(QColor("#999")))
        # Keeps constant on-screen size/orientation regardless of the
        # view's zoom AND its y-flip — without this the label would
        # render mirrored (upside down) because of the `scale(1, -1)`
        # applied in __init__ for a physically-conventional +y-up view.
        label.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        label.setPos(x0 + margin * 0.25, y1 - margin * 0.25)
        self._scene.addItem(label)

    def _fit_to_content(self) -> None:
        bounds = self._content_bounds
        if bounds is None or bounds.isEmpty():
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
        bounds = self._content_bounds
        if bounds is None or bounds.isEmpty():
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
    # Multi-select (native to QGraphicsScene once items are selectable —
    # RubberBandDrag, set in __init__, already lets a drag select several
    # at once; Ctrl/Shift-click extend/toggle a single item into that same
    # selection for free too) + arrow-key nudge on a single selection.
    # ------------------------------------------------------------------ #
    _NUDGE_STEP_M = 0.01e-6  # 10 nm
    _NUDGE_STEP_FINE_M = 0.001e-6  # 1 nm, with Shift held

    def keyPressEvent(self, event) -> None:  # noqa: N802
        key = event.key()
        if key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            sids = self.selected_sids()
            if sids:
                # One checkpoint for the whole batch, not one per item —
                # a single Delete press (even on several items) is one
                # discrete user action, and should undo as one.
                self.window()._checkpoint()
                for sid in sids:
                    self.connector.remove_structure(sid)
                return
        if key in (Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down):
            sids = self.selected_sids()
            if len(sids) == 1 and self._nudge(sids[0], key, fine=bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier), first_press=not event.isAutoRepeat()):
                return
        super().keyPressEvent(event)

    def _nudge(self, sid: str, key, *, fine: bool, first_press: bool) -> bool:
        """Move the selected object by one small step in x/y. Only
        defined for classes geometry_adapters knows how to read a
        position out of (Polygon/CustomSource fall back to no-op, same
        as everywhere else geometry_adapters is used) — returns False for
        those so the caller can fall through to default key handling.
        """
        so = self.connector.get(sid)
        adapter = ga.get_adapter(so.class_name)
        if adapter is None:
            return False

        step = self._NUDGE_STEP_FINE_M if fine else self._NUDGE_STEP_M
        dx = {Qt.Key.Key_Left: -step, Qt.Key.Key_Right: step}.get(key, 0.0)
        dy = {Qt.Key.Key_Down: -step, Qt.Key.Key_Up: step}.get(key, 0.0)

        x, y, _z = ga.get_xyz(so.recipe, adapter)
        changes = ga.xyz_changes(so.recipe, adapter, x=x + dx, y=y + dy)
        if first_press:
            # Coalesce a held-down arrow key's auto-repeated events into
            # ONE undo step for the whole nudge gesture — only the actual
            # first (non-autorepeat) press gets a checkpoint, matching
            # how most graphics editors treat a held nudge as one undo.
            self.window()._checkpoint()
        self.connector.update_param(sid, **changes)
        return True

    def selected_sids(self) -> list[str]:
        return [item.data(SID_KEY) for item in self._scene.selectedItems() if item.data(SID_KEY)]

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
