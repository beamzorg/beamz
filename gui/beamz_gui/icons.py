"""
Small, hand-drawn vector icons for the toolbar — built with QPainter
directly rather than shipping external icon files or an icon-font
dependency, so the app stays fully self-contained. Each function returns
a QIcon sized for toolbar use (see ICON_SIZE).

Kept deliberately simple: a handful of primitive shapes per icon, drawn
at 2x for crisp rendering on high-DPI displays then declared at the
logical ICON_SIZE.
"""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap

ICON_SIZE = 28
_SCALE = 2  # render at 2x, declare at ICON_SIZE, for crisp high-DPI icons
_PX = ICON_SIZE * _SCALE

# A consistent, muted palette distinct from the scene's own object colors
# (structures/sources/monitors/region each already have their own
# on-canvas color — see preview_mesh.py's SOURCE_COLOR/MONITOR_COLOR/
# REGION_COLOR — so the toolbar icons use a neutral foreground instead of
# reusing those, to avoid implying a color-coding relationship that isn't
# actually there).
_FG = Qt.GlobalColor.white
_ACCENT = QColor("#f1c40f")  # matches preview_mesh.REGION_COLOR, reused
# only for the Region icon specifically, where echoing the on-canvas
# highlight color is actually meaningful (this IS the region's color
# everywhere else). Wrapped in QColor explicitly rather than passed as a
# raw string — QPen's constructor overloads expect a QColor/GlobalColor/
# QBrush, and relying on an implicit str->QColor conversion is the kind
# of thing worth just not gambling on.


def _new_painter() -> tuple[QPixmap, QPainter]:
    pix = QPixmap(_PX, _PX)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    return pix, painter


def _finish(pix: QPixmap, painter: QPainter) -> QIcon:
    painter.end()
    return QIcon(pix)


def _pen(width: float = 2.0, color=_FG) -> QPen:
    p = QPen(color)
    p.setWidthF(width * _SCALE)
    p.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    return p


def structure_icon() -> QIcon:
    """A rectangle and a circle, overlapping — a generic "shape" glyph,
    since Structure covers many primitive kinds (Rectangle/Circle/Ring/...).
    """
    pix, p = _new_painter()
    m = 5 * _SCALE
    p.setPen(_pen())
    p.drawRect(QRectF(m, m + 3 * _SCALE, _PX - 2 * m - 5 * _SCALE, _PX - 2 * m - 3 * _SCALE))
    p.drawEllipse(QRectF(m + 8 * _SCALE, m - 2 * _SCALE, _PX - 2 * m - 4 * _SCALE, _PX - 2 * m - 4 * _SCALE))
    return _finish(pix, p)


def source_icon() -> QIcon:
    """A small emitter with radiating arcs — a generic "something radiates
    from here" glyph, since Source covers several beam/mode kinds.
    """
    pix, p = _new_painter()
    cx, cy = 8 * _SCALE, _PX - 8 * _SCALE
    p.setPen(_pen())
    p.setBrush(_FG)
    p.drawEllipse(QPointF(cx, cy), 3 * _SCALE, 3 * _SCALE)
    p.setBrush(Qt.BrushStyle.NoBrush)
    for radius in (9, 15, 21):
        rect = QRectF(cx - radius * _SCALE / 2, cy - radius * _SCALE, radius * _SCALE, radius * _SCALE * 2)
        p.drawArc(rect, -40 * 16, 80 * 16)
    return _finish(pix, p)


def monitor_icon() -> QIcon:
    """A small oscilloscope-style box with a trace inside — Monitor
    records field data over time/frequency, a "waveform readout" glyph.
    """
    pix, p = _new_painter()
    m = 4 * _SCALE
    rect = QRectF(m, m, _PX - 2 * m, _PX - 2 * m)
    p.setPen(_pen())
    p.drawRoundedRect(rect, 3 * _SCALE, 3 * _SCALE)

    path = QPainterPath()
    y_mid = _PX / 2
    path.moveTo(m + 3 * _SCALE, y_mid)
    path.lineTo(m + 8 * _SCALE, y_mid)
    path.lineTo(m + 11 * _SCALE, y_mid - 8 * _SCALE)
    path.lineTo(m + 15 * _SCALE, y_mid + 8 * _SCALE)
    path.lineTo(m + 18 * _SCALE, y_mid)
    path.lineTo(_PX - m - 3 * _SCALE, y_mid)
    p.setPen(_pen(1.6))
    p.drawPath(path)
    return _finish(pix, p)


def region_icon() -> QIcon:
    """A bounding box with corner ticks — the simulation domain itself,
    echoing the same yellow used to highlight it in the 3D/mesh views
    (preview_mesh.REGION_COLOR) so the two are visually associated.
    """
    pix, p = _new_painter()
    m = 5 * _SCALE
    rect = QRectF(m, m, _PX - 2 * m, _PX - 2 * m)
    pen = _pen(1.6, _ACCENT)
    pen.setStyle(Qt.PenStyle.DashLine)
    p.setPen(pen)
    p.drawRect(rect)

    corner = 4 * _SCALE
    p.setPen(_pen(2.2, _ACCENT))
    for x, y, dx, dy in (
        (rect.left(), rect.top(), 1, 1),
        (rect.right(), rect.top(), -1, 1),
        (rect.left(), rect.bottom(), 1, -1),
        (rect.right(), rect.bottom(), -1, -1),
    ):
        p.drawLine(QPointF(x, y), QPointF(x + corner * dx, y))
        p.drawLine(QPointF(x, y), QPointF(x, y + corner * dy))
    return _finish(pix, p)


def mesh_icon() -> QIcon:
    """A simple 3x3 grid — View Mesh shows the discretized simulation
    layout, so a literal grid is the clearest possible glyph.
    """
    pix, p = _new_painter()
    m = 5 * _SCALE
    size = _PX - 2 * m
    p.setPen(_pen(1.6))
    p.drawRect(QRectF(m, m, size, size))
    for i in (1, 2):
        offset = m + size * i / 3
        p.drawLine(QPointF(offset, m), QPointF(offset, m + size))
        p.drawLine(QPointF(m, offset), QPointF(m + size, offset))
    return _finish(pix, p)


def memory_icon() -> QIcon:
    """Three ascending bars — a lightweight "resource usage" glyph for
    Memory Estimate, without needing to render actual text in an icon.
    """
    pix, p = _new_painter()
    base = _PX - 6 * _SCALE
    widths = 5 * _SCALE
    heights = (8 * _SCALE, 13 * _SCALE, 18 * _SCALE)
    p.setPen(_pen(1.4))
    p.setBrush(_FG)
    x = 6 * _SCALE
    for h in heights:
        p.drawRect(QRectF(x, base - h, widths, h))
        x += widths + 3 * _SCALE
    return _finish(pix, p)


def fit_icon() -> QIcon:
    """Four corner brackets framing empty space — the common "fit to
    view"/"zoom to frame" glyph used by camera and CAD/DCC apps.
    """
    pix, p = _new_painter()
    m = 6 * _SCALE
    corner = 6 * _SCALE
    p.setPen(_pen(2.0))
    rect = QRectF(m, m, _PX - 2 * m, _PX - 2 * m)
    for x, y, dx, dy in (
        (rect.left(), rect.top(), 1, 1),
        (rect.right(), rect.top(), -1, 1),
        (rect.left(), rect.bottom(), 1, -1),
        (rect.right(), rect.bottom(), -1, -1),
    ):
        p.drawLine(QPointF(x, y), QPointF(x + corner * dx, y))
        p.drawLine(QPointF(x, y), QPointF(x, y + corner * dy))
    return _finish(pix, p)


def run_icon() -> QIcon:
    """A play triangle — Run Solver is the primary, final action, and
    gets the most universally recognizable glyph available for it.
    """
    pix, p = _new_painter()
    m = 6 * _SCALE
    path = QPainterPath()
    path.moveTo(m, m)
    path.lineTo(_PX - m, _PX / 2)
    path.lineTo(m, _PX - m)
    path.closeSubpath()
    p.setPen(_pen(1.5))
    p.setBrush(_FG)
    p.drawPath(path)
    return _finish(pix, p)