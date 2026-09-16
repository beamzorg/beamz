"""
Maps each beamz placeable class's actual constructor fields onto a
canonical Lumerical-style geometry representation — x, y, z, x span, y
span, z span — so every structure/source/monitor is edited through the
same six fields in the Property Editor regardless of what its underlying
primitive parameters are actually called (`width`/`height` for a
Rectangle, `radius` for a Circle, `outer_radius` for a Ring, `size` for a
source/monitor, ...).

CORNER vs CENTER — audited by reading each class's source directly, not
guessed, because getting this wrong silently moves a structure's center
whenever you edit a span (a real reported bug — see `x_is_corner` etc.
below):
  - Rectangle: `position` is the corner (verts run from `position` to
    `position + (width, height)`, confirmed in `Rectangle.__post_init__`).
    Corner in x, y, AND z (z shares the same "start, extrude by depth"
    pattern as x/width and y/height, though I could not find an explicit
    rasterizer excerpt confirming the z direction — flagged as the one
    part of this audit taken on strong pattern-consistency rather than
    a directly-read confirmation).
  - Taper: x is a corner (starts at `position`, runs to `position +
    length`); y is ALREADY centered (verts are position.y +/- width/2 on
    each end, confirmed in `Taper.__post_init__`) — an asymmetric case.
  - Circle, Ring, CircularBend: `position` IS the true center (Ring/Circle
    obviously; CircularBend confirmed via source — vertices are
    `position + radius * (cos, sin)`, i.e. `position` is the arc's pivot,
    not a bounding-box corner, even though a 90° arc's own bounding box
    happens to sit entirely in one quadrant relative to it, which looks
    corner-like from vertices alone if you don't read the source).
  - Sphere, Box: confirmed true center via source (`Sphere.center =
    property(lambda self: self.position)`; `Box.center` is literally
    named that).
  - Sources/monitors (`center`/`size`) and `GaussianSource` (`position`):
    assumed true center, consistent with field naming and not flagged by
    any evidence to the contrary, but not source-verified as thoroughly
    as the structure classes above.

An adapter says, for a given class:
  - which recipe field holds the position (`position_field`), and how
    many components it carries itself (`position_len`) — some classes
    pack x/y/z into one 3-tuple, others use a 2-tuple position plus a
    separate scalar `z` field (`z_field`).
  - which recipe field (optionally scaled, optionally a specific index of
    a tuple field) each span maps to. `None` means that axis isn't
    meaningfully representable as a span for this class (e.g. a Taper's
    width varies along its length, so it has no single y_span) and the
    Property Editor simply omits that row.
  - whether the stored position is a corner rather than already a center,
    per-axis (`x_is_corner` / `y_is_corner` / `z_is_corner`).

Every function here is pure: given a recipe dict, it reads or returns
*changes* to merge into it — never mutates in place. The Property Editor
applies those changes via `connector.update_param(sid, **changes)`, same
as any other field edit.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Optional, Union

FieldRef = Union[str, tuple[str, int]]  # a scalar field name, or (tuple_field_name, index)


@dataclasses.dataclass
class SpanRef:
    field: FieldRef
    scale: float = 1.0  # displayed_span = underlying_field_value * scale


@dataclasses.dataclass
class GeometryAdapter:
    position_field: str  # e.g. "position" or "center"
    position_len: int  # how many components the position field itself carries (2 or 3)
    z_field: Optional[str] = None  # separate scalar z, when position_len == 2
    x_span: Optional[SpanRef] = None
    y_span: Optional[SpanRef] = None
    z_span: Optional[SpanRef] = None
    # Whether the STORED position coordinate along each axis is a corner
    # (e.g. Rectangle's position is its bottom-left-front corner — verts
    # are computed as (x, y, z) .. (x+width, y+height, z), confirmed by
    # reading Rectangle's __post_init__ source) rather than already being
    # the true center (e.g. Circle/Ring/CircularBend's position IS the
    # center; Sphere/Box likewise, confirmed via their source too — see
    # geometry_adapters.py's module docstring for the full audit). When
    # True, `get_xyz`/`xyz_changes` convert to/from a genuine center for
    # display, and `span_changes` compensates the stored corner so the
    # CENTER stays fixed while a span edit changes size — matching how
    # Lumerical's x/y/z + x-span/y-span/z-span actually behaves. Getting
    # this wrong was a real, reported bug: editing a span silently moved
    # the structure's center away from where the x/y/z fields said it was.
    x_is_corner: bool = False
    y_is_corner: bool = False
    z_is_corner: bool = False
    # Rectangle ALSO has a redundant top-level `z` field (separate from
    # `position`'s 3rd component) that beamz's own `_position(position,
    # z)` helper uses to UNCONDITIONALLY overwrite position[2] whenever
    # z is not None — and its default is 0.0, not None. So writes to
    # position[2] were being silently discarded on every reconstruction,
    # snapping z back to 0 and making displayed z-center track depth/2
    # instead of staying fixed. Confirmed via beamz's own source. When
    # True, z writes go to BOTH position[2] AND this redundant field.
    sync_z_field: bool = False

    def consumed_fields(self) -> set[str]:
        fields = {self.position_field}
        if self.z_field:
            fields.add(self.z_field)
        if self.sync_z_field:
            fields.add("z")
        for span in (self.x_span, self.y_span, self.z_span):
            if span is None:
                continue
            fields.add(span.field[0] if isinstance(span.field, tuple) else span.field)
        return fields


# ---------------------------------------------------------------------- #
# Per-class adapters. Structures first (each genuinely different in how
# they parameterize size), then a single shared shape for every
# center/size-based source/monitor (they're all identical here, since
# that's beamz's own convention for placeable-plane objects).
# ---------------------------------------------------------------------- #
ADAPTERS: dict[str, GeometryAdapter] = {
    "Rectangle": GeometryAdapter(
        position_field="position",
        position_len=3,
        x_span=SpanRef("width"),
        y_span=SpanRef("height"),
        z_span=SpanRef("depth"),
        x_is_corner=True,
        y_is_corner=True,
        z_is_corner=True,
        sync_z_field=True,
    ),
    "Circle": GeometryAdapter(
        position_field="position",
        position_len=2,
        z_field="z",
        # radius drives both spans at once — a circle can't have an
        # independent x/y extent, so editing either span writes the same
        # underlying `radius` field.
        x_span=SpanRef("radius"),
        y_span=SpanRef("radius"),
        z_span=SpanRef("depth"),
        z_is_corner=True,  # position.x/y already centered; separate z field is not
    ),
    "Ring": GeometryAdapter(
        position_field="position",
        position_len=2,
        z_field="z",
        x_span=SpanRef("outer_radius"),
        y_span=SpanRef("outer_radius"),
        z_span=SpanRef("depth"),
        z_is_corner=True,
        # `inner_radius` isn't part of the canonical 6-field box model —
        # it shows up under "Additional Properties" instead.
    ),
    "Box": GeometryAdapter(
        position_field="center",
        position_len=3,
        x_span=SpanRef(("size", 0)),
        y_span=SpanRef(("size", 1)),
        z_span=SpanRef(("size", 2)),
    ),
    "Sphere": GeometryAdapter(
        position_field="position",
        position_len=3,
        x_span=SpanRef("radius"),
        y_span=SpanRef("radius"),
        z_span=SpanRef("radius"),
    ),
    "Taper": GeometryAdapter(
        position_field="position",
        position_len=2,
        z_field="z",
        x_span=SpanRef("length"),
        y_span=None,  # width varies along the taper — no single y span
        z_span=SpanRef("depth"),
        x_is_corner=True,  # position.x is the taper's start point
        z_is_corner=True,
        # position.y is already centered (input/output widths are +/-
        # around it) — y_is_corner stays False.
    ),
    "CircularBend": GeometryAdapter(
        position_field="position",
        position_len=2,
        z_field="z",
        # Bounding-circle approximation, same idea as Ring — the true
        # footprint depends on the bend `angle` too, shown separately
        # under Additional Properties. position itself IS the arc pivot
        # (true center), confirmed via source — see module docstring.
        x_span=SpanRef("outer_radius"),
        y_span=SpanRef("outer_radius"),
        z_span=SpanRef("depth"),
        z_is_corner=True,
    ),
    "GaussianSource": GeometryAdapter(
        # A Gaussian *spot* source: isotropic `width` standing in for all
        # three spans, same idea as Sphere's radius.
        position_field="position",
        position_len=3,
        x_span=SpanRef("width"),
        y_span=SpanRef("width"),
        z_span=SpanRef("width"),
    ),
}

_CENTER_SIZE_ADAPTER = GeometryAdapter(
    position_field="center",
    position_len=3,
    x_span=SpanRef(("size", 0)),
    y_span=SpanRef(("size", 1)),
    z_span=SpanRef(("size", 2)),
)
for _kind in ("GaussianBeamSource", "ModeSource", "FieldMonitor", "FluxMonitor", "ModeMonitor"):
    ADAPTERS[_kind] = _CENTER_SIZE_ADAPTER
# Polygon and CustomSource intentionally have no adapter: Polygon's shape
# is arbitrary vertices with no clean x/y span, and CustomSource has an
# entirely different, lower-level field set. Both fall back to the plain
# per-field editor for every recipe field.


# ---------------------------------------------------------------------- #
def get_adapter(class_name: str) -> Optional[GeometryAdapter]:
    return ADAPTERS.get(class_name)


def _read_tuple_field(recipe: dict[str, Any], name: str, length: int) -> list[float]:
    raw = recipe.get(name) or (0.0,) * length
    values = list(raw) + [0.0] * (length - len(raw))
    return values[:length]


def get_xyz(recipe: dict[str, Any], adapter: GeometryAdapter) -> tuple[float, float, float]:
    """Displayed (always CENTER) x/y/z, converting from a stored corner
    where needed: `displayed = stored + current_span / 2`.
    """
    pos = _read_tuple_field(recipe, adapter.position_field, adapter.position_len)
    raw_x = pos[0]
    raw_y = pos[1] if adapter.position_len >= 2 else 0.0
    if adapter.position_len == 3:
        raw_z = pos[2]
    else:
        raw_z = recipe.get(adapter.z_field, 0.0) if adapter.z_field else 0.0
        raw_z = 0.0 if raw_z is None else raw_z

    x = raw_x + (get_span(recipe, adapter.x_span) or 0.0) / 2 if adapter.x_is_corner else raw_x
    y = raw_y + (get_span(recipe, adapter.y_span) or 0.0) / 2 if adapter.y_is_corner else raw_y
    z = raw_z + (get_span(recipe, adapter.z_span) or 0.0) / 2 if adapter.z_is_corner else raw_z
    return x, y, z


def _write_position_raw(
    adapter: GeometryAdapter,
    recipe: dict[str, Any],
    *,
    x: Optional[float] = None,
    y: Optional[float] = None,
    z: Optional[float] = None,
) -> dict[str, Any]:
    """Write STORED (not display-converted) position/z_field values.
    Internal helper — callers are responsible for any corner<->center
    conversion before calling this.
    """
    pos = _read_tuple_field(recipe, adapter.position_field, adapter.position_len)
    if x is not None:
        pos[0] = x
    if y is not None and adapter.position_len >= 2:
        pos[1] = y
    if adapter.position_len == 3 and z is not None:
        pos[2] = z

    changes: dict[str, Any] = {adapter.position_field: tuple(pos)}
    if adapter.position_len == 2 and adapter.z_field and z is not None:
        changes[adapter.z_field] = z
    if adapter.sync_z_field and z is not None:
        changes["z"] = z
    return changes


def xyz_changes(
    recipe: dict[str, Any],
    adapter: GeometryAdapter,
    *,
    x: Optional[float] = None,
    y: Optional[float] = None,
    z: Optional[float] = None,
) -> dict[str, Any]:
    """Partial dict of recipe changes for an edit to one or more of the
    DISPLAYED (center) x/y/z. Converts each edited axis back to a stored
    corner where needed, using the CURRENT span on that axis (moving the
    center this way does not itself resize anything — the span is
    unchanged, only the corner shifts to keep that span centered on the
    new position).
    """
    displayed = {"x": x, "y": y, "z": z}
    is_corner = {"x": adapter.x_is_corner, "y": adapter.y_is_corner, "z": adapter.z_is_corner}
    span = {"x": adapter.x_span, "y": adapter.y_span, "z": adapter.z_span}

    raw: dict[str, float] = {}
    for axis, value in displayed.items():
        if value is None:
            continue
        if is_corner[axis]:
            current_span = get_span(recipe, span[axis]) or 0.0
            raw[axis] = value - current_span / 2
        else:
            raw[axis] = value

    return _write_position_raw(adapter, recipe, **raw)


def get_span(recipe: dict[str, Any], span: Optional[SpanRef]) -> Optional[float]:
    if span is None:
        return None
    if isinstance(span.field, tuple):
        name, idx = span.field
        values = _read_tuple_field(recipe, name, idx + 1)
        raw = values[idx]
    else:
        raw = recipe.get(span.field, 0.0)
        raw = 0.0 if raw is None else raw
    return raw * span.scale


_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def span_changes(recipe: dict[str, Any], adapter: GeometryAdapter, axis: str, value: float) -> dict[str, Any]:
    """Partial dict of recipe changes for an edit to one span value
    (`axis` is "x"/"y"/"z", selecting `adapter.x_span` etc). For a
    corner-based axis, ALSO recomputes the stored corner so the CENTER
    stays fixed while the span changes — matching Lumerical's behavior,
    and fixing a real reported bug where editing e.g. x span on a
    Rectangle silently dragged its center away from x=0, because the
    corner was left untouched while only `width` grew.
    """
    span = {"x": adapter.x_span, "y": adapter.y_span, "z": adapter.z_span}[axis]
    is_corner = {"x": adapter.x_is_corner, "y": adapter.y_is_corner, "z": adapter.z_is_corner}[axis]

    field_value = value / span.scale
    if isinstance(span.field, tuple):
        name, idx = span.field
        # Read the tuple's CURRENT full length, not just up to `idx` —
        # truncating it here would silently drop other components (a
        # real bug found and fixed during development: editing a Box's
        # y_span this way zeroed out its z_span).
        current = list(recipe.get(name) or ())
        while len(current) <= idx:
            current.append(0.0)
        current[idx] = field_value
        changes: dict[str, Any] = {name: tuple(current)}
    else:
        changes = {span.field: field_value}

    if is_corner:
        center_before = get_xyz(recipe, adapter)[_AXIS_INDEX[axis]]
        new_corner = center_before - value / 2
        changes.update(_write_position_raw(adapter, recipe, **{axis: new_corner}))

    return changes