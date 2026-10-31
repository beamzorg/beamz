from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from ..schema import (
    ExtrudedPolygon,
    Material,
    Object,
    Polygon,
    Scene,
    TaperedExtrudedPolygon,
)

_BUILTIN_MATERIALS_NEAR_1550_NM = {
    "air": Material(1.0),
    "si": Material(3.48**2),
    "silicon": Material(3.48**2),
    "sio2": Material(1.444**2),
    "oxide": Material(1.444**2),
    "sin": Material(2.0**2),
    "si3n4": Material(2.0**2),
}


def _polygons_by_layer(component: Any) -> Mapping[tuple[int, int], Any]:
    try:
        return component.get_polygons_points(by="tuple")
    except TypeError:
        return component.get_polygons_points(by_spec=True)


def _region_polygons(
    component: Any,
    level: Any,
    polygons_by_layer: Mapping[tuple[int, int], Any],
    unit_scale: float,
) -> list[Polygon]:
    layer_expression = getattr(level, "layer", None)
    get_shapes = getattr(layer_expression, "get_shapes", None)
    if callable(get_shapes):
        region: Any = get_shapes(component)
        database_unit = float(component.kcl.dbu)
        scale = database_unit * unit_scale
        result = []
        for polygon in region.each():
            exterior = tuple(
                (float(point.x) * scale, float(point.y) * scale)
                for point in polygon.each_point_hull()
            )
            holes = tuple(
                tuple(
                    (float(point.x) * scale, float(point.y) * scale)
                    for point in polygon.each_point_hole(hole)
                )
                for hole in range(polygon.holes())
            )
            result.append(Polygon(exterior, holes))
        return result

    layer = (
        (int(layer_expression[0]), int(layer_expression[1]))
        if isinstance(layer_expression, (tuple, list)) and len(layer_expression) == 2
        else None
    )
    if layer is None or layer not in polygons_by_layer:
        return []
    return [
        Polygon(
            tuple(
                (float(x) * unit_scale, float(y) * unit_scale)
                for x, y in np.asarray(points, dtype=float)[:, :2]
            )
        )
        for points in polygons_by_layer[layer]
    ]


def _z_to_bias_profile(value: Any) -> tuple[np.ndarray, np.ndarray]:
    """Normalize tabulated bias or approximate a callable with fixed policy."""
    if callable(value):
        samples: dict[float, float] = {}
        knots = {0.0, 1.0}

        def evaluate(z: float) -> float:
            if z not in samples:
                bias = float(cast(Any, value(float(z))))
                if not np.isfinite(bias):
                    raise ValueError("z_to_bias callable returned a non-finite bias.")
                samples[z] = bias
            return samples[z]

        def refine(z0: float, z1: float, depth: int) -> None:
            b0, b1 = evaluate(z0), evaluate(z1)
            probes = (
                z0 + 0.25 * (z1 - z0),
                z0 + 0.5 * (z1 - z0),
                z0 + 0.75 * (z1 - z0),
            )
            error = max(
                abs(evaluate(z) - (b0 + (z - z0) / (z1 - z0) * (b1 - b0)))
                for z in probes
            )
            if error <= 1e-3:
                return
            if depth >= 10:
                raise ValueError(
                    "z_to_bias callable did not converge within the importer policy."
                )
            middle = probes[1]
            knots.add(middle)
            refine(z0, middle, depth + 1)
            refine(middle, z1, depth + 1)

        refine(0.0, 1.0, 0)
        z = np.asarray(sorted(knots), dtype=float)
        bias = np.asarray([samples[value] for value in z], dtype=float)
        return z, bias

    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("z_to_bias must be a (normalized_z, bias) pair or callable.")
    z = np.asarray(value[0], dtype=float)
    bias = np.asarray(value[1], dtype=float)
    if z.ndim != 1 or bias.ndim != 1 or len(z) < 2 or len(z) != len(bias):
        raise ValueError(
            "z_to_bias coordinate and bias arrays must have equal length >= 2."
        )
    if (
        not np.isfinite(z).all()
        or not np.isfinite(bias).all()
        or np.any(np.diff(z) <= 0.0)
        or not np.isclose(z[0], 0.0)
        or not np.isclose(z[-1], 1.0)
        or z[0] < 0.0
        or z[-1] > 1.0
    ):
        raise ValueError(
            "z_to_bias coordinates must be finite, strictly increasing, "
            "and span normalized z=0 through z=1."
        )
    return z, bias


def _offset_polygon(polygon: Polygon, distance: float) -> list[Polygon]:
    if distance == 0.0:
        return [polygon]
    from shapely.geometry import Polygon as ShapelyPolygon

    buffered = ShapelyPolygon(polygon.exterior, polygon.holes).buffer(distance)
    geometries = getattr(buffered, "geoms", (buffered,))
    return [
        Polygon(
            tuple((float(x), float(y)) for x, y in geometry.exterior.coords[:-1]),
            tuple(
                tuple((float(x), float(y)) for x, y in ring.coords[:-1])
                for ring in geometry.interiors
            ),
        )
        for geometry in geometries
        if not geometry.is_empty
    ]


def _level_profile(level, name, downward, unit_scale):
    bias_value = getattr(level, "bias", 0.0)
    if bias_value is not None and not np.isscalar(bias_value):
        raise ValueError(f"Layer {name!r}: only scalar bias is supported.")
    mask_bias = float(cast(Any, bias_value) or 0.0) * unit_scale
    sidewall = float(getattr(level, "sidewall_angle", 0.0) or 0.0)
    reference = float(getattr(level, "width_to_z", 0.0) or 0.0)
    if (
        not np.isfinite([mask_bias, sidewall, reference]).all()
        or abs(sidewall) >= 89
        or not 0 <= reference <= 1
    ):
        raise ValueError(f"Layer {name!r}: invalid bias, sidewall angle or width_to_z.")
    if downward:
        sidewall, reference = -sidewall, 1.0 - reference
    profile = getattr(level, "z_to_bias", None)
    sampled = callable(profile)
    if profile is not None:
        try:
            normalized_z, bias = _z_to_bias_profile(profile)
        except ValueError as exc:
            raise ValueError(f"Layer {name!r}: {exc}") from exc
        # GDSFactory's default zero profile must not disable sidewalls.
        if np.any(bias):
            if sidewall:
                raise ValueError(
                    f"Layer {name!r}: specify sidewall_angle or z_to_bias, not both."
                )
            if downward:
                normalized_z, bias = 1 - normalized_z[::-1], bias[::-1]
        else:
            profile = None
    return (
        mask_bias,
        sidewall,
        reference,
        tuple(tuple(values) for values in (normalized_z, bias))
        if profile is not None
        else None,
        sampled,
    )


def _resolve_levels(
    levels,
    *,
    material_map: Mapping[str, Material | float] | None = None,
    use_builtin_materials: bool = False,
    background: Material | None = None,
    unit_scale: float = 1e-6,
    bounds: tuple[Sequence[float], Sequence[float]] | None = None,
    _padding: tuple[float, float] | None = None,
) -> tuple[Scene, tuple[Mapping[str, Any], ...], Any]:
    """Convert a GDSFactory component and PDK layer stack.

    GDSFactory layout and layer-stack coordinates are conventionally expressed
    in micrometres, hence the default ``unit_scale=1e-6``. PDK material names
    must be resolved through ``material_map``. Set ``use_builtin_materials=True``
    only to opt into nondispersive Si/SiO2/SiN approximations near 1.55 µm.
    """

    unit_scale = float(unit_scale)
    if not np.isfinite(unit_scale) or unit_scale <= 0.0:
        raise ValueError("unit_scale must be finite and positive.")
    if not isinstance(use_builtin_materials, (bool, np.bool_)):
        raise TypeError("use_builtin_materials must be a boolean.")
    background = Material() if background is None else background
    material_lookup = dict(
        _BUILTIN_MATERIALS_NEAR_1550_NM if use_builtin_materials else {}
    )
    material_lookup.update(
        {
            str(name).strip().lower(): (
                value if isinstance(value, Material) else Material(float(value))
            )
            for name, value in (material_map or {}).items()
        }
    )

    materials = [background]
    material_ids = {background: 0}

    def add_material(material: Material) -> int:
        if material not in material_ids:
            material_ids[material] = len(materials)
            materials.append(material)
        return material_ids[material]

    bounds = _validate_bounds(bounds)
    levels = {level.name: level for level in levels}
    # Rank original levels once: low mesh_order/later ties win; users stay above the stack.
    level_order = {
        name: rank
        for rank, name in enumerate(
            sorted(levels, key=lambda n: -_mesh_order(levels[n]))
        )
    }
    objects = []
    reports = []
    object_id = 1

    def is_fill(name):
        return levels[name].background

    # Resolve finite patterned geometry first so native imports can infer bounds.
    for level_name in sorted(levels, key=is_fill):
        level = levels[level_name]
        if is_fill(level_name) and bounds is None and _padding is not None:
            bounds = _geometry_bounds(objects, _padding)

        z_min, z_max = level.z_bounds
        record = dict(
            name=str(level_name),
            z_bounds=(z_min, z_max),
            material=str(getattr(level, "material", "")),
            object_ids=(),
        )

        def skip(reason, record=record):
            reports.append(MappingProxyType({**record, "status": reason}))

        if not np.isfinite([z_min, z_max]).all():
            raise ValueError(
                f"Layer {level_name!r}: zmin and thickness must be finite."
            )
        if z_min == z_max:
            skip("zero thickness")
            continue
        if bounds is not None and (z_max <= bounds[0][2] or z_min >= bounds[1][2]):
            skip("outside bounds")
            continue
        fill = level.background
        if fill:
            if bounds is None:
                raise ValueError(
                    f"Layer {level_name!r}: background fills require bounds."
                )
            level_polygons = []  # resolved after the physical profile below
        else:
            level_polygons = level.polygons
        if not level_polygons and not fill:
            skip("empty mask")
            continue
        mask_bias, sidewall, reference, profile, sampled = level.profile
        if profile is not None:
            normalized_z, bias = profile
        margin = abs(mask_bias) + (z_max - z_min) * abs(np.tan(np.radians(sidewall)))
        if profile is not None:
            margin += float(np.max(np.abs(bias))) * unit_scale
        if fill:
            level_polygons = _fill_polygons(level.exclusions, bounds, margin)
            if not level_polygons:
                skip("empty mask")
                continue
        if bounds is not None and not _intersects_bounds(
            level_polygons, bounds, margin
        ):
            skip("outside bounds")
            continue
        material_name = str(getattr(level, "material", "")).strip().lower()
        material_id = None
        first_id = object_id
        mesh_order = _mesh_order(level)
        for polygon in level_polygons:
            if profile is not None:
                geometries = []
                for index in range(len(normalized_z) - 1):
                    segment_min = z_min + normalized_z[index] * (z_max - z_min)
                    segment_max = z_min + normalized_z[index + 1] * (z_max - z_min)
                    low_bias = mask_bias + bias[index] * unit_scale
                    high_bias = mask_bias + bias[index + 1] * unit_scale
                    angle = np.degrees(
                        np.arctan2(-(high_bias - low_bias), segment_max - segment_min)
                    )
                    for offset in _offset_polygon(polygon, low_bias):
                        geometries.append(
                            ExtrudedPolygon(offset, segment_min, segment_max)
                            if low_bias == high_bias
                            else TaperedExtrudedPolygon(
                                offset,
                                segment_min,
                                segment_max,
                                sidewall_angle_degrees=float(angle),
                            )
                        )
            else:
                geometries = [
                    TaperedExtrudedPolygon(
                        polygon,
                        z_min,
                        z_max,
                        sidewall_angle_degrees=sidewall,
                        width_to_z=reference,
                    )
                    if sidewall != 0.0
                    else ExtrudedPolygon(polygon, z_min, z_max)
                    for polygon in _offset_polygon(polygon, mask_bias)
                ]
            for geometry in geometries:
                if material_id is None:
                    if material_name not in material_lookup:
                        raise ValueError(
                            f"No BeamZ raster material is configured for PDK material "
                            f"{material_name!r} on layer {level_name!r}. Pass material_map=... "
                            "with the intended wavelength-dependent model, or explicitly set "
                            "use_builtin_materials=True for approximate 1.55 µm constants."
                        )
                    material_id = add_material(material_lookup[material_name])
                objects.append(
                    Object(
                        id=object_id,
                        material_id=material_id,
                        priority=level_order[level_name] - len(levels),
                        geometry=geometry,
                    )
                )
                object_id += 1
        if object_id == first_id:
            skip("empty after bias")
            continue
        reports.append(
            MappingProxyType(
                {
                    **record,
                    "status": "included",
                    "mesh_order": mesh_order,
                    "object_ids": tuple(range(first_id, object_id)),
                    "bias_tolerance": 1e-3 * unit_scale if sampled else 0.0,
                    "profile": "sampled callable"
                    if sampled
                    else "tabulated"
                    if profile is not None
                    else "sidewall",
                }
            )
        )

    if not objects:
        raise ValueError("No physical layer-stack levels matched component polygons.")
    level_index = {name: index for index, name in enumerate(levels)}
    if bounds is None and _padding is not None:
        bounds = _geometry_bounds(objects, _padding)
    reports.sort(key=lambda report: level_index[report["name"]])
    return Scene(tuple(materials), tuple(objects), 0), tuple(reports), bounds


def _validate_bounds(bounds):
    if bounds is None:
        return None
    values = np.asarray(bounds, dtype=float)
    if (
        values.shape != (2, 3)
        or not np.isfinite(values).all()
        or np.any(values[1] <= values[0])
    ):
        raise ValueError(
            "bounds must be finite increasing ((xmin, ymin, zmin), (xmax, ymax, zmax)) in metres."
        )
    return (tuple(float(x) for x in values[0]), tuple(float(x) for x in values[1]))


def _fill_polygons(exclusions, bounds, profile_margin):
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.geometry import box
    from shapely.ops import unary_union

    # The outside rectangle is a computational boundary, not a tapered material
    # interface. Extend it beyond the domain before applying an exclusion profile.
    margin = max(np.subtract(bounds[1], bounds[0])) * 2 + profile_margin
    area = box(
        bounds[0][0] - margin,
        bounds[0][1] - margin,
        bounds[1][0] + margin,
        bounds[1][1] + margin,
    )
    excluded = [ShapelyPolygon(p.exterior, p.holes) for p in exclusions]
    if excluded:
        area = area.difference(unary_union(excluded))
    return _shapely_polygons(area)


def _shapely_polygons(area):
    return [
        Polygon(
            tuple(p.exterior.coords)[:-1],
            tuple(tuple(r.coords)[:-1] for r in p.interiors),
        )
        for p in getattr(area, "geoms", (area,))
        if p.geom_type == "Polygon" and not p.is_empty
    ]


def from_gdsfactory(
    component: Any,
    layer_stack: Any,
    *,
    material_map: Mapping[str, Material | float] | None = None,
    use_builtin_materials: bool = False,
    background: Material | None = None,
    unit_scale: float = 1e-6,
    bounds: tuple[Sequence[float], Sequence[float]] | None = None,
    fill_levels: Sequence[str] = (),
) -> Scene:
    """Resolve each physical LayerStack level into metre-valued raster geometry.

    ``bounds`` sets the domain used by background fills and skips outside levels;
    source geometry is retained so a crop never changes a sidewall reference.
    Rasterize on a grid within these bounds. ``fill_levels`` selects full-domain
    slabs in older PDKs; background metadata and exclusions are also supported.
    Materials are explicit unless approximate 1.55 µm constants are opted into.
    """
    return _resolve_levels(
        _read_stack(component, layer_stack, unit_scale, fill_levels),
        material_map=material_map,
        use_builtin_materials=use_builtin_materials,
        background=background,
        unit_scale=unit_scale,
        bounds=bounds,
    )[0]


def _geometry_bounds(objects, padding):
    if not objects:
        raise ValueError(
            "Cannot infer bounds without patterned geometry; pass bounds explicitly."
        )
    lower, upper = [], []
    for obj in objects:
        g = obj.geometry
        xy = np.asarray(g.polygon.exterior)
        growth = 0.0
        if isinstance(g, TaperedExtrudedPolygon):
            h = g.z_max - g.z_min
            slope = np.tan(np.radians(g.sidewall_angle_degrees))
            growth = max(0.0, h * g.width_to_z * slope, -h * (1 - g.width_to_z) * slope)
        lower.append((*(xy.min(axis=0) - growth), g.z_min))
        upper.append((*(xy.max(axis=0) + growth), g.z_max))
    pad = np.array([padding[0], padding[0], padding[1]])
    return _validate_bounds((np.min(lower, axis=0) - pad, np.max(upper, axis=0) + pad))


def _intersects_bounds(polygons, bounds, margin):
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.geometry import box

    domain = box(*bounds[0][:2], *bounds[1][:2])
    for polygon in polygons:
        area = ShapelyPolygon(polygon.exterior, polygon.holes)
        # Conservative envelope across the whole taper/profile, including holes.
        if margin:
            area = area.buffer(margin, join_style="mitre")
        if area.intersection(domain).area > 0:
            return True
    return False


def _mesh_order(level):
    return int(
        (getattr(level, "info", None) or {}).get(
            "mesh_order", getattr(level, "mesh_order", 3)
        )
    )


def _exclusion_polygons(component, level, polygons, unit_scale):
    from types import SimpleNamespace

    result = []
    for layer in getattr(level, "background_exclude_layers", ()):
        from gdsfactory.technology import LogicalLayer

        expression = (
            layer if hasattr(layer, "get_shapes") else LogicalLayer(layer=layer)
        )
        result.extend(
            _region_polygons(
                component, SimpleNamespace(layer=expression), polygons, unit_scale
            )
        )
    return tuple(result)


@dataclass(frozen=True)
class _Layer:
    name: str
    polygons: tuple
    exclusions: tuple | None
    port_layers: frozenset
    z_bounds: tuple[float, float]
    material: str
    mesh_order: int
    profile: tuple

    @property
    def background(self):
        return self.exclusions is not None


def _read_stack(component, stack, unit_scale=1e-6, fill_levels=()):
    unit_scale = float(unit_scale)
    if not np.isfinite(unit_scale) or unit_scale <= 0:
        raise ValueError("unit_scale must be finite and positive.")
    levels = getattr(stack, "layers", {})
    if unknown := set(fill_levels) - set(levels):
        raise ValueError(f"Unknown fill_levels: {sorted(unknown)}.")
    polygons = _polygons_by_layer(component)
    result = []
    for name, level in levels.items():
        fill = name in fill_levels or bool(getattr(level, "background", False))
        masks = (
            ()
            if fill
            else tuple(_region_polygons(component, level, polygons, unit_scale))
        )
        z_start = float(level.zmin) * unit_scale
        z_end = z_start + float(level.thickness) * unit_scale
        profile = (
            _level_profile(level, name, z_end < z_start, unit_scale)
            if masks or fill
            else (0, 0, 0, None, False)
        )
        result.append(
            _Layer(
                str(name),
                masks,
                _exclusion_polygons(component, level, polygons, unit_scale)
                if fill
                else None,
                frozenset(
                    _source_layers(getattr(level, "layer", None))
                    | _source_layers(getattr(level, "derived_layer", None))
                ),
                (min(z_start, z_end), max(z_start, z_end)),
                str(getattr(level, "material", "")),
                _mesh_order(level),
                profile,
            )
        )
    return tuple(result)


def _source_layers(expression):
    if expression is None:
        return set()
    if isinstance(expression, (tuple, list)) and len(expression) == 2:
        return {tuple(expression)}
    if isinstance(expression, (int, str)):
        import gdsfactory as gf

        return {tuple(gf.get_layer_tuple(cast(Any, expression)))}
    if (layer := getattr(expression, "layer", None)) is not None:
        return _source_layers(layer)
    layers = _source_layers(getattr(expression, "layer1", None))
    if getattr(expression, "operation", None) != "not":
        layers |= _source_layers(getattr(expression, "layer2", None))
    return layers
