"""Shared provenance and layout helpers for passive-SOI benchmarks."""

from __future__ import annotations

import hashlib
import json
import warnings
from contextlib import contextmanager
from functools import cache, partial
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from tests.differential.case_schema import DifferentialCase, load_case

CASE_DIR = Path(__file__).parents[1] / "cases"


@contextmanager
def reference_absorber_warning_scope():
    """Allow the paper's intentional waveguide-through-absorber geometry."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Absorber material varies along the absorber normal.*",
            category=RuntimeWarning,
        )
        yield


@cache
def _gdsfactory():
    # GDSFactory 8.18 constructs ``TemporaryDirectory().name`` during import,
    # immediately dropping its owner and emitting this otherwise harmless warning.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Implicitly cleaning up <TemporaryDirectory.*",
            category=ResourceWarning,
        )
        import gdsfactory as gf

    return gf


@cache
def _layout_pdk():
    """Build the generic layer and cross-section definitions used by these cases."""
    gf = _gdsfactory()

    class PassiveSoiLayers(gf.LayerEnum):
        layout = gf.constant(gf.kcl.layout)
        WG: tuple[int, int] = (1, 0)
        SLAB150: tuple[int, int] = (2, 0)

    return gf.Pdk(
        name="beamz_passive_soi_benchmark",
        layers=PassiveSoiLayers,
        cross_sections={
            "strip": partial(
                gf.cross_section.cross_section,
                width=0.5,
                layer="WG",
            )
        },
    )


def load_passive_soi_case(name: str) -> DifferentialCase:
    """Load a named passive-SOI case from its solver-neutral specification."""
    return load_case(CASE_DIR / f"passive_soi_{name}.json")


def domain_bounds_um(case: DifferentialCase) -> dict[str, tuple[float, float]]:
    """Return simulation bounds in the case's layer-stack coordinate system."""
    raw = case.geometry["simulation"]["domain_bounds_um"]
    return {
        axis: (float(raw[axis][0]), float(raw[axis][1])) for axis in ("x", "y", "z")
    }


def domain_size_um(case: DifferentialCase) -> tuple[float, float, float]:
    """Return domain extents derived from the provenance-rich physical bounds."""
    bounds = domain_bounds_um(case)
    return tuple(bounds[axis][1] - bounds[axis][0] for axis in ("x", "y", "z"))


def max_spectrum_difference(
    first_wavelengths_um: tuple[float, ...],
    first_values: tuple[float, ...],
    second_wavelengths_um: tuple[float, ...],
    second_values: tuple[float, ...],
) -> float:
    """Return the largest linearly interpolated difference on the common band."""
    first_wavelengths = np.asarray(first_wavelengths_um, dtype=float)
    second_wavelengths = np.asarray(second_wavelengths_um, dtype=float)
    first_spectrum = np.asarray(first_values, dtype=float)
    second_spectrum = np.asarray(second_values, dtype=float)
    if (
        first_wavelengths.size != first_spectrum.size
        or second_wavelengths.size != second_spectrum.size
    ):
        raise ValueError("each spectrum must match its wavelength samples")
    first_order = np.argsort(first_wavelengths)
    second_order = np.argsort(second_wavelengths)
    first_wavelengths = first_wavelengths[first_order]
    second_wavelengths = second_wavelengths[second_order]
    first_spectrum = first_spectrum[first_order]
    second_spectrum = second_spectrum[second_order]
    lower = max(float(first_wavelengths[0]), float(second_wavelengths[0]))
    upper = min(float(first_wavelengths[-1]), float(second_wavelengths[-1]))
    if lower > upper:
        raise ValueError("spectra have no overlapping wavelength interval")
    samples = np.unique(
        np.concatenate(
            (
                first_wavelengths[
                    (first_wavelengths >= lower) & (first_wavelengths <= upper)
                ],
                second_wavelengths[
                    (second_wavelengths >= lower) & (second_wavelengths <= upper)
                ],
            )
        )
    )
    first_interpolated = np.interp(samples, first_wavelengths, first_spectrum)
    second_interpolated = np.interp(samples, second_wavelengths, second_spectrum)
    return float(np.max(np.abs(first_interpolated - second_interpolated)))


def reference_frequencies(
    case: DifferentialCase, wavelength_span_nm: float
) -> np.ndarray:
    """Return the paper pipeline's uniformly spaced frequency samples."""
    from beamz import LIGHT_SPEED, µm

    protocol = case.geometry["simulation"]
    wavelength_center = float(protocol["wavelength_center_um"]) * µm
    half_span = 0.5 * float(wavelength_span_nm) * 1e-3 * µm
    wavelength_step = float(protocol["wavelength_step_nm"]) * 1e-3 * µm
    sample_count = round(2.0 * half_span / wavelength_step) + 1
    return np.linspace(
        LIGHT_SPEED / (wavelength_center + half_span),
        LIGHT_SPEED / (wavelength_center - half_span),
        num=sample_count,
    )


def port_center_and_direction(
    case: DifferentialCase,
    name: str,
    *,
    inward_offset_um: float,
    z_center: float,
) -> tuple[tuple[float, float, float], str]:
    """Translate a reference GDS port and optionally move it into the device."""
    from beamz import µm

    bounds = domain_bounds_um(case)
    port = case.geometry["ports"][name]
    x_um, y_um = (float(value) for value in port["center_um"])
    orientation = int(round(float(port["orientation_deg"]))) % 360
    if orientation == 180:
        x_um += float(inward_offset_um)
        direction = "+"
    elif orientation == 0:
        x_um -= float(inward_offset_um)
        direction = "-"
    elif orientation == 90:
        y_um -= float(inward_offset_um)
        direction = "-"
    elif orientation == 270:
        y_um += float(inward_offset_um)
        direction = "+"
    else:
        raise ValueError(f"unsupported port orientation {orientation}")
    return (
        (
            (x_um - bounds["x"][0]) * µm,
            (y_um - bounds["y"][0]) * µm,
            z_center,
        ),
        direction,
    )


@contextmanager
def _layout_pdk_scope():
    """Temporarily activate the benchmark PDK for GDSFactory operations."""
    gf = _gdsfactory()
    # get_active_pdk() initializes a generic PDK when none is active. Preserve
    # that uninitialized state too; GDSFactory has no public non-initializing
    # getter or deactivation API.
    previous_pdk = gf.pdk._ACTIVE_PDK
    try:
        _layout_pdk().activate()
        yield
    finally:
        if previous_pdk is None:
            gf.pdk._ACTIVE_PDK = None
        else:
            previous_pdk.activate()


@cache
def _generate_layout(component_name: str, settings_json: str) -> Any:
    gf = _gdsfactory()
    with _layout_pdk_scope():
        try:
            factory = getattr(gf.components, component_name)
        except AttributeError as error:
            raise ValueError(
                f"GDSFactory has no generic component {component_name!r}"
            ) from error
        return factory(**json.loads(settings_json))


def generate_layout(case: DifferentialCase) -> Any:
    """Generate a case layout using the explicit passive-SOI PDK."""
    geometry = case.geometry
    if geometry.get("kind") == "reference_polygons":
        fixture = Path(__file__).parent / "layouts" / geometry["polygon_fixture"]
        raw = fixture.read_bytes()
        if hashlib.sha256(raw).hexdigest() != geometry["polygon_fixture_sha256"]:
            raise ValueError(f"reference polygon checksum mismatch for {case.name}")
        payload = json.loads(raw)
        gf = _gdsfactory()
        with _layout_pdk_scope():
            component = gf.Component()
            for points in payload["polygons_um"]:
                component.add_polygon(points, layer=tuple(payload["layer"]))
            for name, port in geometry["ports"].items():
                component.add_port(
                    name=name,
                    center=tuple(port["center_um"]),
                    width=port["width_um"],
                    orientation=port["orientation_deg"],
                    layer=tuple(payload["layer"]),
                )
        return component
    if geometry.get("kind") != "gdsfactory_component":
        raise ValueError(f"case {case.name!r} is not a GDSFactory component")
    settings_json = json.dumps(geometry.get("settings", {}), sort_keys=True)
    return _generate_layout(str(geometry["component"]), settings_json)


def write_layout_gds(case: DifferentialCase, path: str | Path) -> Path:
    """Generate and serialize a case layout for a solver adapter."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    component = generate_layout(case)
    with _layout_pdk_scope():
        return Path(component.write_gds(gdspath=destination, with_metadata=True))


def component_polygons(component: Any, layer: tuple[int, int]):
    """Read physical polygons without leaking GDSFactory's layer lookup PDK."""
    with _layout_pdk_scope():
        polygons = component.get_polygons_points(by="tuple").get(tuple(layer), ())
    if not polygons:
        raise ValueError(f"component {component.name!r} has no layer {tuple(layer)}")
    return polygons


def layer_union_sha256(component: Any, layer: tuple[int, int]) -> str:
    """Hash the normalized geometric union on one GDS layer.

    The paper's GDSFactory 7.8 artifact stores overlapping arms separately,
    whereas newer GDSFactory versions merge them. Hashing the normalized union
    makes the fingerprint sensitive to physical geometry rather than harmless
    polygon partitioning or ring orientation.
    """
    from shapely import normalize
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    polygons = component_polygons(component, layer)
    geometry = normalize(
        unary_union(
            [
                Polygon(np.asarray(points, dtype=np.float64)[:, :2])
                for points in polygons
            ]
        )
    )
    return hashlib.sha256(geometry.wkb).hexdigest()


def expected_layer_fingerprints(
    case: DifferentialCase,
) -> Mapping[tuple[int, int], str]:
    """Return the physical layer fingerprints pinned by a case specification."""
    layers = case.geometry.get("layers", {})
    return {
        tuple(int(value) for value in layer["gds"]): str(layer["union_sha256"])
        for layer in layers.values()
    }
