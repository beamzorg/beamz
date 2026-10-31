"""Small, synthetic process stacks; layout and process dimensions are in µm.

Material constants illustrate import semantics, not a calibrated foundry PDK.
No global PDK activation or layout-grid mutation happens at import time.
"""

from dataclasses import dataclass
from typing import Any

import gdsfactory as gf
from gdsfactory.technology import LayerLevel, LayerStack, LogicalLayer

from beamz import Material

UM = 1e-6
MATERIALS = {
    "si": Material(12.0),
    "sin": Material(4.0),
    "sio2": Material(2.25),
    # Positive-epsilon lossy dielectric proxy, deliberately not an optical metal.
    "heater_proxy": Material(6.0, conductivity=100.0),
}


@dataclass(frozen=True)
class Probe:
    name: str
    xyz_um: tuple[float, float, float]
    epsilon: float


@dataclass(frozen=True)
class Case:
    component: Any
    stack: Any
    bounds_um: tuple[tuple[float, float, float], tuple[float, float, float]]
    probes: tuple[Probe, ...]
    description: str


def rectangle(component, layer, x0, y0, x1, y1):
    component.add_polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], layer=layer)


def rib_etch() -> Case:
    """Two physical levels share one mask; a derived etch leaves a rib and slab."""
    c = gf.Component()
    rectangle(c, (1, 0), 0, 0, 4, 2)
    rectangle(c, (2, 0), 0, 0, 4, 0.6)
    rectangle(c, (2, 0), 0, 1.4, 4, 2)
    wg, etch = LogicalLayer(layer=(1, 0)), LogicalLayer(layer=(2, 0))
    stack = LayerStack(
        layers={
            "slab": LayerLevel(
                layer=wg, zmin=0, thickness=0.09, material="si", mesh_order=3
            ),
            "core": LayerLevel(
                layer=wg - etch,
                derived_layer=LogicalLayer(layer=(101, 0)),
                zmin=0,
                thickness=0.30,
                material="si",
                mesh_order=1,
                sidewall_angle=10,
                width_to_z=0.5,
            ),
        }
    )
    c.add_port("o1", center=(0, 1), width=0.8, orientation=180, layer=(1, 0))
    c.add_port("o2", center=(4, 1), width=0.8, orientation=0, layer=(1, 0))
    return Case(
        c,
        stack,
        ((-0.2, -0.2, -0.1), (4.2, 2.2, 0.5)),
        (
            Probe("rib", (2, 1, 0.20), 12),
            Probe("remaining slab", (2, 0.3, 0.04), 12),
            Probe("etched above slab", (2, 0.3, 0.20), 1),
        ),
        "Derived shallow etch and overlapping silicon rib/slab with 10 degree sidewalls.",
    )


def stacked_heater() -> Case:
    """Patterned silicon, offset upper SiN and a lossy heater proxy."""
    c = gf.Component()
    rectangle(c, (1, 0), 0, 0.8, 4, 1.2)
    rectangle(c, (3, 0), 0, 1.0, 4, 1.6)
    rectangle(c, (4, 0), 1, 0.5, 3, 1.5)
    rectangle(c, (5, 0), 0, 0, 4, 2)
    stack = LayerStack(
        layers={
            "oxide": LayerLevel(
                layer=(5, 0), zmin=-0.2, thickness=1.5, material="sio2", mesh_order=10
            ),
            "core": LayerLevel(
                layer=(1, 0),
                zmin=0,
                thickness=0.22,
                material="si",
                mesh_order=1,
                sidewall_angle=8,
            ),
            "upper": LayerLevel(
                layer=(3, 0),
                zmin=0.55,
                thickness=0.30,
                material="sin",
                mesh_order=2,
                sidewall_angle=-6,
                width_to_z=1,
            ),
            "heater": LayerLevel(
                layer=(4, 0),
                zmin=1.0,
                thickness=0.1,
                material="heater_proxy",
                mesh_order=1,
            ),
        }
    )
    return Case(
        c,
        stack,
        ((-0.2, -0.2, -0.3), (4.2, 2.2, 1.4)),
        (
            Probe("silicon wins oxide overlap", (2, 1, 0.1), 12),
            Probe("upper nitride", (2, 1.3, 0.7), 4),
            Probe("heater proxy", (2, 1, 1.05), 6),
            Probe("beside heater", (0.5, 1, 1.05), 2.25),
            Probe("oxide only", (2, 0.3, 0.4), 2.25),
            Probe("negative angle narrows below top reference", (2, 1.01, 0.60), 2.25),
            Probe("negative angle wider near top reference", (2, 1.01, 0.82), 4),
        ),
        "Four patterned physical levels, positive/negative taper and overlap precedence.",
    )


def sidewall_references() -> Case:
    """Same source layer at several heights, including width_to_z=0.37."""
    c = gf.Component()
    rectangle(c, (1, 0), 0, 0, 4, 1)
    layers = {}
    probes = []
    for index, reference in enumerate((0.0, 0.37, 0.5, 1.0)):
        zmin = index * 0.8
        layers[f"reference_{reference}"] = LayerLevel(
            layer=(1, 0),
            zmin=zmin,
            thickness=0.5,
            material="si",
            sidewall_angle=20,
            width_to_z=reference,
        )
        probes.append(Probe(f"inside reference {reference}", (2, 0.5, zmin + 0.25), 12))
    # For positive 20 degree taper the top edge moves inward by h*tan(20).
    # At z=0.45 this excludes y=0.10 for a bottom reference, while a
    # top reference at z=2.9 makes that same point remain inside.
    probes.extend(
        (
            Probe("bottom reference narrows", (2, 0.1, 0.45), 1),
            Probe("top reference remains wide", (2, 0.1, 2.85), 12),
            Probe("0.37 reference differs from midpoint", (2, 0.01, 1.05), 1),
            Probe("midpoint preserves mask at midpoint", (2, 0.01, 1.85), 12),
        )
    )
    return Case(
        c,
        LayerStack(layers=layers),
        ((-0.2, -0.3, -0.1), (4.2, 1.3, 3.1)),
        tuple(probes),
        "Continuous sidewall reference heights without rounding to three faces.",
    )


def bias_profiles() -> Case:
    """Constant mask bias, tabulated profile and adaptively sampled callable."""
    c = gf.Component()
    rectangle(c, (1, 0), 0, 0, 4, 1)
    stack = LayerStack(
        layers={
            "constant": LayerLevel(
                layer=(1, 0), zmin=0, thickness=0.4, material="si", bias=0.10
            ),
            "tabulated": LayerLevel(
                layer=(1, 0),
                zmin=0.8,
                thickness=0.4,
                material="si",
                z_to_bias=((0, 0.5, 1), (0, 0.12, 0)),
            ),
        }
    )
    # GDSFactory 8's LayerLevel model only accepts tabulated profiles. BeamZ's
    # importer also accepts level-like objects with callable z_to_bias.
    from types import SimpleNamespace

    stack.layers["callable"] = SimpleNamespace(
        layer=LogicalLayer(layer=(1, 0)),
        zmin=1.6,
        thickness=0.4,
        material="si",
        mesh_order=3,
        z_to_bias=lambda z: 0.12 * z * z,
    )
    return Case(
        c,
        stack,
        ((-0.3, -0.3, -0.1), (4.3, 1.3, 2.1)),
        (
            Probe("constant bias grows footprint", (2, -0.05, 0.2), 12),
            Probe("tabulated bulge", (2, -0.08, 1.0), 12),
            Probe("outside tabulated base", (2, -0.08, 0.82), 1),
            Probe("callable upper expansion", (2, -0.05, 1.96), 12),
            Probe("outside callable base", (2, -0.05, 1.64), 1),
        ),
        "Mask growth and non-monotonic/curved z bias; callable uses a level-like adapter.",
    )


def hierarchy_holes() -> Case:
    """A transformed child contains a boolean hole and a DBU-sized region."""
    child = gf.Component()
    rectangle(child, (1, 0), 0, 0, 2, 2)
    rectangle(child, (2, 0), 0.5, 0.5, 1.5, 1.5)
    c = gf.Component()
    ref = c.add_ref(child)
    ref.drotate(90)
    ref.dmove((3, 0))  # (x, y) -> (3-y, x)
    outer, hole = LogicalLayer(layer=(1, 0)), LogicalLayer(layer=(2, 0))
    # LogicalLayer.sized() takes database units, unlike LayerLevel.bias (µm).
    grown = outer.sized(int(round(0.1 / c.kcl.dbu)))
    stack = LayerStack(
        layers={
            "ring": LayerLevel(
                layer=outer - hole,
                derived_layer=LogicalLayer(layer=(102, 0)),
                zmin=0,
                thickness=0.3,
                material="si",
            ),
            "grown": LayerLevel(layer=grown, zmin=0.6, thickness=0.2, material="sin"),
        }
    )
    return Case(
        c,
        stack,
        ((0.5, -0.5, -0.1), (3.5, 2.5, 1.0)),
        (
            Probe("transformed ring", (2.75, 1, 0.15), 12),
            Probe("preserved hole", (2, 1, 0.15), 1),
            Probe("sized extension", (0.95, 1, 0.7), 4),
            Probe("outside extension", (0.8, 1, 0.7), 1),
        ),
        "Rotated hierarchy, boolean subtraction with a hole and database-unit sizing.",
    )


BUILDERS = {
    "rib_etch": rib_etch,
    "stacked_heater": stacked_heater,
    "sidewall_references": sidewall_references,
    "bias_profiles": bias_profiles,
    "hierarchy_holes": hierarchy_holes,
}
