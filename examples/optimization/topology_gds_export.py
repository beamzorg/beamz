"""Optimize a small mode converter and inspect its actual GDS round trip.

Run from the repository root:
    JAX_PLATFORMS=cpu uv run --extra gds python examples/optimization/topology_gds_export.py

Outputs a before/after figure, masks, checkpoint, and numerical measurements.
The coarse 2D example demonstrates export behavior, not fabrication readiness.
"""

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.path import Path as PlotPath
from shapely.geometry import GeometryCollection, LineString
from shapely.geometry import Polygon as SPolygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union
from skimage.measure import find_contours

from beamz import (
    LIGHT_SPEED,
    PML,
    Design,
    GaussianPulse,
    Material,
    ModeMonitor,
    ModeSource,
    ModeSpec,
    Rectangle,
    Simulation,
)
from beamz.design import export_gds
from beamz.optimization import (
    FilterProject,
    InverseDesign,
    LinearSchedule,
    OptimizationSchedule,
    Optimizer,
    RandomInitializationSpec,
    TopologyDesignRegion,
    density_to_shapely_geometry,
    shapely_geometry_to_polygons,
)

CORE, CLAD = Material(4), Material(1)


def make_design():
    design = Design(width=6e-6, height=4e-6, material=CLAD)
    design += Rectangle(position=(0, 1.6e-6), width=2e-6, height=0.8e-6, material=CORE)
    design += Rectangle(
        position=(4e-6, 1.4e-6), width=2e-6, height=1.2e-6, material=CORE
    )
    frequency = LIGHT_SPEED / 1.55e-6
    source = ModeSource(
        center=(1e-6, 2e-6, 0),
        size=(0, 2.8e-6, 1e-6),
        direction="+",
        mode_spec=ModeSpec(num_modes=1, polarization="tm"),
        source_time=GaussianPulse(freq0=frequency, fwidth=0.15 * frequency, offset=1.5),
    )
    output = ModeMonitor(
        center=(5e-6, 2e-6, 0),
        size=(0, 2.8e-6, 1e-6),
        freqs=[frequency],
        mode_spec=ModeSpec(num_modes=2, polarization="tm"),
        name="output",
    )
    incoming = output.updated_copy(
        center=(1.6e-6, 2e-6, 0),
        name="input",
        mode_spec=ModeSpec(num_modes=1, polarization="tm"),
    )
    simulation = Simulation(
        design=design,
        resolution=100e-9,
        run_time=200e-15,
        polarization="tm",
        sources=[source],
        monitors=[incoming, output],
        boundaries=[PML(thickness=0.6e-6, formulation="cpml")],
    )
    solid = np.zeros((16, 20), dtype=bool)
    solid[4:12, :2] = True
    solid[2:14, -2:] = True
    void = np.zeros_like(solid)
    void[:, :2] = ~solid[:, :2]
    void[:, -2:] = ~solid[:, -2:]
    region = TopologyDesignRegion(
        center=(3e-6, 2e-6, 0),
        size=(2e-6, 1.6e-6, np.inf),
        eps_bounds=(1, 4),
        pixel_size=100e-9,
        transformations=(FilterProject(200e-9, beta=8),),
        initialization_spec=RandomInitializationSpec(0.45, 0.55, seed=7),
        fixed_solid=solid,
        fixed_void=void,
    )
    return InverseDesign(simulation, region, gradient_backend="autodiff")


def transmission(data):
    out = data["output"].amps.sel(direction="+", mode_index=1).values
    incoming = data["input"].amps.sel(direction="+", mode_index=0).values
    return jnp.mean(jnp.abs(out / incoming) ** 2)


def legacy_geometry(density, dx, x0, y0):
    """Frozen reproduction of the pre-#331 contour padding for comparison only."""
    geometry = GeometryCollection()
    for ring in find_contours(np.pad(density, 1, constant_values=-0.5), 0.5):
        points = [(x0 + (x - 0.5) * dx, y0 + (y - 0.5) * dx) for y, x in ring]
        geometry = geometry.symmetric_difference(SPolygon(points).buffer(0))
    return geometry


def core_geometry(design):
    return unary_union(
        [
            SPolygon(
                [(x, y) for x, y, _ in s.vertices],
                [[(x, y) for x, y, _ in h] for h in getattr(s, "interiors", ())],
            )
            for s in design.structures
            if s.material == CORE
        ]
    )


def read_gds_geometry(path):
    # Preserve absolute coordinates instead of recentering the layout bounding box.
    import gdsfactory as gf

    layout = gf.kdb.Layout()
    layout.read(str(path))
    region = gf.kdb.Region(layout.top_cell().begin_shapes_rec(layout.layer(1, 0)))
    region.merge()
    scale = layout.dbu * 1e-6
    return unary_union(
        [
            SPolygon(
                [(v.x * scale, v.y * scale) for v in p.each_point_hull()],
                [
                    [(v.x * scale, v.y * scale) for v in p.each_point_hole(i)]
                    for i in range(p.holes())
                ],
            )
            for p in region.each()
        ]
    )


def port_gap_nm(geometry):
    """Measure uncovered length through the center of the input connection."""
    start, stop = 1.95e-6, 2.08e-6
    line = LineString([(start, 2e-6), (stop, 2e-6)])
    # Length, rather than distance to a clipped polygon, is insensitive to the
    # sub-ulp slivers that SI/GDS unit conversion can leave at a clipping plane.
    return max(0.0, line.length - geometry.intersection(line).length) * 1e9


def design_from_geometry(geometry):
    design = Design(width=6e-6, height=4e-6, material=CLAD)
    for polygon in shapely_geometry_to_polygons(geometry, material=CORE):
        design += polygon
    return design


def evaluate(simulation):
    data = simulation.run(backend="jax", performance=False)
    out = data.mode("output").amps.sel(direction="+", mode_index=1).item()
    incoming = data.mode("input").amps.sel(direction="+", mode_index=0).item()
    return float(abs(out / incoming) ** 2)


def parts(geometry):
    if geometry.geom_type == "Polygon":
        return [geometry]
    return list(geometry.geoms)


def draw(ax, geometry, color, *, scale=1e6):
    for polygon in parts(geometry):
        p = orient(polygon)
        vertices, codes = [], []
        for ring in [p.exterior, *p.interiors]:
            xy = np.asarray(ring.coords) * scale
            vertices.extend(xy)
            codes.extend(
                [PlotPath.MOVETO]
                + [PlotPath.LINETO] * (len(xy) - 2)
                + [PlotPath.CLOSEPOLY]
            )
        ax.add_patch(
            PathPatch(
                PlotPath(vertices, codes),
                facecolor=color,
                edgecolor=color,
                linewidth=0.7,
            )
        )


def figures(output, geometries, measurements):
    colors = {
        "legacy": "#cd5a4d",
        "pixel_exact": "#64748b",
        "contour": "#168b8a",
        "refined": "#4764d4",
    }
    names = {
        "legacy": "Previous export",
        "pixel_exact": "Exact pixels",
        "contour": "Corrected contours",
        "refined": "Contours · 10 nm sampling",
    }
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(
        2, 4, figsize=(15, 7), gridspec_kw={"height_ratios": [1.2, 1]}
    )
    for i, (key, geometry) in enumerate(geometries.items()):
        draw(axes[0, i], geometry, colors[key])
        axes[0, i].set(
            xlim=(1.6, 4.4),
            ylim=(1.05, 2.95),
            aspect="equal",
            xlabel="x (µm)",
            ylabel="y (µm)",
            title=names[key],
        )
        axes[0, i].axvline(2, color="#abb6c0", ls=":", lw=1)
        axes[0, i].axvline(4, color="#abb6c0", ls=":", lw=1)
        draw(axes[1, i], geometry, colors[key], scale=1e9)
        axes[1, i].set(
            xlim=(1940, 2080), ylim=(1820, 2180), xlabel="x (nm)", ylabel="y (nm)"
        )
        axes[1, i].axvline(2000, color="#334155", ls=":", lw=1)
        gap = measurements[key]["port_gap_nm"]
        axes[1, i].set_title(f"Input seam: {gap:.2f} nm gap", fontsize=10)
    fig.suptitle(
        "Topology → GDS: preserve the connection",
        x=0.04,
        ha="left",
        fontsize=20,
        weight="bold",
    )
    fig.text(
        0.04,
        0.905,
        "Same optimized density · 100 nm design cells · 1 nm GDS precision · masks shown after file reimport",
        color="#526173",
    )
    fig.tight_layout(rect=(0.015, 0.015, 1, 0.89))
    fig.savefig(output / "comparison.png", dpi=180)
    fig.savefig(output / "comparison.svg")
    plt.close(fig)

    # An analytic bilinear field makes the effect of finer contour sampling visible.
    yy, xx = np.mgrid[:8, :10] + 0.5
    density = xx * yy / 100
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    for ax, mode, spacing, title in zip(
        axes,
        ["pixel_exact", "contour", "contour"],
        [None, None, 0.1],
        ["Exact pixels", "Native contour", "10× contour sampling"],
        strict=True,
    ):
        g = density_to_shapely_geometry(
            density, level=0.25, mode=mode, sample_spacing=spacing
        )
        draw(ax, g, "#168b8a", scale=1)
        x = np.linspace(25 / 7.5, 9.5, 600)
        ax.plot(x, 25 / x, color="#e79c35", lw=1.7, label="Analytic interface")
        ax.set(
            xlim=(3, 10),
            ylim=(2.5, 8),
            aspect="equal",
            title=title,
            xlabel="x (cell units)",
            ylabel="y (cell units)",
        )
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(
        "Interpolation refines curves without inventing new design information",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(output / "interpolation.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/gds-export"))
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    design = make_design()
    optimizer = Optimizer(
        design,
        learning_rate=0.04,
        schedule=OptimizationSchedule(beta=LinearSchedule(1, 12, args.steps)),
    )
    checkpoint = args.output / "optimization.npz"
    result = optimizer.run(
        transmission,
        steps=args.steps,
        checkpoint=checkpoint,
        resume=checkpoint if args.resume else None,
        callback=lambda r: print(
            f"step {r.completed_steps}: objective={r.history[-1].post_process_val:.5f}",
            flush=True,
        ),
    )
    rho = np.asarray(
        design.design_region.material_density(result.final_params, beta=result.beta)
    )
    np.savez(args.output / "density.npz", params=result.final_params, density=rho)
    old = legacy_geometry(rho, 100e-9, 2e-6, 1.2e-6)
    legacy = design.simulation.design
    for polygon in shapely_geometry_to_polygons(old, material=CORE):
        legacy += polygon
    layouts = {
        "legacy": legacy,
        "pixel_exact": result.export_design(mode="pixel_exact"),
        "contour": result.export_design(),
        "refined": result.export_design(sample_spacing=10e-9),
    }
    measurements, geometries = {}, {}
    for name, layout in layouts.items():
        path = export_gds(
            layout,
            args.output / f"{name}.gds",
            precision=1e-9,
            layer_map={CORE: (1, 0)},
        )
        original = core_geometry(layout)
        reloaded = read_gds_geometry(path)
        geometries[name] = reloaded
        metrics = dict(
            port_gap_nm=port_gap_nm(reloaded),
            connected_components=len(parts(reloaded)),
            holes=sum(len(p.interiors) for p in parts(reloaded)),
            roundtrip_boundary_error_nm=original.hausdorff_distance(reloaded) * 1e9,
            roundtrip_area_difference_fraction=original.symmetric_difference(
                reloaded
            ).area
            / original.area,
        )
        reloaded_design = design_from_geometry(reloaded)
        for spacing in [100e-9, 50e-9]:
            label = f"transmission_{spacing * 1e9:.0f}nm"
            metrics[label] = evaluate(
                design.simulation.updated_copy(
                    design=reloaded_design, resolution=spacing
                )
            )
        metrics["before_serialization_transmission_50nm"] = evaluate(
            design.simulation.updated_copy(design=layout, resolution=50e-9)
        )
        measurements[name] = metrics
        print(name, json.dumps(metrics), flush=True)
    measurements["density_reference"] = dict(
        transmission_100nm=evaluate(result.to_simulation())
    )
    (args.output / "measurements.json").write_text(
        json.dumps(measurements, indent=2) + "\n"
    )
    figures(args.output, geometries, measurements)
    print(f"Showcase saved to {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
