"""Run with python -m examples.gdsfactory_imports.run --output .cache/gdsfactory.

Exports GDS, cross-section plots and analytical material probes.
No FDTD solve is needed to check geometry import.
"""

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import gdsfactory as gf
import numpy as np

from beamz.design import raster as r
from beamz.design.gds import import_component
from beamz.design.raster.importers._beamz import from_beamz

from .cases import BUILDERS, MATERIALS, UM, Probe


def activate_pdk():
    try:
        gf.get_active_pdk()
    except ValueError:
        from gdsfactory.gpdk import PDK

        PDK.activate()


def sample_epsilon(scene, xyz_um):
    center = np.asarray(xyz_um) * UM
    grid = r.Grid.uniform(
        tuple(center - 0.0005 * UM), tuple(center + 0.0005 * UM), (1, 1, 1)
    )
    result = scene.rasterize(
        grid, options=r.RasterOptions(quality="reference", smoothing="volume")
    )
    return float(result.tensors["epsilon"][0, 0, 0, 0])


def check_probes(scene, probes):
    checks = []
    compiled = r.compile_scene(scene)
    for probe in probes:
        actual = sample_epsilon(compiled, probe.xyz_um)
        checks.append(
            {
                "name": probe.name,
                "xyz_um": probe.xyz_um,
                "expected_epsilon": probe.epsilon,
                "actual_epsilon": actual,
                "passed": bool(np.isclose(actual, probe.epsilon, atol=2e-5, rtol=0)),
            }
        )
    return checks


def export_case(name, case, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    case.component.write_gds(output / f"{name}.gds")
    imported = import_component(
        case.component,
        layer_stack=case.stack,
        material_map=MATERIALS,
        bounds=np.asarray(case.bounds_um) * UM,
    )
    scene = from_beamz(imported.design)
    origin = np.asarray(imported.source_origin)
    local_probes = tuple(
        Probe(p.name, tuple(np.asarray(p.xyz_um) - origin / UM), p.epsilon)
        for p in case.probes
    )
    checks = check_probes(scene, local_probes)
    if not all(check["passed"] for check in checks):
        raise AssertionError(f"{name}: failed geometry probes: {checks}")
    source_lower, source_upper = np.asarray(case.bounds_um)
    lower, upper = np.asarray(case.bounds_um) * UM - origin
    x_mid = 0.5 * (lower[0] + upper[0])
    z_cut = local_probes[0].xyz_um[2] * UM
    cuts = {
        "xy": r.Grid.uniform(
            (lower[0], lower[1], z_cut - 0.0005 * UM),
            (upper[0], upper[1], z_cut + 0.0005 * UM),
            (64, 48, 1),
        ),
        "yz": r.Grid.uniform(
            (x_mid - 0.0005 * UM, lower[1], lower[2]),
            (x_mid + 0.0005 * UM, upper[1], upper[2]),
            (1, 48, 64),
        ),
    }
    compiled = r.compile_scene(scene)
    arrays = {
        axis: compiled.rasterize(
            grid, options=r.RasterOptions(quality="fast", smoothing="volume")
        )
        .tensors["epsilon"][0]
        .squeeze()
        for axis, grid in cuts.items()
    }
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    for axis, label, extent in zip(
        axes,
        ("xy", "yz"),
        (
            (source_lower[0], source_upper[0], source_lower[1], source_upper[1]),
            (source_lower[1], source_upper[1], source_lower[2], source_upper[2]),
        ),
        strict=True,
    ):
        artist = axis.imshow(
            arrays[label],
            origin="lower",
            extent=extent,
            vmin=1,
            vmax=12,
            aspect="auto",
            interpolation="nearest",
        )
        axis.set(
            xlabel=f"{label[0]} (µm)",
            ylabel=f"{label[1]} (µm)",
            title=f"{label.upper()} at {'z' if label == 'xy' else 'x'}="
            f"{((z_cut + origin[2]) if label == 'xy' else (x_mid + origin[0])) / UM:.3g} µm",
        )
    fig.colorbar(artist, ax=axes, label="Relative permittivity (volume average)")
    fig.savefig(output / f"{name}.png", dpi=160)
    plt.close(fig)
    return {
        "description": case.description,
        "objects": len(scene.objects),
        "probes": checks,
        "source_origin_m": imported.source_origin,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["all", *BUILDERS], default="all")
    parser.add_argument(
        "--output", type=Path, default=Path(".cache/gdsfactory-imports")
    )
    args = parser.parse_args()
    activate_pdk()
    names = BUILDERS if args.case == "all" else [args.case]
    report = {
        "versions": {
            name: version(name) for name in ("beamz", "gdsfactory", "kfactory")
        },
        "examples": {
            name: export_case(name, BUILDERS[name](), args.output) for name in names
        },
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
