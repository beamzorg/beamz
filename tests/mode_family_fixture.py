"""Translate labeled synthetic fixtures to the array-native solver contract."""

import numpy as np

from beamz.devices.modes.models import Grid, Materials
from beamz.devices.modes.result import Result


def mode_family(*, n_complex, field_components, solver_info=None):
    first = next(iter(field_components.values()))
    axes, normal = first.dims[:2], first.dims[2]
    edges = []
    for axis in axes:
        centers = np.asarray(first.coords[axis], dtype=float)
        width = centers[1] - centers[0] if centers.size > 1 else 1.0
        bounds = [centers[0] - width / 2]
        for center in centers:
            bounds.append(2 * center - bounds[-1])
        edges.append(tuple(bounds))
    return Result(
        np.asarray(n_complex),
        {
            name: np.asarray(data.transpose("f", "mode_index", *axes, normal))[..., 0]
            for name, data in field_components.items()
        },
        Grid(*edges, normal_axis="xyz".index(normal)),
        np.asarray(n_complex.coords["f"]),
        solver_info=solver_info,
    )


def mode_grid_inputs(**inputs):
    """Bind component-array fixtures into the canonical validated solver input."""
    keys = {
        k
        for k in inputs
        if k.startswith(("eps_", "mu_"))
        or k in {"x_edges", "y_edges", "normal_axis", "normal_coordinate"}
    }
    return dict(
        material_grid=Materials.from_components(**{k: inputs[k] for k in keys}),
        **{k: v for k, v in inputs.items() if k not in keys},
    )
