"""Material names and active Yee supports shared by validation and conversion."""

from types import MappingProxyType

import numpy as np

from beamz.lattice import component_shapes

CELL_TENSORS = ("epsilon", "mu", "conductivity")

# solver name, raster name, field component, tensor diagonal, cell property
SUPPORTS = (
    ("eps_x", "epsilon_ex", "Ex", 0, "permittivity"),
    ("eps_y", "epsilon_ey", "Ey", 1, "permittivity"),
    ("eps_z", "epsilon_ez", "Ez", 2, "permittivity"),
    ("sig_x", "conductivity_ex", "Ex", 0, "conductivity"),
    ("sig_y", "conductivity_ey", "Ey", 1, "conductivity"),
    ("sig_z", "conductivity_ez", "Ez", 2, "conductivity"),
    ("mu_hx", "mu_hx", "Hx", 0, "permeability"),
    ("mu_hy", "mu_hy", "Hy", 1, "permeability"),
    ("mu_hz", "mu_hz", "Hz", 2, "permeability"),
)


def active_supports(dimensions, polarization="tm"):
    """Retain canonical ordering while selecting the solver's static family."""
    if dimensions == 3:
        return SUPPORTS
    components = {"Ez", "Hx", "Hy"} if polarization == "tm" else {"Ex", "Ey", "Hz"}
    return tuple(spec for spec in SUPPORTS if spec[2] in components)


def supports_by_cell(dimensions, polarization="tm"):
    groups = {}
    for target, _, _, _, cell in active_supports(dimensions, polarization):
        groups.setdefault(cell, set()).add(target)
    return groups


def support_shapes(shape, polarization="tm", *, raster=False):
    """Canonical component support sizes in solver or native tensor names."""
    components = component_shapes(shape, polarization)
    shapes = {
        source if raster else target: components[component]
        for target, source, component, _, _ in SUPPORTS
    }
    shapes["epsilon_node" if raster else "eps_node"] = tuple(n + 1 for n in shape)
    return shapes


def material_fields(
    values, shapes, *, compact=False, copy=True, label="material fields"
):
    """Validate and freeze a material field group at either ingress boundary."""
    values = {str(name): value for name, value in values.items()}
    unknown = set(values) - shapes.keys()
    if unknown:
        raise ValueError(f"Unknown {label}: {', '.join(sorted(unknown))}.")
    result = {}
    for name, value in values.items():
        array = np.array(value, copy=True) if copy else np.asarray(value)
        expected = shapes[name]
        actual = array.shape[1:] if compact else array.shape
        valid = not compact or array.shape[:1] in ((1,), (3,), (6,))
        if not valid or actual != expected:
            prefix = "compact (1|3|6, " if compact else ""
            raise ValueError(
                f"{name} tensor has shape {array.shape}, expected {prefix}{expected}."
            )
        array.setflags(write=False)
        result[str(name)] = array
    return MappingProxyType(result)
