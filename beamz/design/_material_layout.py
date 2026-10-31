"""Material names and active Yee supports shared by validation and conversion."""

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
