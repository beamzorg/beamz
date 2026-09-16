"""
Minimal material presets for v1.

beamz's `Material` dataclass is just `permittivity`/`permeability`/
`conductivity` — no wavelength dependence built in. A real dispersive
material database (measured n,k data + fits, like Lumerical's) is future
work; for now every preset is a constant (non-dispersive) refractive
index. This is enough to assign materials and get correct index contrast
in a simulation — just not spectrally accurate across a broad bandwidth.
"""
from __future__ import annotations

from typing import Optional

import beamz as bz

# label -> constant refractive index
PRESETS: dict[str, float] = {
    "Air / Vacuum (n=1.0)": 1.0,
    "SiO2 (n=1.444)": 1.444,
    "Si3N4 (n=2.0)": 2.0,
    "Si (n=3.48)": 3.48,
}

CUSTOM_LABEL = "Custom index..."


def material_from_index(n: float) -> bz.Material:
    return bz.Material(permittivity=n**2)


def index_from_material(material: Optional[bz.Material]) -> float:
    """Best-effort round trip for display purposes: sqrt(permittivity) for
    the scalar case. Tensor or otherwise non-scalar permittivity has no
    single defining index, so it falls back to 1.0 for display rather than
    guessing.
    """
    if material is None:
        return 1.0
    eps = getattr(material, "permittivity", None)
    if isinstance(eps, (int, float)) and not isinstance(eps, bool):
        return float(eps) ** 0.5
    return 1.0


def label_for_index(n: float, tol: float = 1e-3) -> str:
    for label, val in PRESETS.items():
        if abs(val - n) < tol:
            return label
    return CUSTOM_LABEL
