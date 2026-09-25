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

import colorsys
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

# Practically-relevant photonics index range used to map n -> a color —
# air/oxide-like at the low end up through silicon-like at the high end.
# Values outside this range still get a sensible color (clamped), just
# without further hue separation past the ends.
_INDEX_COLOR_MIN = 1.0
_INDEX_COLOR_MAX = 4.0


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


def color_for_material(material: Optional[bz.Material]) -> str:
    """Hex color DERIVED FROM the structure's actual material index,
    deterministic (same index always -> same color) — replaces beamz's
    own `.color` field as what the 2D canvas / 3D preview actually draw.
    That field turned out to be a fixed, identical default
    ("#6699cc") on every structure class regardless of `material` —
    confirmed directly (`dataclasses.fields(bz.Circle)` shows `color`
    and `material` as two completely independent fields with no relation
    to each other) — which is exactly why two structures with
    deliberately different materials/indices were rendering as the
    identical shade of blue. Low index (air-like) maps toward blue, high
    index (silicon-like) toward red, via a hue sweep in HSV space with
    saturation/value held constant so only hue (which the eye reads as
    "different color", not "different brightness") carries the
    information — deliberately NOT keyed on object identity/class, so
    two structures sharing one material/index still look like the same
    thing, which is the actually useful signal ("what material is this
    made of") rather than an arbitrary per-object color.
    """
    n = index_from_material(material)
    t = max(0.0, min(1.0, (n - _INDEX_COLOR_MIN) / (_INDEX_COLOR_MAX - _INDEX_COLOR_MIN)))
    hue = (1.0 - t) * 0.62  # 0.62 (blue) at low n -> 0.0 (red) at high n
    r, g, b = colorsys.hsv_to_rgb(hue, 0.55, 0.85)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
