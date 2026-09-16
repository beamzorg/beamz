"""
Generates a sensible default recipe (constructor kwargs) for any beamz
placeable class, used by the dynamically-built Add menus in the toolbar —
menus populated straight from `model.STRUCTURE_KINDS` /
`model.SOURCE_KINDS` / `model.MONITOR_KINDS`, so a new class added to
those tuples gets a working "Add" action for free without touching the
UI code.

Strategy: check the hand-picked _SCALED_DEFAULTS table FIRST, THEN fall
back to `cls()` (the class's own built-in defaults) if nothing is
registered. An earlier version tried `cls()` first and only consulted a
fallback table when that raised — reasonable-sounding, but wrong in
practice: `cls()` succeeding doesn't mean the result is a SENSIBLE
default, just a legal one. `Rectangle()`'s own built-in default is a
literal 1x1x1 METRE cube — legal (every field has *a* default), but
catastrophically oversized next to a micron-scale photonic domain, and
guaranteed to swallow the entire PML region on every "+ Structure ->
Rectangle" toolbar click. Confirmed directly: `build_default_recipe`
returned `{}` for Rectangle (meaning "trust the class defaults"), and the
resulting object really was 1 metre wide. So _SCALED_DEFAULTS is checked
unconditionally first for anything registered in it, and `cls()` is only
trusted for classes with no entry there.
"""
from __future__ import annotations

from typing import Any

import numpy as np

import beamz as bz

# A representative telecom-band default (1550 nm / 193.5 THz) used
# throughout, since it's the single most common wavelength photonics
# examples center on.
_DEFAULT_FREQ = 193.5e12
_DEFAULT_FWIDTH = 20e12


def _default_pulse() -> "bz.GaussianPulse":
    return bz.GaussianPulse(freq0=_DEFAULT_FREQ, fwidth=_DEFAULT_FWIDTH)


# Hand-picked, photonics-scale (micron-range) defaults — checked BEFORE
# trusting a class's own `cls()` defaults, for exactly the reason in the
# module docstring above. Covers every STRUCTURE_KINDS / SOURCE_KINDS /
# MONITOR_KINDS member; add an entry here for any future class added to
# those registries in model.py too, or its toolbar "+" action will fall
# through to beamz's own (possibly meter-scale) defaults.
_SCALED_DEFAULTS: dict[str, dict[str, Any]] = {
    # --- structures --------------------------------------------------
    "Rectangle": dict(position=(0.0, 0.0, 0.0), width=1e-6, height=0.5e-6, depth=0.22e-6),
    "Circle": dict(position=(0.0, 0.0), radius=0.5e-6, depth=0.22e-6),
    "Ring": dict(position=(0.0, 0.0), inner_radius=0.5e-6, outer_radius=1e-6, depth=0.22e-6),
    "Taper": dict(position=(0.0, 0.0), input_width=0.5e-6, output_width=1e-6, length=2e-6, depth=0.22e-6),
    "CircularBend": dict(position=(0.0, 0.0), inner_radius=1e-6, outer_radius=1.5e-6, angle=90.0, depth=0.22e-6),
    "Box": dict(center=(0.0, 0.0, 0.0), size=(1e-6, 1e-6, 1e-6)),
    "Sphere": dict(position=(0.0, 0.0, 0.0), radius=0.5e-6),
    # Polygon has no positional/span concept to scale — a small triangle
    # near the origin is at least a visible, sensible starting point.
    "Polygon": dict(vertices=((0.0, 0.0, 0.0), (1e-6, 0.0, 0.0), (0.5e-6, 1e-6, 0.0))),
    # --- sources -------------------------------------------------------
    "GaussianSource": dict(
        position=(0.0, 0.0, 0.0),
        width=0.5e-6,
        signal=np.sin(2 * np.pi * _DEFAULT_FREQ * np.linspace(0.0, 1e-13, 200)),
    ),
    "GaussianBeamSource": dict(
        center=(0.0, 0.0, 0.0),
        size=(0.0, 3e-6, 1.5e-6),
        source_time=_default_pulse(),
        waist_radius=1e-6,  # compiling (needed for run/memory-estimate,
        # not just construction) rejects `None` here with "waist_radius
        # must be positive" — found by actually running memory_estimate()
        # against a default-constructed source, not just instantiating it.
    ),
    "ModeSource": dict(
        center=(0.0, 0.0, 0.0),
        size=(0.0, 3e-6, 1.5e-6),
        source_time=_default_pulse(),
        direction="+",  # ModeSource: just "+"/"-" along the zero-extent
        # axis of `size`, unlike GaussianBeamSource's "+x"/"-x" form —
        # found by reading its docstring after `cls()`/a guessed "+x"
        # both failed with "direction must be '+' or '-'".
    ),
    "CustomSource": dict(
        component="Ez",
        timing="harmonic",
        index=(0, 0, 0),
        coeff=1.0,
        waveform=np.zeros(10),
        target_shape=(1,),
    ),
    # --- monitors --------------------------------------------------
    "FieldMonitor": dict(
        center=(0.0, 0.0, 0.0),
        size=(0.0, 3e-6, 1.5e-6),
        freqs=np.array([_DEFAULT_FREQ]),
    ),
    "FluxMonitor": dict(
        center=(0.0, 0.0, 0.0),
        size=(0.0, 3e-6, 1.5e-6),
        freqs=np.array([_DEFAULT_FREQ]),
    ),
    "ModeMonitor": dict(
        center=(0.0, 0.0, 0.0),
        size=(0.0, 3e-6, 1.5e-6),
        freqs=np.array([_DEFAULT_FREQ]),
    ),
}


def build_default_recipe(class_name: str) -> dict[str, Any]:
    """Photonics-scale default constructor kwargs for `class_name`.
    Checks `_SCALED_DEFAULTS` first; only trusts the class's own no-arg
    `cls()` defaults for classes with no entry there. Raises the
    underlying TypeError if neither works, rather than masking a class
    this table hasn't been taught about yet.
    """
    if class_name in _SCALED_DEFAULTS:
        return dict(_SCALED_DEFAULTS[class_name])

    cls = getattr(bz, class_name)
    try:
        cls()  # if this doesn't raise, every field has a usable default
        return {}
    except TypeError:
        raise TypeError(
            f"{class_name} has required constructor fields and no scaled "
            f"default is registered in defaults.py — add one to _SCALED_DEFAULTS."
        ) from None