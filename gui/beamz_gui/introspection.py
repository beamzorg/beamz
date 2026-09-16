"""
Turns a placeable's *recipe* (canonical constructor kwargs) into a list of
`FieldSpec`s the Property Editor can render generically.

Deliberately introspects the recipe — via a freshly constructed canonical
instance, `cls(**recipe)` — rather than `SceneObject.obj` directly. `obj`
may currently be a rotated derivative (e.g. a `Polygon` standing in for a
rotated `Rectangle`; see model.py's module docstring) that no longer has
`width`/`height` fields at all. The recipe is always the pre-rotation
truth, so this is what keeps every field editable regardless of rotation
state.
"""
from __future__ import annotations

import dataclasses
from typing import Any

import beamz as bz
import numpy as np

# Fields that exist on beamz placeables but shouldn't appear in the
# generic editor: `vertices`/`interiors` are derived geometry output, not
# constructor input (see model.py's `_recipe_from_instance`), and `color`
# gets its own swatch widget rather than a text field.
HIDDEN_FIELDS = {"vertices", "interiors", "color"}


@dataclasses.dataclass
class FieldSpec:
    name: str
    # "float" | "int" | "bool" | "str" | "tuple2" | "tuple3" | "material" |
    # "nested" (a beamz dataclass instance, e.g. GaussianPulse) |
    # "array" (a numpy array or a plain numeric list/tuple of any length,
    # e.g. `freqs`, `signal`) | "readonly" (genuinely unhandled — e.g. an
    # unset Optional field of unknown type; see property_editor.py's
    # NESTED_FIELD_TYPES for how known ones still get a usable editor)
    kind: str
    default: Any


def introspect_fields(class_name: str, recipe: dict[str, Any]) -> list[FieldSpec]:
    """Build FieldSpecs for every constructor field of `class_name`, with
    current values taken from `recipe` where present and the class's own
    default otherwise (exactly what `cls(**recipe)` gives you for free).
    """
    cls = getattr(bz, class_name)
    base = cls(**recipe)
    if not dataclasses.is_dataclass(base):
        return []

    specs: list[FieldSpec] = []
    for f in dataclasses.fields(base):
        if not f.init or f.name.startswith("_") or f.name in HIDDEN_FIELDS:
            continue
        value = getattr(base, f.name)
        specs.append(FieldSpec(name=f.name, kind=_classify(f.name, value), default=value))
    return specs


def _classify(field_name: str, value: Any) -> str:
    # Classify by FIELD NAME first for material, not just the current
    # value's type: every freshly-created structure has `material=None`
    # (beamz's own default), and `isinstance(None, bz.Material)` is False,
    # so relying on the value alone silently misclassified every
    # never-yet-assigned material field as "readonly" — meaning the
    # material dropdown never appeared until *something else* had already
    # set a real Material instance. A real reported bug: "property viewer
    # does not have material property" turned out to be exactly this.
    if field_name == "material":
        return "material"
    if isinstance(value, bz.Material):
        return "material"
    # bool must be checked before int (bool is an int subclass in Python).
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, tuple) and 2 <= len(value) <= 3 and all(
        isinstance(x, (int, float)) and not isinstance(x, bool) for x in value
    ):
        return f"tuple{len(value)}"
    if isinstance(value, str):
        return "str"
    # A nested beamz object (e.g. `source_time=GaussianPulse(...)`) — its
    # OWN fields get introspected the same way, recursively, rather than
    # being shown as an opaque repr string. Checked before the array case
    # since dataclasses are never also arrays.
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return "nested"
    # A numpy array, or a plain list/tuple of arbitrary length holding
    # only numbers (as opposed to the tuple2/tuple3 case above, which is
    # specifically a 2- or 3-component coordinate) — e.g. `freqs`,
    # `signal`, `waveform`. Editable as a comma-separated list.
    if isinstance(value, np.ndarray):
        return "array"
    if isinstance(value, (list, tuple)) and len(value) > 0 and all(
        isinstance(x, (int, float, complex)) and not isinstance(x, bool) for x in value
    ):
        return "array"
    return "readonly"