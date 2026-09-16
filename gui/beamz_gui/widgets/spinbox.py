"""
QDoubleSpinBox's default behavior is a real usability problem for
photonics values: it neither accepts typing scientific notation like
"1e-6" (the validator rejects intermediate states like "1e-" while
you're still typing it, so you're forced to type "0.000001" instead), nor
does it display compactly — a fixed `decimals()` count means every field
shows the same number of trailing zeros regardless of the actual value
("0.000000" for an unset field is a common complaint).

SmartDoubleSpinBox fixes both, while keeping full double precision
internally (only the DISPLAY text is compacted, not the stored value):
  - `validate()` accepts partial scientific-notation input ("1e", "1e-",
    "1.5e-") as Intermediate rather than Invalid, so typing "1e-6"
    character by character actually works.
  - `textFromValue()` shows up to 6 significant figures, trimmed of
    trailing zeros, switching to scientific notation for very small/large
    magnitudes (Python's `%g` format) instead of a fixed decimal count.
"""
from __future__ import annotations

import re

from PySide6.QtGui import QValidator
from PySide6.QtWidgets import QDoubleSpinBox

_PARTIAL_FLOAT_RE = re.compile(r"[+-]?\d*\.?\d*(e[+-]?\d*)?$", re.IGNORECASE)


class SmartDoubleSpinBox(QDoubleSpinBox):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        # High internal precision ceiling so the stored value is never
        # rounded away — compactness is purely a DISPLAY concern, handled
        # by textFromValue below, not by lowering this.
        super().setDecimals(15)

    def setDecimals(self, decimals: int) -> None:  # noqa: N802 (Qt override naming)
        # Ignored on purpose: callers used to set a small decimals() count
        # for display purposes, which also clamps the STORED value's
        # precision — exactly the bug this class exists to avoid. Display
        # compactness is handled by textFromValue() instead.
        pass

    def textFromValue(self, value: float) -> str:  # noqa: N802
        if value == 0:
            return "0"
        return f"{value:.6g}"

    def valueFromText(self, text: str) -> float:  # noqa: N802
        try:
            return float(text)
        except ValueError:
            return 0.0

    def validate(self, text: str, pos: int):  # noqa: N802
        stripped = text.strip()
        if stripped in ("", "-", "+"):
            return (QValidator.State.Intermediate, text, pos)
        try:
            float(stripped)
            return (QValidator.State.Acceptable, text, pos)
        except ValueError:
            if _PARTIAL_FLOAT_RE.fullmatch(stripped):
                return (QValidator.State.Intermediate, text, pos)
            return (QValidator.State.Invalid, text, pos)