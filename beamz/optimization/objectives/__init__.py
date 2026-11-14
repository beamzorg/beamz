"""Composable modal objectives."""

# Restore the forward type reference after loading the mutually dependent classes.
from . import base as _base
from .base import ModalObjective
from .composite import SoftMinModePower, WeightedObjective
from .mode import ModePower

_base.ModePower = ModePower  # type: ignore[misc]
