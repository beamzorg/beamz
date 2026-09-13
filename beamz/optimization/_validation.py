"""Validation shared by density APIs and optimizer continuation."""

from numbers import Integral

import numpy as np


def step_count(value, name="step", *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        bound = "positive" if minimum == 1 else "nonnegative"
        raise ValueError(f"{name} must be a {bound} integer.")
    return int(value)


def check_density(values, shape, name="Density"):
    values = np.asarray(values)
    if (
        values.shape != shape
        or not np.isfinite(values).all()
        or np.any((values < 0) | (values > 1))
    ):
        raise ValueError(f"{name} must match shape {shape} and be finite in [0, 1].")
