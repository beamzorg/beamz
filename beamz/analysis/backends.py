"""Selection of the Matplotlib (default) and optional interactive XY backend."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

import numpy as np

_BACKEND = ContextVar("beamz_plotting_backend", default="matplotlib")


def _validate_backend(backend):
    if backend not in ("matplotlib", "xy"):
        raise ValueError(
            f"Unknown plotting backend {backend!r}; choose 'matplotlib' or 'xy'."
        )
    return backend


def get_plotting_backend():
    """Return the plotting backend active in this context."""
    return _BACKEND.get()


def get_pyplot(backend=None):
    """Return the selected pyplot module, importing XY only when requested."""
    backend = _validate_backend(backend or get_plotting_backend())
    if backend == "xy":
        try:
            import xy.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "The xy plotting backend requires XY. Install it with pip install 'beamz[xy]'."
            ) from exc
    else:
        import matplotlib.pyplot as plt
    return plt


def set_plotting_backend(backend):
    """Select 'matplotlib' or 'xy' for subsequent BeamZ plots in this context."""
    get_pyplot(_validate_backend(backend))
    _BACKEND.set(backend)


@contextmanager
def plotting_backend(backend):
    """Temporarily select a backend; restore it even if plotting fails."""
    get_pyplot(_validate_backend(backend))
    token = _BACKEND.set(backend)
    try:
        yield
    finally:
        _BACKEND.reset(token)


def axes_backend(ax):
    """Identify native axes without importing the optional backend."""
    first = np.asarray(ax, dtype=object).flat[0]
    return "xy" if type(first).__module__.startswith("xy.") else "matplotlib"


def with_plotting_backend(function):
    """Scope a public plot call and infer its backend from supplied axes."""

    @wraps(function)
    def wrapped(*args, backend=None, **kwargs):
        ax = kwargs.get("ax")
        selected = backend or (
            axes_backend(ax) if ax is not None else get_plotting_backend()
        )
        _validate_backend(selected)
        if ax is not None and axes_backend(ax) != selected:
            raise ValueError(
                "The supplied axes do not belong to the requested plotting backend."
            )
        with plotting_backend(selected):
            return function(*args, **kwargs)

    wrapped.__doc__ = (function.__doc__ or "") + (
        "\n\n    backend : {'matplotlib', 'xy'}, optional\n"
        "        Override the active plotting backend for this call. Supplied axes\n"
        "        select their own backend when this option is omitted.\n"
    )
    return wrapped
