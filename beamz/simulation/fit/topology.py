"""Metric-free incidence operators on complete Cartesian cell complexes.

Arrays use (y, x) or (z, y, x) order. Electric voltages live on primal
edges; magnetic fluxes live on primal faces. The transpose is the actual
algebraic transpose, including the boundary contributions.
"""

import jax.numpy as jnp


def difference_transpose(values, axis):
    """Transpose of the forward difference with zero exterior extension."""
    low = [(0, 0)] * values.ndim
    high = [(0, 0)] * values.ndim
    low[axis] = (1, 0)
    high[axis] = (0, 1)
    return jnp.pad(values, low) - jnp.pad(values, high)


def curl(e, *, dimension, polarization="tm"):
    """Apply primal edge-to-face incidence C without metric factors."""
    if dimension == 3:
        return {
            "Bx": jnp.diff(e["Ez"], axis=1) - jnp.diff(e["Ey"], axis=0),
            "By": jnp.diff(e["Ex"], axis=0) - jnp.diff(e["Ez"], axis=2),
            "Bz": jnp.diff(e["Ey"], axis=2) - jnp.diff(e["Ex"], axis=1),
        }
    if polarization == "tm":
        return {"Bx": jnp.diff(e["Ez"], axis=0), "By": -jnp.diff(e["Ez"], axis=1)}
    return {"Bz": jnp.diff(e["Ey"], axis=1) - jnp.diff(e["Ex"], axis=0)}


def curl_transpose(h, *, dimension, polarization="tm"):
    """Apply dual face-to-edge incidence C transpose."""
    dt = difference_transpose
    if dimension == 3:
        return {
            "Ex": dt(h["By"], 0) - dt(h["Bz"], 1),
            "Ey": dt(h["Bz"], 2) - dt(h["Bx"], 0),
            "Ez": dt(h["Bx"], 1) - dt(h["By"], 2),
        }
    if polarization == "tm":
        return {"Ez": dt(h["Bx"], 0) - dt(h["By"], 1)}
    return {"Ex": -dt(h["Bz"], 0), "Ey": dt(h["Bz"], 1)}


def divergence(b, *, dimension, polarization="tm"):
    """Apply face-to-cell incidence S; S C is identically zero."""
    if dimension == 3:
        return (
            jnp.diff(b["Bx"], axis=2)
            + jnp.diff(b["By"], axis=1)
            + jnp.diff(b["Bz"], axis=0)
        )
    if polarization == "tm":
        return jnp.diff(b["Bx"], axis=1) + jnp.diff(b["By"], axis=0)
    # TE has only Bz, with no z dependence.
    return jnp.zeros_like(b["Bz"])


def gradient(potential, *, dimension, polarization="tm"):
    """Apply node-to-edge incidence G; C G is identically zero."""
    if dimension == 3:
        return {
            "Ex": jnp.diff(potential, axis=2),
            "Ey": jnp.diff(potential, axis=1),
            "Ez": jnp.diff(potential, axis=0),
        }
    if polarization == "tm":
        return {"Ez": jnp.zeros_like(potential)}
    return {"Ex": jnp.diff(potential, axis=1), "Ey": jnp.diff(potential, axis=0)}


def dual_divergence(d, *, dimension, polarization="tm"):
    """Apply -G transpose to dual electric fluxes (outward node flux)."""
    dt = difference_transpose
    if dimension == 3:
        return -dt(d["Ex"], 2) - dt(d["Ey"], 1) - dt(d["Ez"], 0)
    if polarization == "tm":
        return jnp.zeros_like(d["Ez"])
    return -dt(d["Ex"], 1) - dt(d["Ey"], 0)
