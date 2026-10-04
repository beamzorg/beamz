"""Independent Bessel-function benchmark for circular dielectric slab bends.

Hiremath and Hammer, https://www.siio.eu/NaisWP3/Bend/ (TE, core-center radius).
See docs/bent-waveguide-modes.md for the mesh and PML refinement study.
"""

import numpy as np
import pytest

from beamz.devices.modes import solve_grid


def _published_slab(radius, spacing=0.025, outer=20.0, pml_width=4.0):
    edges = np.linspace(-10.0, outer, round((outer + 10.0) / spacing) + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    eps = np.where(np.abs(centers) < 0.5, 1.7**2, 1.6**2)[:, None]
    modes = solve_grid(
        eps_xx=eps,
        x_edges=edges,
        y_edges=[0.0, 1.0],  # Singleton transverse direction: invariant slab.
        wavelength=1.3,
        num_modes=2,
        target_neff=1.66,
        pml=(round(pml_width / spacing), 0),
        bend_radius=radius,
    )
    # TE has E along the invariant axis; avoid depending on eigenvalue ordering.
    ey = np.sum(np.abs(modes.field_components["Ey"].values), axis=(0, 1, 2, 3))
    return complex(modes.n_complex.values[0, np.argmax(ey)])


@pytest.mark.parametrize(
    ("radius", "reference_real", "reference_loss"),
    [
        (50, 1.66303, 3.309e-4),
        (100, 1.66096, 1.987e-6),
        (150, 1.66061, 1.020e-8),
        (200, 1.66049, 5.066e-11),
    ],
)
def test_published_bent_slab(
    radius, reference_real, reference_loss, validation_metrics
):
    measured = _published_slab(radius)
    for name, value, reference, tolerance in [
        ("phase index", measured.real, reference_real, "bend_neff"),
        ("radiation loss index", measured.imag, reference_loss, "bend_loss"),
    ]:
        validation_metrics.check(
            name,
            measured=value,
            reference=reference,
            tolerance=tolerance,
            resolution="25 nm",
            metadata={"radius_um": radius},
        )


def test_published_bend_mesh_refinement(validation_metrics):
    values = [_published_slab(50, spacing=dx) for dx in (0.05, 0.025, 0.0125)]
    # Self-convergence avoids the five-decimal rounding of the published table.
    order = np.log2(
        abs(values[0].real - values[1].real) / abs(values[1].real - values[2].real)
    )
    validation_metrics.check_lower("mesh convergence order", order, 1.7)
    validation_metrics.check_upper("mesh convergence order", order, 2.3)
    validation_metrics.check(
        "fine-mesh phase index",
        values[-1].real,
        1.66303,
        tolerance="bend_neff",
        resolution="12.5 nm",
    )


def test_published_bend_pml_domain_refinement(validation_metrics):
    values = [
        _published_slab(50, outer=outer, pml_width=width)
        for outer, width in [(15, 3), (20, 4), (25, 5)]
    ]
    validation_metrics.check_upper(
        "phase index domain sensitivity",
        max(abs(v.real - values[-1].real) for v in values),
        1e-6,
    )
    validation_metrics.check_upper(
        "loss index relative domain sensitivity",
        max(abs(v.imag / values[-1].imag - 1) for v in values),
        0.001,
    )
