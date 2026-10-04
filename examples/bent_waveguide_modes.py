"""Compare straight and circular-bend modes of a generic rectangular guide.

Run with: uv run python examples/bent_waveguide_modes.py
All direct mode-solver lengths below are in micrometres.
"""

import numpy as np

from beamz.devices.modes import solve_grid


def main():
    x_edges = np.linspace(-2.0, 2.0, 81)
    y_edges = np.linspace(-1.5, 1.5, 61)
    x = (x_edges[:-1] + x_edges[1:]) / 2
    y = (y_edges[:-1] + y_edges[1:]) / 2
    core = (np.abs(x[:, None]) < 0.5) & (np.abs(y[None, :]) < 0.3)
    eps = np.where(core, 3.48**2, 1.44**2)
    inputs = dict(eps_xx=eps, x_edges=x_edges, y_edges=y_edges, wavelength=1.55)
    straight = solve_grid(**inputs, target_neff=3.2)
    reference = float(straight.n_eff.values[0, 0])
    print(f"Straight: n_eff = {reference:.8f}")
    for radius in (10.0, 50.0, 500.0):
        bent = solve_grid(**inputs, target_neff=reference, bend_radius=radius)
        index = float(bent.n_eff.values[0, 0])
        print(
            f"R={radius:6.1f} um: n_eff = {index:.8f}, delta = {index - reference:+.3e}"
        )


if __name__ == "__main__":
    main()
