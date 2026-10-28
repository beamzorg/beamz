"""Constants carried with the MicroMode SciPy reference solver.

Imported with ``scipy_reference.py`` (see ``UPSTREAM.md``). These are the
reference-solver values, including the nondimensional ``MU0`` used to form
``EPSILON0``. They are not the SI constants in ``beamz.const``.
"""

ETA0 = 376.730_313_666_853_5
MU0 = 1.256_637_062_12e-12
C0 = 2.997_924_58e14
EPSILON0 = 1.0 / (MU0 * C0 * C0)
