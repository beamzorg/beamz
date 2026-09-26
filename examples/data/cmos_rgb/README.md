# CMOS RGB material inputs

The Al_Rakic1995, aSi_Horiba, SiN_Horiba, and SiO2_Palik_LowLoss pole/residue coefficients in `materials.json` were extracted from Flexcompute's open-source Tidy3D material library:
https://github.com/flexcompute/tidy3d/blob/c7f41d17ac5de2f7a87e811bbfa1d051f6f9c762/tidy3d/material_library/material_library.py

The RGB CSV files come from the accompanying reference notebook repository (wavelength in micrometres, n, k):
https://github.com/flexcompute/tidy3d-notebooks/tree/c37c785d52e9258c9d048a781524b8e8d7c758ca/misc

Retrieved 2026-09-25. The material-library repository uses LGPL-2.1; the notebook/data repository uses AGPL-3.0. Copies of their license notices are included alongside these inputs. Material coefficients retain the reference model names and validity bands. No Tidy3D runtime dependency is needed. The CSV absorption baseline of 0.01 is added during fitting, as in the reference CMOSRGBSensor notebook. Fitted filter spectra are approximations; consult the saved fit errors and plots.

Reference notebook: https://www.flexcompute.com/tidy3d/examples/notebooks/CMOSRGBSensor/

## Exact published-example snapshot (2026-09-26)

The active `materials.json` and `filters.json` now come from the complete
simulation embedded in the current reference page's 3D viewer, saved as
`tidy3d_reference_simulation.json`. `reference_provenance.json` records its URL,
retrieval date, version, and checksums. Extraction: decode the HTML
`data-simulation` attribute from base64, decompress gzip, then read the HDF5
`JSON_STRING` dataset. Reproduce conversion offline with
`python scripts/validation/import_cmos_reference.py` (no Tidy3D dependency).

The embedded RGB coefficients reproduce the reference's actual fitted materials,
including their nonconstant real index. We check maximum k <= 0.02 in the CSV's
transparent bands. This is a passband-specific acceptance condition, not a claim
that the hypothetical CSV data are physically exact. The old positive-Lorentz
approximations are retained in `filters_lorentz_legacy.json`.

**Aluminum version mismatch:** the embedded `Al_Rakic1995` has four real pole
pairs and differs materially from the five-pair library version previously
imported. At 550 nm their permittivities are approximately -21.90+74.86i and
-42.72+13.34i, respectively. The old file is retained as
`materials_library_legacy.json`. Matching a library name alone was insufficient
for reproducing this example. The silica, SiN, and aSi spectra match the old
snapshot to floating-point precision.
