# Bundled optical material data

`catalog.json` stores BeamZ PoleResidue specifications in SI units (frequencies
in Hz; pole locations and residues in rad/s), named variants, and source metadata.
The public API is `from beamz import material_library`. All data ship with BeamZ
and are available offline.

The silica, silicon-nitride, amorphous-silicon, synthetic RGB filter, and
`Al/Rakic1995_CMOS` coefficients were imported from a published CMOS RGB simulation
snapshot retrieved on 2026-09-26, version `2.12.0.dev3`. Its SHA-256 identifier is
`4e3bc436bdff467be70c837a068d27c05fd93effadbb1aee5b18085d6c3cb2b3`.
Independent source coefficients and snapshot checksums are retained in
`tests/fixtures/materials/cmos_reference.json`, without refitting.

`Al/Rakic1995` is an imported five-pole optical-material fit, pinned to Git
revision `c7f41d17ac5de2f7a87e811bbfa1d051f6f9c762`, symbol `Al_Rakic1995`.
The four-real-pole CMOS fit is kept under a separate variant name to preserve
that example's material specification; it is not the aluminum default.

The CSVs contain wavelength [micrometres], n, and k, imported unchanged from
published notebook data at Git revision
`c37c785d52e9258c9d048a781524b8e8d7c758ca`, files `misc/{red,green,blue}_eps.csv`.
These are hypothetical filter inputs, not measured optical constants.
`MaterialVariant.nk_data` converts wavelengths to metres and adds the recorded
`k_offset = 0.01`, matching the input used for the bundled fits.

The imported library fit carries an LGPL-2.1 notice; the imported notebook data
carry AGPL-3.0 notices. Unmodified license texts are included as
`LICENSE-LGPL-2.1.txt` and `LICENSE-AGPL-3.0.txt`. Each catalog variant records its
source identifier and license. BeamZ's own source retains its existing license.
