# Reference silicon polygons

These JSON files contain polygon coordinates in micrometres on GDS layer
`(1, 0)`, extracted from the matching GDS files in
[JPPhotonics/fdtd-pipeline](https://github.com/JPPhotonics/fdtd-pipeline/tree/622e0a9b7429eaf2335b1000b39e283544a198c4/gds_library/cells_from_gds/gdsfactory_generic_pdk).
The revision and SHA-256 checksums are recorded in each device manifest.

Extraction uses `gf.import_gds(path).get_polygons_points(by="tuple")[(1, 0)]`.
The fixture object contains `layer: [1, 0]` and `polygons_um`, a list of polygon
vertex arrays. Coordinates and polygon partitioning are preserved. Export
with `json.dumps(payload, separators=(",", ":")) + "\n"` to reproduce the
fixture checksum. The normalized geometric union is checked separately,
so an exact fixture-byte check is not the only geometry safeguard.

Port positions, widths, and orientations come from the accompanying reference
YAML, not from newer component factories. No annotation or pin layers are
extruded into the silicon stack.

The conversion cases solve four candidate eigenmodes on the physical source
and every physical mode monitor before selecting polarization and mode order.
A single candidate cannot distinguish TE0 from TM0; increasing only the
analysis port's candidate count does not update its physical monitor.
