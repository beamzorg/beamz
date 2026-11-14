"""Export optimized density as material geometry."""

from __future__ import annotations

from beamz.optimization.design.geometry.polygons import density_to_polygons


def _export_density(
    design,
    density,
    *,
    eps_bounds,
    origin,
    dx,
    depth=0.0,
    layer_depth=None,
    region_mask=None,
    background=None,
    fixed_solid=None,
    fixed_void=None,
    threshold=0.5,
    mode="contour",
    sample_spacing=None,
):
    """Overlay background and solid density using one planar/layered export path.

    Background is a 2D mask on the same grid. The physical density may include
    outside-mask context; only the resulting solid geometry is clipped.
    """
    from beamz.design.materials import Material

    if not 0 < threshold < 1:
        raise ValueError("threshold must lie strictly between zero and one.")
    grid = dict(x0=origin[0], y0=origin[1], dx=dx)
    if background is not None:
        for polygon in density_to_polygons(
            background,
            level=0.5,
            material=Material(eps_bounds[0]),
            mode="pixel_exact",
            **grid,
        ):
            design += polygon.updated_copy(depth=depth).shift(0, 0, origin[2])
    volumetric = density.ndim == 3
    layers = density if volumetric else density[None]
    layer_depth = depth if layer_depth is None else layer_depth
    for index, layer in enumerate(layers):
        z = origin[2] + index * layer_depth

        def mask_layer(mask, index=index):
            return mask[index] if volumetric and mask is not None else mask

        for polygon in density_to_polygons(
            layer,
            level=threshold,
            material=Material(eps_bounds[1]),
            mode=mode,
            sample_spacing=sample_spacing,
            region_mask=mask_layer(region_mask),
            fixed_solid=mask_layer(fixed_solid),
            fixed_void=mask_layer(fixed_void),
            **grid,
        ):
            design += polygon.updated_copy(depth=layer_depth).shift(0, 0, z)
    return design
