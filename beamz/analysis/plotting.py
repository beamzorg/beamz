"""Small matplotlib plotting helpers used by examples and notebooks.

The public helpers stay importable from this module. Their implementations live
in the private modules beside this file.
"""

from beamz.analysis._plot_design import draw_geometry, plot_design
from beamz.analysis._plot_fields import (
    AxisSlice,
    FieldView,
    extract_axis_aligned_slice,
    plot_field_view,
)
from beamz.analysis._plot_monitors import (
    plot_dft_field,
    plot_mode_field_components,
    plot_result_field,
)
from beamz.analysis._plot_sections import overlay_boundaries, overlay_simulation_devices
from beamz.analysis._plot_simulation import (
    plot_simulation,
    plot_simulation_permittivity,
    view_simulation_3d,
)

__all__ = [
    "AxisSlice",
    "FieldView",
    "draw_geometry",
    "extract_axis_aligned_slice",
    "overlay_boundaries",
    "overlay_simulation_devices",
    "plot_design",
    "plot_dft_field",
    "plot_field_view",
    "plot_mode_field_components",
    "plot_result_field",
    "plot_simulation",
    "plot_simulation_permittivity",
    "view_simulation_3d",
]
