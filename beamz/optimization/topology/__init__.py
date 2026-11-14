"""Topology gradients, density filters, and geometry transforms."""

from .filtering import masked_morphological_filter
from .gradient import compute_overlap_gradient
from .morphology import (
    grayscale_closing,
    grayscale_dilation,
    grayscale_erosion,
    grayscale_opening,
    smooth_max,
    smooth_min,
)
from .projections import project_density, smoothed_heaviside
from .transforms.conic import (
    _conic_filter,
    _conic_kernel,
    generate_conic_kernel,
    masked_conic_filter,
)
from .transforms.density import (
    _TRANSFORM_STATIC_ARGS,
    apply_fixed_density,
    transform_density,
)
from .transforms.differentiation import compute_parameter_gradient_vjp
from .transforms.mask import create_optimization_mask
from .transforms.padding import fold_high_side_yee_padding_to_shape
