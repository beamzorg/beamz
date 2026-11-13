"""Runtime scalar aliases exclude scene values and reject unsafe equality."""

import numpy as np
import pytest

from scripts.cuda_space_time_ir import IllegalSchedule, yee_operator
from scripts.cuda_stencil_trace import coefficient_reads
from scripts.cuda_trace_common_e import (
    CommonElectricScalarClass,
    common_electric_scalar_class,
    specialize_common_e,
    verify_common_e,
)
from scripts.cuda_trace_runtime_uniform import (
    UniformElectricRegionClass,
    specialize_runtime_e,
)


def test_value_independent_common_runtime_leaf_is_exactly_one_scalar():
    operator, _ = specialize_runtime_e(yee_operator(cpml_axes=(0, 1, 2)), UniformElectricRegionClass())
    a, b = np.full(3, .125, np.float32), np.full(3, 2.75, np.float32)
    fact = common_electric_scalar_class(a)
    assert fact == common_electric_scalar_class(b) == CommonElectricScalarClass()
    verify_common_e(fact, a)
    verify_common_e(fact, b)
    first, proof_a = specialize_common_e(operator, fact)
    second, proof_b = specialize_common_e(operator, common_electric_scalar_class(b))
    assert proof_a == proof_b
    names = {r.name for u in first.updates for r in coefficient_reads(u.value) if r.name.startswith("source_E")}
    assert names == {"source_Ex"}
    assert [u.reads for u in first.updates] == [u.reads for u in second.updates] == [u.reads for u in operator.updates]


@pytest.mark.parametrize("case", ["bit", "signed-zero", "nan", "inf", "float64", "shape", "strided"])
def test_unproven_or_stale_runtime_equality_rejects(case):
    value = np.ones(3, np.float32)
    if case == "bit":
        value.view(np.uint32)[1] ^= 1
    elif case == "signed-zero":
        value[:] = [0., -0., 0.]
    elif case == "nan":
        value[:] = np.nan
    elif case == "inf":
        value[:] = np.inf
    elif case == "float64":
        value = value.astype(np.float64)
    elif case == "shape":
        value = value.reshape(1, 3)
    elif case == "strided":
        value = np.ones(6, np.float32)[::2]
    assert common_electric_scalar_class(value) is None
    with pytest.raises(IllegalSchedule):
        verify_common_e(CommonElectricScalarClass(), value)


def test_matching_negative_zeros_preserve_runtime_bit_identity():
    assert common_electric_scalar_class(np.full(3, -0., np.float32)) == CommonElectricScalarClass()


def test_equality_does_not_replace_missing_spatial_region_proof():
    with pytest.raises(IllegalSchedule, match="region certificate"):
        specialize_common_e(yee_operator(cpml_axes=(0, 1, 2)), CommonElectricScalarClass())
