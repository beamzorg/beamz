"""Runtime-valued facts may change leaf placement, never Yee arithmetic/order."""

import numpy as np
import pytest

from scripts.cuda_native_geometry_ir import SUPPORT_DELTA
from scripts.cuda_space_time_ir import IllegalSchedule, compile_stream, yee_operator
from scripts.cuda_stencil_trace import coefficient_reads
from scripts.cuda_trace_runtime_uniform import (
    UniformElectricRegionClass,
    certify_region,
    specialize_runtime_e,
    verify_region,
)


def packet(value=.125):
    inputs = [None] * 43
    for c, delta in enumerate(SUPPORT_DELTA):
        shape = tuple(32 + d for d in delta)
        inputs[c] = np.zeros(shape, np.float32)
    for c in range(3):
        inputs[37 + c] = np.array([value, .25], np.float32)
        inputs[40 + c] = np.zeros((inputs[3 + c].size + 3) // 4, np.int32)
    return inputs


def witness(inputs, value=.125):
    return certify_region(inputs, (14, 15, 31), (7, 15, 31), 4,
                          np.full(3, value, np.float32))


def test_actual_trajectory_bits_are_bound_without_baking_them_into_kernel():
    a, b = packet(.125), packet(.0625)
    wa, wb = witness(a, .125), witness(b, .0625)
    assert wa.scalar_bits != wb.scalar_bits
    verify_region(wa, a, np.full(3, .125, np.float32))
    verify_region(wb, b, np.full(3, .0625, np.float32))
    operator = yee_operator(cpml_axes=(0, 1, 2))
    specialized, evidence = specialize_runtime_e(operator, UniformElectricRegionClass())
    assert evidence == specialize_runtime_e(operator, UniformElectricRegionClass())[1]
    assert [u.reads for u in specialized.updates] == [u.reads for u in operator.updates]
    assert compile_stream(specialized)["frontier"] == compile_stream(operator)["frontier"]
    e = {r for u in specialized.updates for r in coefficient_reads(u.value) if r.name.startswith("source_E")}
    assert {r.name for r in e} == {"source_Ex", "source_Ey", "source_Ez"}
    assert all(r.axes == () for r in e)


def test_arithmetic_nodes_and_operand_positions_are_identical():
    old = yee_operator(cpml_axes=(0, 1, 2))
    new, _ = specialize_runtime_e(old, UniformElectricRegionClass())

    def compare(a, b):
        assert a.op == b.op
        if a.op == "coefficient" and a.args[0].name.startswith("source_E"):
            assert a.args[0].name == b.args[0].name and b.args[0].axes == ()
            return
        assert len(a.args) == len(b.args)
        for x, y in zip(a.args, b.args, strict=True):
            if hasattr(x, "op"):
                compare(x, y)
            else:
                assert x == y

    for a, b in zip(old.updates, new.updates, strict=True):
        assert a.output == b.output
        compare(a.value, b.value)


def test_changed_inside_material_value_rejects_region_dispatch():
    inputs = packet()
    fact = witness(inputs)
    lo = fact.bounds[0][0]
    shape = inputs[3].shape
    linear = (lo[0] * shape[1] + lo[1]) * shape[2] + lo[2]
    inputs[40].view(np.uint32)[linear >> 2] |= np.uint32(1 << (8 * (linear & 3)))
    with pytest.raises(IllegalSchedule, match="not bit-uniform"):
        verify_region(fact, inputs, np.full(3, .125, np.float32))


def test_immutable_snapshot_change_outside_region_is_not_silently_reused():
    inputs = packet()
    fact = witness(inputs)
    inputs[37][1] = np.float32(.5)  # Unused entry; geometry equality still holds.
    with pytest.raises(IllegalSchedule, match="became stale"):
        verify_region(fact, inputs, np.full(3, .125, np.float32))


def test_signed_zero_is_a_bit_fact_and_out_of_range_codes_are_positive_zero():
    inputs = packet(-0.)
    fact = witness(inputs, -0.)
    with pytest.raises(IllegalSchedule):
        verify_region(fact, inputs, np.zeros(3, np.float32))
    for c in range(3):
        inputs[40 + c].view(np.uint32)[:] = np.uint32(0xffffffff)
    certify_region(inputs, (14, 15, 31), (7, 15, 31), 4, np.zeros(3, np.float32))


@pytest.mark.parametrize("values", [np.ones(3, np.float64), np.array([np.nan] * 3, np.float32), np.zeros(2, np.float32)])
def test_runtime_scalar_precision_shape_and_finiteness_fail_closed(values):
    with pytest.raises(IllegalSchedule):
        certify_region(packet(), (14, 15, 31), (7, 15, 31), 4, values)


def test_all_empty_trajectory_is_not_a_positive_witness():
    with pytest.raises(IllegalSchedule, match="no supported E update"):
        certify_region(packet(), (100, 100, 100), (7, 15, 31), 4,
                       np.full(3, .125, np.float32))
