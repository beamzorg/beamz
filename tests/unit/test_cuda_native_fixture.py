"""Full native ABI layout and state-parity acceptance guards, without JAX."""

import numpy as np

from scripts.benchmark_cuda_native_runtime import (
    chunks_for,
    compare_state,
    logical_native_state,
)
from scripts.export_cuda_native_fixture import (
    PERSISTENT_INPUTS,
    WORKSPACE_INPUTS,
    abi_input_names,
    buffer_digest,
    state_output_mapping,
)


def test_exact_134_buffer_layout_and_copy_roles():
    names = abi_input_names()
    assert len(set(names)) == len(names) == 134
    for index, expected in {
        0: "hx",
        31: "h_psi_0",
        62: "e_psi_0",
        74: "final_hx",
        80: "scratch_hx",
        86: "source_pre_e_ex_coefficients",
        113: "monitor_indices",
        120: "monitor_dft_re",
        126: "monitor_origin",
        127: "monitor_current_step",
        128: "monitor_elapsed_steps",
        129: "plan_bounds",
    }.items():
        assert names[index] == expected
    assert not PERSISTENT_INPUTS & WORKSPACE_INPUTS
    assert len(PERSISTENT_INPUTS) == 21


def test_buffer_digest_handles_empty_metrics_and_distinguishes_scalar_shape():
    assert buffer_digest(np.empty((0,), np.float32)) == buffer_digest(
        np.empty((0,), np.float32)
    )
    assert buffer_digest(np.float32(0)) != buffer_digest(np.zeros((1,), np.float32))
    assert buffer_digest(np.empty((0,), np.float32)) != buffer_digest(
        np.empty((0, 1), np.float32)
    )


def test_chunk_policy_preserves_boundary_and_odd_tail():
    assert chunks_for(259, "chunk256") == [256, 3]
    assert chunks_for(259, "whole") == [259]
    assert chunks_for(256, "chunk256") == [256]
    assert sum(chunks_for(6559, "chunk256")) == 6559


def test_parity_cannot_hide_cpml_dft_or_clock_failure():
    reference = {
        "cpml_psi_h_terms.0": np.ones(2, np.float32),
        "dft_vec_im": np.ones(2, np.float32),
        "current_step": np.int32(7),
    }
    actual = {
        "cpml_psi_h_terms.0": np.zeros(2, np.float32),
        "dft_vec_im": np.zeros(2, np.float32),
        "current_step": np.int32(8),
    }
    report = compare_state(actual, reference)
    assert not report["passed"]
    assert len(report["leaves"]) == 3
    assert all(not row["passed"] for row in report["leaves"])


def test_logical_fields_follow_reported_bank_not_total_step_parity():
    leaves = [
        {
            "name": name,
            "dtype": "<f4",
            "initial_key": name,
            "mapping": state_output_mapping(name),
        }
        for name in ("ex", "hx", "t")
    ]
    fixture = {"state_leaves": leaves}
    outputs = [np.full(2, i, np.float32) for i in range(36)]
    clocks = {"time": 1.25, "current_step": 7}
    a = logical_native_state(fixture, {}, outputs, clocks, 0)
    b = logical_native_state(fixture, {}, outputs, clocks, 1)
    np.testing.assert_array_equal(a["ex"], outputs[3])
    np.testing.assert_array_equal(b["ex"], outputs[21])
    np.testing.assert_array_equal(a["hx"], outputs[0])
    assert a["t"] == np.float32(1.25)
