import numpy as np

from scripts.validate_cuda_native_profile_facts import (
    PROFILE_INDICES,
    equal_axis_profiles,
    profile_packet,
)


def packet():
    values = [np.array(0, dtype=np.float32) for _ in range(134)]
    # Deliberately alias each phase's host profile arrays. A changed-term witness
    # must split those aliases, otherwise its paired term changes accidentally.
    h = np.array([.125, .25, .5], dtype=np.float32)
    e = np.array([.375, .625, .875], dtype=np.float32)
    for i in PROFILE_INDICES:
        values[i] = h if i < 31 else e
    return values


def test_profile_witness_splits_host_aliases_without_changing_initial_packet():
    source = packet()
    changed = profile_packet(source, "unequal-paired-H-profile")
    np.testing.assert_array_equal(source[13], [.125, .25, .5])
    np.testing.assert_array_equal(changed[13], source[13])
    assert not np.array_equal(changed[28], source[28])
    assert not equal_axis_profiles(changed)
    assert equal_axis_profiles(source)
    assert all(changed[i] is source[i] for i in range(134) if i not in PROFILE_INDICES)


def test_signed_zero_and_phase_boundaries_are_bitwise_facts():
    source = packet()
    assert not np.array_equal(source[13], source[44])
    assert equal_axis_profiles(source)  # H and E must not alias across phases.
    changed = profile_packet(source, "signed-zero-paired-mismatch")
    assert changed[13].view(np.uint32)[0] == 0
    assert changed[28].view(np.uint32)[0] == 0x80000000
    assert not equal_axis_profiles(changed)
    changed = profile_packet(source, "distinct-H-E-phase-profiles")
    assert equal_axis_profiles(changed)
    np.testing.assert_array_equal(changed[13], source[13])
    assert not np.array_equal(changed[44], source[44])


def test_matching_nonfinite_profile_bits_are_not_a_valid_alias_certificate():
    for bits in (0x7F800000, 0xFF800000, 0x7FC00001):
        source = packet()
        for index in (13, 28):
            source[index] = source[index].copy()
            source[index].view(np.uint32)[0] = bits
        assert source[13].tobytes() == source[28].tobytes()
        assert not equal_axis_profiles(source)
