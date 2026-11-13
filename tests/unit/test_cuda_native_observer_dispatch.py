import numpy as np

from scripts.validate_cuda_native_observer_dispatch import separated_probe_packet


def test_separate_observer_keeps_both_source_groups_at_original_yee_site():
    n = 40
    arrays = [np.zeros((), np.float32) for _ in range(134)]
    shapes = [(n, n, n + 1), (n, n + 1, n), (n + 1, n, n),
              (n + 1, n + 1, n), (n + 1, n, n + 1), (n, n + 1, n + 1)]
    arrays[:6] = [np.zeros(shape, np.float32) for shape in shapes]
    arrays[113] = np.full((2, 6, 3, 2), -1, np.int32)
    arrays[114] = np.zeros((2, 6, 3, 2), np.float32)
    source, observer = (32, 32, 14), (32, 32, 15)
    values, attrs = separated_probe_packet(arrays, {}, source, observer, 35)
    assert attrs["coincident_source_group_mask"] == 160
    for group in (5, 7):
        assert values[88 + 3 * group].tolist() == [list(source), list(source)]
        assert values[87 + 3 * group].shape == (2, 35)
    for monitor in range(2):
        for component in range(6):
            native = (component + 3) % 6
            flat = values[113][monitor, component, 0, 0]
            assert tuple(np.unravel_index(flat, shapes[native])) == observer
    assert np.count_nonzero(values[114]) == 12
    assert np.all(arrays[113] == -1)
    assert not np.any(arrays[114])
    assert all(values[i] is arrays[i] for i in range(6))


def test_witness_reaches_an_observer_only_safe_owner_trajectory():
    # Owned X31 begins at physical15 over a16-step block. Its trajectory has
    # the observer but not the source, and lies beyond12-cell CPML atN136.
    low, high = 31 - 16, 31 + 31 - 2
    assert 14 < low <= 15 <= 16 <= high
    assert low >= 12 and high < 136 - 12
    # Z and Y owner choices serving the same point likewise stay interior.
    assert all(origin - 16 >= 12 and origin + 7 - 2 < 136 - 12
               for origin in (28, 35, 42))
    assert all(origin - 16 >= 12 and origin + 15 - 2 < 136 - 12
               for origin in (30, 45))
