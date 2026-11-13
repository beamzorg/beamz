"""Trail lifetimes must survive asynchronous order within a rank frontier."""

import pytest

from scripts.prove_cuda_systolic_ring import prove_ring


@pytest.mark.parametrize("cpml", [False, True])
def test_four_frontiers_retain_every_versioned_operand(cpml):
    result = prove_ring((4, 5, 6), (2, 3, 4), 4, cpml=cpml)
    assert result["checked_consumer_values"] > 0
    assert result["peak_live_tokens"] * 4 <= result["ring_payload_bound_bytes"]


def test_reusing_an_epoch_before_its_last_consumer_is_rejected():
    with pytest.raises(AssertionError, match="Retired or wrong-version"):
        prove_ring(cpml=True, epochs=3)


def test_reading_the_wrong_history_version_is_rejected():
    with pytest.raises(AssertionError, match="Retired or wrong-version"):
        prove_ring(cpml=True, wrong_version=True)
