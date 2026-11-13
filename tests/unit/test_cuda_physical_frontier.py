"""Legality checks for replacing versioned trails by physical working fields."""
import pytest

from scripts.prove_cuda_physical_frontier import componentwise_theorem, token_schedule


@pytest.mark.parametrize("owned", [(3,5,7),(7,15,31),(15,15,15)])
@pytest.mark.parametrize("depth", [4,16])
def test_full_rank_raw_war_and_seeded_version_schedule(owned,depth):
    proof=componentwise_theorem(owned,depth)
    assert proof["checks"]["field_dependences"]==30
    result=token_schedule((5,7,9),owned,depth,seed=83)
    assert result["traced_reads"]>0
    assert result["publications"]>0


@pytest.mark.parametrize("negative", ["reverse-ranks","output-cleared-before-bootstrap","stale-ghost-version","publish-future-version"])
def test_destructive_reuse_and_wrong_order_are_rejected(negative):
    with pytest.raises(AssertionError):
        token_schedule((5,7,9),(3,5,7),4,negative=negative)


@pytest.mark.parametrize("seed", [13,37,83])
def test_three_parent_only_order_can_overtake_ranks_without_destroying_values(seed):
    result=token_schedule((5,7,9),(3,5,7),16,seed=seed,order_kind="three-parent-DAG")
    assert result["rank_overtakes"]>0


@pytest.mark.parametrize("depth", [4,16])
def test_mixed_parent_and_two_children_share_physical_frontier(depth):
    result=token_schedule((9,11,13),(6,5,7),depth,seed=13,order_kind="three-parent-DAG",heterogeneous_z=True)
    assert len(result["local_owned_shapes"])==2


def test_reversing_internal_children_destroys_a_pending_version():
    with pytest.raises(AssertionError):
        token_schedule((9,11,13),(6,5,7),16,seed=13,order_kind="three-parent-DAG",heterogeneous_z=True,negative="reverse-internal-children")
