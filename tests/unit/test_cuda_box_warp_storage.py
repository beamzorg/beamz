"""Check whole-register/shared cuts and resource bounds before CUDA staging."""
import pytest

from scripts.cuda_space_time_ir import IllegalSchedule
from scripts.cuda_trace_box_warp_storage import BoxWarpStorage


@pytest.mark.parametrize("fields,slots,private,shared", [
    (None,12,96,61516),('Hx',12,80,79948),(('Hx','Ex'),10,64,98380),
])
def test_exact_provider_cut_and_dense_field_placement(fields,slots,private,shared):
    report=BoxWarpStorage(shared_field=fields,profile_slots=slots).prove()
    assert report['checked_routes']==245760
    assert report['private_field_register_requirement']==private
    assert report['shared_bytes']==shared
    assert report['cached_shared_bytes']<=101376


@pytest.mark.parametrize("storage", [
    BoxWarpStorage(private_z=0),BoxWarpStorage(private_y=3),
    BoxWarpStorage(shared_field=('Hx','Hx')),BoxWarpStorage(shared_field='unknown'),
])
def test_invalid_or_aliasing_register_rectangles_rejected(storage):
    with pytest.raises(IllegalSchedule):storage.validate()


def test_two_dense_fields_cannot_cache_all12terms_within_block_capacity():
    report=BoxWarpStorage(shared_field=('Hx','Ex')).prove()
    assert report['cached_shared_bytes']>101376


@pytest.mark.parametrize('slot_major',[False,True])
def test_provider_primary_homes_remove_private_mirrors_without_extra_shared(slot_major):
    mirrored=BoxWarpStorage((8,16,32),4,2,None,slot_major=slot_major)
    primary=BoxWarpStorage((8,16,32),4,2,None,provider_home=True,slot_major=slot_major)
    a,b=mirrored.prove(),primary.prove()
    assert a['private_field_register_requirement']==48
    assert b['private_field_register_requirement']==26
    assert b['shared_bytes']==a['shared_bytes']==45132
    assert b['cached_shared_bytes']==48588
    assert b['checked_routes']==122880
    assert b['routes']['shared']>a['routes']['shared']
