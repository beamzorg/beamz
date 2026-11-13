"""Prove exact packet addresses and support masks for the low-X call domain."""
from pathlib import Path

import pytest

from scripts.patch_cuda_native_persistent_lowx_trace import lowx_trace


def surface(z,y,x):
    if z==6:return y*31+x
    if y==14:return 15*31+z*31+x
    if x==30:return 15*31+6*31+z*14+y
    return -1


def support(n,z,y,x):
    if min(z,y,x)<0 or max(z,y,x)>n:return 0
    iz,iy,ix=z<n,y<n,x<n
    return int(iz and iy)|(int(iz and ix)<<1)|(int(iy and ix)<<2)|(int(ix)<<3)|(int(iy)<<4)|(int(iz)<<5)


@pytest.mark.parametrize('depth',[4,16])
def test_all_active_stage_threads_have_exact_neighbor4_and_surface(depth):
    checked=0
    for tid in range(1024):
        active=list(range(tid,105,1024))
        assert len(active)==(1 if tid<105 else 0)
        for i in active:
            dz,dy,dx=1+i//15,1+i%15,0
            assert 1<=dz<=7 and 1<=dy<=15
            neighbor=int(dz==0)|(int(dy==0)<<1)|(int(dx==0)<<2)
            assert neighbor==4
            old_surface=surface(dz-1 if dz else 6,dy-1 if dy else 14,dx-1 if dx else 30)
            new_surface=surface(dz-1,dy-1,30)
            assert old_surface==new_surface>=0
            for n in (25,96,512,564):
                for oz,oy,ox in ((0,0,0),(7,15,31),(14,30,62),(n//7*7,n//15*15,n//31*31)):
                    for step in range(1,depth+1):
                        old_mask=support(n,oz-1+dz-step,oy-1+dy-step,ox-1+dx-step)
                        new_mask=support(n,oz-1+dz-step,oy-1+dy-step,ox-1-step)
                        assert old_mask==new_mask
                        for field in range(6):
                            if old_mask&(1<<field):
                                surface_points=15*31+6*31+6*14
                                assert (step-1)*6*surface_points+field*surface_points+old_surface==(step-1)*6*surface_points+field*surface_points+new_surface
                                checked+=1
    assert checked>0


def test_private_helpers_keep_original_masks_cg_loads_and_stage_writers():
    source=Path('.cache/cuda-two-hour-2026-11-13-0819/source-plane-axis-cg-v1/source/src/asymmetric_native.cuh')
    generated=Path('.cache/cuda-five-hour-2026-11-13-1059/persistent-specialized-g4-floor-v2/source/src/native_persistent.cuh')
    if not(source.exists() and generated.exists()):pytest.skip('Private parent unavailable')
    old=generated.read_text();new,packet=lowx_trace(old,source.read_text())
    assert 'const int neighbor=4;' in packet
    assert 'Surface(dz-1,dy-1,BX-1)' in packet
    assert packet.count('__ldcg(point+')==6
    assert 'NativeSupportMask(p,oz-1+dz-version,oy-1+dy-version,ox-1-version)' in packet
    assert old.count('__syncthreads()')==new.count('__syncthreads()')
    assert 'PersistentStageRegisterLow<Depth,SupportSafe,T>(p,low_stage' in new
    assert 'PersistentStageSourcePendingH<Depth,SupportSafe,T>(p,low_stage' in new
    assert 'StageRegisterLow<Depth,SupportSafe,T>(p,low_stage' not in new.replace('PersistentStageRegisterLow','PrivateLow')
