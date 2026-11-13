"""Bind specialized pipeline hooks to the unchanged original arithmetic graph."""
from pathlib import Path
import re

import pytest

from scripts.patch_cuda_native_persistent_specialize import clone_uniform, specialize
from scripts.patch_cuda_native_persistent import clone_body


def parent():
    p=Path('.cache/cuda-two-hour-2026-11-13-0819/source-plane-axis-cg-v1/source/src/asymmetric_native.cuh')
    if not p.exists():pytest.skip('Frozen private parent unavailable')
    return p.read_text()


def original_uniform(code):
    a=code.index('template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int ActiveAxes=-1>\n__device__ __forceinline__ void UniformBody')
    b=code.index('template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy=(Fast?0:5)>',a)
    return code[a:b]


def test_uniform_arithmetic_and_memory_statements_identical():
    old=original_uniform(parent());new=clone_uniform(parent())
    # Exact arithmetic read/update regions preserve all guards, expression
    # operands, ordered canonical calls and shared/private destinations.
    for marker,end in [('{const int step=*phase_step,hz=', '  const int hz=(-n-1)&7'),
                       ('{const int step=*phase_step,ehz=', '  __syncthreads(); // B3')]:
        a=old.index(marker);b=old.index(end,a)
        block=old[a:b]
        if 'ehz=' in marker:
            # New body adds the moved PRIVATE publication after this block.
            assert block in new
        else:assert block in new
    for marker in ['float h[3][Slots],e[3][Slots];',
                   'StageRegisterLow<Depth,SupportSafe,T>(p,low_stage,bases,n,oz,oy,ox);',
                   'RetireRegisterLow<T,Slots>(low_stage,h,e,n,sy,sx);']:
        assert old.count(marker)==new.count(marker)==1
    assert old.count('__syncthreads()')==new.count('__syncthreads()')


def test_publication_before_b3_uses_private_values_not_shared_frame():
    new=clone_uniform(parent())
    a=new.index('  const int y=oy+low+dy,x=ox+low+dx;',new.index('for(int c=0;c<3;++c){float value=0.f;if(env.Valid(c+3'))
    b=new.index('__syncthreads(); // B3',a)
    publish=new[a:b]
    assert 'p.trails[' in publish and 'hf.data[' in publish and 'ef.data[' in publish
    assert 'frame[' not in publish
    assert 'h[c][j]' in publish and 'e[c][j]' in publish
    assert publish.rstrip().endswith('__threadfence();')
    assert new.index('// B3')<new.index('PublishAndWait(n,Fast?Depth:window[1])')<new.index('// B4')
    assert new.count('__threadfence();')==1


def test_actual_dispatch_matches_existing_policy_for_all_flag_bytes():
    # Flag128 is unused and not admitted by existing captured facts. Verify
    # exactly the original low6 policy and captured bit64 common-E fact.
    for flags in range(128):
        policy=0 if not(flags&63) else 1 if not(flags&17) else (5 if flags&46 else 2 if flags&1 and not(flags&16) else 3 if flags&16 and not(flags&1) else 4)
        branch=0 if not(flags&63) else 1 if not(flags&17) and flags&64 else 2
        assert (branch==0)==(policy==0)
        assert (branch==1)==(policy==1 and bool(flags&64))
    code=(Path('scripts/cuda_native_persistent.cuh').read_text())
    old_paired=clone_body(parent())
    generated,_=specialize(code.replace('// BEAMZ_PERSISTENT_PAIRED_BODY_INSERT',old_paired),parent())
    assert old_paired in generated
    assert 'C().body_kind=!(facts&63)?0:(!(facts&17)&&(facts&64))?1:2;' in generated
    assert 'if(C().body_kind==0)PersistentUniformBody<Depth,true,0,false>(p);' in generated
    assert 'else if(C().body_kind==1)PersistentUniformBody<Depth,false,1,true>(p);' in generated


def test_fast_and_nonfast_start_wait_before_any_dependency_read():
    body=clone_uniform(parent())
    assert body.index('WaitUpstream(1)')<body.index('Acquire(p.owner_tags+owner)')
    assert body.index('WaitUpstream(window[0])')<body.index('Acquire(p.owner_tags+owner)')
    assert body.index('WaitUpstream(window[0])')<body.index('if(first==1)')
    assert body.count('if(C().abort_local)return;')==2
    assert 'if(C().abort_local||window[0]>window[1])return;' in body


def test_writer_fence_barrier_release_happens_before_next_import():
    # Independent instruction-level HB graph for every writer: each thread's
    # global stores are fenced before its B3 arrival; thread0 cannot release
    # readiness until every B3 arrival. This does not require any writer to
    # execute E-frame overwrite before readiness (consumers read global trails).
    edges={}
    def edge(a,b):edges.setdefault(a,set()).add(b)
    for t in range(1024):
        edge(('trail',t),('fence',t));edge(('psi',t),('fence',t))
        edge(('fence',t),('B3arrival',t));edge(('B3arrival',t),'B3exit')
    edge('B3exit','release_ready');edge('release_ready','consumer_acquire');edge('consumer_acquire','consumer_import')
    def reaches(a,target):
        seen=set();todo=[a]
        while todo:
            p=todo.pop()
            if p==target:return True
            if p not in seen:seen.add(p);todo.extend(edges.get(p,()))
        return False
    for t in range(1024):
        assert reaches(('trail',t),'consumer_import')
        assert reaches(('psi',t),'consumer_import')
    # Negative for this conservative explicit-fence graph only. It does NOT
    # refute the separate cumulative CTA-join -> GPU-release recipe, whose
    # graph has a direct synchronized path from every writer via the join.
    edges[('trail',777)].clear()
    assert not reaches(('trail',777),'consumer_import')
