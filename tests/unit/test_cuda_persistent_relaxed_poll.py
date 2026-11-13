"""Failed relaxed polls never authorize reads of mutable payload."""
from pathlib import Path

from scripts.patch_cuda_native_persistent_relaxed_poll import relaxed_poll


def test_relaxed_polls_retain_all_successful_acquires_and_releases():
    old=Path('scripts/cuda_native_persistent.cuh').read_text();new=relaxed_poll(old)
    assert 'const uint32_t seen=Relaxed(C().completed+tile);' in new
    assert 'if(seen==C().epoch&&Acquire(C().completed+tile)==C().epoch)return true;' in new
    assert 'const uint64_t tag=Relaxed(C().tags+owner);' in new
    assert 'uint32_t ready=Relaxed(C().ready+owner);' in new
    assert 'if(ready>=uint32_t(step)&&Acquire(C().tags+owner)==expected){\n    ready=Acquire(C().ready+owner);if(ready>=uint32_t(step))return true;' in new
    assert 'if(ready!=previous_ready){previous_ready=ready;last=Timer();}' in new
    assert 'if(Relaxed(C().abort_global))' in new
    assert old.count('Release(')==new.count('Release(')
    # The complete physics bodies and queue publication code are outside the
    # controller rewrite, including all global fences and release stores.
    marker='struct PersistentEnvironment:'
    assert old[old.index(marker):]==new[new.index(marker):]


def test_success_tag_acquire_orders_ready_reset_before_ready_load():
    # C++ memory-order HB graph, independent of scheduling simulation. Atomic
    # write/read coherence forbids a read older than a HB-preceding reset.
    def before(edges,a,b):
        seen=set();todo=[a]
        while todo:
            p=todo.pop()
            if p==b:return True
            if p not in seen:seen.add(p);todo.extend(v for u,v in edges if u==p)
        return False
    edges=[('oldReady','reset0'),('reset0','releaseTag'),
           ('releaseTag','acquireTag'),('acquireTag','acquireReady')]
    assert before(edges,'reset0','acquireReady')
    # Invalid lowering: testing ready before acquire-tag can observe the old
    # holder's depth value, even if the later tag acquire sees the new holder.
    negative=[('oldReady','reset0'),('reset0','releaseTag'),
              ('releaseTag','acquireTag'),('acquireReady','acquireTag')]
    assert not before(negative,'reset0','acquireReady')
    # Speculative relaxed values have no authorization edge into payload.
    assert not before(edges,'relaxedReady','acquireReady')


def test_only_atomic_polls_no_plain_l1_reads():
    new=relaxed_poll(Path('scripts/cuda_native_persistent.cuh').read_text())
    a=new.index('template<class T>__device__ __forceinline__ T Relaxed(')
    b=new.index('template<class T>__device__ __forceinline__ void Release(',a)
    assert '::cuda::atomic_ref<T,::cuda::thread_scope_device>' in new[a:b]
    assert 'a.load(::cuda::memory_order_relaxed)' in new[a:b]


def test_watchdog_sees_current_progress_after_speculative_old_high_value():
    observations=[16,0,1,2,4,8,16]
    previous=0;resets=[]
    for tick,ready in enumerate(observations):
        if ready!=previous:previous=ready;resets.append(tick)
    assert resets==list(range(len(observations)))
    previous=0;old_resets=[]
    for tick,ready in enumerate(observations):
        if ready>previous:previous=ready;old_resets.append(tick)
    assert old_resets==[0]  # Previous high-water policy could falsely abort.
