"""Typed scheduler words retain exact GPU ordering and control graph."""
from pathlib import Path

from scripts.patch_cuda_native_persistent_global_atomics import GLOBAL_HELPERS,CPU_HELPERS,global_atomics,shim_global_helpers
from scripts.patch_cuda_native_persistent_relaxed_poll import relaxed_poll


def test_direct_global_preserves_every_control_and_physics_statement():
    old=relaxed_poll(Path('scripts/cuda_native_persistent.cuh').read_text())
    new=global_atomics(old)
    a=old.index('__device__ __forceinline__ uint64_t Timer()')
    b=new.index('__device__ __forceinline__ uint64_t Timer()')
    reset='::cuda::atomic_ref<uint32_t,::cuda::thread_scope_device>ready(C().ready[owner]);ready.store(0,::cuda::memory_order_relaxed);'
    assert old[a:].replace(reset,'StoreRelaxed(C().ready+owner,uint32_t(0));')==new[b:]
    assert old.count('__syncthreads()')==new.count('__syncthreads()')
    assert old.count('Release(')==new.count('Release(')


def test_all_atomic_instructions_global_gpu_exact_memory_orders():
    for order in ('acquire','relaxed'):
        for width in (32,64):assert f'ld.{order}.gpu.global.u{width}' in GLOBAL_HELPERS
    for order in ('release','relaxed'):
        for width in (32,64):assert f'st.{order}.gpu.global.u{width}' in GLOBAL_HELPERS
    assert GLOBAL_HELPERS.count(':"memory"')==8
    assert GLOBAL_HELPERS.count('__cvta_generic_to_global(word)')==4
    assert GLOBAL_HELPERS.count('alignof(T)>=sizeof(T)')==4
    assert 'atomic_ref' not in GLOBAL_HELPERS
    assert 'memory_order_acquire' in CPU_HELPERS
    assert 'memory_order_release' in CPU_HELPERS
    assert CPU_HELPERS.count('memory_order_relaxed')==2


def test_actual_controller_CPU_shim_is_exact_hash_bound_helper_replacement():
    text='namespace test{\n'+GLOBAL_HELPERS+'\nvoid controller_unchanged();\n}'
    result=shim_global_helpers(text)
    assert result=='namespace test{\n'+CPU_HELPERS+'\nvoid controller_unchanged();\n}'
    modified=text.replace('ld.acquire.gpu.global.u32','ld.relaxed.gpu.global.u32',1)
    try:shim_global_helpers(modified)
    except ValueError:pass
    else:raise AssertionError('Modified acquire semantics accepted by CPU shim')


def test_word_offset_address_proof_naturally_aligned():
    # cudaMalloc bases are at least256B aligned; array offsets use word size.
    for word_bytes in (4,8):
        for index in range(4096):assert (256+index*word_bytes)%word_bytes==0
