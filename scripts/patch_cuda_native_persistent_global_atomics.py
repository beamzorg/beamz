"""Trace-type scheduler atomics as device-global, naturally aligned words.

NativeSession constructs every scheduler pointer from its private DeviceMemory
cudaMalloc allocations. Direct PTX preserves GPU scope and acquire/release
semantics while removing impossible generic local/shared address-space cases.
"""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path

try:
    from scripts.patch_cuda_native_persistent import sha,once
except ModuleNotFoundError:
    from patch_cuda_native_persistent import sha,once


GLOBAL_HELPERS=r'''// TRACE_TYPED_GLOBAL_ATOMICS_BEGIN
template<class T>__device__ __forceinline__ T Acquire(const T*word){
 static_assert((sizeof(T)==4||sizeof(T)==8)&&alignof(T)>=sizeof(T),"Aligned32/64 scheduler words only");
 const uint64_t address=__cvta_generic_to_global(word);
 if constexpr(sizeof(T)==4){uint32_t value;asm volatile("ld.acquire.gpu.global.u32 %0, [%1];":"=r"(value):"l"(address):"memory");return T(value);}
 else{uint64_t value;asm volatile("ld.acquire.gpu.global.u64 %0, [%1];":"=l"(value):"l"(address):"memory");return T(value);}
}
template<class T>__device__ __forceinline__ T Relaxed(const T*word){
 static_assert((sizeof(T)==4||sizeof(T)==8)&&alignof(T)>=sizeof(T),"Aligned32/64 scheduler words only");
 const uint64_t address=__cvta_generic_to_global(word);
 if constexpr(sizeof(T)==4){uint32_t value;asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];":"=r"(value):"l"(address):"memory");return T(value);}
 else{uint64_t value;asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];":"=l"(value):"l"(address):"memory");return T(value);}
}
template<class T>__device__ __forceinline__ void Release(T*word,T value){
 static_assert((sizeof(T)==4||sizeof(T)==8)&&alignof(T)>=sizeof(T),"Aligned32/64 scheduler words only");
 const uint64_t address=__cvta_generic_to_global(word);
 if constexpr(sizeof(T)==4){asm volatile("st.release.gpu.global.u32 [%0], %1;"::"l"(address),"r"(uint32_t(value)):"memory");}
 else{asm volatile("st.release.gpu.global.u64 [%0], %1;"::"l"(address),"l"(uint64_t(value)):"memory");}
}
template<class T>__device__ __forceinline__ void StoreRelaxed(T*word,T value){
 static_assert((sizeof(T)==4||sizeof(T)==8)&&alignof(T)>=sizeof(T),"Aligned32/64 scheduler words only");
 const uint64_t address=__cvta_generic_to_global(word);
 if constexpr(sizeof(T)==4){asm volatile("st.relaxed.gpu.global.u32 [%0], %1;"::"l"(address),"r"(uint32_t(value)):"memory");}
 else{asm volatile("st.relaxed.gpu.global.u64 [%0], %1;"::"l"(address),"l"(uint64_t(value)):"memory");}
}
// TRACE_TYPED_GLOBAL_ATOMICS_END
'''


CPU_HELPERS=r'''template<class T>__device__ __forceinline__ T Acquire(const T*word){
 ::cuda::atomic_ref<T,::cuda::thread_scope_device>a(*const_cast<T*>(word));return a.load(::cuda::memory_order_acquire);
}
template<class T>__device__ __forceinline__ T Relaxed(const T*word){
 ::cuda::atomic_ref<T,::cuda::thread_scope_device>a(*const_cast<T*>(word));return a.load(::cuda::memory_order_relaxed);
}
template<class T>__device__ __forceinline__ void Release(T*word,T value){
 ::cuda::atomic_ref<T,::cuda::thread_scope_device>a(*word);a.store(value,::cuda::memory_order_release);
}
template<class T>__device__ __forceinline__ void StoreRelaxed(T*word,T value){
 ::cuda::atomic_ref<T,::cuda::thread_scope_device>a(*word);a.store(value,::cuda::memory_order_relaxed);
}
'''


def global_atomics(code):
    a=code.index('template<class T>__device__ __forceinline__ T Acquire(')
    b=code.index('__device__ __forceinline__ uint64_t Timer()',a)
    if code[a:b].count('::cuda::atomic_ref')!=3:raise ValueError('Expected acquired/relaxed/release wrappers')
    code=code[:a]+GLOBAL_HELPERS+code[b:]
    code=once(code,'::cuda::atomic_ref<uint32_t,::cuda::thread_scope_device>ready(C().ready[owner]);ready.store(0,::cuda::memory_order_relaxed);',
        'StoreRelaxed(C().ready+owner,uint32_t(0));')
    return code


def shim_global_helpers(control):
    a=control.index('// TRACE_TYPED_GLOBAL_ATOMICS_BEGIN')
    b=control.index('// TRACE_TYPED_GLOBAL_ATOMICS_END',a)+len('// TRACE_TYPED_GLOBAL_ATOMICS_END\n')
    if control[a:b]!=GLOBAL_HELPERS:raise ValueError('Unexpected direct-global PTX atomic definitions')
    return control[:a]+CPU_HELPERS+control[b:]


def patch(source,output):
    source,output=source.resolve(),output.resolve()
    if output.exists() or source==output or source in output.parents or output in source.parents:
        raise ValueError('New disjoint copied source required')
    old=(source/'src/native_persistent.cuh').read_text();new=global_atomics(old)
    host=(source/'src/native_extension.cc').read_text()
    here=Path(__file__).resolve().parent;session=(here/'patch_cuda_native_persistent_session.py').read_text()
    required=['Check(cudaMalloc(&data, std::max(size, size_t{1})), "cudaMalloc");',
        'moving_workspace_.owner_tags=static_cast<uint64_t*>(upload(',
        'moving_workspace_.epoch=static_cast<uint32_t*>(upload(',
        'moving_workspace_.bad=static_cast<int32_t*>(upload(']
    if not all(x in host for x in required):raise ValueError('NativeSession moving metadata provenance changed')
    for member in ('next_job','abort','claimed_epoch','completed_epoch','ready_step'):
        if f'pw.{member}=' not in session:raise ValueError('Private persistent metadata allocation changed')
    if 'auto memory=std::make_unique<DeviceMemory>(size);' not in session:
        raise ValueError('Persistent allocator is no longer native cudaMalloc DeviceMemory')
    shutil.copytree(source,output);(output/'src/native_persistent.cuh').write_text(new)
    proof={'parent_persistent_consumer_sha256':sha(old.encode()),'persistent_consumer_sha256':sha(new.encode()),
        'patcher_sha256':sha(Path(__file__).read_bytes()),'global_helpers_sha256':sha(GLOBAL_HELPERS.encode()),
        'cudaMalloc_native_session_parent_sha256':sha(host.encode()),'persistent_session_patcher_sha256':sha(session.encode()),
        'workspace_abi_changed':False,'expected_entries':65,'old_body_unchanged':old[old.index('__device__ __forceinline__ uint64_t Timer()'):].replace('::cuda::atomic_ref<uint32_t,::cuda::thread_scope_device>ready(C().ready[owner]);ready.store(0,::cuda::memory_order_relaxed);','StoreRelaxed(C().ready+owner,uint32_t(0));')==new[new.index('__device__ __forceinline__ uint64_t Timer()'):],
        'provenance':'NativeSession DeviceMemory cudaMalloc; private moving owner_tags/epoch/bad and persistent next/abort/claimed/completed/ready allocations',
        'types':'uint32_t/int32_t, uint64_t, naturally aligned offset arrays; static_assert size4/8 and alignment; cvta.to.global conversion',
        'semantics':'PTX ld.acquire.gpu.global, ld.relaxed.gpu.global, st.release.gpu.global, st.relaxed.gpu.global; volatile asm + compiler memory clobbers',
        'cpu_proof_shim':'Exact hash-bound PTX wrapper block maps to std atomic_ref with same memory orders; actual controller unchanged',
        'requires':['PTX ISA review','actual CPU controller proof','zero/budgeted resources','all30 exact','sanitizers','paired timings'],'gpu_executed':False}
    (output.parent/'persistent-global-atomics-proof.json').write_text(json.dumps(proof,indent=2)+'\n');return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();print(json.dumps(patch(a.source,a.output)))
