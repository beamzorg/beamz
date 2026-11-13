"""Generate complete research backends with power-of-two moving-cache shapes.

The cell trace, arithmetic, wavefront epochs, CPML and monitor semantics remain
unchanged. Source-specific paired storage is deliberately replaced by the
generic source-capable body so that it cannot assume the old cache dimensions.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import re
import shutil

from scripts.cuda_trace_phase_cycle import ROOT, PARENT, PARENT_CONSUMER, block, sha


def verify_mapping(cache, threads):
    cz, cy, cx = cache
    if cx != 32 or cz not in (4, 8, 16) or cy not in (8, 16):
        raise ValueError("Reviewed power-of-two Z/Y geometry and full X warp required")
    plane = cy*cx
    volume = cz*plane
    if threads % plane or volume % threads or threads not in (256, 512, 1024):
        raise ValueError("Whole XY planes per block and integral private slots required")
    slots, zstep = volume//threads, threads//plane
    owners = {}
    for tid, j in itertools.product(range(threads), range(slots)):
        point = tid//plane+j*zstep, (tid//32)&(cy-1), tid&31
        if point in owners:
            raise AssertionError("Multiple owners for a cached point")
        owners[point] = tid,j
    if len(owners) != volume:
        raise AssertionError("Missing cache owners")
    checked = 0
    bz, by, bx = cz-1, cy-1, cx-1
    for n in range(1, 17):
        ghost = tuple((-n-1)&(c-1) for c in cache)
        owned = set()
        staged = set()
        for point in owners:
            relative = tuple((q-g)&(c-1) for q,g,c in zip(point,ghost,cache))
            if all(relative):
                owned.add(relative)
            if relative[0] and relative[1] and not relative[2]:
                key = (relative[0]-1)*by+relative[1]-1
                if key in staged:
                    raise AssertionError("Low-X packet has multiple owners")
                staged.add(key)
            checked += 1
        if owned != set(itertools.product(range(1,cz),range(1,cy),range(1,cx))):
            raise AssertionError("Moving owner extent differs")
        if staged != set(range(bz*by)):
            raise AssertionError("Low-X packet mapping differs")
    return {"cache_points":volume,"owner_checks":checked,"slots":slots,
            "zstep":zstep,"field_floats_per_thread":6*slots,
            "owned_zyx":[bz,by,bx],"surface_points":bz*by*bx-(bz-1)*(by-1)*(bx-1)}


def stage(source, output, cache, threads, *, min_blocks=1, direct_low=False, expected_consumer=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    canonical = ROOT / "cuda"
    if output.exists() or output == canonical or canonical in output.parents or source in output.parents or output in source.parents:
        raise ValueError("New disjoint copied source required")
    text = (source/"src/asymmetric_native.cuh").read_text()
    expected_consumer = PARENT_CONSUMER if expected_consumer is None else expected_consumer
    if sha(text.encode()) != expected_consumer:
        raise ValueError("Exact current winner required")
    proof = verify_mapping(cache, threads)
    cz,cy,cx=cache; bz,by,bx=proof['owned_zyx']
    # Remove the old-shape-specific paired-source definitions and dispatch.
    begin=text.index("template<int Depth,bool Safe,int T>\n__device__ __forceinline__ void StageSourcePendingH")
    end=text.index("inline bool MakeSource(",begin)
    text=text[:begin]+text[end:]
    begin,end,body=block(text,"inline cudaError_t LaunchSourcePending(")
    prefix=body[:body.index('{')+1]
    text=text[:begin]+prefix+"\n return Launch<Depth,Layout,false,IsotropicH,Policy>(stream,p,ids,count);\n}"+text[end:]
    text=text.replace("constexpr int BZ=7,BY=15,BX=31,CZ=8,CY=16,CX=32,Threads=512;",
                      f"constexpr int BZ={bz},BY={by},BX={bx},CZ={cz},CY={cy},CX=32,Threads={threads},ResearchThreads={threads};")
    text=text.replace("constexpr int T=(Fast||(Layout==1&&(Policy==1||Policy==2)&&IsotropicH&&UniformE))?1024:512,Slots=PAD/T,ZStep=T/512;",
                      "constexpr int T=ResearchThreads,Slots=PAD/T,ZStep=T/(CY*CX);")
    text=text.replace("threadIdx.x/512", "threadIdx.x/(CY*CX)").replace("(T/512)","(T/(CY*CX))")
    text=text.replace("&7", "&(CZ-1)").replace("&15", "&(CY-1)")
    # Observation coarse cells are always 8x16, independently of the moving
    # tile. Their masks must never follow the research cache dimensions.
    _,_,observe=block((source/'src/asymmetric_native.cuh').read_text(), 'void Observe(')
    begin,end,_=block(text,'void Observe(')
    text=text[:begin]+observe+text[end:]
    text=text.replace("y&=15;", "y&=(CY-1);")
    text=text.replace("zs==7?", "zs==(CZ-1)?").replace("sy==15?", "sy==(CY-1)?")
    text=text.replace("__launch_bounds__((Fast?1024:512),1)",f"__launch_bounds__(ResearchThreads,{min_blocks})")
    text=text.replace("__launch_bounds__(1024,1)",f"__launch_bounds__(ResearchThreads,{min_blocks})")
    text=text.replace("<<<count,(Fast?1024:512),", "<<<count,ResearchThreads,")
    text=text.replace(",1024,FixedCubeSharedBytes,",",ResearchThreads,FixedCubeSharedBytes,")
    text=text.replace(",1024,ProfileCachedSharedBytes,",",ResearchThreads,ProfileCachedSharedBytes,")
    text=text.replace(",1024,SharedBytes,",",ResearchThreads,SharedBytes,")
    text=text.replace("if(threadIdx.x<512)for(int i=threadIdx.x;i<864;i+=512)",
                      "for(int i=threadIdx.x;i<864;i+=blockDim.x)")
    # The shared phase variable is redundant; preserve all barrier positions.
    text=text.replace("step=*phase_step", "step=n")
    for old in ("if(!threadIdx.x)*phase_step=1;", "if(!threadIdx.x)*phase_step=first;", "if(!threadIdx.x)*phase_step=n+1;"):
        text=text.replace(old, "")
    if direct_low:
        # Every low-X ghost imports directly into its own private values. These
        # are disjoint from the owned H points and old shared E readers.
        for signature in ("void UniformBody(","void FixedCubeCachedBody("):
            begin,end,body=block(text,signature)
            old="if(zs==hz||sy==hy){const int dz=(zs-hz)&(CZ-1);"
            if body.count(old)!=1:
                raise ValueError("Low-ghost import footprint changed")
            body=body.replace(old,"if(zs==hz||sy==hy||sx==hx){const int dz=(zs-hz)&(CZ-1);")
            body=body.replace("  StageRegisterLow<Depth,SupportSafe,T>(p,low_stage,bases,n,oz,oy,ox);", "")
            body=body.replace("  RetireRegisterLow<T,Slots>(low_stage,h,e,n,sy,sx);", "")
            text=text[:begin]+body+text[end:]
        text=text.replace("+6ULL*LowXPoints*sizeof(float)","")
    shared=12*cz*cy*cx+64+12+(0 if direct_low else 24*bz*by)
    cached=shared+3456
    if cached>101376:
        # Reuse the independent global profiles when a large cache leaves no
        # room for their optional shared copies. Physics pointers stay distinct.
        text=text.replace("constexpr size_t ProfileCachedSharedBytes=SharedBytes+864ULL*sizeof(float);",
                          "constexpr size_t ProfileCachedSharedBytes=SharedBytes;")
        text=text.replace("constexpr size_t FixedCubeSharedBytes=SharedBytes+864ULL*sizeof(float);",
                          "constexpr size_t FixedCubeSharedBytes=SharedBytes;")
        text=text.replace("CachedProfiles=true;", "CachedProfiles=false;")
        text=text.replace("ActiveAxes,true>", "ActiveAxes,false>")
        text=text.replace("UniformBody<Depth,1,false,true,1,true,1,true>","UniformBody<Depth,1,false,true,1,true,1,false>")
        text=text.replace("UniformBody<Depth,1,false,true,1,true,2,true>","UniformBody<Depth,1,false,true,1,true,2,false>")
        text=text.replace("UniformBody<Depth,1,false,true,1,true,4,true>","UniformBody<Depth,1,false,true,1,true,4,false>")
        text=text.replace("UniformBody<Depth,1,false,true,1,true,-1,true>","UniformBody<Depth,1,false,true,1,true,-1,false>")
        # Fixed geometry coefficient override must honor the same choice.
        text=text.replace("if(kind>=2)return ReadIndependentProfileCache(profiles,index,kind,Packed(index,z,y,x));",
                          "if(kind>=2){const auto&psi=p.phase[index/6].psi[index%6];const int packed=Packed(index,z,y,x);return kind==2?__ldg(psi.a+packed):kind==3?__ldg(psi.b+packed):__ldg(psi.inv+packed);}")
        cached=shared
    text=text.replace('static_assert(FixedCubeSharedBytes==55204&&FixedCubeSharedBytes==ProfileCachedSharedBytes,"Same independent864-profile allocation");',
                      f'static_assert(FixedCubeSharedBytes=={cached}&&FixedCubeSharedBytes==ProfileCachedSharedBytes,"Research cache allocation");')
    text=text.replace('static_assert(SharedBytes==51748&&ProfileCachedSharedBytes==55204,"Exact appended profile layout");',
                      f'static_assert(SharedBytes=={shared}&&ProfileCachedSharedBytes=={cached},"Research cache/profile layout");')
    if shared>101376:
        raise ValueError("Cache exceeds sm86 per-block shared capacity")
    # Both host plan and runtime workspace agree with the generated consumer.
    header=(source/'src/native_moving.h').read_text()
    header=header.replace("kOwnedZ=7,kOwnedY=15,kOwnedX=31",f"kOwnedZ={bz},kOwnedY={by},kOwnedX={bx}")
    header=header.replace("owned[3]{7,15,31}",f"owned[3]{{{bz},{by},{bx}}}")
    header=header.replace("owned{7,15,31}",f"owned{{{bz},{by},{bx}}}")
    shutil.copytree(source,output)
    (output/'src/asymmetric_native.cuh').write_text(text)
    (output/'src/native_moving.h').write_text(header)
    extension=(output/'src/native_extension.cc').read_text()
    extension=extension.replace('NB_MODULE(_cuda_native, module) {',
        f'NB_MODULE(_cuda_native, module) {{\n module.attr("research_cache_shape")=nb::make_tuple({cz},{cy},{cx});\n module.attr("research_block_threads")={threads};\n module.attr("research_generic_source_storage")=true;')
    (output/'src/native_extension.cc').write_text(extension)
    proof.update(schema="beamz.trace-cache-shape.v1",threads=threads,min_blocks=min_blocks,
                 direct_low=direct_low,shared_bytes=shared,cached_shared_bytes=cached,
                 parent_sha256=expected_consumer,consumer_sha256=sha(text.encode()),
                 header_sha256=sha(header.encode()),producer_sha256=sha(Path(__file__).read_bytes()),
                 source_paired_storage_disabled=True,expected_kernels=56,GPU_executed=False)
    (output.parent/'box-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=PARENT)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cache',type=int,nargs=3,required=True)
    p.add_argument('--threads',type=int,required=True)
    p.add_argument('--min-blocks',type=int,default=1)
    p.add_argument('--direct-low',action='store_true')
    a=p.parse_args()
    print(json.dumps(stage(a.source,a.output,tuple(a.cache),a.threads,min_blocks=a.min_blocks,direct_low=a.direct_low)))
