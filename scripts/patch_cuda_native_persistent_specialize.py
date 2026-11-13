"""Copy-only persistent pipeline with original bulk/NoIO fast bodies.

The baseline P5-only prototype stays untouched. All old consumer source remains
byte identical; only the new persistent entry selects a CTA-uniform body.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.patch_cuda_native_persistent import patch_copy, sha, once
except ModuleNotFoundError:
    from patch_cuda_native_persistent import patch_copy, sha, once


def clone_uniform(original):
    start = original.index("template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int ActiveAxes=-1>\n__device__ __forceinline__ void UniformBody")
    end = original.index("template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy=(Fast?0:5)>", start)
    parent = original[start:end]
    body = once(parent,
        "template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int ActiveAxes=-1>\n__device__ __forceinline__ void UniformBody(const Parameters&p,const int32_t*tiles){",
        "template<int Depth,bool Fast,int Policy,bool UniformE>\n__device__ __forceinline__ void PersistentUniformBody(const Parameters&p){\n constexpr int Layout=1,ActiveAxes=-1;constexpr bool IsotropicH=true;")
    body = once(body,"const int tile=tiles[blockIdx.x]","const int tile=C().tile")
    body = once(body,"const uint64_t epoch=uint64_t(*p.epoch)<<32;","const uint64_t epoch=uint64_t(C().epoch)<<32;")
    body = once(body,
        " if(!threadIdx.x){const int owner=(rank&3)*p.frontier+p.slots[tile];p.owner_tags[owner]=epoch|uint32_t(tile);}\n __syncthreads();",
        " if(!threadIdx.x){Claim();if constexpr(Fast){if(!C().abort_local)WaitUpstream(1);}}\n __syncthreads();\n if(C().abort_local)return;")
    body = once(body,
        "   window[1]=min(Depth,min(oz+BZ-1,min(oy+BY-1,ox+BX-1)));\n  }",
        "   window[1]=min(Depth,min(oz+BZ-1,min(oy+BY-1,ox+BX-1)));\n   if(window[0]<=window[1])WaitUpstream(window[0]);\n  }")
    body = once(body,"if(window[0]>window[1])return;","if(C().abort_local||window[0]>window[1])return;")
    body = once(body,"if(bits&&p.owner_tags[owner]!=(epoch|uint32_t(producer)))",
                     "if(bits&&Acquire(p.owner_tags+owner)!=(epoch|uint32_t(producer)))")
    body = once(body,"const uint8_t flags=Policy==0?0:Policy==2?1:Policy==3?16:Policy==4?17:p.flags[tile];",
                     "const uint8_t flags=p.flags[tile];")
    body = once(body,"Environment<Layout,Fast,IsotropicH,Policy,UniformE,-1,false,ActiveAxes>env{p,flags};",
                     "PersistentUniformEnvironment<Fast,Policy,UniformE>env{p,flags};")
    # Publications read current PRIVATE registers. Moving before B3 cannot
    # overwrite the shared H frame while another warp is still consuming it.
    a = body.index("  const int y=oy+low+dy,x=ox+low+dx;\n", body.index("// B4"))
    b = body.index("\n }\n}\n", a)
    publication = body[a:b]
    body = body[:a] + body[b:]
    marker = "  __syncthreads(); // B3 retires current-H frame readers before E overwrite."
    body = once(body,marker,publication+"\n  __threadfence();\n"+marker)
    body = once(body,
        "  if(!threadIdx.x)*phase_step=n+1;\n  __syncthreads(); // B4 publishes E and the next phase counter.",
        "  if(!threadIdx.x){*phase_step=n+1;PublishAndWait(n,Fast?Depth:window[1]);}\n  __syncthreads(); // B4 publishes E and the next phase counter.\n  if(C().abort_local)return;")
    for phase in (0,1):
        for component in range(3):
            token=f"beamz::cuda::trace_native::AdvanceNativeCell<{phase},{component},false>"
            if body.count(token)!=parent.count(token):raise ValueError("Canonical ordered cell calls changed")
    if body.count("__syncthreads()")!=parent.count("__syncthreads()"):
        raise ValueError("Uniform-body barrier count changed")
    if "tiles[blockIdx.x]" in body or "p.owner_tags[owner]=" in body:
        raise ValueError("Nonpersistent control remains")
    return body


ENV = r'''
template<bool Fast,int Policy,bool UniformE>
struct PersistentUniformEnvironment:Environment<1,Fast,true,Policy,UniformE>{
 using Base=Environment<1,Fast,true,Policy,UniformE>;
 __device__ __forceinline__ PersistentUniformEnvironment(const Parameters&physics,uint8_t facts):Base{physics,facts}{}
 __device__ __forceinline__ float LoadLocal(int slot,int z,int y,int x)const{
  return __ldcg(this->p.phase[slot/6].psi[slot%6].input+this->PsiOffset(slot,z,y,x));
 }
 __device__ __forceinline__ float Coefficient(int kind,int index,int z,int y,int x)const{
  if(kind==1&&index>=3&&index<6&&(this->flags&64))return this->p.common_e[index-3];
  return Base::Coefficient(kind,index,z,y,x);
 }
};
'''


def specialize(code, original):
    body=clone_uniform(original)
    code=once(code,"int tile,owner,abort_local;","int tile,owner,abort_local,body_kind;")
    code=once(code,"template<int Depth>\n__global__ __launch_bounds__(1024,1) void PersistentTileStep(",
              ENV+"\n"+body+"\ntemplate<int Depth>\n__global__ __launch_bounds__(1024,1) void PersistentTileStep(")
    code=once(code,
        "else{const int job=atomicAdd(C().next,1);C().tile=job<C().count?C().jobs[job]:-1;}",
        """else{const int job=atomicAdd(C().next,1);C().tile=job<C().count?C().jobs[job]:-1;
    if(C().tile>=0){const uint8_t facts=p.flags[C().tile];
     // Low six bits reproduce existing Bulk/NoIO policy. Common-E bit64
     // is the original captured certificate, not an inferred material fact.
     C().body_kind=!(facts&63)?0:(!(facts&17)&&(facts&64))?1:2;
    }
   }""")
    code=once(code,"  PersistentPairedBody<Depth>(p);",
        """  if(C().body_kind==0)PersistentUniformBody<Depth,true,0,false>(p);
  else if(C().body_kind==1)PersistentUniformBody<Depth,false,1,true>(p);
  else PersistentPairedBody<Depth>(p);""")
    return code,body


def patch_specialized(source,output):
    result=patch_copy(source,output)
    original=(source/"src/asymmetric_native.cuh").read_text()
    generated,body=specialize((output/"src/native_persistent.cuh").read_text(),original)
    (output/"src/native_persistent.cuh").write_text(generated)
    result.update({"variant":"persistent CTA-uniform original bulk + certified NoIO + generic P5",
        "persistent_consumer_sha256":sha(generated.encode()),"cloned_uniform_body_sha256":sha(body.encode()),
        "specialization_patcher_sha256":sha(Path(__file__).read_bytes()),
        "body_dispatch":"low6bits zero => Fast; no source/observer and bit64 => certified NoIO; else P5",
        "uniform_readiness":"private trail/final publication moved before B3; all-writer fence before B3; releaseSTEP and waitnext before B4",
        "workspace_abi_changed":False,"original_p5_prototype_changed":False})
    (output.parent/"persistent-specialized-proof.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",required=True,type=Path);p.add_argument("--output",required=True,type=Path)
    a=p.parse_args();print(json.dumps(patch_specialized(a.source.resolve(),a.output.resolve())))
