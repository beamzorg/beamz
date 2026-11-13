"""Private NoIO+Support copy: unchecked startup only after full-cache proof."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

CONSUMER = "45a832f23ccabfeffd4afc8bd0ad2c339a97d16a831f579a3e537d662edae3ef"
EXTENSION = "d0a71bf792c2bed4f7ae77ae365d7906c2ead32abe875d73cd09152d8578e706"


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Initial-fact anchor changed: {old}")
    return text.replace(old, new, 1)


def patch_consumer(text):
    if hashlib.sha256(text.encode()).hexdigest() != CONSUMER:
        raise ValueError("Exact NoIO+Support consumer required")
    anchor = "__device__ __forceinline__ float Initial(const Parameters&p,int field,int z,int y,int x){"
    helper = """__device__ __forceinline__ float InitialUnchecked(const Parameters&p,int field,int z,int y,int x){
 const auto&f=InitialField(p,field);return f.data[z*f.plane+y*f.row+x];
}
"""
    text = once(text, anchor, helper + anchor)
    start = text.index(" if constexpr(Fast){\n #pragma unroll\n for(int j=0;j<Slots;++j)")
    end = text.index(" const uint8_t flags=", start)
    original = text[start:end]
    prefix = """ // Bit128 is certified independently for the FULL initial cache, including
 // its evicted high/low ghost slots; the branch is immutable and CTA-uniform.
 if(SupportSafe&&(p.flags[tile]&128)){
  #pragma unroll
  for(int j=0;j<Slots;++j){const int zs=zbase+j*ZStep,z=oz+(zs==7?-1:zs),y=oy+(sy==15?-1:sy),x=ox+(sx==31?-1:sx);
   #pragma unroll
   for(int c=0;c<3;++c){h[c][j]=InitialUnchecked(p,c,z,y,x);e[c][j]=InitialUnchecked(p,c+3,z,y,x);frame[c*PAD+Slot(zs,sy,sx)]=e[c][j];}
  }
  if(!threadIdx.x)*phase_step=1;__syncthreads();
 }else{
"""
    return text[:start] + prefix + original + " }\n" + text[end:]


def patch_extension(text):
    if hashlib.sha256(text.encode()).hexdigest() != EXTENSION:
        raise ValueError("Exact NoIO+Support native extension required")
    anchor = "    BeamzNativeMovingSparseObservation sparse;"
    fact = """    moving_safe_initial_tiles_=0;
    for(size_t tile=0;tile<moving_plan_.tile_flags.size();++tile){
      const int policy=beamz::cuda::moving::Policy(moving_plan_.tile_flags[tile]);
      if(policy!=0&&(policy<2||policy>4))continue;
      const int q[3]={int(tile/(moving_plan_.counts[1]*moving_plan_.counts[2])),int((tile/moving_plan_.counts[2])%moving_plan_.counts[1]),int(tile%moving_plan_.counts[2])};
      bool safe=true;
      for(int c=0;c<6;++c)for(int axis=0;axis<3;++axis){
        const int64_t origin=int64_t(q[axis])*moving_plan_.owned[axis];
        safe&=origin-1>=0&&origin+moving_plan_.owned[axis]-1<base_inputs_[c].dims[axis];
      }
      if(safe){moving_plan_.tile_flags[tile]|=128;++moving_safe_initial_tiles_;}
    }
"""
    text = once(text, anchor, fact + anchor)
    text = once(text, "size_t moving_uniform_bulk_=0,moving_uniform_noio_=0,moving_uniform_prefix_bytes_=0;",
                "size_t moving_safe_initial_tiles_=0,moving_uniform_bulk_=0,moving_uniform_noio_=0,moving_uniform_prefix_bytes_=0;")
    text = once(text, 'result["moving_uniform_noio_tiles"]=moving_uniform_noio_;',
                'result["moving_safe_initial_tiles"]=moving_safe_initial_tiles_;result["moving_uniform_noio_tiles"]=moving_uniform_noio_;')
    return once(text, "NB_MODULE(_cuda_native, module) {", "NB_MODULE(_cuda_native, module) {\n  module.attr(\"moving_safe_initial_fact\")=true;")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    if args.source.resolve() == Path(__file__).resolve().parents[1] / "cuda":
        parser.error("Only private copied sources allowed")
    c, e = (args.source / "src" / n for n in ("asymmetric_native.cuh", "native_extension.cc"))
    cs, es = patch_consumer(c.read_bytes().decode()), patch_extension(e.read_bytes().decode())
    c.write_text(cs)
    e.write_text(es)
    print(json.dumps({"consumer_sha256": hashlib.sha256(cs.encode()).hexdigest(),
                      "extension_sha256": hashlib.sha256(es.encode()).hexdigest(),
                      "flag": 128, "policies": [0, 2, 3, 4],
                      "certificate": "ALL6 actual supports contain inclusive [origin-1,origin+B-1]",
                      "scope": "One CTA-uniform startup branch; all arithmetic, later phases, geometry and checked fallback unchanged"}))


if __name__ == "__main__":
    main()
