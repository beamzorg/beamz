"""Emit an opt-in six-policy native moving consumer, leaving input unchanged.

Requires the matching native host policy ranges when the policy macro is ON.
The cell arithmetic, frontier layout, epoch protocol, and barriers are unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Expected exactly one policy anchor: {old!r}")
    return source.replace(old, new, 1)


def patch(source):
    source = once(
        source,
        "#pragma once",
        "#pragma once\n#ifndef BEAMZ_CUDA_NATIVE_MOVING_POLICIES\n#define BEAMZ_CUDA_NATIVE_MOVING_POLICIES 0\n#endif",
    )
    start = source.index(
        "template<int Layout,bool Fast,bool IsotropicH> struct Environment"
    )
    end = source.index("// Canonical staggering was proved", start)
    env = source[start:end]
    env = once(
        env,
        "template<int Layout,bool Fast,bool IsotropicH> struct Environment {",
        """template<int Layout,bool Fast,bool IsotropicH,int Policy=(Fast?0:5)> struct Environment {
 static constexpr bool SupportSafe=Fast||(Policy>=2&&Policy<=4);
 static constexpr bool HasCpml=Policy==1||Policy==5;
 static constexpr bool HasSources=Policy==3||Policy==4||Policy==5;
 static constexpr bool HasObserve=Policy==2||Policy==4||Policy==5;""",
    )
    env = once(
        env, "if constexpr(Fast)return true;", "if constexpr(SupportSafe)return true;"
    )
    env = once(
        env, "if constexpr(Fast)return false;", "if constexpr(!HasCpml)return false;"
    )
    env = once(
        env,
        "if constexpr(!Fast){const auto&f=InitialField",
        "if constexpr(!SupportSafe){const auto&f=InitialField",
    )
    env = once(
        env,
        "if constexpr(!Fast){\n   if(flags&",
        "if constexpr(HasSources||HasObserve){\n   if constexpr(HasSources)if(flags&",
    )
    env = once(
        env,
        "if(flags&beamz::cuda::moving::kMayObserve)Observe",
        "if constexpr(HasObserve)if(flags&beamz::cuda::moving::kMayObserve)Observe",
    )
    source = source[:start] + env + source[end:]
    start = source.index(
        "template<int Depth,int Layout,bool Fast,bool IsotropicH>\n__global__"
    )
    end = source.index("inline bool MakeSource(", start)
    kernel = source[start:end]
    kernel = once(
        kernel,
        "template<int Depth,int Layout,bool Fast,bool IsotropicH>",
        "template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy=(Fast?0:5)>",
    )
    kernel = once(
        kernel,
        "const __grid_constant__ Parameters p,const int32_t*tiles){",
        "const __grid_constant__ Parameters p,const int32_t*tiles){\n constexpr bool SupportSafe=Fast||(Policy>=2&&Policy<=4);",
    )
    kernel = once(
        kernel,
        "const uint8_t flags=Fast?0:p.flags[tile];Environment<Layout,Fast,IsotropicH>env{p,flags};",
        "const uint8_t flags=Policy==0?0:Policy==2?1:Policy==3?16:Policy==4?17:p.flags[tile];Environment<Layout,Fast,IsotropicH,Policy>env{p,flags};",
    )
    kernel = kernel.replace(
        "ImportPacket<Depth,Fast>(", "ImportPacket<Depth,SupportSafe>("
    )
    kernel = once(kernel, "StageLowX<Depth,Fast>(", "StageLowX<Depth,SupportSafe>(")
    source = source[:start] + kernel + source[end:]
    source = once(
        source,
        "template<int Depth,int Layout,bool Fast,bool IsotropicH>\ninline cudaError_t Launch",
        "template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy=(Fast?0:5)>\ninline cudaError_t Launch",
    )
    source = source.replace(
        "Asymmetric<Depth,Layout,Fast,IsotropicH>",
        "Asymmetric<Depth,Layout,Fast,IsotropicH,Policy>",
    )
    source = once(
        source,
        " const size_t owners=4ULL*w.max_frontier;",
        """#if BEAMZ_CUDA_NATIVE_MOVING_POLICIES
 if(!w.policy_offsets_host)return false;
 for(int rank=0;rank<w.rank_count;++rank){const auto*offsets=w.policy_offsets_host+rank*7;
  if(offsets[0]!=w.rank_offsets_host[rank]||offsets[6]!=w.rank_offsets_host[rank+1])return false;
  for(int policy=0;policy<6;++policy)if(offsets[policy]>offsets[policy+1])return false;
 }
#endif
 const size_t owners=4ULL*w.max_frontier;""",
    )
    source = once(
        source,
        "inline cudaError_t LaunchRanks(cudaStream_t stream,const Parameters&p,const BeamzNativeMovingWorkspace&w){",
        """inline cudaError_t LaunchRanks(cudaStream_t stream,const Parameters&p,const BeamzNativeMovingWorkspace&w){
#if BEAMZ_CUDA_NATIVE_MOVING_POLICIES
 for(int rank=0;rank<w.rank_count;++rank){const auto*offsets=w.policy_offsets_host+rank*7;
  for(int policy=0;policy<6;++policy){const int start=offsets[policy],count=offsets[policy+1]-start;
   if(!count)continue;cudaError_t status=cudaSuccess;
   switch(policy){
    case 0:status=Launch<Depth,Layout,true,IsotropicH,0>(stream,p,w.tile_ids_device+start,count);break;
    case 1:status=Launch<Depth,Layout,false,IsotropicH,1>(stream,p,w.tile_ids_device+start,count);break;
    case 2:status=Launch<Depth,Layout,false,IsotropicH,2>(stream,p,w.tile_ids_device+start,count);break;
    case 3:status=Launch<Depth,Layout,false,IsotropicH,3>(stream,p,w.tile_ids_device+start,count);break;
    case 4:status=Launch<Depth,Layout,false,IsotropicH,4>(stream,p,w.tile_ids_device+start,count);break;
    case 5:status=Launch<Depth,Layout,false,IsotropicH,5>(stream,p,w.tile_ids_device+start,count);break;
   }
   if(status!=cudaSuccess)return status;
  }
 }
 return cudaSuccess;
#else""",
    )
    source = once(
        source,
        "template<int Depth,int Layout>\ninline cudaError_t DispatchClass",
        "#endif\n\ntemplate<int Depth,int Layout>\ninline cudaError_t DispatchClass",
    )
    # The conditional above must close inside LaunchRanks, before its function
    # brace rather than enclosing that brace in the disabled alternative.
    source = once(
        source,
        " return cudaSuccess;\n}\n#endif\n\ntemplate<int Depth,int Layout>\ninline cudaError_t DispatchClass",
        " return cudaSuccess;\n#endif\n}\n\ntemplate<int Depth,int Layout>\ninline cudaError_t DispatchClass",
    )
    return source


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.input.resolve() == a.output.resolve():
        p.error("Input is frozen; output must be a different path")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(patch(a.input.read_text()))
    print(
        json.dumps(
            {
                "input": str(a.input),
                "input_sha256": hashlib.sha256(a.input.read_bytes()).hexdigest(),
                "output": str(a.output),
                "output_sha256": hashlib.sha256(a.output.read_bytes()).hexdigest(),
                "policy_macro_default": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
