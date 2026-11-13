"""Trace the captured four-component, six-contribution coincident mode source.

All six FP32 additions retain their order. The native constructor proves equal
starts within each component. Bounds and time are evaluated once per component,
and a paired-plane register/shared schedule replaces the generic source body.
"""
from pathlib import Path
from build_cuda_block_uniform import once


def patch(source: Path):
    path = source / "src/asymmetric_native.cuh"
    text = path.read_text()
    text = once(text,
        "bool CachedProfiles=false> struct Environment",
        "bool CachedProfiles=false,int SourceTerms=0> struct Environment")
    anchor = "    for(int source=0;source<s.count;++source){const int sz=z-s.starts[3*source],sy=y-s.starts[3*source+1],sx=x-s.starts[3*source+2];"
    replacement = """    if constexpr(SourceTerms>0){
     // Admission proves every contribution has this same immutable start.
     const int sz=z-s.starts[0],sy=y-s.starts[1],sx=x-s.starts[2];
     if(sz>=0&&sy>=0&&sx>=0&&sz<s.nz&&sy<s.ny&&sx<s.nx){
      int time=*s.current_step+p.macro_step+step_offset+version-1;
      time=max(0,min(time,s.ntime-1));
      const int spatial=(sz*s.ny+sy)*s.nx+sx,volume=s.nz*s.ny*s.nx;
      #pragma unroll
      for(int source=0;source<SourceTerms;++source)
       value=__fadd_rn(value,__fmul_rn(__ldg(s.coefficients+source*volume+spatial),__ldg(s.waveforms+source*s.ntime+time)));
     }
    }else{
""" + anchor
    text = once(text, anchor, replacement)
    text = once(text,
        "    }\n   }\n   if constexpr(HasObserve)",
        "    }\n    }\n   }\n   if constexpr(HasObserve)")
    text = once(text,
        "int ActiveAxes=-1,bool CachedProfiles=false>\n__device__ __forceinline__ void SourcePendingPairedBody",
        "int ActiveAxes=-1,bool CachedProfiles=false,int SourceTerms=0>\n__device__ __forceinline__ void SourcePendingPairedBody")
    text = once(text,
        "Environment<Layout,Fast,IsotropicH,Policy,UniformE,SourceMask,PlanePair,ActiveAxes,CachedProfiles>env",
        "Environment<Layout,Fast,IsotropicH,Policy,UniformE,SourceMask,PlanePair,ActiveAxes,CachedProfiles,SourceTerms>env")
    anchor = "inline bool MakeSource(const BeamzSourceGroupLaunch&g,Source*out){"
    kernels = """template<int Depth,int Layout,int Policy>
__global__ __launch_bounds__(1024,1) void AsymmetricModeSource54Six(
 const __grid_constant__ Parameters p,const int32_t*ids){
 static_assert(Policy>=3&&Policy<=5,"Source-bearing policies only");
 if(p.flags[PhysicalTileID(p,ids[blockIdx.x])]&64)
  SourcePendingPairedBody<Depth,Layout,false,true,Policy,true,1024,54,false,-1,false,6>(p,ids);
 else SourcePendingPairedBody<Depth,Layout,false,true,Policy,false,1024,54,false,-1,false,6>(p,ids);
}
inline bool ModeSource54SixEligible(const Parameters&p){
 for(int c=0;c<6;++c){
  const bool present=bool(54u&(1u<<c));const auto&s=p.sources[c];
  if(s.count!=(present?6:0)||(present&&s.nx!=1))return false;
 }
 return true;
}

"""
    text = once(text, anchor, kernels + anchor)
    anchor = "inline cudaError_t LaunchSourcePending(cudaStream_t stream,const Parameters&p,const int32_t*ids,int count,uint32_t presence,bool plane_pair){\n if(!count)return cudaSuccess;"
    text = once(text, anchor, anchor + """
 if constexpr(IsotropicH){
  if(presence==54u&&ModeSource54SixEligible(p)){
   const auto kernel=AsymmetricModeSource54Six<Depth,Layout,Policy>;
   auto status=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,SourcePendingSharedBytes);
   if(status!=cudaSuccess)return status;
   AsymmetricModeSource54Six<Depth,Layout,Policy><<<count,1024,SourcePendingSharedBytes,stream>>>(p,ids);
   return cudaGetLastError();
  }
 }
""")
    path.write_text(text)
    path = source / "src/native_extension.cc"
    text = once(path.read_text(), '  module.attr("moving_block_uniform_e")=true;',
                '  module.attr("moving_mode_source54_six")=true;\n  module.attr("moving_block_uniform_e")=true;')
    path.write_text(text)
