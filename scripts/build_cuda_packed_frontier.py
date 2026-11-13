"""Generate an AoSoA32 physical time frontier with timed pack/unpack boundaries.

The arithmetic, local tile map and complete rank joins stay fixed. Six fields
share one spatial address and differ by a constant32-word component offset.
The old trail arena becomes a padded physical field bank. Canonical uploaded
state, ordinary tails and download/reset contracts stay in their native layout.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from scripts.build_cuda_physical_frontier import stage as stage_frontier
from scripts.cuda_trace_phase_cycle import block


LAYOUT='''// Generated physical-field layout; dimensions are physical minima.
__host__ __device__ __forceinline__ int PhysicalFrontierIndex(int ny,int nx,int z,int y,int x){
 const int groups=(nx+32)/32;
 return ((z*(ny+1)+y)*groups+(x>>5))*192+(x&31);
}
__host__ __device__ __forceinline__ size_t PhysicalFrontierWords(int nz,int ny,int nx){
 return size_t(nz+1)*(ny+1)*((nx+32)/32)*192;
}
'''

IMPORT='''FieldPacket ImportPacket(const Parameters&p,const size_t*bases,int version,
 int oz,int oy,int ox,int dz,int dy,int dx){
 (void)bases;
 const int z=oz-1+dz-version,y=oy-1+dy-version,x=ox-1+dx-version;
 unsigned mask=Fast?63u:NativeSupportMask(p,z,y,x);
 if constexpr(ReuseTerminal)if(version==Depth)mask&=7u;
 if(!mask)return {};
 const float*address=p.trails+PhysicalFrontierIndex(p.physical_n[1],p.physical_n[2],z,y,x);
 return {mask&1?__ldcg(address+0*32):0.f,mask&2?__ldcg(address+1*32):0.f,
         mask&4?__ldcg(address+2*32):0.f,mask&8?__ldcg(address+3*32):0.f,
         mask&16?__ldcg(address+4*32):0.f,mask&32?__ldcg(address+5*32):0.f};
}'''

PUBLISH='''template<bool Safe>__device__ __forceinline__ void PublishPhysicalFrontier(
 const Parameters&p,int z,int y,int x,const FieldPacket&v){
 const unsigned mask=Safe?63u:NativeSupportMask(p,z,y,x);if(!mask)return;
 float*address=p.trails+PhysicalFrontierIndex(p.physical_n[1],p.physical_n[2],z,y,x);
 if(mask&1)address[0*32]=v.hx;if(mask&2)address[1*32]=v.hy;if(mask&4)address[2*32]=v.hz;
 if(mask&8)address[3*32]=v.ex;if(mask&16)address[4*32]=v.ey;if(mask&32)address[5*32]=v.ez;
}
'''

TRANSFER='''namespace beamz::cuda::moving_native {
__global__ void AsymmetricPhysicalFrontierTransfer(const __grid_constant__ Parameters p,bool unpack){
 const int width=((p.physical_n[2]+32)/32)*32;
 const size_t words=size_t(p.physical_n[0]+1)*(p.physical_n[1]+1)*width;
 const size_t q=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(q>=words)return;
 const int x=q%width,y=(q/width)%(p.physical_n[1]+1),z=q/(size_t(width)*(p.physical_n[1]+1));
 const unsigned mask=NativeSupportMask(p,z,y,x);
 const int common=PhysicalFrontierIndex(p.physical_n[1],p.physical_n[2],z,y,x);
 #pragma unroll
 for(int field=0;field<6;++field){
  const auto&f=InitialField(p,field);const int offset=z*f.plane+y*f.row+x;
  if(unpack){if(mask&(1u<<field))f.data[offset]=p.trails[common+field*32];}
  else p.trails[common+field*32]=(mask&(1u<<field))?f.data[offset]:0.f;
 }
}
}
int BeamzEnqueueNativePhysicalFrontierTransfer(void*opaque,const BeamzProgramLaunch&program,
 const BeamzBuffer*final_fields,const BeamzNativeMovingWorkspace&w,bool unpack){
 using namespace beamz::cuda::moving_native;
 Parameters p;int layout=0;if(!Prepare(program,final_fields,w,0,&p,&layout))return cudaErrorInvalidValue;
 const size_t points=PhysicalFrontierWords(p.physical_n[0],p.physical_n[1],p.physical_n[2])/6;
 AsymmetricPhysicalFrontierTransfer<<<(points+255)/256,256,0,static_cast<cudaStream_t>(opaque)>>>(p,unpack);
 return cudaGetLastError();
}
'''


def once(text,old,new):
    if text.count(old)!=1:raise ValueError('Unique packed-frontier binding required:'+old[:100])
    return text.replace(old,new,1)


def stage(source,output):
    source,output=Path(source).resolve(),Path(output).resolve()
    parent=stage_frontier(source,output,in_place=True)
    text=(output/'src/asymmetric_native.cuh').read_text()
    anchor='__device__ __forceinline__ float Initial('
    text=once(text,anchor,LAYOUT+'\n'+anchor)
    begin,end,_=block(text,anchor)
    text=text[:begin]+'''__device__ __forceinline__ float Initial(const Parameters&p,int field,int z,int y,int x){
 const auto&f=InitialField(p,field);if(!Contains(f,z,y,x))return 0.f;
 return __ldcg(p.trails+PhysicalFrontierIndex(p.physical_n[1],p.physical_n[2],z,y,x)+field*32);
}'''+text[end:]
    begin,end,_=block(text,'FieldPacket ImportPacket(')
    text=text[:begin]+IMPORT+'\n'+PUBLISH+text[end:]
    # Bind each of the four original publication sites as one packet.
    for signature in ('void UniformBody(','void FixedCubeCachedBody(','void SourcePendingPairedBody('):
        begin,end,body=block(text,signature)
        expected=2 if 'Paired' in signature else 1
        for index in range(expected):
            a,b,_=block(body,'if(surface>=0||n==Depth)')
            if expected==1:
                packet='h[0][j],h[1][j],h[2][j],e[0][j],e[1][j],e[2][j]'
            else:
                packet=f'h[0][si],h[1][si],h[2][si],ex{index}[gy],get_ey(zs,ys,sx,si),ez{index}[gy]'
            replacement='if(surface>=0||n==Depth)PublishPhysicalFrontier<SupportSafe>(p,z,y,x,FieldPacket{'+packet+'});'
            # A different anchor prevents matching this replacement next time.
            replacement=replacement.replace('if(surface>=0||n==Depth)','if(n==Depth||surface>=0)')
            body=body[:a]+replacement+body[b:]
        text=text[:begin]+body+text[end:]
    old='if(owners>std::numeric_limits<size_t>::max()/w.depth/6/SurfacePoints||w.trail_floats<owners*w.depth*6*SurfacePoints)return false;'
    new='if(w.trail_floats!=PhysicalFrontierWords(minimum[0],minimum[1],minimum[2])||w.trail_floats>size_t(INT32_MAX))return false;'
    text=once(text,old,new)
    text+='\n'+TRANSFER
    (output/'src/asymmetric_native.cuh').write_text(text)
    header=(output/'src/native_moving.h').read_text()
    declaration='''
// Research-only physical-field conversion, included in solve timing.
int BeamzEnqueueNativePhysicalFrontierTransfer(void*stream,const BeamzProgramLaunch&program,
 const BeamzBuffer*final_fields,const BeamzNativeMovingWorkspace&w,bool unpack);
'''
    header=once(header,'#endif',declaration+'\n#endif')
    (output/'src/native_moving.h').write_text(header)
    extension=(output/'src/native_extension.cc').read_text()
    planner='  const auto p=BeamzBuildNativeMovingPlan(fields.data(),depth,points,groups,starts,12,bool(BEAMZ_CUDA_NATIVE_MOVING_POLICIES));'
    extension=once(extension,planner,'''  auto p=BeamzBuildNativeMovingPlan(fields.data(),depth,points,groups,starts,12,bool(BEAMZ_CUDA_NATIVE_MOVING_POLICIES));
  p.trail_floats=size_t(p.extent[0])*p.extent[1]*((p.extent[2]+31)/32)*192;
  if(p.trail_floats>size_t(INT32_MAX))throw std::invalid_argument("packed frontier plan exceeds proven32-bit addressing");''')
    # Size the replacement arena before the unchanged actual-free-memory guard.
    anchor='    moving_observation_bytes_=observation.keys.size()*sizeof(int64_t)'
    replacement='''    // AoSoA32 physical field bank replaces versioned face histories.
    moving_plan_.trail_floats=size_t(minimum[0]+1)*(minimum[1]+1)*((minimum[2]+32)/32)*192;
    if(moving_plan_.trail_floats>size_t(INT32_MAX))throw std::invalid_argument("packed physical frontier exceeds proven32-bit addressing");
'''+anchor
    extension=once(extension,anchor,replacement)
    begin,end,body=block(extension,'nb::dict RunMoving(')
    before='''        const auto enqueue=[&]()->cudaError_t{
          for(int offset=0;offset+depth<=count;offset+=depth){'''
    after='''        const auto enqueue=[&]()->cudaError_t{
          if(count>=depth){const int error=BeamzEnqueueNativePhysicalFrontierTransfer(stream_.value,identity.program,other_.data(),moving_workspace_,false);if(error)return static_cast<cudaError_t>(error);}
          for(int offset=0;offset+depth<=count;offset+=depth){'''
    body=once(body,before,after)
    anchor='          BindOutputs();BeamzNativeLaunchStorage tail;'
    body=once(body,anchor,'''          if(count>=depth){const int error=BeamzEnqueueNativePhysicalFrontierTransfer(stream_.value,identity.program,other_.data(),moving_workspace_,true);if(error)return static_cast<cudaError_t>(error);}
'''+anchor)
    body=once(body,'"moving_rank_kernels_sources_raw_observations_DFT_tails_and_clock_advance"',
              '"packed_frontier_conversion_moving_sources_raw_observations_DFT_tails_and_clock_advance"')
    extension=extension[:begin]+body+extension[end:]
    marker=' module.attr("moving_physical_frontier_inplace")=true;'
    extension=once(extension,marker,marker+'\n module.attr("moving_physical_frontier_layout")="AoSoA32";')
    metadata='    result["moving_observation_bytes"]=moving_observation_bytes_;'
    extension=once(extension,metadata,'    result["moving_physical_frontier_layout"]="AoSoA32";result["moving_versioned_trail_payload_bytes"]=0;\n'+metadata)
    (output/'src/native_extension.cc').write_text(extension)
    proof=dict(parent, schema='beamz.packed-physical-frontier-lowering.v1',layout='AoSoA32',
               consumer_sha256=hashlib.sha256(text.encode()).hexdigest(),
               extension_sha256=hashlib.sha256(extension.encode()).hexdigest(),
               generated_layout=LAYOUT,expected_kernels=69,
               native_allocation_unchanged=False,versioned_trail_payload_bytes=0,
               bootstrap='Canonical working fields packed into the AoSoA32 physical time frontier',
               planner_and_session_allocation_match=True,
               replacement_allocation='6*(Nz+1)*(Ny+1)*ceil((Nx+1)/32)*32 FP32 words',
               conversion='Once before/after full macros in each chunk, captured and timed; tails use canonical state',
               overflow_guard='Total packed words<=INT32_MAX before allocation and in CUDA Prepare',
               generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    # The old physical-frontier compiler proof is the required common admission
    # binding; retain its ancestry, and explicitly describe the new arena role.
    (output.parent/'physical-frontier-compiler-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(stage(a.source,a.output)))
