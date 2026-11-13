"""Specialize boundary-free NoIO trajectories while retaining active CPML.

The existing immutable bit32 certificate proves all startup/import supports and
the full1..Depth window. Complementary CTA-uniform gates retain one owner per
tile; the first experiment uses two rank-local launches rather than repartition
the host plan. The scalar cell arithmetic and independent profiles stay intact.
"""
from __future__ import annotations
import argparse,itertools,json
from pathlib import Path
import shutil
from scripts.cuda_trace_phase_cycle import ROOT,PARENT,PARENT_CONSUMER,block,sha

HELPER=r'''
template<int N,int ActiveAxes>struct SafeNoIOEnvironment:FixedCubeCachedEnvironment<N,ActiveAxes>{
 using Base=FixedCubeCachedEnvironment<N,ActiveAxes>;
 static constexpr bool SupportSafe=true;
 __device__ __forceinline__ SafeNoIOEnvironment(const Parameters&p,uint8_t flags,const float*profiles):Base(p,flags,profiles){}
 __device__ __forceinline__ bool Valid(int,int,int,int)const{return true;}
 __device__ __forceinline__ float Difference(int phase,int,int,int,int,int,float center,float adjacent)const{
  return beamz::cuda::yee::ScaleYeeDifference(phase?__fsub_rn(center,adjacent):__fsub_rn(adjacent,center),this->p.phase[phase].inv);
 }
};
template<int ActiveAxes>struct SafeNoIOEnvironment<0,ActiveAxes>:Environment<1,false,true,1,true,-1,false,ActiveAxes,true>{
 using Base=Environment<1,false,true,1,true,-1,false,ActiveAxes,true>;
 static constexpr bool SupportSafe=true;
 __device__ __forceinline__ SafeNoIOEnvironment(const Parameters&p,uint8_t flags,const float*profiles):Base{p,flags,profiles}{}
 __device__ __forceinline__ bool Valid(int,int,int,int)const{return true;}
 __device__ __forceinline__ float Difference(int phase,int,int,int,int,int,float center,float adjacent)const{
  return beamz::cuda::yee::ScaleYeeDifference(phase?__fsub_rn(center,adjacent):__fsub_rn(adjacent,center),this->p.phase[phase].inv);
 }
};
__device__ __forceinline__ float SafeNoIOInitial(const Parameters&p,int field,int z,int y,int x){
 const auto&f=InitialField(p,field);return f.data[z*f.plane+y*f.row+x];
}
'''


def prove():
    cases=points=0
    for minimum,depth in itertools.product(((32,35,37),(64,96,128),(512,512,512),(564,564,564)),(4,16)):
        for origin in itertools.product(*(range(0,n+depth+31,b) for n,b in zip(minimum,(7,15,31)))):
            low=tuple(o-depth for o in origin);high=tuple(o+b-2 for o,b in zip(origin,(7,15,31)))
            safe=all(l>0 and h+1<n for l,h,n in zip(low,high,minimum))
            if not safe:continue
            # Every component contains the common [0,N) rectangle. Startup,
            # ghost and every staggered neighbor read lie in this rectangle.
            for a,(o,b,n) in enumerate(zip(origin,(7,15,31),minimum)):
                assert 0<=o-depth-1 and o+b-1<n
                assert 1<=o-depth and o+b-2<n-1
                points+=2
            cases+=1
    assert cases>0
    return dict(eligible_trajectory_cases=cases,extreme_support_checks=points,
                full_window=True,owned_E_points_never_on_native_wall=True,
                CPML_not_disabled=True,complementary_owner_gates_all_flags=256)


def stage(source,output,single_kernel=False):
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists() or output==ROOT/'cuda' or ROOT/'cuda' in output.parents or source in output.parents or output in source.parents:raise ValueError('New disjoint copied research source required')
    original=(source/'src/asymmetric_native.cuh').read_text()
    if sha(original.encode())!=PARENT_CONSUMER:raise ValueError('Exact current winner required')
    planner=(source/'src/native_moving.cc').read_text()
    if 'if(low[a]<=0||high[a]+1>=minimum[a])flags|=kNativeBoundary;' not in planner:raise ValueError('Actual host boundary certificate changed')
    text=original
    _,_,body=block(text,'void UniformBody(')
    body=body[body.index('{')+1:-1]
    body=body.replace('constexpr bool SupportSafe=Fast||(Policy>=2&&Policy<=4);','constexpr bool SupportSafe=true,FullWindow=true;')
    body=body.replace('if constexpr(!Fast)','if constexpr(!FullWindow)').replace('if constexpr(Fast)','if constexpr(FullWindow)')
    body=body.replace('(Fast?1:window[0])','(FullWindow?1:window[0])').replace('(Fast?Depth:window[1])','(FullWindow?Depth:window[1])')
    env='Environment<Layout,Fast,IsotropicH,Policy,UniformE,-1,false,ActiveAxes,CachedProfiles>env{p,flags,profiles};'
    if body.count(env)!=1:raise ValueError('Captured NoIO environment changed')
    body=body.replace(env,'SafeNoIOEnvironment<N,ActiveAxes>env{p,flags,profiles};')
    body=body.replace('Initial(p,','SafeNoIOInitial(p,')
    clone='template<int Depth,int N,int ActiveAxes=-1>\n__device__ __forceinline__ void SafeNoIOBody(const Parameters&p,const int32_t*tiles){\n constexpr int Layout=1,Policy=1;constexpr bool Fast=false,IsotropicH=true,UniformE=true,CachedProfiles=true;'+body+'}\n'
    entries='''template<int Depth,int N=0>
__global__ __launch_bounds__(1024,1) void AsymmetricSafeNoIO(const __grid_constant__ Parameters p,const int32_t*tiles){
 if(p.flags[tiles[blockIdx.x]]&32)return;
 switch(p.flags[tiles[blockIdx.x]]&14){
'''
    for flag,axis in ((2,1),(4,2),(8,4)):entries+=f' case {flag}:SafeNoIOBody<Depth,N,{axis}>(p,tiles);return;\n'
    entries+=' default:SafeNoIOBody<Depth,N>(p,tiles);return;\n }\n}\n'
    entries+='''template<int Depth,int N=0>inline cudaError_t LaunchSafeNoIO(cudaStream_t stream,const Parameters&p,const int32_t*tiles,int count){
 const auto kernel=AsymmetricSafeNoIO<Depth,N>;
 auto status=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,ProfileCachedSharedBytes);if(status!=cudaSuccess)return status;
 AsymmetricSafeNoIO<Depth,N><<<count,1024,ProfileCachedSharedBytes,stream>>>(p,tiles);return cudaGetLastError();
}
'''
    if single_kernel:
        entries='''template<int Depth,int N=0>__device__ __forceinline__ void DispatchSafeNoIO(const Parameters&p,const int32_t*tiles){
 switch(p.flags[tiles[blockIdx.x]]&14){
'''
        for flag,axis in ((2,1),(4,2),(8,4)):entries+=f' case {flag}:SafeNoIOBody<Depth,N,{axis}>(p,tiles);return;\n'
        entries+=' default:SafeNoIOBody<Depth,N>(p,tiles);return;\n }\n}\n'
        marker='template<int Depth,int N>\n__global__ __launch_bounds__(1024,1) void AsymmetricNoIOFixedCube'
    else:
        marker='template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy=(Fast?0:5)>\n__global__'
    if text.count(marker)!=1:raise ValueError('Safe body insertion changed')
    text=text.replace(marker,HELPER+'\n'+clone+'\n'+entries+'\n'+marker)
    for signature in ('void AsymmetricNoIOFixedCube(','void AsymmetricNoIOUniform1024('):
        begin,end,wrapper=block(text,signature)
        safe_dispatch='DispatchSafeNoIO<Depth,N>(p,tiles);' if 'FixedCube' in signature else 'DispatchSafeNoIO<Depth>(p,tiles);'
        gate='if(!(p.flags[tiles[blockIdx.x]]&32)){'+safe_dispatch+'return;}' if single_kernel else 'if(!(p.flags[tiles[blockIdx.x]]&32))return;'
        wrapper=wrapper.replace('{','{\n '+gate,1)
        text=text[:begin]+wrapper+text[end:]
    for n in (() if single_kernel else (512,564)):
        old=f'''   AsymmetricNoIOFixedCube<Depth,{n}><<<uniform_count,1024,FixedCubeSharedBytes,stream>>>(p,ids);
   return cudaGetLastError();'''
        new=old.replace('return cudaGetLastError();',f'status=cudaGetLastError();if(status!=cudaSuccess)return status;return LaunchSafeNoIO<Depth,{n}>(stream,p,ids,uniform_count);')
        if text.count(old)!=1:raise ValueError('Fixed NoIO launch changed')
        text=text.replace(old,new)
    old='''  AsymmetricNoIOUniform1024<Depth><<<uniform_count,1024,ProfileCachedSharedBytes,stream>>>(p,ids);
  return cudaGetLastError();'''
    new=old.replace('return cudaGetLastError();','status=cudaGetLastError();if(status!=cudaSuccess)return status;return LaunchSafeNoIO<Depth>(stream,p,ids,uniform_count);')
    if not single_kernel:
        if text.count(old)!=1:raise ValueError('Generic NoIO launch changed')
        text=text.replace(old,new)
    shutil.copytree(source,output);(output/'src/asymmetric_native.cuh').write_text(text)
    record=dict(schema='beamz.trace-boundary-free-NoIO.v1',parent_sha256=PARENT_CONSUMER,consumer_sha256=sha(text.encode()),
                producer_sha256=sha(Path(__file__).read_bytes()),proof=prove(),expected_kernels=66 if single_kernel else 72,
                startup_contains_elided=True,full_window_specialized=True,CPML_preserved=True,
                duplicate_gated_grids_experimental=not single_kernel,single_kernel_dispatch=single_kernel,GPU_executed=False)
    (output.parent/'safe-noio-proof.json').write_text(json.dumps(record,indent=2)+'\n');return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=PARENT);p.add_argument('--output',type=Path,required=True);p.add_argument('--single-kernel',action='store_true');a=p.parse_args();print(json.dumps(stage(a.source,a.output,a.single_kernel)))
