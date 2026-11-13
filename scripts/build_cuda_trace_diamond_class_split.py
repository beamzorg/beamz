"""Separate traced bulk and single-axis CPML ownership classes into kernels.

Existing rank-local ranges remain intact. Complementary guards partition the
same tile IDs; every branch finishes before the next rank. This prototype pays
for empty CTAs to avoid changing the native workspace ABI while measuring the
compiler's actual class-specific register and instruction requirements.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from scripts.build_cuda_trace_diamond_private import stage as stage_private
from scripts.build_cuda_trace_tile import replace_body
from scripts.cuda_trace_phase_cycle import block


KERNELS=r'''
template<int Depth>__global__ __launch_bounds__(DiamondPrivateThreads,1) void AsymmetricDiamondUniformBulk(
 const __grid_constant__ Parameters p,const int32_t*ids){
 const auto flags=p.flags[ids[blockIdx.x]];if(flags!=64)return;
 Environment<1,true,true,0,true,-1,false,-1,false>env{p,flags,nullptr};
 RunDiamondPrivate<Depth,1>(p,ids,env);
}
template<int Depth,int N,int Axis>__global__ __launch_bounds__(DiamondPrivateThreads,1) void AsymmetricDiamondNoIOFace(
 const __grid_constant__ Parameters p,const int32_t*ids){
 const auto flags=p.flags[ids[blockIdx.x]];
 if(!(flags&64)||(flags&17)||((flags&14)!=(2<<Axis)))return;
 extern __shared__ float frame[];float*profiles=frame+DiamondPrivateSharedBytes/sizeof(float);
 for(int i=threadIdx.x;i<864;i+=DiamondPrivateThreads){
  const int profile=i/24,packed=i%24,slot=profile/3,kind=profile%3;const auto&psi=p.phase[slot/6].psi[slot%6];
  const float*input=kind==0?psi.a:kind==1?psi.b:psi.inv;profiles[i]=__ldg(input+packed);
 }
 FixedCubeCachedEnvironment<N,(1<<Axis)>env{p,flags,profiles};
 RunDiamondPrivate<Depth,1>(p,ids,env);
}
template<int Depth,int N>inline cudaError_t LaunchDiamondNoIOFaces(cudaStream_t stream,const Parameters&p,const int32_t*ids,int count){
 constexpr size_t bytes=DiamondPrivateSharedBytes+3456;cudaError_t status;
 const auto z=AsymmetricDiamondNoIOFace<Depth,N,0>,y=AsymmetricDiamondNoIOFace<Depth,N,1>,x=AsymmetricDiamondNoIOFace<Depth,N,2>;
 status=cudaFuncSetAttribute(z,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);if(status!=cudaSuccess)return status;
 status=cudaFuncSetAttribute(y,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);if(status!=cudaSuccess)return status;
 status=cudaFuncSetAttribute(x,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);if(status!=cudaSuccess)return status;
 AsymmetricDiamondNoIOFace<Depth,N,0><<<count,DiamondPrivateThreads,bytes,stream>>>(p,ids);status=cudaGetLastError();if(status!=cudaSuccess)return status;
 AsymmetricDiamondNoIOFace<Depth,N,1><<<count,DiamondPrivateThreads,bytes,stream>>>(p,ids);status=cudaGetLastError();if(status!=cudaSuccess)return status;
 AsymmetricDiamondNoIOFace<Depth,N,2><<<count,DiamondPrivateThreads,bytes,stream>>>(p,ids);return cudaGetLastError();
}
'''


def stage(source,output,expected,plus=4,minus=2,*,parity_providers=False,shared_field=None):
    output=Path(output).resolve()
    stage_private(source,output,expected,plus,minus,dynamic_parity=True,region_clock=True,storage_gauge='physical',factorized_cell=True,parity_providers=parity_providers,shared_field=shared_field)
    text=(output/'src/asymmetric_native.cuh').read_text();proof=json.loads((output.parent/'diamond-private-proof.json').read_text());size=proof['shared_bytes']
    text=text.replace(f'ProfileCachedSharedBytes={size},FixedCubeSharedBytes={size}', 'ProfileCachedSharedBytes=98380,FixedCubeSharedBytes=98380')
    text=text.replace('((Fast&&Layout==1)?DiamondPrivateSharedBytes:SharedBytes),','SharedBytes,')
    text=replace_body(text,'void UniformBody(','''
 const auto flags=p.flags[tiles[blockIdx.x]];
 Environment<Layout,Fast,IsotropicH,Policy,UniformE,-1,false,ActiveAxes,false>env{p,flags,nullptr};
 RunTraceTile<Depth,Layout,false,false,1>(p,tiles,env);
''')
    text=replace_body(text,'void FixedCubeCachedBody(','''
 const auto flags=p.flags[tiles[blockIdx.x]];
 Environment<1,false,true,1,true,-1,false,ActiveAxes,false>env{p,flags,nullptr};
 RunTraceTile<Depth,1,false,false,1>(p,tiles,env);
''')
    start=text.index('template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy=(Fast?0:5)>\n__global__')
    text=text[:start]+KERNELS+'\n'+text[start:]
    _,_,entry=block(text,'void Asymmetric(');body=entry[entry.index('{')+1:-1]
    text=replace_body(text,'void Asymmetric(','''
 if constexpr(Layout==1&&Fast&&IsotropicH&&Policy==0){
  if(p.flags[tiles[blockIdx.x]]==64)return;
  UniformBody<Depth,Layout,Fast,IsotropicH,Policy,false>(p,tiles);
 }else{
'''+body+'\n }\n')
    _,_,entry=block(text,'void AsymmetricNoIOFixedCube(');body=entry[entry.index('{')+1:-1]
    # Complement of the exact admission used by all three face kernels.
    text=replace_body(text,'void AsymmetricNoIOFixedCube(','''
 const auto flags=p.flags[tiles[blockIdx.x]];const int axes=flags&14;
 if((flags&64)&&!(flags&17)&&(axes==2||axes==4||axes==8))return;
'''+body)
    start,end,launch=block(text,'inline cudaError_t Launch(')
    old='status=cudaGetLastError();if(status!=cudaSuccess)return status;\n return cudaSuccess;'
    new='''status=cudaGetLastError();if(status!=cudaSuccess)return status;
 if constexpr(Layout==1&&Fast&&IsotropicH&&Policy==0){
  const auto specialized=AsymmetricDiamondUniformBulk<Depth>;
  status=cudaFuncSetAttribute(specialized,cudaFuncAttributeMaxDynamicSharedMemorySize,DiamondPrivateSharedBytes);if(status!=cudaSuccess)return status;
  AsymmetricDiamondUniformBulk<Depth><<<count,DiamondPrivateThreads,DiamondPrivateSharedBytes,stream>>>(p,ids);
  return cudaGetLastError();
 }
 return cudaSuccess;'''
    if launch.count(old)!=1:raise ValueError('Bulk native launch binding changed')
    text=text[:start]+launch.replace(old,new)+text[end:]
    for n in (512,564):
        old=f'AsymmetricNoIOFixedCube<Depth,{n}><<<uniform_count,TraceBulkThreads,FixedCubeSharedBytes,stream>>>(p,ids);\n   return cudaGetLastError();'
        new=f'''AsymmetricNoIOFixedCube<Depth,{n}><<<uniform_count,TraceBulkThreads,FixedCubeSharedBytes,stream>>>(p,ids);
   status=cudaGetLastError();if(status!=cudaSuccess)return status;
   return LaunchDiamondNoIOFaces<Depth,{n}>(stream,p,ids,uniform_count);'''
        if text.count(old)!=1:raise ValueError('Fixed NoIO native launch binding changed')
        text=text.replace(old,new)
    (output/'src/asymmetric_native.cuh').write_text(text)
    extension=(output/'src/native_extension.cc').read_text().replace('NB_MODULE(_cuda_native, module) {','''NB_MODULE(_cuda_native, module) {
 module.attr("moving_diamond_split_classes")=true;
 module.attr("moving_diamond_noio_face_threads")=DiamondPrivateThreads;
'''.replace('=DiamondPrivateThreads;',f'={proof["threads"]};'))
    (output/'src/native_extension.cc').write_text(extension)
    proof.update(class_split=True,expected_kernels=82,class_scope='Dedicated uniform isotropic bulk and single-axis fixed512/564 CPML faces; other classes retain full-shared physical-gauge diamond',
                 consumer_sha256=hashlib.sha256(text.encode()).hexdigest(),generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 scheduling='Complementary guards over identical rank-local ranges; extra empty CTAs; host workspace ABI unchanged')
    (output.parent/'diamond-private-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--expected-consumer-sha256',required=True);p.add_argument('--parity-providers',action='store_true');p.add_argument('--shared-field',choices=('Hx','Hy','Hz','Ex','Ey','Ez'));p.add_argument('--plus',type=int,default=4);p.add_argument('--minus',type=int,default=2)
    a=p.parse_args();print(json.dumps(stage(a.source,a.output,a.expected_consumer_sha256,a.plus,a.minus,parity_providers=a.parity_providers,shared_field=a.shared_field)))
