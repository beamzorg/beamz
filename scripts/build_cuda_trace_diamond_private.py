"""Bind compiler-derived diamond register storage to the complete native runtime."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from scripts.build_cuda_trace_diamond import stage as stage_diamond
from scripts.build_cuda_trace_tile import replace_body
from scripts.cuda_trace_diamond_private_codegen import generate
from scripts.cuda_trace_diamond_storage import DiamondWarpTopology
from scripts.cuda_trace_diamond_geometry import DiamondGeometry


def stage(source,output,expected,plus=4,minus=2,*,dynamic_parity=False,region_clock=False,storage_gauge="retimed",fixed_uniform_classes=False,factorized_cell=False,parity_providers=False,shared_field=None):
    output=Path(output).resolve();topology=DiamondWarpTopology(plus,minus,DiamondGeometry((16,16,32),"yz",storage_gauge),parity_providers,shared_field)
    if parity_providers and not factorized_cell:raise ValueError('Parity providers require factorized cell lowering')
    if factorized_cell:
        from scripts.cuda_trace_diamond_cell_codegen import generate as generate_cell
        generated,proof=generate_cell(topology,region_clock=region_clock)
    else:generated,proof=generate(topology,dynamic_parity=dynamic_parity,region_clock=region_clock)
    stage_diamond(source,output,expected,classify=True,orientation='yz',storage_gauge=storage_gauge)
    original=(output/'src/trace_tile_generated.cuh').read_text()
    original=original.replace('TraceBulkThreads=1024','TraceBulkThreads=DiamondPrivateThreads')
    # Declaration needed by the existing launch mappings; helper body follows
    # DiamondPosition and DiamondImport in the second generated header.
    original=f'constexpr int DiamondPrivateThreads={topology.threads};\n'+original
    generated=generated.replace(f'DiamondPrivateSlots={topology.slots},DiamondPrivateThreads={topology.threads}',f'DiamondPrivateSlots={topology.slots}')
    (output/'src/trace_tile_generated.cuh').write_text(original)
    (output/'src/trace_diamond_private_generated.cuh').write_text(generated)
    proof['source_sha256']=hashlib.sha256(generated.encode()).hexdigest()
    text=(output/'src/asymmetric_native.cuh').read_text()
    text=text.replace('#include "trace_tile_generated.cuh"','#include "trace_tile_generated.cuh"\n#include "trace_diamond_private_generated.cuh"')
    size=proof['shared_bytes']
    old='SharedBytes=98380,ProfileCachedSharedBytes=98380,FixedCubeSharedBytes=98380,TraceSourceFrameBytes=98380'
    if text.count(old)!=1:raise ValueError('Full shared reference allocation changed')
    text=text.replace(old,f'SharedBytes=98380,ProfileCachedSharedBytes={size},FixedCubeSharedBytes={size},TraceSourceFrameBytes=98380')
    text=replace_body(text,'void UniformBody(','''
 const auto flags=p.flags[tiles[blockIdx.x]];
 Environment<Layout,Fast,IsotropicH,Policy,UniformE,-1,false,ActiveAxes,false>env{p,flags,nullptr};
 if constexpr((Fast&&Layout==1)||(Layout==1&&(Policy==1||Policy==2)&&IsotropicH&&UniformE))
  RunDiamondPrivate<Depth,Layout>(p,tiles,env);
 else RunTraceTile<Depth,Layout,false,false,1>(p,tiles,env);
''')
    text=replace_body(text,'void FixedCubeCachedBody(','''
 const auto flags=p.flags[tiles[blockIdx.x]];
 Environment<1,false,true,1,true,-1,false,ActiveAxes,false>env{p,flags,nullptr};
 RunDiamondPrivate<Depth,1>(p,tiles,env);
''')
    text=text.replace('<<<count,((Fast&&Layout==1)?TraceBulkThreads:TraceBaseThreads),SharedBytes,',
                      '<<<count,((Fast&&Layout==1)?TraceBulkThreads:TraceBaseThreads),((Fast&&Layout==1)?DiamondPrivateSharedBytes:SharedBytes),')
    if fixed_uniform_classes:
        text=text.replace(f'ProfileCachedSharedBytes={size},FixedCubeSharedBytes={size}',
                          f'ProfileCachedSharedBytes=98380,FixedCubeSharedBytes={size+3456}')
        text=text.replace('if constexpr((Fast&&Layout==1)||(Layout==1&&(Policy==1||Policy==2)&&IsotropicH&&UniformE))',
                          'if constexpr(Fast&&Layout==1&&IsotropicH&&UniformE)')
        text=text.replace('((Fast&&Layout==1)?DiamondPrivateSharedBytes:SharedBytes),', 'SharedBytes,')
        text=replace_body(text,'void FixedCubeCachedBody(','''
 extern __shared__ float frame[];
 float*profiles=frame+DiamondPrivateSharedBytes/sizeof(float);
 StageIndependentCpmlProfiles(p,profiles);
 const auto flags=p.flags[tiles[blockIdx.x]];
 FixedCubeCachedEnvironment<N,ActiveAxes>env{p,flags,profiles};
 RunDiamondPrivate<Depth,1>(p,tiles,env);
''')
    (output/'src/asymmetric_native.cuh').write_text(text)
    extension=(output/'src/native_extension.cc').read_text()
    extension=extension.replace('module.attr("moving_noio_uniform_threads")=1024;',
                                f'module.attr("moving_noio_uniform_threads")={topology.threads};')
    extension=extension.replace('module.attr("moving_noio_uniform_1024")=true;',f'module.attr("moving_noio_uniform_1024")={str(topology.threads==1024).lower()};')
    extension=extension.replace('module.attr("moving_observer_uniform_1024")=true;',f'module.attr("moving_observer_uniform_1024")={str(topology.threads==1024).lower()};')
    extension=extension.replace('NB_MODULE(_cuda_native, module) {',f'''NB_MODULE(_cuda_native, module) {{
 module.attr("moving_trace_diamond_private_bulk")=true;
 module.attr("moving_trace_diamond_private_bulk_threads")={topology.threads};
 module.attr("moving_trace_diamond_private_bulk_shared_bytes")={size};
''')
    (output/'src/native_extension.cc').write_text(extension)
    proof.update(consumer_sha256=hashlib.sha256(text.encode()).hexdigest(),
                 class_scope='Certified uniform isotropic bulk and fixed512/564 CPML only' if fixed_uniform_classes else 'Certified bulk, uniform CPML and observer; general/source retain full-shared retimed diamond',
                 fixed_uniform_classes=fixed_uniform_classes,
                 generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output.parent/'diamond-private-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    # The general-diamond proof must describe its actual header after launch
    # constants change. Private storage has its own independent emitted proof.
    path=output.parent/'diamond-compiler-proof.json';general=json.loads(path.read_text())
    general['source_sha256']=hashlib.sha256(original.encode()).hexdigest()
    general['consumer_sha256']=proof['consumer_sha256']
    general['private_bulk_proof']='diamond-private-proof.json'
    path.write_text(json.dumps(general,indent=2)+'\n')
    return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--expected-consumer-sha256',required=True)
    p.add_argument('--plus',type=int,default=4);p.add_argument('--minus',type=int,default=2)
    p.add_argument('--dynamic-parity',action='store_true')
    p.add_argument('--region-clock',action='store_true')
    p.add_argument('--storage-gauge',choices=('retimed','physical'),default='retimed')
    p.add_argument('--fixed-uniform-classes',action='store_true')
    p.add_argument('--factorized-cell',action='store_true')
    p.add_argument('--parity-providers',action='store_true')
    p.add_argument('--shared-field',choices=('Hx','Hy','Hz','Ex','Ey','Ez'))
    a=p.parse_args();print(json.dumps(stage(a.source,a.output,a.expected_consumer_sha256,a.plus,a.minus,dynamic_parity=a.dynamic_parity,region_clock=a.region_clock,storage_gauge=a.storage_gauge,fixed_uniform_classes=a.fixed_uniform_classes,factorized_cell=a.factorized_cell,parity_providers=a.parity_providers,shared_field=a.shared_field)))
