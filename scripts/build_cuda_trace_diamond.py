"""Build a complete native diamond-ownership reference lowering.

The first binding keeps policy classification conservative and disables the old
box-shaped material certificate. It is intended to establish complete physics
under transformed ownership before resource/performance specialization.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from scripts.build_cuda_trace_tile import stage as stage_scaffold,replace_body,replace_last_bounds
from scripts.cuda_trace_phase_cycle import block,sha
from scripts.cuda_trace_diamond_codegen import generate
from scripts.cuda_trace_diamond_geometry import DiamondGeometry
from scripts.cuda_trace_diamond_regions import cpp_helpers


def stage(source,output,expected,*,classify=False,orientation="xy",storage_gauge="retimed"):
    source,output=Path(source).resolve(),Path(output).resolve()
    geometry=DiamondGeometry((16,16,32),"yz",storage_gauge) if orientation=="yz" else DiamondGeometry()
    generated,proof=generate(geometry)
    scaffold=stage_scaffold(source,output,expected,cache=(16,32,16),bulk_groups=2,base_groups=2)
    (output.parent/'tile-compiler-proof.json').rename(output.parent/'scaffold-binding-proof.json')
    text=(output/'src/asymmetric_native.cuh').read_text();shared=proof['shared_bytes']
    if orientation=='yz':
        old='BZ=15,BY=31,BX=15,CZ=16,CY=32,CX=16'
        if text.count(old)!=1:raise ValueError('Diamond geometry scaffold changed')
        text=text.replace(old,'BZ=15,BY=15,BX=31,CZ=16,CY=16,CX=32')
        header=(output/'src/native_moving.h').read_text()
        for old,new in [('kOwnedZ=15,kOwnedY=31,kOwnedX=15','kOwnedZ=15,kOwnedY=15,kOwnedX=31'),
                        ('owned[3]{15,31,15}','owned[3]{15,15,31}'),('owned{15,31,15}','owned{15,15,31}')]:
            if header.count(old)!=1:raise ValueError('Diamond host scaffold changed')
            header=header.replace(old,new)
        (output/'src/native_moving.h').write_text(header)
    begin=text.index('constexpr size_t SharedBytes=');end=text.index('struct Source {',begin)
    text=text[:begin]+f'''constexpr size_t SharedBytes={shared},ProfileCachedSharedBytes={shared},FixedCubeSharedBytes={shared},TraceSourceFrameBytes={shared};
static_assert(BEAMZ_NATIVE_ASYMMETRIC_SWIZZLE==0,"Diamond cache owns its packed layout");
'''+text[end:]
    text=replace_body(text,'void UniformBody(','''
 const auto flags=p.flags[tiles[blockIdx.x]];
 // The box certificate is disabled. Keep all physical support checks active.
 Environment<Layout,false,IsotropicH,Policy,false,-1,false,ActiveAxes,false>env{p,flags,nullptr};
 RunTraceTile<Depth,Layout,false,false,1>(p,tiles,env);
''')
    text=replace_body(text,'void FixedCubeCachedBody(','''
 const auto flags=p.flags[tiles[blockIdx.x]];
 Environment<1,false,true,1,false,-1,false,ActiveAxes,false>env{p,flags,nullptr};
 RunTraceTile<Depth,1,false,false,1>(p,tiles,env);
''')
    text=replace_body(text,'void SourcePendingPairedBody(','''
 const auto flags=p.flags[tiles[blockIdx.x]];
 Environment<Layout,false,IsotropicH,Policy,false,SourceMask,PlanePair,ActiveAxes,false>env{p,flags,nullptr};
 RunTraceTile<Depth,Layout,false,false,1>(p,tiles,env);
''')
    if classify:
        text=text.replace('Environment<Layout,false,IsotropicH,Policy,false,',
                          'Environment<Layout,Fast,IsotropicH,Policy,UniformE,')
        text=text.replace('Environment<1,false,true,1,false,',
                          'Environment<1,false,true,1,true,')
    # Exact all-shared frame leaves no room for the optional profile cache.
    old='template<int Policy>constexpr size_t ResearchSourceSharedBytes=SourcePendingSharedBytes+(Policy==5?864ULL*sizeof(float):0);'
    if text.count(old)!=1:raise ValueError('Source allocation binding changed')
    text=text.replace(old,'template<int Policy>constexpr size_t ResearchSourceSharedBytes=SourcePendingSharedBytes;')
    old='if(w.counts[a]!=(int64_t(maximum[a])+w.depth+sizes[a]-1)/sizes[a])'
    if text.count(old)!=1:raise ValueError('Native prepare count check changed')
    text=text.replace('const int sizes[3]={BZ,BY,BX};int64_t tiles=1;',
        'const int sizes[3]={BZ,BY,BX},virtual_extent[3]={maximum[0],minimum[1]+minimum[2]+1,minimum[1]+minimum[2]+2};int64_t tiles=1;')
    text=text.replace(old,'if(w.counts[a]!=(int64_t(virtual_extent[a])+w.depth+sizes[a]-1)/sizes[a])')
    if orientation=='yz':
        text=text.replace('virtual_extent[3]={maximum[0],minimum[1]+minimum[2]+1,minimum[1]+minimum[2]+2}',
                          'virtual_extent[3]={minimum[0]+minimum[1]+1,minimum[0]+minimum[1]+2,maximum[2]}')
    (output/'src/asymmetric_native.cuh').write_text(text)
    (output/'src/trace_tile_generated.cuh').write_text(generated)
    host=(output/'src/native_moving.cc').read_text()
    old='for(int a=0;a<3;++a){p.counts[a]=(int64_t(p.extent[a])+depth+p.owned[a]-1)/p.owned[a];'
    if host.count(old)!=1:raise ValueError('Native host count declaration changed')
    host=host.replace(old,'''const int virtual_extent[3]={p.extent[0],minimum[1]+minimum[2]+1,minimum[1]+minimum[2]+2};
  for(int a=0;a<3;++a){p.counts[a]=(int64_t(virtual_extent[a])+depth+p.owned[a]-1)/p.owned[a];''')
    if host.count('p.tile_flags[tile]=flags;')!=1:raise ValueError('Host policy binding changed')
    if classify:
        start=host.index('      const int origin[]=')
        end=host.index('      if(groups&&host_starts)',start)
        host=host[:start]+'''      const int origin[]={z*p.owned[0],y*p.owned[1],x*p.owned[2]};int64_t low[3],high[3];uint8_t flags=0;
      BeamzNativeDiamondBounds(origin,p.owned.data(),depth,minimum[1],low,high);
      for(int a=0;a<3;++a){
        if(low[a]<thickness||high[a]>=minimum[a]-thickness)flags|=uint8_t(2<<a);
        if(low[a]<=0||high[a]+1>=minimum[a])flags|=kNativeBoundary;
      }
'''+host[end:]
    else:
        host=host.replace('p.tile_flags[tile]=flags;','p.tile_flags[tile]=63; // Conservative complete-physics reference classification.')
    start=host.index('  for(const auto&point:observed_points)for(int version=1;version<=depth;++version)')
    end=host.index('  // Preserve rank antichains.',start)
    host=host[:start]+'''  for(const auto&point:observed_points){
    for(int a=0;a<3;++a)if(point[a]<0||point[a]>=p.extent[a])throw std::invalid_argument("diamond observation outside physical fields");
    for(int version=1;version<=depth;++version)for(int r=0;r<2;++r){
      const int owner[3]={(point[0]+version)/p.owned[0],(point[1]+point[2]+version)/p.owned[1],(point[2]-point[1]+minimum[1]+r+version)/p.owned[2]};
      const int tile=(owner[0]*ny+owner[1])*nx+owner[2];p.tile_flags[tile]|=kMayObserve;
    }
  }
'''+host[end:]
    if not classify:
        host=replace_body(host,'BeamzCertifyNativeMovingUniformE(','''
 BeamzNativeMovingUniformE result;result.admitted.assign(plan.tile_flags.size(),0);
 // The previous prefix queries describe boxes, not retimed diamonds.
 result.supported=false;return result;
''')
    else:
        old='''      for(int a=0;a<3;++a){const int64_t origin=int64_t(q[a])*plan.owned[a];lo[a]=int(std::clamp<int64_t>(origin-plan.depth,0,f.dims[a]));hi[a]=int(std::clamp<int64_t>(origin+plan.owned[a]-1,0,f.dims[a]));}'''
        if host.count(old)!=1:raise ValueError('Material prefix query changed')
        host=host.replace(old,'''      const int origin[3]={q[0]*plan.owned[0],q[1]*plan.owned[1],q[2]*plan.owned[2]},retiming_e[3]={1,0,1};int64_t physical_low[3],physical_high[3];
      BeamzNativeDiamondBounds(origin,plan.owned.data(),plan.depth,int(fields[0].dims[1]),physical_low,physical_high,retiming_e[c]);
      for(int a=0;a<3;++a){lo[a]=int(std::clamp<int64_t>(physical_low[a],0,f.dims[a]));hi[a]=int(std::clamp<int64_t>(physical_high[a]+1,0,f.dims[a]));}''')
        header=(output/'src/native_moving.h').read_text()
        header=header.replace('namespace beamz::cuda::moving {',cpp_helpers(orientation)+'\nnamespace beamz::cuda::moving {',1)
        (output/'src/native_moving.h').write_text(header)
    if orientation=='yz':
        host=host.replace('virtual_extent[3]={p.extent[0],minimum[1]+minimum[2]+1,minimum[1]+minimum[2]+2}',
                          'virtual_extent[3]={minimum[0]+minimum[1]+1,minimum[0]+minimum[1]+2,p.extent[2]}')
        host=host.replace('owner[3]={(point[0]+version)/p.owned[0],(point[1]+point[2]+version)/p.owned[1],(point[2]-point[1]+minimum[1]+r+version)/p.owned[2]}',
                          'owner[3]={(point[0]+point[1]+version)/p.owned[0],(point[0]-point[1]+minimum[1]+r+version)/p.owned[1],(point[2]+version)/p.owned[2]}')
    (output/'src/native_moving.cc').write_text(host)
    extension=(output/'src/native_extension.cc').read_text()
    if orientation=='yz':
        extension=extension.replace('research_cache_shape"]=nb::make_tuple(16,32,16)',
                                    'research_cache_shape"]=nb::make_tuple(16,16,32)')
    extension=extension.replace(f'result["moving_source_pending_shared_bytes"]={scaffold["source_shared_bytes"]};result["moving_original_shared_bytes"]={scaffold["shared_bytes"]};',
                                 f'result["moving_source_pending_shared_bytes"]={shared};result["moving_original_shared_bytes"]={shared};')
    extension=extension.replace('module.attr("moving_source_pending_mask20_1024")=true;',
                                'module.attr("moving_source_pending_mask20_1024")=false;')
    # The header defines the actual per-class1024/512 mapping for this lowering.
    extension=extension.replace('module.attr("moving_source_pending_threads")=1024;',
                                'module.attr("moving_source_pending_threads")=512;')
    extension=extension.replace('NB_MODULE(_cuda_native, module) {','''NB_MODULE(_cuda_native, module) {
 module.attr("moving_trace_diamond_gpu")=true;
 module.attr("research_cache_lattice_determinant")=2;
 module.attr("research_uniform_e_certificate_disabled")=true;
 module.attr("moving_source_plane_pair_threads")=1024;
''')
    if classify:
        extension=extension.replace('module.attr("research_uniform_e_certificate_disabled")=true;',
                                    'module.attr("research_uniform_e_certificate_disabled")=false;')
        start,end,_=block(extension,'for(int c=0;c<3;++c){int64_t first=1,last=depth;')
        extension=extension[:start]+'''for(int c=0;c<3;++c){const auto&f=base_inputs_[3+c];
          const int origin[3]={q[0]*moving_plan_.owned[0],q[1]*moving_plan_.owned[1],q[2]*moving_plan_.owned[2]};
          const int shape[3]={int(f.dims[0]),int(f.dims[1]),int(f.dims[2])},r[3]={1,0,1};
          has_e_update|=BeamzNativeDiamondHasUpdate(origin,moving_plan_.owned.data(),depth,minimum[1],shape,r[c]);
        }'''+extension[end:]
    (output/'src/native_extension.cc').write_text(extension)
    proof.update(parent_source=str(source),parent_consumer_sha256=expected,
                 consumer_sha256=sha(text.encode()),host_plan_sha256=sha(host.encode()),
                 generator_sha256=sha(Path(__file__).read_bytes()),expected_kernels=68,
                 physics_header_unchanged=True,classify=classify,
                 classification='Conservative physical bounds, exact nonempty ownership and per-field material queries' if classify else 'All general; box material certificate disabled')
    (output.parent/'diamond-compiler-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--expected-consumer-sha256',required=True)
    p.add_argument('--classify',action='store_true')
    p.add_argument('--orientation',choices=('xy','yz'),default='xy')
    p.add_argument('--storage-gauge',choices=('retimed','physical'),default='retimed')
    a=p.parse_args();print(json.dumps(stage(a.source,a.output,a.expected_consumer_sha256,classify=a.classify,orientation=a.orientation,storage_gauge=a.storage_gauge)))
