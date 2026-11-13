"""Lower a traced private-Z/warp-X/shared-Y schedule into immutable CUDA sources."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil

from scripts.cuda_trace_phase_cycle import ROOT, block, sha
from scripts.cuda_trace_systolic_compiler import (ThreadMap, compile_column, prove_routes,
                                                prove_compact_providers, prove_z_providers,
                                                prove_deduplicated_z)

PARENT = ROOT / '.cache/cuda-trace-150min-2026-11-13-1620/phase-grouped-sourcefacts-v1/source'
PARENT_SHA = 'cce32cfee8c4497d05c30585672935d03515bbf6fcf2bfc179c45865a2f3b7e1'


def emit_body(text, *, separate_families=True, compact=False, z_groups=1, deduplicate=False,
              fused_publication=False):
    mapping = ThreadMap(threads=512*z_groups, z_groups=z_groups)
    schedule = compile_column(mapping=mapping, separate_families=separate_families)
    _, _, original = block(text, 'void UniformBody(')
    prefix = original[original.index('{') + 1:original.index(' const uint8_t flags=')]
    prefix = prefix.replace('constexpr int T=(Fast||(Layout==1&&(Policy==1||Policy==2)&&IsotropicH&&UniformE))?1024:512,Slots=PAD/T,ZStep=T/512;',
                            'constexpr int T=512,Slots=CZ,ZStep=1;')
    prefix = prefix.replace('frame+3*PAD', 'frame+ColumnFieldBanks*PAD')
    prefix = prefix.replace('frame+SharedBytes/sizeof(float)', 'frame+ColumnBaseBytes/sizeof(float)')
    prefix = prefix.replace('zbase=threadIdx.x/512', 'zbase=0')
    # Only Ex/Ez are read through the Y-neighbour shared buffer. H/E buffers
    # are distinct in the two-barrier schedule and aliased in the four-barrier one.
    prefix = prefix.replace('frame[c*PAD+Slot(zs,sy,sx)]=e[c][j];',
                            'if(c!=1)frame[(ColumnEBase+(c==2))*PAD+Slot(zs,sy,sx)]=e[c][j];')
    prefix = prefix.replace('frame[Slot(zs,sy,sx)]=v.ex;frame[PAD+Slot(zs,sy,sx)]=v.ey;frame[2*PAD+Slot(zs,sy,sx)]=v.ez;',
                            'frame[ColumnEBase*PAD+Slot(zs,sy,sx)]=v.ex;frame[(ColumnEBase+1)*PAD+Slot(zs,sy,sx)]=v.ez;')
    # The prefix contains the reviewed native admission/late-start initialization.
    # Replace only its environment selection with a compile-time geometry choice.
    parts = ['template<int Depth,int N=0,int ActiveAxes=-1,bool Fast=false,bool UniformE=true>',
             '__device__ __forceinline__ void TraceColumnBody(const Parameters&p,const int32_t*tiles){',
             ' constexpr int Layout=1,Policy=Fast?0:1;constexpr bool IsotropicH=true,CachedProfiles=!Fast&&ColumnCacheProfiles;',
             prefix,
             ' const uint8_t flags=p.flags[tile];',
             ' auto env=[&] __device__(){if constexpr(Fast)return Environment<1,true,true,0,UniformE,-1,false,0,false>{p,0,nullptr};',
             ' else if constexpr(N>0&&ColumnCacheProfiles)return FixedCubeCachedEnvironment<N,ActiveAxes>{p,flags,profiles};',
             ' else return Environment<1,false,true,1,true,-1,false,ActiveAxes,ColumnCacheProfiles>{p,flags,profiles};}();',
             ' #pragma unroll 1',
             ' for(int n=(Fast?1:window[0]);n<=(Fast?Depth:window[1]);++n){',
             '  int step;asm volatile("mov.b32 %0,%1;":"=r"(step):"r"(n));',
             '  const int hz=(-step-1)&7,hy=(-step-1)&15,hx=(-step-1)&31,low=-step-1;',
             '  const int dy=(sy-hy)&15,dx=(sx-hx)&31,y=oy+low+dy,x=ox+low+dx;',
             '  #pragma unroll',
             '  for(int j=0;j<Slots;++j){const int zs=j,z=oz+low+((zs-hz)&7);',
             '   const float x1=__shfl_sync(0xffffffffu,e[1][j],(sx+1)&31);',
             '   const float x2=__shfl_sync(0xffffffffu,e[2][j],(sx+1)&31);',
             '   if(zs!=hz&&sy!=hy&&sx!=hx){',
             '    const auto read=[&] __device__(int field,int,int dz,int ddy,int ddx){',
             '     if(field<3)return h[field][j];const int c=field-3;',
             '     if(dz)return e[c][(j+dz)&7];',
             '     if(ddy)return frame[(ColumnEBase+(c==2))*PAD+Slot(zs,sy+ddy,sx)];',
             '     if(ddx)return c==1?x1:x2;return e[c][j];',
             '    };',
             '    #pragma unroll',
             '    for(int c=0;c<3;++c){float value=0.f;if(env.Valid(c,z,y,x)){',
             '     if(c==0)value=beamz::cuda::trace_native::AdvanceNativeCell<0,0,false>(env,read,n,z,y,x);',
             '     else if(c==1)value=beamz::cuda::trace_native::AdvanceNativeCell<0,1,false>(env,read,n,z,y,x);',
             '     else value=beamz::cuda::trace_native::AdvanceNativeCell<0,2,false>(env,read,n,z,y,x);',
             '    }h[c][j]=value;}',
             '   }',
             '  }',
             '  // All old-E reads have finished in this thread before ghost replacement.',
             '  #pragma unroll',
             '  for(int j=0;j<Slots;++j){const int zs=j;if(zs==hz||sy==hy||sx==hx){',
             '   const auto v=ImportPacket<Depth,Fast>(p,bases,n,oz,oy,ox,(zs-hz)&7,dy,dx);',
             '   h[0][j]=v.hx;h[1][j]=v.hy;h[2][j]=v.hz;e[0][j]=v.ex;e[1][j]=v.ey;e[2][j]=v.ez;',
             '  }}']
    if not separate_families:
        parts += ['  __syncthreads(); // Retire old-E readers before aliased H stores.']
    parts += ['  #pragma unroll',
              '  for(int j=0;j<Slots;++j){frame[Slot(j,sy,sx)]=h[0][j];frame[PAD+Slot(j,sy,sx)]=h[2][j];}',
              '  __syncthreads(); // Publish current H, retire all old-E readers.',
              '  #pragma unroll',
              '  for(int j=0;j<Slots;++j){const int zs=j,z=oz+low+((zs-hz)&7);',
              '   const float x1=__shfl_sync(0xffffffffu,h[1][j],(sx-1)&31);',
              '   const float x2=__shfl_sync(0xffffffffu,h[2][j],(sx-1)&31);',
              '   if(zs!=hz&&sy!=hy&&sx!=hx){',
              '    const auto read=[&] __device__(int field,int,int dz,int ddy,int ddx){',
              '     if(field>=3)return e[field-3][j];',
              '     if(dz)return h[field][(j+dz)&7];',
              '     if(ddy)return frame[(field==2)*PAD+Slot(zs,sy+ddy,sx)];',
              '     if(ddx)return field==1?x1:x2;return h[field][j];',
              '    };',
              '    #pragma unroll',
              '    for(int c=0;c<3;++c){float value=0.f;if(env.Valid(c+3,z,y,x)){',
              '     if(c==0)value=beamz::cuda::trace_native::AdvanceNativeCell<1,0,false>(env,read,n,z,y,x);',
              '     else if(c==1)value=beamz::cuda::trace_native::AdvanceNativeCell<1,1,false>(env,read,n,z,y,x);',
              '     else value=beamz::cuda::trace_native::AdvanceNativeCell<1,2,false>(env,read,n,z,y,x);',
              '    }e[c][j]=value;}',
              '   }',
              '  }']
    if not separate_families:
        parts += ['  __syncthreads(); // Retire H readers before aliased E stores.']
    parts += ['  #pragma unroll',
              '  for(int j=0;j<Slots;++j){frame[ColumnEBase*PAD+Slot(j,sy,sx)]=e[0][j];frame[(ColumnEBase+1)*PAD+Slot(j,sy,sx)]=e[2][j];}',
              '  __syncthreads(); // Publish current E, retire all current-H readers.']
    # Preserve the current grouped global publication and terminal stores exactly.
    _, _, loop = block(original, 'for(int n=(Fast?1:window[0]);n<=(Fast?Depth:window[1]);++n)')
    publication = loop[loop.index('  const int y=oy+low+dy,x=ox+low+dx;'): -1]
    publication = publication.replace('  const int y=oy+low+dy,x=ox+low+dx;', '')
    parts += [publication, ' }', '}']
    result = '\n'.join(parts)
    if z_groups != 1:
        if z_groups != 2:
            raise ValueError('Native column lowering currently admits one or two Z groups')
        result = result.replace('constexpr int T=512,Slots=CZ,ZStep=1;',
                                'constexpr int T=1024,Slots=CZ/ColumnZGroups,ZStep=1;')
        result = result.replace('zbase=0', 'zbase=(threadIdx.x/512)*Slots')
        result = result.replace('const int zs=j,', 'const int zs=zbase+j,')
        result = result.replace('const int zs=j;', 'const int zs=zbase+j;')
        result = result.replace('if(dz)return e[c][(j+dz)&7];',
            'if(dz){if(j+dz>=0&&j+dz<Slots)return e[c][j+dz];return frame[ColumnZOffset+(ColumnEBase+(c==1))*ColumnZPoints+ColumnZReadIndex(zs,sy,sx,n,dz)];}')
        result = result.replace('if(dz)return h[field][(j+dz)&7];',
            'if(dz){if(j+dz>=0&&j+dz<Slots)return h[field][j+dz];return frame[ColumnZOffset+(field==1)*ColumnZPoints+ColumnZReadIndex(zs,sy,sx,n,dz)];}')
        result = result.replace('frame+ColumnFieldBanks*PAD', 'frame+ColumnZOffset+ColumnFieldBanks*ColumnZPoints')
        # Initial z-edge state must be visible to the very first H update.
        result = result.replace('if(c!=1)frame[(ColumnEBase+(c==2))*PAD+Slot(zs,sy,sx)]=e[c][j];',
            'if(c!=1)frame[(ColumnEBase+(c==2))*PAD+Slot(zs,sy,sx)]=e[c][j];if(c!=2)ColumnStoreZComponent<true>(frame,c,zs,sy,sx,0,e[c][j]);')
        init = 'frame[ColumnEBase*PAD+Slot(zs,sy,sx)]=v.ex;frame[(ColumnEBase+1)*PAD+Slot(zs,sy,sx)]=v.ez;'
        result = result.replace(init, init+'ColumnStoreZComponent<true>(frame,0,zs,sy,sx,prev,v.ex);ColumnStoreZComponent<true>(frame,1,zs,sy,sx,prev,v.ey);')
        result = result.replace('for(int j=0;j<Slots;++j){frame[Slot(j,sy,sx)]=h[0][j];frame[PAD+Slot(j,sy,sx)]=h[2][j];}',
            'for(int j=0;j<Slots;++j){const int zs=zbase+j;frame[Slot(zs,sy,sx)]=h[0][j];frame[PAD+Slot(zs,sy,sx)]=h[2][j];ColumnStoreZComponent<false>(frame,0,zs,sy,sx,n,h[0][j]);ColumnStoreZComponent<false>(frame,1,zs,sy,sx,n,h[1][j]);}')
        result = result.replace('for(int j=0;j<Slots;++j){frame[ColumnEBase*PAD+Slot(j,sy,sx)]=e[0][j];frame[(ColumnEBase+1)*PAD+Slot(j,sy,sx)]=e[2][j];}',
            'for(int j=0;j<Slots;++j){const int zs=zbase+j;frame[ColumnEBase*PAD+Slot(zs,sy,sx)]=e[0][j];frame[(ColumnEBase+1)*PAD+Slot(zs,sy,sx)]=e[2][j];ColumnStoreZComponent<true>(frame,0,zs,sy,sx,n,e[0][j]);ColumnStoreZComponent<true>(frame,1,zs,sy,sx,n,e[1][j]);}')
    if compact:
        # Replace initialization publications using the actual prior version.
        # Fast is false in this admitted lowering, but its dead C++ branch
        # must remain syntactically well-formed as well.
        result = result.replace('frame+ColumnFieldBanks*PAD', 'frame+ColumnFieldBanks*ColumnPlanePoints')
        initial_fast = 'if(c!=1)frame[(ColumnEBase+(c==2))*PAD+Slot(zs,sy,sx)]=e[c][j];'
        result = result.replace(initial_fast,
            'if(c!=1)ColumnStoreComponent<true>(frame,c,zs,sy,sx,0,e[c][j]);')
        initial_late = 'frame[ColumnEBase*PAD+Slot(zs,sy,sx)]=v.ex;frame[(ColumnEBase+1)*PAD+Slot(zs,sy,sx)]=v.ez;'
        if result.count(initial_late) != 1:
            raise ValueError('Exact late-start initial publication required')
        result = result.replace(initial_late,
            'ColumnStoreComponent<true>(frame,0,zs,sy,sx,prev,v.ex);ColumnStoreComponent<true>(frame,2,zs,sy,sx,prev,v.ez);')
        result = result.replace('frame[(ColumnEBase+(c==2))*PAD+Slot(zs,sy+ddy,sx)]',
            'frame[(ColumnEBase+(c==2))*ColumnPlanePoints+ColumnReadIndex(zs,sy,sx,n)]')
        result = result.replace('frame[(field==2)*PAD+Slot(zs,sy+ddy,sx)]',
            'frame[(field==2)*ColumnPlanePoints+ColumnReadIndex(zs,sy,sx,n)]')
        result = result.replace('frame[Slot(j,sy,sx)]=h[0][j];frame[PAD+Slot(j,sy,sx)]=h[2][j];',
            'ColumnStoreComponent<false>(frame,0,j,sy,sx,n,h[0][j]);ColumnStoreComponent<false>(frame,2,j,sy,sx,n,h[2][j]);')
        result = result.replace('frame[ColumnEBase*PAD+Slot(j,sy,sx)]=e[0][j];frame[(ColumnEBase+1)*PAD+Slot(j,sy,sx)]=e[2][j];',
            'ColumnStoreComponent<true>(frame,0,j,sy,sx,n,e[0][j]);ColumnStoreComponent<true>(frame,2,j,sy,sx,n,e[2][j]);')
        if z_groups > 1:
            result = result.replace('frame[Slot(zs,sy,sx)]=h[0][j];frame[PAD+Slot(zs,sy,sx)]=h[2][j];',
                'ColumnStoreComponent<false>(frame,0,zs,sy,sx,n,h[0][j]);ColumnStoreComponent<false>(frame,2,zs,sy,sx,n,h[2][j]);')
            result = result.replace('frame[ColumnEBase*PAD+Slot(zs,sy,sx)]=e[0][j];frame[(ColumnEBase+1)*PAD+Slot(zs,sy,sx)]=e[2][j];',
                'ColumnStoreComponent<true>(frame,0,zs,sy,sx,n,e[0][j]);ColumnStoreComponent<true>(frame,2,zs,sy,sx,n,e[2][j]);')
    if result.count('__shfl_sync(') != 4:
        raise ValueError('All four shuffle sites must remain outside owner predicates')
    if deduplicate:
        result = result.replace('frame+ColumnZOffset+ColumnFieldBanks*ColumnZPoints',
                                'frame+ColumnZOffset+ColumnZTotalWords')
        result = result.replace('frame[ColumnZOffset+(ColumnEBase+(c==1))*ColumnZPoints+ColumnZReadIndex(zs,sy,sx,n,dz)]',
                                'ColumnReadZValue<true>(frame,c,zs,sy,sx,n,dz)')
        result = result.replace('frame[ColumnZOffset+(field==1)*ColumnZPoints+ColumnZReadIndex(zs,sy,sx,n,dz)]',
                                'ColumnReadZValue<false>(frame,field,zs,sy,sx,n,dz)')
    if fused_publication:
        if not separate_families or z_groups != 2 or not compact:
            raise ValueError('First fused publication lowering requires separate compact two-group banks')
        for electric, marker in ((False,'  __syncthreads(); // Publish current H,'),
                                 (True,'  __syncthreads(); // Publish current E,')):
            end=result.index(marker)
            start=result.rfind('  #pragma unroll',0,end)
            region=result[start:end]
            _,_,loop=block(region,'for(int j=0;j<Slots;++j)')
            statements=loop[loop.index('{')+1:-1]
            if not statements.startswith('const int zs=zbase+j;'):
                raise ValueError('Exact separate publication loop required')
            statements=statements.removeprefix('const int zs=zbase+j;')
            result=result[:start]+result[end:]
            if not electric:
                old='    }h[c][j]=value;}\n   }'
                if result.count(old)!=1:raise ValueError('Unique H compute publication site required')
                result=result.replace(old,'    }h[c][j]=value;}\n    '+statements+'\n   }')
                old='h[0][j]=v.hx;h[1][j]=v.hy;h[2][j]=v.hz;e[0][j]=v.ex;e[1][j]=v.ey;e[2][j]=v.ez;\n  }}'
                if result.count(old)!=1:raise ValueError('Unique current ghost import required')
                result=result.replace(old,old[:-5]+statements+'\n  }}')
            else:
                old='    }e[c][j]=value;}\n   }\n  }'
                if result.count(old)!=1:raise ValueError('Unique E compute publication site required')
                result=result.replace(old,'    }e[c][j]=value;}\n   }\n   '+statements+'\n  }')
    return result, schedule


def stage(source, output, *, separate_families=True, compact=False, z_groups=1, scope='noio',
          deduplicate=False, cache_profiles=True, generic_geometry=False, fused_publication=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    if (output.exists() or output == ROOT / 'cuda' or ROOT / 'cuda' in output.parents
            or source in output.parents or output in source.parents):
        raise ValueError('New disjoint copied source required')
    original = (source / 'src/asymmetric_native.cuh').read_text()
    if sha(original.encode()) != PARENT_SHA:
        raise ValueError('Explicit frozen temporal winner required')
    if scope not in ('noio', 'fast', 'all'):
        raise ValueError('Explicit certified NoIO/fast scope required')
    if deduplicate and (not compact or z_groups != 2):
        raise ValueError('Deduplicated frontiers require compact providers and two Z groups')
    body, schedule = emit_body(original, separate_families=separate_families, compact=compact,
                               z_groups=z_groups,deduplicate=deduplicate,fused_publication=fused_publication)
    field_banks = 4 if separate_families else 2
    definitions = (f'constexpr bool ColumnCacheProfiles={str(cache_profiles).lower()};\n'
                   f'constexpr int ColumnFieldBanks={field_banks},ColumnEBase={field_banks-2},ColumnPlanePoints={"BZ*BY*BX" if compact else "PAD"},ColumnZGroups={z_groups};\n'
                   'constexpr int ColumnZOffset=ColumnFieldBanks*ColumnPlanePoints,ColumnZPoints=ColumnZGroups>1?ColumnZGroups*BY*BX:0;\n'
                   'constexpr size_t ColumnBaseBytes=(ColumnZOffset+ColumnFieldBanks*ColumnZPoints)*sizeof(float)+8*sizeof(size_t)+3*sizeof(int);\n'
                   'constexpr size_t ColumnSharedBytes=ColumnBaseBytes+(ColumnCacheProfiles?864ULL*sizeof(float):0);\n')
    if compact:
        definitions += '''
__device__ __forceinline__ int ColumnReadIndex(int z,int y,int x,int version){
 const int dz=(z+version+1)&7,dy=(y+version+1)&15,dx=(x+version+1)&31;
 return ((dz-1)*BY+dy-1)*BX+dx-1;
}
template<bool Electric>__device__ __forceinline__ void ColumnStoreComponent(float*frame,int c,int z,int y,int x,int version,float value){
 const int dz=(z+version+1)&7,dy=(y+version+1)&15,dx=(x+version+1)&31;
 if constexpr(Electric){if(dz<BZ&&dy&&dx<BX){const int index=(dz*BY+dy-1)*BX+dx;
  frame[(ColumnEBase+(c==2))*ColumnPlanePoints+index]=value;}}
 else {if(dz&&dy<BY&&dx){const int index=((dz-1)*BY+dy)*BX+dx-1;
  frame[(c==2)*ColumnPlanePoints+index]=value;}}
}
'''
    if z_groups > 1:
        definitions += '''
__device__ __forceinline__ int ColumnZReadIndex(int z,int y,int x,int version,int delta){
 const int dy=(y+version+1)&15,dx=(x+version+1)&31,group=((z+delta)&7)/(CZ/ColumnZGroups);
 return (group*BY+dy-1)*BX+dx-1;
}
template<bool Electric>__device__ __forceinline__ void ColumnStoreZComponent(float*frame,int c,int z,int y,int x,int version,float value){
 const int dy=(y+version+1)&15,dx=(x+version+1)&31,group=z/(CZ/ColumnZGroups);
 if constexpr(Electric){if(z%(CZ/ColumnZGroups)==0&&dy<BY&&dx<BX){const int index=(group*BY+dy)*BX+dx;
  frame[ColumnZOffset+(ColumnEBase+(c==1))*ColumnZPoints+index]=value;}}
 else {if(z%(CZ/ColumnZGroups)==(CZ/ColumnZGroups)-1&&dy&&dx){const int index=(group*BY+dy-1)*BX+dx-1;
  frame[ColumnZOffset+(c==1)*ColumnZPoints+index]=value;}}
}
'''
    if deduplicate:
        definitions = definitions.replace(
            'constexpr size_t ColumnBaseBytes=(ColumnZOffset+ColumnFieldBanks*ColumnZPoints)*sizeof(float)+8*sizeof(size_t)+3*sizeof(int);',
            f'constexpr int ColumnZFamilyWords=ColumnZGroups*BY*BX+BY*BX+ColumnZGroups*BX,ColumnZTotalWords={2 if separate_families else 1}*ColumnZFamilyWords;\n'
            'constexpr size_t ColumnBaseBytes=(ColumnZOffset+ColumnZTotalWords)*sizeof(float)+8*sizeof(size_t)+3*sizeof(int);')
        start, end, _ = block(definitions, 'void ColumnStoreZComponent(')
        definitions = definitions[:start]+'''void ColumnStoreZComponent(float*frame,int c,int z,int y,int x,int version,float value){
 const int dz=(z+version+1)&7,dy=(y+version+1)&15,dx=(x+version+1)&31,group=z/(CZ/ColumnZGroups);
 if constexpr(Electric){if(z%(CZ/ColumnZGroups)||dy>=BY||dx>=BX)return;}
 else {if(z%(CZ/ColumnZGroups)!=(CZ/ColumnZGroups)-1||!dy||!dx)return;}
 const int familybase=Electric?(ColumnEBase?ColumnZFamilyWords:0):0;
 int index;
 if(c==1){index=(group*BY+(Electric?dy:dy-1))*BX+(Electric?dx:dx-1);}
 else {
  const bool inY=Electric?(dz<BZ&&dy):(dz&&dy<BY);
  if(inY)return;
  const bool ghost=Electric?dz==BZ:dz==0;
  index=ColumnZGroups*BY*BX+(ghost?((Electric?dy:dy-1)*BX+(Electric?dx:dx-1)):(BY*BX+group*BX+(Electric?dx:dx-1)));
 }
 frame[ColumnZOffset+familybase+index]=value;
}
template<bool Electric>__device__ __forceinline__ float ColumnReadZValue(const float*frame,int c,int z,int y,int x,int target_version,int delta){
 const int version=Electric?target_version-1:target_version;
 const int sourcez=(z+delta)&7,dz=(sourcez+version+1)&7,dy=(y+version+1)&15,dx=(x+version+1)&31,group=sourcez/(CZ/ColumnZGroups);
 const int familybase=Electric?(ColumnEBase?ColumnZFamilyWords:0):0;
 if(c==0){
  const bool inY=Electric?(dz<BZ&&dy):(dz&&dy<BY);
  if(inY){const int index=Electric?((dz*BY+dy-1)*BX+dx):(((dz-1)*BY+dy)*BX+dx-1);
   return frame[(Electric?ColumnEBase:0)*ColumnPlanePoints+index];}
  const bool ghost=Electric?dz==BZ:dz==0;
  const int index=ColumnZGroups*BY*BX+(ghost?((Electric?dy:dy-1)*BX+(Electric?dx:dx-1)):(BY*BX+group*BX+(Electric?dx:dx-1)));
  return frame[ColumnZOffset+familybase+index];
 }
 const int index=(group*BY+(Electric?dy:dy-1))*BX+(Electric?dx:dx-1);
 return frame[ColumnZOffset+familybase+index];
}
'''+definitions[end:]
    anchor = 'template<int Depth,int N>\n__global__ __launch_bounds__(1024,1) void AsymmetricNoIOFixedCube('
    if original.count(anchor) != 1:
        raise ValueError('Frozen NoIO entrypoint anchor changed')
    text = original.replace(anchor, definitions + body + '\n\n' + anchor)
    # Only the certified NoIO kernels use the new thread/storage schedule.
    for signature in (() if scope == 'fast' else ('void AsymmetricNoIOFixedCube(', 'void AsymmetricNoIOUniform1024(')):
        start, end, entry = block(text, signature)
        entry = entry.replace('FixedCubeCachedBody<Depth,N,', 'TraceColumnBody<Depth,N,')
        entry = entry.replace('FixedCubeCachedBody<Depth,N>', 'TraceColumnBody<Depth,N>')
        if generic_geometry:
            entry = entry.replace('TraceColumnBody<Depth,N', 'TraceColumnBody<Depth,0')
        for axes in ('1', '2', '4', '-1'):
            entry = entry.replace(f'UniformBody<Depth,1,false,true,1,true,{axes},true>',
                                  f'TraceColumnBody<Depth,0,{axes}>')
        before = text[:start]
        bounds = before.rfind('__launch_bounds__(1024,1)')
        if bounds < 0 or before[bounds:].count('__launch_bounds__') != 1:
            raise ValueError('NoIO launch bound anchor changed')
        before = before[:bounds] + before[bounds:].replace('__launch_bounds__(1024,1)', f'__launch_bounds__({512*z_groups},1)', 1)
        text = before + entry + text[end:]
    if scope != 'fast':
        start, end, launch = block(text, 'inline cudaError_t LaunchNoIO(')
        launch = launch.replace('FixedCubeSharedBytes', 'ColumnSharedBytes').replace('ProfileCachedSharedBytes', 'ColumnSharedBytes')
        launch = launch.replace('<<<uniform_count,1024,', f'<<<uniform_count,{512*z_groups},')
        text = text[:start] + launch + text[end:]
    if scope in ('fast', 'all'):
        start, end, entry = block(text, 'void Asymmetric(')
        anchor = 'if constexpr(Layout==1&&Policy==1&&IsotropicH)'
        if entry.count(anchor) != 1:
            raise ValueError('Exact fast-policy dispatch anchor required')
        entry = entry.replace(anchor, '''if constexpr(Fast&&Layout==1&&IsotropicH){
  if(p.flags[tiles[blockIdx.x]]&64)TraceColumnBody<Depth,0,0,true,true>(p,tiles);
  else TraceColumnBody<Depth,0,0,true,false>(p,tiles);
 }else '''+anchor)
        text = text[:start]+entry+text[end:]
        start, end, launch = block(text, 'inline cudaError_t Launch(')
        launch = launch.replace(' auto status=cudaFuncSetAttribute(',
                                ' constexpr size_t ResearchSharedBytes=Fast?ColumnBaseBytes:SharedBytes;\n auto status=cudaFuncSetAttribute(')
        launch = launch.replace('cudaFuncAttributeMaxDynamicSharedMemorySize,SharedBytes',
                                'cudaFuncAttributeMaxDynamicSharedMemorySize,ResearchSharedBytes')
        launch = launch.replace('<<<count,(Fast?1024:512),SharedBytes,',
                                f'<<<count,(Fast?{512*z_groups}:512),ResearchSharedBytes,')
        text = text[:start]+launch+text[end:]
    shutil.copytree(source, output)
    (output / 'src/asymmetric_native.cuh').write_text(text)
    proof = dict(schema='beamz.trace-column-lowering.v1', schedule=schedule,
                 routing_proof=prove_routes(ThreadMap(threads=512*z_groups,z_groups=z_groups),steps=16), parent_source=str(source),
                 parent_consumer_sha256=PARENT_SHA, consumer_sha256=sha(text.encode()),
                 producer_sha256=sha(Path(__file__).read_bytes()),
                 source_observer_fallback_paths_unchanged=True, expected_kernels=66,
                 column_shared_bytes=(field_banks*(3255 if compact else 4096)
                    +( (2 if separate_families else 1)*1457 if deduplicate else field_banks*(z_groups*15*31 if z_groups>1 else 0)))*4+64+12+(864*4 if cache_profiles else 0),
                 deduplicated_frontiers=deduplicate, cache_profiles=cache_profiles,
                 fused_publication=fused_publication,
                 publication_order_proof=('H owned stores follow their current value; H ghost stores follow imports; E stores follow B1 and their private update; distinct H/E banks; original B1/B2 retained' if fused_publication else None),
                 generic_geometry=generic_geometry,deduplication_proof=prove_deduplicated_z() if deduplicate else None,
                 z_groups=z_groups, scope=scope,z_provider_proof=prove_z_providers() if z_groups>1 else None,
                 compact_shared_providers=compact,
                 compact_provider_proof=prove_compact_providers() if compact else None,
                 numerical_cell_header_sha256=sha((source/'src/asymmetric_native_cell_generated.cuh').read_bytes())
                     if (source/'src/asymmetric_native_cell_generated.cuh').exists() else None,
                 GPU_executed=False)
    (output.parent / 'column-schedule-proof.json').write_text(json.dumps(proof, indent=2) + '\n')
    return proof


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=PARENT)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--aliased-families', action='store_true')
    p.add_argument('--compact-providers', action='store_true')
    p.add_argument('--z-groups', type=int, choices=(1,2), default=1)
    p.add_argument('--scope', choices=('noio','fast','all'), default='noio')
    p.add_argument('--deduplicate-frontiers', action='store_true')
    p.add_argument('--uncached-profiles', action='store_true')
    p.add_argument('--generic-geometry', action='store_true')
    p.add_argument('--fused-publication', action='store_true')
    a = p.parse_args()
    print(json.dumps(stage(a.source, a.output, separate_families=not a.aliased_families,
        compact=a.compact_providers,z_groups=a.z_groups,scope=a.scope,
        deduplicate=a.deduplicate_frontiers,cache_profiles=not a.uncached_profiles,
        generic_geometry=a.generic_geometry,fused_publication=a.fused_publication)))
