"""Generate a complete temporal tile loop from a lane/provider topology.

The numerical cell implementation and native environment contract are reused;
the cache loop, private ownership, lane exchanges and shared layout are emitted
afresh. This is a new lowering, not a rewrite of the old CUDA compute loop.
"""
from __future__ import annotations

import hashlib
import json

from scripts.cuda_trace_tile_topology import TileTopology,emit_provider_helpers,verify_topology,prove_shared_e_effects
from scripts.cuda_trace_tile_bindings import inline_point_section


def phase_code(electric,*,inline_cells=False,cache=(16,16,16),shared_e=False):
    phase=int(electric);family='h' if electric else 'e';output='e' if electric else 'h'
    sign=-1 if electric else 1;source_electric=not electric
    own=('if(field>=3)return e[field-3][j];const int c=field;'
         if electric else 'if(field<3)return h[field][j];const int c=field-3;')
    args='env,read,n,z,y,x'
    result=f'''
  #pragma unroll
  for(int j=0;j<Slots;++j){{
   const int zs=zbase+j,z=oz+low+((zs-hz)&(CZ-1));
   const float wx1=__shfl_sync(0xffffffffu,{family}[1][j],(lane&~(CX-1))|((sx+({sign}))&(CX-1)));
   const float wx2=__shfl_sync(0xffffffffu,{family}[2][j],(lane&~(CX-1))|((sx+({sign}))&(CX-1)));
   float wy0=0.f,wy2=0.f;
   if constexpr(CX<32){{
    wy0=__shfl_sync(0xffffffffu,{family}[0][j],(lane+({sign})*CX)&31);
    wy2=__shfl_sync(0xffffffffu,{family}[2][j],(lane+({sign})*CX)&31);
   }}
   if(zs!=hz&&sy!=hy&&sx!=hx){{
    const auto read=[&] __device__(int field,int,int dz,int ddy,int ddx){{
     {own}
     if(dz){{
      if(j+dz>=0&&j+dz<Slots)return {family}[c][j+dz];
      return Storage::template Read<{str(source_electric).lower()}>(frame,c,zs+dz,sy,sx);
     }}
     if(ddy){{
      if constexpr(CX<32)if(sy/Rows==((sy+ddy)&(CY-1))/Rows)return c==0?wy0:wy2;
      return Storage::template Read<{str(source_electric).lower()}>(frame,c,zs,sy+ddy,sx);
     }}
     if(ddx)return c==1?wx1:wx2;
     return {family}[c][j];
    }};
    #pragma unroll
    for(int c=0;c<3;++c){{float value=0.f;if(env.Valid(c+{3*phase},z,y,x)){{
     if(c==0)value=beamz::cuda::trace_native::AdvanceNativeCell<{phase},0,false>({args});
     else if(c==1)value=beamz::cuda::trace_native::AdvanceNativeCell<{phase},1,false>({args});
     else value=beamz::cuda::trace_native::AdvanceNativeCell<{phase},2,false>({args});
    }}{output}[c][j]=value;}}
   }}
  }}
'''
    if inline_cells:return inline_point_section(result,electric,cache,shared_e=shared_e)
    return result


def generate(cache=(16,16,16),bulk_groups=4,base_groups=2,*,shared_e_sources=False,source_groups=None,generic_source_groups=None,inline_cells=False):
    source_groups=base_groups if source_groups is None else source_groups
    generic_source_groups=source_groups if generic_source_groups is None else generic_source_groups
    for groups in (bulk_groups,base_groups,source_groups,generic_source_groups):
        if groups<2:raise ValueError('First generated binding requires at least two Z groups')
        verify_topology(TileTopology(cache,groups))
    source=['// Generated from cuda_trace_tile_topology.py; do not edit.',
            '// Included inside beamz::cuda::moving_native after the native callbacks.',
            '#pragma once',emit_provider_helpers(cache),
            f'constexpr int TraceBulkGroups={bulk_groups},TraceBaseGroups={base_groups},TraceSourceGroups={source_groups},TraceGenericSourceGroups={generic_source_groups};',
            'constexpr int TraceBulkThreads=CY*CX*TraceBulkGroups,TraceBaseThreads=CY*CX*TraceBaseGroups,TraceSourceThreads=CY*CX*TraceSourceGroups,TraceGenericSourceThreads=CY*CX*TraceGenericSourceGroups;',
            f'constexpr int TraceSharedWords={max(verify_topology(TileTopology(cache,g))["shared_words"] for g in (bulk_groups,base_groups,source_groups,generic_source_groups))};',
            '''
template<int Depth,int Layout,bool Fast,bool CachedProfiles,int Groups,class Env>
__device__ __forceinline__ void RunTraceTile(const Parameters&p,const int32_t*tiles,Env env){
 using Storage=TraceTileStorage<Groups>;
 constexpr int Slots=CZ/Groups,Rows=32/CX;
 constexpr bool Safe=Env::SupportSafe;
 extern __shared__ float frame[];
 size_t*bases=reinterpret_cast<size_t*>(frame+TraceSharedWords);
 volatile int*window=reinterpret_cast<volatile int*>(bases+8)+1;
 if constexpr(CachedProfiles)StageIndependentCpmlProfiles(p,frame+SharedBytes/sizeof(float));
 const int tile=tiles[blockIdx.x],bz=tile/(p.nx*p.ny),by=(tile/p.nx)%p.ny,bx=tile%p.nx;
 const int oz=bz*BZ,oy=by*BY,ox=bx*BX,rank=bz+by+bx;
 const uint64_t epoch=uint64_t(*p.epoch)<<32;
 if(!threadIdx.x){const int owner=(rank&3)*p.frontier+p.slots[tile];p.owner_tags[owner]=epoch|uint32_t(tile);}
 __syncthreads();
 if constexpr(!Fast){
  if(!threadIdx.x){
   const int mz=p.phase[0].own[0].z+1,my=p.phase[0].own[0].y+1,mx=p.phase[0].own[1].x+1;
   window[0]=max(1,max(oz-mz+1,max(oy-my+1,ox-mx+1)));
   window[1]=min(Depth,min(oz+BZ-1,min(oy+BY-1,ox+BX-1)));
  }
  __syncthreads();if(window[0]>window[1])return;
 }
 if(threadIdx.x<8){const int bits=threadIdx.x,dz=bits&1,dy=(bits>>1)&1,dx=(bits>>2)&1;size_t base=size_t(-1);
  if(bz>=dz&&by>=dy&&bx>=dx){const int producer=tile-dz*p.nx*p.ny-dy*p.nx-dx;const int owner=((rank-dz-dy-dx)&3)*p.frontier+p.slots[producer];
   if(bits&&p.owner_tags[owner]!=(epoch|uint32_t(producer)))atomicOr(p.bad,1);else base=size_t(owner)*Depth*6*SurfacePoints;
  }bases[bits]=base;
 }
 const int sx=threadIdx.x&(CX-1),sy=(threadIdx.x/CX)&(CY-1),zbase=(threadIdx.x/(CY*CX))*Slots,lane=threadIdx.x&31;
 float h[3][Slots],e[3][Slots];
 if constexpr(!Fast)__syncthreads();
 const int first=Fast?1:window[0],prev=first-1;
 #pragma unroll
 for(int j=0;j<Slots;++j){const int zs=zbase+j;FieldPacket v{};
  if(first==1){const int z=oz+(zs==CZ-1?-1:zs),y=oy+(sy==CY-1?-1:sy),x=ox+(sx==CX-1?-1:sx);
   v={Initial(p,0,z,y,x),Initial(p,1,z,y,x),Initial(p,2,z,y,x),Initial(p,3,z,y,x),Initial(p,4,z,y,x),Initial(p,5,z,y,x)};
  }else{const int hz=(-prev-1)&(CZ-1),hy=(-prev-1)&(CY-1),hx=(-prev-1)&(CX-1);
   if(zs==hz||sy==hy||sx==hx)v=ImportPacket<Depth,false,(Layout==1)>(p,bases,prev,oz,oy,ox,(zs-hz)&(CZ-1),(sy-hy)&(CY-1),(sx-hx)&(CX-1));
  }
  h[0][j]=v.hx;h[1][j]=v.hy;h[2][j]=v.hz;e[0][j]=v.ex;e[1][j]=v.ey;e[2][j]=v.ez;
  Storage::template Store<true>(frame,zs,sy,sx,v.ex,v.ey,v.ez);
 }
 __syncthreads();
 #pragma unroll 1
 for(int n=first;n<=(Fast?Depth:window[1]);++n){
  int step;asm volatile("mov.b32 %0,%1;":"=r"(step):"r"(n));
  const int hz=(-step-1)&(CZ-1),hy=(-step-1)&(CY-1),hx=(-step-1)&(CX-1),low=-step-1;
  const int dy=(sy-hy)&(CY-1),dx=(sx-hx)&(CX-1),y=oy+low+dy,x=ox+low+dx;
''',phase_code(False,inline_cells=inline_cells,cache=cache),'''
  // The complete local H loop has retired private old-E reads before imports.
  #pragma unroll
  for(int j=0;j<Slots;++j){const int zs=zbase+j;if(zs==hz||sy==hy||sx==hx){
   const auto v=ImportPacket<Depth,Safe,(Layout==1)>(p,bases,n,oz,oy,ox,(zs-hz)&(CZ-1),dy,dx);
   h[0][j]=v.hx;h[1][j]=v.hy;h[2][j]=v.hz;e[0][j]=v.ex;e[1][j]=v.ey;e[2][j]=v.ez;
  }}
  #pragma unroll
  for(int j=0;j<Slots;++j)Storage::template Store<false>(frame,zbase+j,sy,sx,h[0][j],h[1][j],h[2][j]);
  __syncthreads(); // H visible; all old-E shared readers retired.
''',phase_code(True,inline_cells=inline_cells,cache=cache),'''
  #pragma unroll
  for(int j=0;j<Slots;++j)Storage::template Store<true>(frame,zbase+j,sy,sx,e[0][j],e[1][j],e[2][j]);
  __syncthreads(); // E visible; all current-H shared readers retired.
  #pragma unroll
  for(int j=0;j<Slots;++j){const int zs=zbase+j;if(zs!=hz&&sy!=hy&&sx!=hx){
   const int dz=(zs-hz)&(CZ-1),z=oz+low+dz,surface=Surface(dz-1,dy-1,dx-1);
   if(surface>=0&&(Layout!=1||n<Depth)){
    #pragma unroll
    for(int c=0;c<3;++c){p.trails[bases[0]+((n-1)*6+c)*SurfacePoints+surface]=h[c][j];p.trails[bases[0]+((n-1)*6+c+3)*SurfacePoints+surface]=e[c][j];}
   }
   if(n==Depth){
    #pragma unroll
    for(int c=0;c<3;++c){const auto&hf=FinalField(p,c);const auto&ef=FinalField(p,c+3);
     if(Contains(hf,z,y,x))hf.data[z*hf.plane+y*hf.row+x]=h[c][j];
     if(Contains(ef,z,y,x))ef.data[z*ef.plane+y*ef.row+x]=e[c][j];
    }
   }
  }}
 }
}
''']
    if shared_e_sources:
        # Apply a storage lowering to compiler-owned loop IR sections. Numerical
        # statements are unchanged; E's home and import lifetime are different.
        prefix=source[7].split(' float h[3][Slots],e[3][Slots];')[0]
        prefix=prefix.replace('void RunTraceTile(','void RunTraceTileSharedE(')
        prefix=prefix.replace('frame+TraceSharedWords','frame+3*PAD+Storage::Words/2')
        prefix=prefix.replace('frame+SharedBytes/sizeof(float)','frame+3*PAD+Storage::Words/2+19')
        initial=source[7].split(' float h[3][Slots],e[3][Slots];')[1]
        initial=initial.replace('e[0][j]=v.ex;e[1][j]=v.ey;e[2][j]=v.ez;',
            'frame[Storage::Words/2+Slot(zs,sy,sx)]=v.ex;frame[Storage::Words/2+PAD+Slot(zs,sy,sx)]=v.ey;frame[Storage::Words/2+2*PAD+Slot(zs,sy,sx)]=v.ez;')
        initial=initial.replace('  Storage::template Store<true>(frame,zs,sy,sx,v.ex,v.ey,v.ez);','')
        hphase='''
  #pragma unroll
  for(int j=0;j<Slots;++j){const int zs=zbase+j,z=oz+low+((zs-hz)&(CZ-1));
   if(zs!=hz&&sy!=hy&&sx!=hx){
    const auto read=[&] __device__(int field,int,int dz,int ddy,int ddx){
     if(field<3)return h[field][j];
     return frame[Storage::Words/2+(field-3)*PAD+Slot(zs+dz,sy+ddy,sx+ddx)];
    };
    #pragma unroll
    for(int c=0;c<3;++c){float value=0.f;if(env.Valid(c,z,y,x)){
     if(c==0)value=beamz::cuda::trace_native::AdvanceNativeCell<0,0,false>(env,read,n,z,y,x);
     else if(c==1)value=beamz::cuda::trace_native::AdvanceNativeCell<0,1,false>(env,read,n,z,y,x);
     else value=beamz::cuda::trace_native::AdvanceNativeCell<0,2,false>(env,read,n,z,y,x);
    }h[c][j]=value;}
   }
  }
'''
        imports='''
  #pragma unroll
  for(int j=0;j<Slots;++j){const int zs=zbase+j;if(zs==hz||sy==hy||sx==hx){
   const auto v=ImportPacket<Depth,Safe,(Layout==1)>(p,bases,n,oz,oy,ox,(zs-hz)&(CZ-1),dy,dx);
   h[0][j]=v.hx;h[1][j]=v.hy;h[2][j]=v.hz;
  }}
  #pragma unroll
  for(int j=0;j<Slots;++j)Storage::template Store<false>(frame,zbase+j,sy,sx,h[0][j],h[1][j],h[2][j]);
  __syncthreads(); // Publish H; retire every shared old-E neighbour reader.
  #pragma unroll
  for(int j=0;j<Slots;++j){const int zs=zbase+j;if(zs==hz||sy==hy||sx==hx){
   const auto v=ImportPacket<Depth,Safe,(Layout==1)>(p,bases,n,oz,oy,ox,(zs-hz)&(CZ-1),dy,dx);
   frame[Storage::Words/2+Slot(zs,sy,sx)]=v.ex;
   frame[Storage::Words/2+PAD+Slot(zs,sy,sx)]=v.ey;
   frame[Storage::Words/2+2*PAD+Slot(zs,sy,sx)]=v.ez;
  }}
'''
        if inline_cells:
            hphase=inline_point_section(hphase,False,cache,shared_e=True)
            ephase=phase_code(True,inline_cells=True,cache=cache,shared_e=True)
        else:
            ephase=phase_code(True).replace('if(field>=3)return e[field-3][j];',
                'if(field>=3)return frame[Storage::Words/2+(field-3)*PAD+Slot(zs,sy,sx)];')
            ephase=ephase.replace('e[c][j]=value;',
                'frame[Storage::Words/2+c*PAD+Slot(zs,sy,sx)]=value;')
        terminal=source[11]
        begin=terminal.index('  __syncthreads();')
        terminal=terminal[begin:].replace('e[c][j]',
            'frame[Storage::Words/2+c*PAD+Slot(zs,sy,sx)]')
        source += [prefix,' float h[3][Slots];',initial,hphase,imports,ephase,terminal]
    result='\n'.join(source)+'\n'
    return result,dict(schema='beamz.generated-temporal-tile.v1',cache=cache,bulk_groups=bulk_groups,source_groups=source_groups,generic_source_groups=generic_source_groups,
                       base_groups=base_groups,bulk=verify_topology(TileTopology(cache,bulk_groups)),
                       base=verify_topology(TileTopology(cache,base_groups)),barriers_per_step=2,
                       source=verify_topology(TileTopology(cache,source_groups)),
                       generic_source=verify_topology(TileTopology(cache,generic_source_groups)),
                       shared_e_sources=shared_e_sources,
                       inline_cell_reads=inline_cells,
                       source_storage_effects=prove_shared_e_effects(cache) if shared_e_sources else None,
                       source_private_field_floats_per_thread=(3 if shared_e_sources else 6)*(cache[0]//source_groups),
                       effects='Existing point-local CPML/source/observation callbacks; retained rank trails/final stores',
                       generated_sha256=hashlib.sha256(result.encode()).hexdigest(),GPU_executed=False)
