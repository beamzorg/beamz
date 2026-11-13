"""Emit a field-retimed diamond CUDA loop, initially using full shared fields.

The initial storage is deliberately simple so transformed ownership, lattice
parity, packed CPML and source/observation chronology can be validated together.
It is not a performance claim or a lowering of the published DiamondCandy code.
"""
from __future__ import annotations
import hashlib

from scripts.cuda_trace_diamond_geometry import DiamondGeometry,DiamondReadBinding
from scripts.generate_cuda_native_cell import generate_native_cell


def generate(geometry=DiamondGeometry()):
    geometry.validate();blocks=generate_native_cell(read_binding=DiamondReadBinding(geometry))['point_blocks']
    source=['''// Generated retimed diamond ownership; full shared-field reference lowering.
#pragma once
constexpr int DiamondHalfX=CX/2,DiamondPoints=PAD/2;
constexpr int TraceBulkThreads=1024,TraceBaseThreads=512,TraceSourceThreads=1024,TraceGenericSourceThreads=512;
constexpr int TraceBulkGroups=2,TraceBaseGroups=1,TraceSourceGroups=2,TraceGenericSourceGroups=1;
template<int Field>__device__ __forceinline__ int DiamondRetiming(){return Field==1||Field==3||Field==5?1:0;}
struct DiamondSite {int z,y,x,dz,dp,dm,m;bool owned;};
template<int Field>__device__ __forceinline__ DiamondSite DiamondPosition(int q,int version,int oz,int op,int om,int bias){
 const int qz=q/(CY*DiamondHalfX),qp=(q/DiamondHalfX)&(CY-1),qm=q&(DiamondHalfX-1);
 const int r=DiamondRetiming<Field>(),m=2*qm+((op+om-bias+qp-r)&1),low=-version-1;
 const int dz=(qz-low)&(CZ-1),dp=(qp-low)&(CY-1),dm=(m-low)&(CX-1);
 const int plus=op+low+dp,minus=om+low+dm-r-bias;
 return {oz+low+dz,(plus-minus)/2,(plus+minus)/2,dz,dp,dm,m,dz&&dp&&dm};
}
template<int Depth,int Layout,int Field>__device__ __forceinline__ float DiamondImport(
 const Parameters&p,const size_t*bases,int version,const DiamondSite&s){
 if(!(NativeSupportMask(p,s.z,s.y,s.x)&(1u<<Field)))return 0.f;
 if constexpr(Layout==1)if(version==Depth){
  if constexpr(Field>=3)return 0.f;
  else{const auto&f=FinalField(p,Field);return __ldcg(f.data+s.z*f.plane+s.y*f.row+s.x);}
 }
 const int neighbor=(s.dz==0)|((s.dp==0)<<1)|((s.dm==0)<<2);
 const int surface=Surface(s.dz?s.dz-1:BZ-1,s.dp?s.dp-1:BY-1,s.dm?s.dm-1:BX-1);
 const size_t base=bases[neighbor];
 if(!neighbor||surface<0||base==size_t(-1)){atomicOr(p.bad,2);return __int_as_float(0x7fffffff);}
 return __ldcg(p.trails+base+((version-1)*6+Field)*SurfacePoints+surface);
}
template<int Depth,int Layout,bool Fast,bool CachedProfiles,int Groups,class Env>
__device__ __forceinline__ void RunTraceTile(const Parameters&p,const int32_t*tiles,Env env){
 extern __shared__ float frame[];
 size_t*bases=reinterpret_cast<size_t*>(frame+6*DiamondPoints);
 volatile int*window=reinterpret_cast<volatile int*>(bases+8);
 const int tile=tiles[blockIdx.x],bz=tile/(p.nx*p.ny),bp=(tile/p.nx)%p.ny,bm=tile%p.nx;
 const int oz=bz*BZ,op=bp*BY,om=bm*BX,rank=bz+bp+bm,bias=p.physical_n[1];
 const uint64_t epoch=uint64_t(*p.epoch)<<32;
 if(!threadIdx.x){const int owner=(rank&3)*p.frontier+p.slots[tile];p.owner_tags[owner]=epoch|uint32_t(tile);}
 __syncthreads();
 if(!threadIdx.x){
  const int sum=op+om-bias;
  window[0]=max(1,max(oz-p.physical_n[0],(sum-1-2*p.physical_n[2])/2));
  window[1]=min(Depth,min(oz+BZ-1,(sum+BY+BX-2)/2));
  if(op-om+BY-1+bias+1<0||op-om+1-BX+bias>2*p.physical_n[1])window[1]=0;
 }
 __syncthreads();if(window[0]>window[1])return;
 if(threadIdx.x<8){const int bits=threadIdx.x,dz=bits&1,dp=(bits>>1)&1,dm=(bits>>2)&1;size_t base=size_t(-1);
  if(bz>=dz&&bp>=dp&&bm>=dm){const int producer=tile-dz*p.nx*p.ny-dp*p.nx-dm;
   const int owner=((rank-dz-dp-dm)&3)*p.frontier+p.slots[producer];
   if(bits&&p.owner_tags[owner]!=(epoch|uint32_t(producer)))atomicOr(p.bad,1);else base=size_t(owner)*Depth*6*SurfacePoints;
  }bases[bits]=base;
 }
 __syncthreads();
 const int first=window[0],last=window[1];
 for(int q=threadIdx.x;q<DiamondPoints;q+=blockDim.x){
''']
    for field in range(6):
        source.append(f'''  {{const auto s=DiamondPosition<{field}>(q,first-1,oz,op,om,bias);float value=0.f;
   if(first==1)value=Initial(p,{field},s.z,s.y,s.x);
   else if(!s.owned)value=DiamondImport<Depth,Layout,{field}>(p,bases,first-1,s);
   frame[{field}*DiamondPoints+q]=value;
  }}''')
    source+=[' }',' __syncthreads();',' #pragma unroll 1',' for(int n=first;n<=last;++n){']
    for phase in (0,1):
        # Ghost writes are disjoint from same-family old-self readers. E ghost
        # writes occur only after B1 has retired all old-E neighbour readers.
        source+=['  for(int q=threadIdx.x;q<DiamondPoints;q+=blockDim.x){']
        for c in range(3):
            field=3*phase+c
            source.append(f'''   {{const auto s=DiamondPosition<{field}>(q,n,oz,op,om,bias);
    if(!s.owned)frame[{field}*DiamondPoints+q]=DiamondImport<Depth,Layout,{field}>(p,bases,n,s);
   }}''')
        source+=['  }','  for(int q=threadIdx.x;q<DiamondPoints;q+=blockDim.x){',
                 '   const int qz=q/(CY*DiamondHalfX),qp=(q/DiamondHalfX)&(CY-1);']
        for c in range(3):
            field=3*phase+c;name=('E' if phase else 'H')+'xyz'[c]
            source.append(f'''   {{const auto s=DiamondPosition<{field}>(q,n,oz,op,om,bias);
    if(s.owned){{const int z=s.z,y=s.y,x=s.x,m=s.m;float value=0.f;
     if(env.Valid({field},z,y,x)){{
{blocks[name]}
      value=result;
     }}frame[{field}*DiamondPoints+q]=value;
    }}
   }}''')
        source+=['  }','  __syncthreads(); // Opposite family visible; old-family neighbour readers retired.']
    source+=['  for(int q=threadIdx.x;q<DiamondPoints;q+=blockDim.x){']
    for field in range(6):
        source.append(f'''   {{const auto s=DiamondPosition<{field}>(q,n,oz,op,om,bias);if(s.owned){{
    const float value=frame[{field}*DiamondPoints+q];
    const int surface=Surface(s.dz-1,s.dp-1,s.dm-1);
    if(surface>=0&&(Layout!=1||n<Depth))p.trails[bases[0]+((n-1)*6+{field})*SurfacePoints+surface]=value;
    if(n==Depth){{const auto&f=FinalField(p,{field});if(Contains(f,s.z,s.y,s.x))f.data[s.z*f.plane+s.y*f.row+s.x]=value;}}
   }}}}''')
    source+=['  }',' }','}','']
    text='\n'.join(source)
    if geometry.orientation=='yz':
        begin=text.index('constexpr int DiamondHalfX=')
        end=text.index('template<int Depth,int Layout,int Field>',begin)
        retimed=[str(i) for i,f in enumerate(('Hx','Hy','Hz','Ex','Ey','Ez')) if geometry.retiming(f)]
        expr='||'.join('Field=='+i for i in retimed)
        text=text[:begin]+f'''constexpr int DiamondHalfY=CY/2,DiamondPoints=PAD/2;
constexpr int TraceBulkThreads=1024,TraceBaseThreads=512,TraceSourceThreads=1024,TraceGenericSourceThreads=512;
constexpr int TraceBulkGroups=2,TraceBaseGroups=1,TraceSourceGroups=2,TraceGenericSourceGroups=1;
template<int Field>__device__ __forceinline__ int DiamondRetiming(){{return {expr}?1:0;}}
struct DiamondSite {{int z,y,x,dz,dp,dm,m;bool owned;}};
template<int Field>__device__ __forceinline__ DiamondSite DiamondPosition(int q,int version,int oz,int op,int om,int bias){{
 const int q0=q/(DiamondHalfY*CX),qm=(q/CX)&(DiamondHalfY-1),q2=q&(CX-1);
 const int r=DiamondRetiming<Field>(),m=2*qm+((oz+op-bias+q0-r)&1),low=-version-1;
 const int d0=(q0-low)&(CZ-1),d1=(m-low)&(CY-1),d2=(q2-low)&(CX-1);
 const int plus=oz+low+d0,minus=op+low+d1-r-bias;
 return {{(plus+minus)/2,(plus-minus)/2,om+low+d2,d0,d1,d2,m,d0&&d1&&d2}};
}}
'''+text[end:]
        old='''  const int sum=op+om-bias;
  window[0]=max(1,max(oz-p.physical_n[0],(sum-1-2*p.physical_n[2])/2));
  window[1]=min(Depth,min(oz+BZ-1,(sum+BY+BX-2)/2));
  if(op-om+BY-1+bias+1<0||op-om+1-BX+bias>2*p.physical_n[1])window[1]=0;'''
        new='''  const int sum=oz+op-bias;
  window[0]=max(1,max(om-p.physical_n[2],(sum-1-2*p.physical_n[0])/2));
  window[1]=min(Depth,min(om+BX-1,(sum+BZ+BY-2)/2));
  if(oz-op+BZ-1+bias+1<0||oz-op+1-BY+bias>2*p.physical_n[1])window[1]=0;'''
        if text.count(old)!=1:raise ValueError('Diamond window binding changed')
        text=text.replace(old,new)
        text=text.replace('const int qz=q/(CY*DiamondHalfX),qp=(q/DiamondHalfX)&(CY-1);',
                          'const int q0=q/(DiamondHalfY*CX),q2=q&(CX-1);')
        text=text.replace('x=s.x,m=s.m;', 'x=s.x,q1=s.m;')

    if geometry.storage_gauge=='physical':
        if geometry.orientation!='yz':raise ValueError('Physical gauge needs YZ lowering')
        old=''' const int r=DiamondRetiming<Field>(),m=2*qm+((oz+op-bias+q0-r)&1),low=-version-1;
 const int d0=(q0-low)&(CZ-1),d1=(m-low)&(CY-1),d2=(q2-low)&(CX-1);'''
        new=''' const int r=DiamondRetiming<Field>(),m=2*qm+((oz+op-bias+q0)&1),low=-version-1;
 const int d0=(q0-low)&(CZ-1),d1=(m+r-low)&(CY-1),d2=(q2-low)&(CX-1);'''
        if text.count(old)!=1:raise ValueError('Physical storage decode binding changed')
        text=text.replace(old,new)
        text=text.replace('const int q0=q/(DiamondHalfY*CX),q2=q&(CX-1);',
                          'const int q0=q/(DiamondHalfY*CX),q2=q&(CX-1);const auto cell=DiamondPosition<0>(q,n,oz,op,om,bias);')
        text=text.replace('z=s.z,y=s.y,x=s.x,q1=s.m;', 'z=cell.z,y=cell.y,x=cell.x,q1=cell.m;')
    return text,dict(schema='beamz.retimed-diamond-cuda.v1',cache=geometry.cache,
                     orientation=geometry.orientation,storage_gauge=geometry.storage_gauge,field_points=geometry.field_points,shared_bytes=24*geometry.field_points+76,
                     source_sha256=hashlib.sha256(text.encode()).hexdigest(),retiming=geometry.plan['tensor_retiming'],
                     source_barriers_per_step=2,storage='Six full shared fields; no cached CPML profiles',
                     limitations=['Dense transformed owner grid and uncompressed parity-bearing trails',
                                  'Conservative all-general policy classification; uniform-E certification disabled',
                                  'Initial/final field I/O not yet separately coalesced'],GPU_executed=False)
