"""Factor the traced Yee cell's ownership, imports and publications by address.

Fields share physical coordinates in the physical storage gauge. Retiming
still governs per-field ownership and ghost locations. A single phase event
imports ghosts, computes owned components, then publishes shared providers.
This avoids separately expanding six scalar address/control programs per site.
"""
from __future__ import annotations
import hashlib

from scripts.cuda_space_time_ir import FIELDS,IllegalSchedule
from scripts.cuda_trace_diamond_geometry import DiamondGeometry
from scripts.cuda_trace_diamond_storage import DiamondWarpTopology
from scripts.cuda_trace_diamond_private_codegen import helpers,DynamicPrivateReadBinding
from scripts.generate_cuda_native_cell import generate_native_cell
from scripts.cuda_trace_diamond_parity_storage import generate_helpers,store_lines


def target(t,field,slot):
    if FIELDS[field]==t.shared_field:return 'frame[DiamondSharedFieldOffset+q]'
    return f'{"e" if field>=3 else "h"}[{field%3}][{slot}]'


def generate(topology=DiamondWarpTopology(4,2,DiamondGeometry((16,16,32),'yz','physical')),*,region_clock=True):
    if topology.geometry.storage_gauge!='physical':raise IllegalSchedule('Factorized cell requires common physical storage coordinates')
    proof=topology.prove();groups={r:tuple(i for i,f in enumerate(FIELDS) if topology.geometry.retiming(f)==r) for r in (0,1)}
    source=[generate_helpers(topology) if topology.parity_providers else helpers(topology),'''template<int R>__device__ __forceinline__ DiamondSite DiamondCellSite(const DiamondSite&cell){
 if constexpr(R==0)return cell;
 else {const bool wrap=cell.dp==CY-1;return {cell.z-(wrap?CY/2:0),cell.y+(wrap?CY/2:0),cell.x,
   cell.dz,(cell.dp+1)&(CY-1),cell.dm,cell.m,bool(cell.dz&&((cell.dp+1)&(CY-1))&&cell.dm)};}
}
''']
    if region_clock:
        source.append('''__device__ __forceinline__ DiamondSite DiamondCellPosition(int q,int version,int oz,int op,int om,int bias){
 int resident_version;
#ifdef __CUDA_ARCH__
 asm volatile("mov.b32 %0,%1;":"=r"(resident_version):"r"(version));
#else
 resident_version=version;
#endif
 return DiamondPosition<0>(q,resident_version,oz,op,om,bias);
}
''')
    else:source.append('''__device__ __forceinline__ DiamondSite DiamondCellPosition(int q,int version,int oz,int op,int om,int bias){return DiamondPosition<0>(q,version,oz,op,om,bias);}
''')
    source.append('''// Begin GPU memory effects.
struct DiamondCellPacket{float v0,v1,v2,v3,v4,v5;};
template<int Depth,int Layout,int Mask,bool SupportSafe=false>__device__ __forceinline__ DiamondCellPacket DiamondImportCell(
 const Parameters&p,const size_t*bases,int version,const DiamondSite&s){
 DiamondCellPacket v{};const unsigned support=SupportSafe?unsigned(Mask):(NativeSupportMask(p,s.z,s.y,s.x)&Mask);
 if(!support)return v;
 if constexpr(Layout==1){if(version==Depth){
''')
    for field in range(3):source.append(f'''  if constexpr(Mask&{1<<field})if(support&{1<<field}){{const auto&f=FinalField(p,{field});v.v{field}=__ldcg(f.data+s.z*f.plane+s.y*f.row+s.x);}}''')
    source+=['  return v;',' }}', ''' const int neighbor=(s.dz==0)|((s.dp==0)<<1)|((s.dm==0)<<2);
 const int surface=Surface(s.dz?s.dz-1:BZ-1,s.dp?s.dp-1:BY-1,s.dm?s.dm-1:BX-1);
 const size_t base=bases[neighbor];
 if(!neighbor||surface<0||base==size_t(-1)){atomicOr(p.bad,2);
''']
    for field in range(6):source.append(f'  if constexpr(Mask&{1<<field})if(support&{1<<field})v.v{field}=__int_as_float(0x7fffffff);')
    source+=['  return v;',' }',' const float*address=p.trails+base+(version-1)*6*SurfacePoints+surface;']
    for field in range(6):source.append(f' if constexpr(Mask&{1<<field})if(support&{1<<field})v.v{field}=__ldcg(address+{field}*SurfacePoints);')
    source+=[' return v;','}', '''template<int Depth,int Layout,class Env>
__device__ __forceinline__ void RunDiamondPrivate(const Parameters&p,const int32_t*tiles,Env env){
 extern __shared__ float frame[];
 size_t*bases=reinterpret_cast<size_t*>(frame+DiamondProviderWords);
 volatile int*window=reinterpret_cast<volatile int*>(bases+8);
 const int tile=tiles[blockIdx.x],bz=tile/(p.nx*p.ny),bp=(tile/p.nx)%p.ny,bm=tile%p.nx;
 const int oz=bz*BZ,op=bp*BY,om=bm*BX,rank=bz+bp+bm,bias=p.physical_n[1],lane=threadIdx.x&31;
 const int parity=(oz+op-bias)&1,warp=threadIdx.x>>5;
 const uint64_t epoch=uint64_t(*p.epoch)<<32;
 if(!threadIdx.x){const int owner=(rank&3)*p.frontier+p.slots[tile];p.owner_tags[owner]=epoch|uint32_t(tile);}
 __syncthreads();
 if(!threadIdx.x){if constexpr(Env::SupportSafe){window[0]=1;window[1]=Depth;}else{const int sum=oz+op-bias;
  window[0]=max(1,max(om-p.physical_n[2],(sum-1-2*p.physical_n[0])/2));
  window[1]=min(Depth,min(om+BX-1,(sum+BZ+BY-2)/2));
  if(oz-op+BZ-1+bias+1<0||oz-op+1-BY+bias>2*p.physical_n[1])window[1]=0;
 }}
 __syncthreads();if(window[0]>window[1])return;
 if(threadIdx.x<8){const int bits=threadIdx.x,dz=bits&1,dp=(bits>>1)&1,dm=(bits>>2)&1;size_t base=size_t(-1);
  if(bz>=dz&&bp>=dp&&bm>=dm){const int producer=tile-dz*p.nx*p.ny-dp*p.nx-dm;
   const int owner=((rank-dz-dp-dm)&3)*p.frontier+p.slots[producer];
   if(bits&&p.owner_tags[owner]!=(epoch|uint32_t(producer)))atomicOr(p.bad,1);else base=size_t(owner)*Depth*6*SurfacePoints;
  }bases[bits]=base;
 }
 __syncthreads();const int first=Env::SupportSafe?1:window[0],last=Env::SupportSafe?Depth:window[1];
 float h[3][DiamondPrivateSlots],e[3][DiamondPrivateSlots];
''']
    for slot in range(topology.slots):
        source.append(f' {{const int q=DiamondPrivateQ<{slot}>(threadIdx.x);const auto cell=DiamondCellPosition(q,first-1,oz,op,om,bias);')
        for r,fields in groups.items():
            mask=sum(1<<i for i in fields)
            source.append(f'  {{const auto s=DiamondCellSite<{r}>(cell);DiamondCellPacket values'+'{};')
            source.append('   if(first==1){')
            for field in fields:source.append(f'    if constexpr(Env::SupportSafe){{const auto&f=InitialField(p,{field});values.v{field}=__ldcg(f.data+s.z*f.plane+s.y*f.row+s.x);}}else values.v{field}=Initial(p,{field},s.z,s.y,s.x);')
            source+=['   }',f'   else if(!s.owned)values=DiamondImportCell<Depth,Layout,{mask},Env::SupportSafe>(p,bases,first-1,s);']
            for field in fields:
                family='e' if field>=3 else 'h';source.append(f'   {target(topology,field,slot)}=values.v{field};')
            source.append('  }')
        for field,name in enumerate(FIELDS):
            family='e' if field>=3 else 'h'
            source+=store_lines(topology,field,slot,f'{family}[{field%3}][{slot}]','  ')
        source.append(' }')
    source+=[' __syncthreads();',' #pragma unroll 1',' for(int n=first;n<=last;++n){']
    for phase in (0,1):
        family='e' if phase else 'h'
        for slot in range(topology.slots):
            source.append(f'''  {{const int q=DiamondPrivateQ<{slot}>(threadIdx.x),q0=q/(DiamondHalfY*CX),q2=q&(CX-1);
   const auto cell=DiamondCellPosition(q,n,oz,op,om,bias);const int q1=cell.m,z=cell.z,y=cell.y,x=cell.x;''')
            for r in (0,1):
                fields=tuple(f for f in groups[r] if f//3==phase);mask=sum(1<<f for f in fields)
                source.append(f'''   {{const auto s=DiamondCellSite<{r}>(cell);if(!s.owned){{const auto values=DiamondImportCell<Depth,Layout,{mask},Env::SupportSafe>(p,bases,n,s);''')
                for field in fields:source.append(f'    {target(topology,field,slot)}=values.v{field};')
                source.append('   }}')
            binding=DynamicPrivateReadBinding(topology,slot)
            blocks=generate_native_cell(read_binding=binding)['point_blocks'];used='\n'.join(blocks[FIELDS[phase*3+c]] for c in range(3))
            for shuffle,value,dx in binding.shuffles.values():
                if shuffle in used:source.append(f'   const float {shuffle}=__shfl_sync(0xffffffffu,{value},(lane+({dx}))&31);')
            for c in range(3):
                field=phase*3+c;name=FIELDS[field];r=topology.geometry.retiming(name)
                source.append(f'''   {{const bool owned=cell.dz&&cell.dm&&((cell.dp+{r})&(CY-1));if(owned){{float value=0.f;
    if(env.Valid({field},z,y,x)){{''')
                source+=[blocks[name],'     value=result;}',f'    {target(topology,field,slot)}=value;','   }}']
            for c in range(3):
                field=phase*3+c
                source+=store_lines(topology,field,slot,f'{family}[{c}][{slot}]')
            source.append('  }')
        source.append('  __syncthreads(); // Family published; opposite old-family readers retired.')
    for slot in range(topology.slots):
        source.append(f'  {{const int q=DiamondPrivateQ<{slot}>(threadIdx.x);const auto cell=DiamondCellPosition(q,n,oz,op,om,bias);')
        for r,fields in groups.items():
            source.append(f'''   {{const auto s=DiamondCellSite<{r}>(cell);if(s.owned){{const int surface=Surface(s.dz-1,s.dp-1,s.dm-1);
    if(surface>=0&&(Layout!=1||n<Depth)){{float*address=p.trails+bases[0]+(n-1)*6*SurfacePoints+surface;''')
            for field in fields:
                family='e' if field>=3 else 'h';source.append(f'     address[{field}*SurfacePoints]={target(topology,field,slot)};')
            source.append('    }')
            source.append('    if(n==Depth){')
            for field in fields:
                family='e' if field>=3 else 'h';source.append(f'     {{const auto&f=FinalField(p,{field});if(Env::SupportSafe||Contains(f,s.z,s.y,s.x))f.data[s.z*f.plane+s.y*f.row+s.x]={target(topology,field,slot)};}}')
            source+=['    }','   }}']
        source.append('  }')
    source+=[' }','}','']
    text='\n'.join(source)
    proof.update(schema='beamz.factorized-diamond-cell.v1',storage_gauge='physical',region_clock=region_clock,
                 dynamic_parity_operands=True,factorized_cell=True,
                 support_safe_admission='CPML-free physical owned bounds with12-cell margin imply all cached cells atversions0..Depth lie within minimum physical supports',
                 per_phase='One cell address, retiming-class grouped imports, component SSA then shared provider publication',
                 retiming_groups=groups,source_barriers_per_step=2,source_sha256=hashlib.sha256(text.encode()).hexdigest(),CUDA_executed=False)
    return text,proof
