"""Lower traced diamond updates to registers, warp shuffles and shared providers."""
from __future__ import annotations
import hashlib

from scripts.cuda_space_time_ir import FIELDS,IllegalSchedule
from scripts.cuda_trace_diamond_storage import DiamondWarpTopology
from scripts.generate_cuda_native_cell import generate_native_cell


class PrivateReadBinding:
    def __init__(self,topology,slot,parity):
        self.topology=topology;self.slot=slot;self.parity=parity;self.shuffles={}

    def __call__(self,output,read):
        phase=int(output.startswith('E'));source_phase=int(read.field.startswith('E'))
        if read.version!=(0 if source_phase else phase):raise IllegalSchedule('Wrong private field version')
        if read.field==self.topology.shared_field:
            shift=[sum(a*b for a,b in zip(row,read.offset)) for row in self.topology.geometry.basis]
            a,b,c=shift
            return (f'frame[DiamondSharedFieldOffset+(((((q0+({a}))&(CZ-1))*DiamondHalfY+'
                    f'(((q1+({b}))&(CY-1))>>1))*CX)+((q2+({c}))&(CX-1)))]')
        route,slot,dx=self.topology.route(output,read,self.slot,self.parity)
        field=FIELDS.index(read.field);family='e' if source_phase else 'h';component=field%3
        value=f'{family}[{component}][{slot}]'
        if route=='register':return value
        if route=='shuffle':
            key=(read.field,slot,dx);name=f'diamond_shuffle_{field}_{slot}_{dx+1}'
            if key not in self.shuffles:self.shuffles[key]=(name,value,dx)
            return self.shuffles[key][0]
        if self.topology.parity_providers:
            from scripts.cuda_trace_diamond_parity_storage import shared_expression
            return shared_expression(self.topology,output,read,self.slot,self.parity)
        shift=[sum(a*b for a,b in zip(row,read.offset)) for row in self.topology.geometry.basis]
        if self.topology.geometry.storage_gauge=='retimed':shift[1]+=self.topology.geometry.retiming(read.field)-self.topology.geometry.retiming(output)
        a,b,c=shift
        return (f'frame[DiamondProviderIndex<{field}>(((q0+({a}))&(CZ-1)),'
                f'(((q1+({b}))&(CY-1))>>1),((q2+({c}))&(CX-1)))]')


class DynamicPrivateReadBinding:
    """Select physical lattice parity per operand without cloning the time loop."""
    def __init__(self,topology,slot):
        self.zero=PrivateReadBinding(topology,slot,0);self.one=PrivateReadBinding(topology,slot,1)

    @property
    def shuffles(self):return self.zero.shuffles|self.one.shuffles

    def __call__(self,output,read):
        zero=self.zero(output,read);one=self.one(output,read)
        return zero if zero==one else f'(parity?({one}):({zero}))'


def helpers(topology):
    slots=topology.slots;bm=topology.minus;bp=topology.plus;mb=topology.geometry.packed_shape[1]//bm
    lines=[f'constexpr int DiamondPrivateSlots={slots},DiamondPrivateThreads={topology.threads};',
           f'constexpr int DiamondProviderWords={topology.providers["words"]};',
           'constexpr size_t DiamondPrivateSharedBytes=4*DiamondProviderWords+76;',
           f'''template<int Slot>__device__ __forceinline__ int DiamondPrivateQ(int tid){{
 const int warp=tid>>5,lane=tid&31;
 const int plus=(warp/{mb})*{bp}+Slot/{bm},minus=(warp%{mb})*{bm}+Slot%{bm};
 return (plus*DiamondHalfY+minus)*CX+lane;
}}
template<int Field>__device__ __forceinline__ int DiamondProviderIndex(int plus,int minus,int x){{
 const int slot=(plus%{bp})*{bm}+minus%{bm},warp=(plus/{bp})*{mb}+minus/{bm};
 int rank=-1;
''']
    for field,name in enumerate(FIELDS):
        table=topology.providers['slots'][name]
        lines.append(f' if constexpr(Field=={field})'+'{switch(slot){'+''.join(f'case {s}:rank={i};break;' for i,s in enumerate(table))+f'}}return {topology.providers["field_offsets"][name]}+(warp*{len(table)}+rank)*32+x;'+'}')
    lines+=[' return -1;','}']
    return '\n'.join(lines)+'\n'


def generate(topology=DiamondWarpTopology(),*,dynamic_parity=False,region_clock=False):
    proof=topology.prove();source=[helpers(topology)]
    source+=['''template<int Depth,int Layout,int Parity,class Env>
__device__ __forceinline__ void RunDiamondPrivateParity(const Parameters&p,const int32_t*tiles,Env env){
 extern __shared__ float frame[];
 size_t*bases=reinterpret_cast<size_t*>(frame+DiamondProviderWords);
 volatile int*window=reinterpret_cast<volatile int*>(bases+8);
 const int tile=tiles[blockIdx.x],bz=tile/(p.nx*p.ny),bp=(tile/p.nx)%p.ny,bm=tile%p.nx;
 const int oz=bz*BZ,op=bp*BY,om=bm*BX,rank=bz+bp+bm,bias=p.physical_n[1],lane=threadIdx.x&31;
 const uint64_t epoch=uint64_t(*p.epoch)<<32;
 if(!threadIdx.x){const int owner=(rank&3)*p.frontier+p.slots[tile];p.owner_tags[owner]=epoch|uint32_t(tile);}
 __syncthreads();
 if(!threadIdx.x){
  const int sum=oz+op-bias;
  window[0]=max(1,max(om-p.physical_n[2],(sum-1-2*p.physical_n[0])/2));
  window[1]=min(Depth,min(om+BX-1,(sum+BZ+BY-2)/2));
  if(oz-op+BZ-1+bias+1<0||oz-op+1-BY+bias>2*p.physical_n[1])window[1]=0;
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
 float h[3][DiamondPrivateSlots],e[3][DiamondPrivateSlots];
''']
    for slot in range(topology.slots):
        source.append(f' {{const int q=DiamondPrivateQ<{slot}>(threadIdx.x);')
        for field,name in enumerate(FIELDS):
            family='e' if field>=3 else 'h';component=field%3
            source.append(f'''  {{const auto s=DiamondPosition<{field}>(q,first-1,oz,op,om,bias);float value=0.f;
   if(first==1)value=Initial(p,{field},s.z,s.y,s.x);
   else if(!s.owned)value=DiamondImport<Depth,Layout,{field}>(p,bases,first-1,s);
   {family}[{component}][{slot}]=value;
''')
            if slot in topology.providers['slots'][name]:
                source.append(f'   frame[DiamondProviderIndex<{field}>(q/(DiamondHalfY*CX),(q/CX)&(DiamondHalfY-1),lane)]=value;')
            source.append('  }')
        source.append(' }')
    source+=[' __syncthreads();',' #pragma unroll 1',' for(int n=first;n<=last;++n){']
    for phase in (0,1):
        family='e' if phase else 'h'
        # Import ghosts before any phase computes. Shared providers of this
        # family are dead after the preceding phase's retirement barrier.
        for slot in range(topology.slots):
            source.append(f'  {{const int q=DiamondPrivateQ<{slot}>(threadIdx.x);')
            for c in range(3):
                field=phase*3+c;name=FIELDS[field]
                source.append(f'''   {{const auto s=DiamondPosition<{field}>(q,n,oz,op,om,bias);
    if(!s.owned){{const float value=DiamondImport<Depth,Layout,{field}>(p,bases,n,s);{family}[{c}][{slot}]=value;
''')
                if slot in topology.providers['slots'][name]:
                    source.append(f'     frame[DiamondProviderIndex<{field}>(q/(DiamondHalfY*CX),(q/CX)&(DiamondHalfY-1),lane)]=value;')
                source.append('    }\n   }')
            source.append('  }')
        for slot in range(topology.slots):
            aligned=topology.geometry.storage_gauge=='physical'
            if aligned:
                source.append(f'''  {{const int q=DiamondPrivateQ<{slot}>(threadIdx.x),q0=q/(DiamondHalfY*CX),q2=q&(CX-1);
   const auto cell=DiamondPosition<0>(q,n,oz,op,om,bias);const int q1=cell.m,z=cell.z,y=cell.y,x=cell.x;''')
            for c in range(3):
                field=phase*3+c;name=FIELDS[field]
                if aligned:
                    r=topology.geometry.retiming(name)
                    source.append(f'   {{const bool owned=cell.dz&&cell.dm&&((cell.dp+{r})&(CY-1));')
                else:
                    source.append(f'  {{const int q=DiamondPrivateQ<{slot}>(threadIdx.x),q0=q/(DiamondHalfY*CX),q2=q&(CX-1);')
                    source.append(f'   const auto s=DiamondPosition<{field}>(q,n,oz,op,om,bias);const int q1=s.m,z=s.z,y=s.y,x=s.x;')
                for parity in ((None,) if dynamic_parity else (0,1)):
                    binding=DynamicPrivateReadBinding(topology,slot) if dynamic_parity else PrivateReadBinding(topology,slot,parity)
                    point=generate_native_cell(read_binding=binding)['point_blocks'][name]
                    source.append('   {' if dynamic_parity else f'   if constexpr(Parity=={parity})'+'{')
                    for shuffle,value,dx in binding.shuffles.values():
                        if shuffle not in point:continue
                        source.append(f'    const float {shuffle}=__shfl_sync(0xffffffffu,{value},(lane+({dx}))&31);')
                    source.append('    if(owned){float value=0.f;' if aligned else '    if(s.owned){float value=0.f;')
                    source+= [f'     if(env.Valid({field},z,y,x))'+'{',point,'      value=result;}',f'     {family}[{c}][{slot}]=value;']
                    if slot in topology.providers['slots'][name]:
                        source.append(f'     frame[DiamondProviderIndex<{field}>(q0,(q/CX)&(DiamondHalfY-1),lane)]=value;')
                    source+=['    }','   }']
                source.append('  }')
            if aligned:source.append('  }')
        source.append('  __syncthreads(); // Publish opposite family; retire old-family readers.')
    for slot in range(topology.slots):
        source.append(f'  {{const int q=DiamondPrivateQ<{slot}>(threadIdx.x);')
        for field,name in enumerate(FIELDS):
            family='e' if field>=3 else 'h';c=field%3
            source.append(f'''   {{const auto s=DiamondPosition<{field}>(q,n,oz,op,om,bias);if(s.owned){{
    const float value={family}[{c}][{slot}];const int surface=Surface(s.dz-1,s.dp-1,s.dm-1);
    if(surface>=0&&(Layout!=1||n<Depth))p.trails[bases[0]+((n-1)*6+{field})*SurfacePoints+surface]=value;
    if(n==Depth){{const auto&f=FinalField(p,{field});if(Contains(f,s.z,s.y,s.x))f.data[s.z*f.plane+s.y*f.row+s.x]=value;}}
   }}}}''')
        source.append('  }')
    source+=[' }','}', '''template<int Depth,int Layout,class Env>
__device__ __forceinline__ void RunDiamondPrivate(const Parameters&p,const int32_t*tiles,Env env){
 const int tile=tiles[blockIdx.x],bz=tile/(p.nx*p.ny),bp=(tile/p.nx)%p.ny;
 if((bz*BZ+bp*BY-p.physical_n[1])&1)RunDiamondPrivateParity<Depth,Layout,1>(p,tiles,env);
 else RunDiamondPrivateParity<Depth,Layout,0>(p,tiles,env);
}
''']
    text='\n'.join(source)
    if dynamic_parity:
        text=text[:text.index('template<int Depth,int Layout,class Env>')]
        text=text.replace('template<int Depth,int Layout,int Parity,class Env>',
                          'template<int Depth,int Layout,class Env>')
        text=text.replace('void RunDiamondPrivateParity(', 'void RunDiamondPrivate(')
        text=text.replace('const int first=window[0],last=window[1];',
                          'const int first=window[0],last=window[1],parity=(oz+op-bias)&1;')
    if region_clock:
        text=text.replace('DiamondPosition<','DiamondPositionOpaque<')
        marker='template<int Depth,int Layout,class Env>' if dynamic_parity else 'template<int Depth,int Layout,int Parity,class Env>'
        helper='''template<int Field>__device__ __forceinline__ DiamondSite DiamondPositionOpaque(int q,int version,int oz,int op,int om,int bias){
 int resident_version;
#ifdef __CUDA_ARCH__
 asm volatile("mov.b32 %0,%1;":"=r"(resident_version):"r"(version));
#else
 resident_version=version;
#endif
 return DiamondPosition<Field>(q,resident_version,oz,op,om,bias);
}
'''
        text=text.replace(marker,helper+marker,1)
    proof.update(storage_gauge=topology.geometry.storage_gauge,region_clock=region_clock,dynamic_parity_operands=dynamic_parity,
                source_sha256=hashlib.sha256(text.encode()).hexdigest(),source_barriers_per_step=2,
                 CUDA_executed=False,scope='Generated native register/warp/shared field placement; separate arithmetic, visibility and full-physics validation required')
    return text,proof
