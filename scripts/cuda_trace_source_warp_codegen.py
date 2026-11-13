"""Generate source-oriented native CUDA with separate compute and I/O ownership."""
from __future__ import annotations
import hashlib

from scripts.cuda_space_time_ir import FIELDS,IllegalSchedule
from scripts.cuda_trace_source_warp_topology import SourceWarpTopology
from scripts.generate_cuda_native_cell import generate_native_cell


class SourceWarpReadBinding:
    def __init__(self,topology,slot,shared_e=False):self.topology=topology;self.slot=slot;self.shared_e=shared_e;self.shuffles={}

    def __call__(self,output,read):
        phase=int(output.startswith('E'));source_phase=int(read.field.startswith('E'))
        if read.version!=(0 if source_phase else phase):raise IllegalSchedule('Wrong source-warp field version')
        field=FIELDS.index(read.field);c=field%3;family='e' if source_phase else 'h';dz,dy,dx=read.offset
        if self.shared_e and source_phase:
            return f'frame[{c}*PAD+SourceWarpESlot(zs+({dz}),sy+({dy}),sx+({dx}))]'
        if sum(abs(d) for d in read.offset)>1:raise IllegalSchedule('Nearest axial Yee source route required')
        value=f'{family}[{c}][{self.slot}]'
        if not any(read.offset):return value
        if dx and 0<=self.slot+dx<self.topology.slots:return f'{family}[{c}][{self.slot+dx}]'
        shared=f'frame[{"3*PAD+" if self.shared_e else ""}SourceWarpProviderIndex<{field}>((zs+({dz}))&7,(sy+({dy}))&15,(sx+({dx}))&31)]'
        if dx:return shared
        axis=0 if dz else 1;direction=dz or dy;name=f'source_shuffle_{field}_{self.slot}_{axis}_{direction+1}'
        target='(lane^16)' if dz else f'((lane&16)|((sy+({dy}))&15))'
        self.shuffles[name]=(value,target)
        if dy:return name
        return f'(((lane>>4)=={0 if dz>0 else 1})?{name}:({shared}))'


def helpers(t,shared_e=False):
    lines=[f'''constexpr int SourceWarpSlots={t.slots},SourceWarpThreads={t.threads},SourceWarpProviderWords={t.providers['words']};
constexpr int SourceWarpScratchWords=3*PAD>SourceWarpProviderWords?3*PAD:SourceWarpProviderWords;
constexpr size_t SourceWarpFrameBytes=4*SourceWarpScratchWords+76;
template<int Field>__device__ __forceinline__ int SourceWarpProviderIndex(int z,int y,int x){{
 const int slot=(z%2)*SourceWarpSlots+x%SourceWarpSlots,warp=(z/2)*(CX/SourceWarpSlots)+x/SourceWarpSlots;int rank=-1;
''']
    for field,name in enumerate(FIELDS):
        slots=t.providers['slots'][name]
        lines.append(f' if constexpr(Field=={field})'+'{switch(slot){'+''.join(f'case {s}:rank={i};break;' for i,s in enumerate(slots))+f'}}return {t.providers["field_offsets"][name]}+(warp*{len(slots)}+rank)*16+y;'+'}')
    lines+=[' return -1;','}', '''struct SourceWarpSite{int z,y,x,dz,dy,dx;bool owned;};
__device__ __forceinline__ SourceWarpSite SourceWarpPosition(int zs,int sy,int sx,int version,int oz,int oy,int ox){
 const int low=-version-1,dz=(zs-low)&7,dy=(sy-low)&15,dx=(sx-low)&31;
 return {oz+low+dz,oy+low+dy,ox+low+dx,dz,dy,dx,dz&&dy&&dx};
}
''']
    text='\n'.join(lines)
    if shared_e:
        hwords=t.providers['field_offsets']['Ex']
        text=text.replace(f'SourceWarpProviderWords={t.providers["words"]}',f'SourceWarpProviderWords={hwords}')
        text=text.replace('SourceWarpScratchWords=3*PAD>SourceWarpProviderWords?3*PAD:SourceWarpProviderWords',
                          'SourceWarpScratchWords=3*PAD+SourceWarpProviderWords')
        text+='\n__device__ __forceinline__ int SourceWarpESlot(int z,int y,int x){z&=7;y&=15;x&=31;return (z*CY+y)*CX+(x^(y|((z&1)<<4)));}\n'
    return text


def generate(topology=SourceWarpTopology(),*,shared_e=False,region_clock=False):
    proof=topology.prove();slots=topology.slots
    lines=[helpers(topology,shared_e),'''template<int Depth,int Layout,bool Fast,bool IsotropicH,int Policy,bool UniformE,int SourceMask,bool PlanePair,int ActiveAxes,bool CachedProfiles>
__device__ __forceinline__ void RunSourceWarp(const Parameters&p,const int32_t*tiles){
 extern __shared__ float frame[];
 size_t*bases=reinterpret_cast<size_t*>(frame+SourceWarpScratchWords);
 volatile int*window=reinterpret_cast<volatile int*>(bases+8);
 float*profiles=CachedProfiles?frame+SourceWarpFrameBytes/sizeof(float):nullptr;
 if constexpr(CachedProfiles)for(int i=threadIdx.x;i<864;i+=blockDim.x){
  const int slot=i/72,kind=(i/24)%3,packed=i%24;const auto&psi=p.phase[slot/6].psi[slot%6];
  const float*input=kind==0?psi.a:kind==1?psi.b:psi.inv;profiles[i]=__ldg(input+packed);
 }
 const int tile=tiles[blockIdx.x],bz=tile/(p.nx*p.ny),by=(tile/p.nx)%p.ny,bx=tile%p.nx;
 const int oz=bz*BZ,oy=by*BY,ox=bx*BX,rank=bz+by+bx,lane=threadIdx.x&31;
 const int zs=(threadIdx.x/32/(CX/SourceWarpSlots))*2+(lane>>4),sy=lane&15,xbase=(threadIdx.x/32%(CX/SourceWarpSlots))*SourceWarpSlots;
 const uint64_t epoch=uint64_t(*p.epoch)<<32;
 if(!threadIdx.x){const int owner=(rank&3)*p.frontier+p.slots[tile];p.owner_tags[owner]=epoch|uint32_t(tile);}
 __syncthreads();
 if(!threadIdx.x){window[0]=max(1,max(oz-p.physical_n[0],max(oy-p.physical_n[1],ox-p.physical_n[2])));
  window[1]=min(Depth,min(oz+BZ-1,min(oy+BY-1,ox+BX-1)));}
 __syncthreads();if(window[0]>window[1])return;
 if(threadIdx.x<8){const int bits=threadIdx.x,dz=bits&1,dy=(bits>>1)&1,dx=(bits>>2)&1;size_t base=size_t(-1);
  if(bz>=dz&&by>=dy&&bx>=dx){const int producer=tile-dz*p.nx*p.ny-dy*p.nx-dx;
   const int owner=((rank-dz-dy-dx)&3)*p.frontier+p.slots[producer];
   if(bits&&p.owner_tags[owner]!=(epoch|uint32_t(producer)))atomicOr(p.bad,1);else base=size_t(owner)*Depth*6*SurfacePoints;
  }bases[bits]=base;
 }
 __syncthreads();
 const int first=window[0],last=window[1];
 constexpr bool SupportSafe=Fast||(Policy>=2&&Policy<=4);
 float h[3][SourceWarpSlots],e[3][SourceWarpSlots];
''']
    # Coalesced linear-X transfer, then redistribute to source-oriented lanes.
    for phase in (0,1):
        family='e' if phase else 'h'
        lines+=[' for(int q=threadIdx.x;q<PAD;q+=blockDim.x){',
                '  const auto s=SourceWarpPosition(q/(CY*CX),(q/CX)&15,q&31,first-1,oz,oy,ox);']
        for c in range(3):
            field=3*phase+c;member=('e' if phase else 'h')+'xyz'[c]
            target=f'frame[{c}*PAD+SourceWarpESlot(q/(CY*CX),(q/CX)&15,q&31)]' if shared_e and phase else f'frame[{c}*PAD+q]'
            lines.append(f'''  {{float value=0.f;if(first==1)value=Initial(p,{field},s.z,s.y,s.x);
   else if(!s.owned)value=ImportPacket<Depth,false,(Layout==1)>(p,bases,first-1,oz,oy,ox,s.dz,s.dy,s.dx).{member};
   {target}=value;}}''')
        lines+=[' }',' __syncthreads();']
        for slot in (() if shared_e and phase else range(slots)):
            for c in range(3):lines.append(f' {family}[{c}][{slot}]=frame[{c}*PAD+(zs*CY+sy)*CX+xbase+{slot}];')
        lines.append(' __syncthreads(); // All private reads retire before scratch reuse.')
    for slot in range(slots):
        for field,name in enumerate(FIELDS[:3] if shared_e else FIELDS):
            family='e' if field>=3 else 'h';c=field%3
            # The two Z lanes have different provider membership, uniform per
            # half-warp and independent of physical cache motion.
            zlanes=[z for z in (0,1) if z*slots+slot in topology.providers['slots'][name]]
            if zlanes:
                guard='true' if len(zlanes)==2 else f'(lane>>4)=={zlanes[0]}'
                lines.append(f' if({guard})frame[{"3*PAD+" if shared_e else ""}SourceWarpProviderIndex<{field}>(zs,sy,xbase+{slot})]={family}[{c}][{slot}];')
    lines+=[' __syncthreads();',
            ' const auto flags=p.flags[tile];',
            ' Environment<Layout,Fast,IsotropicH,Policy,UniformE,SourceMask,PlanePair,ActiveAxes,CachedProfiles>env{p,flags,profiles};',
            ' #pragma unroll 1',' for(int n=first;n<=last;++n){']
    for phase in (0,1):
        family='e' if phase else 'h'
        for slot in range(slots):
            lines.append(f'''  {{const int sx=xbase+{slot};const auto s=SourceWarpPosition(zs,sy,sx,n,oz,oy,ox);
   if(!s.owned){{const auto v=ImportPacket<Depth,SupportSafe,(Layout==1)>(p,bases,n,oz,oy,ox,s.dz,s.dy,s.dx);''')
            for c in range(3):
                field=3*phase+c;name=FIELDS[field];member=family+'xyz'[c]
                target=f'frame[{c}*PAD+SourceWarpESlot(zs,sy,sx)]' if shared_e and phase else f'{family}[{c}][{slot}]'
                lines.append(f'    {target}=v.{member};')
                zlanes=[z for z in (0,1) if z*slots+slot in topology.providers['slots'][name]]
                if zlanes and not(shared_e and phase):
                    guard='true' if len(zlanes)==2 else f'(lane>>4)=={zlanes[0]}'
                    lines.append(f'    if({guard})frame[{"3*PAD+" if shared_e else ""}SourceWarpProviderIndex<{field}>(zs,sy,sx)]=v.{member};')
            lines+=['   }','  }']
        for slot in range(slots):
            lines.append(f'  {{const int sx=xbase+{slot};const auto s=SourceWarpPosition(zs,sy,sx,n,oz,oy,ox);const int z=s.z,y=s.y,x=s.x;')
            binding=SourceWarpReadBinding(topology,slot,shared_e);blocks=generate_native_cell(read_binding=binding)['point_blocks']
            used='\n'.join(blocks[FIELDS[phase*3+c]] for c in range(3))
            for name,(value,target) in binding.shuffles.items():
                if name in used:lines.append(f'   const float {name}=__shfl_sync(0xffffffffu,{value},{target});')
            lines.append('   if(s.owned){')
            for c in range(3):
                field=phase*3+c;name=FIELDS[field]
                lines += ['    {float value=0.f;',f'     if(env.Valid({field},z,y,x))'+'{',blocks[name],'      value=result;}',f'     '+(f'frame[{c}*PAD+SourceWarpESlot(zs,sy,sx)]' if shared_e and phase else f'{family}[{c}][{slot}]')+'=value;']
                zlanes=[z for z in (0,1) if z*slots+slot in topology.providers['slots'][name]]
                if zlanes and not(shared_e and phase):
                    guard='true' if len(zlanes)==2 else f'(lane>>4)=={zlanes[0]}'
                    lines.append(f'     if({guard})frame[{"3*PAD+" if shared_e else ""}SourceWarpProviderIndex<{field}>(zs,sy,sx)]=value;')
                lines.append('    }')
            lines+=['   }','  }']
        lines.append('  __syncthreads(); // Publish family and retire opposite-family readers.')
    for slot in range(slots):
        lines.append(f'''  {{const int sx=xbase+{slot};const auto s=SourceWarpPosition(zs,sy,sx,n,oz,oy,ox);
   if(s.owned){{const int surface=Surface(s.dz-1,s.dy-1,s.dx-1);
    if(surface>=0&&(Layout!=1||n<Depth)){{''')
        for field in range(6):
            family='e' if field>=3 else 'h';c=field%3
            value=f'frame[{c}*PAD+SourceWarpESlot(zs,sy,sx)]' if shared_e and field>=3 else f'{family}[{c}][{slot}]'
            lines.append(f'     p.trails[bases[0]+((n-1)*6+{field})*SurfacePoints+surface]={value};')
        lines+=['    }','   }','  }']
    lines+=[' }',' if(last==Depth){']
    # Reuse the same half-field scratch for final X-coalesced world stores.
    for phase in ((1,0) if shared_e else (0,1)):
        family='e' if phase else 'h'
        for slot in (() if shared_e and phase else range(slots)):
            for c in range(3):lines.append(f'  frame[{c}*PAD+(zs*CY+sy)*CX+xbase+{slot}]={family}[{c}][{slot}];')
        lines+=['  __syncthreads();','  for(int q=threadIdx.x;q<PAD;q+=blockDim.x){',
                '   const auto s=SourceWarpPosition(q/(CY*CX),(q/CX)&15,q&31,Depth,oz,oy,ox);',
                '   if(s.owned){']
        for c in range(3):
            field=phase*3+c
            lines.append(f'    {{const auto&f=FinalField(p,{field});if(Contains(f,s.z,s.y,s.x))f.data[s.z*f.plane+s.y*f.row+s.x]=frame[{c}*PAD+{"SourceWarpESlot(q/(CY*CX),(q/CX)&15,q&31)" if shared_e and phase else "q"}];}}')
        lines+=['   }','  }']
        if phase==(1 if shared_e else 0):lines.append('  __syncthreads(); // Retire output reads before scratch reuse.')
    lines+=[' }','}','']
    text='\n'.join(lines)
    if shared_e:
        text=text.replace('float h[3][SourceWarpSlots],e[3][SourceWarpSlots];','float h[3][SourceWarpSlots];')
        proof['private_topology_reference_routes']=proof.pop('routes')
        proof.update(provider_words=topology.providers['field_offsets']['Ex'],provider_bytes=4*topology.providers['field_offsets']['Ex'],private_field_registers=3*slots,
                     shared_frame_bytes=4*(3*topology.points+topology.providers['field_offsets']['Ex'])+76)
    if region_clock:
        text=text.replace('const auto s=SourceWarpPosition(', 'const auto s=SourceWarpPositionOpaque(')
        marker='template<int Depth,int Layout,bool Fast'
        helper='''__device__ __forceinline__ SourceWarpSite SourceWarpPositionOpaque(int zs,int sy,int sx,int version,int oz,int oy,int ox){
 int resident_version;
#ifdef __CUDA_ARCH__
 asm volatile("mov.b32 %0,%1;":"=r"(resident_version):"r"(version));
#else
 resident_version=version;
#endif
 return SourceWarpPosition(zs,sy,sx,resident_version,oz,oy,ox);
}
'''
        text=text.replace(marker,helper+marker,1)
    proof.update(region_clock=region_clock,shared_e=shared_e,source_sha256=hashlib.sha256(text.encode()).hexdigest(),
            barriers_per_step=2,profile_bytes=3456,compute_IO_mapping_separate=True,CUDA_executed=False)
    return text,proof
