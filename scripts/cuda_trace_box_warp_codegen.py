"""Generate a two-barrier box schedule with private Z/Y warp rectangles."""
from __future__ import annotations
import hashlib

from scripts.cuda_space_time_ir import FIELDS
from scripts.cuda_trace_box_warp_storage import BoxWarpStorage,BoxReadBinding
from scripts.generate_cuda_native_cell import generate_native_cell


def target(t,field,slot,q='q'):
    if FIELDS[field] in t.dense_fields:return f'frame[BoxDenseOffset+{t.dense_fields.index(FIELDS[field])*t.points}+{q}]'
    if t.provider_home and slot in t.providers['slots'][FIELDS[field]]:return f'frame[BoxProviderIndex<{field},{slot}>(warp,lane)]'
    return f'{"e" if field>=3 else "h"}[{field%3}][{slot}]'


def generate(t=BoxWarpStorage(),*,late_addresses=False,volatile_addresses=False,group_publications=False,bounded_warp=False,terminal_epilogue=False):
    if volatile_addresses:late_addresses=True
    proof=t.prove();cz,cy,cx=t.cache;zb=cz//t.private_z;yb=cy//t.private_y
    if volatile_addresses:
        proof['shared_bytes']+=t.threads//32*4
        proof['cached_shared_bytes']+=t.threads//32*4
    source=[f'''constexpr int TraceBulkThreads={t.threads},TraceBaseThreads={t.threads},TraceSourceThreads={t.threads},TraceGenericSourceThreads={t.threads};
constexpr int BoxSlots={t.slots},BoxProviderWords={t.shared_words},BoxDenseOffset={t.providers['words']};
template<int S>__device__ __forceinline__ int BoxQ(int tid){{
 const int warp=tid>>5,z=(warp/{yb})*{t.private_z}+S/{t.private_y},y=(warp%{yb})*{t.private_y}+S%{t.private_y};
 return (z*CY+y)*CX+(tid&31);
}}
template<int Field,int S,int ZDelta=0,int YDelta=0>__device__ __forceinline__ int BoxProviderIndex(int warp,int lane){{
 const int provider=(((warp/{yb}+ZDelta)&{zb-1})*{yb})+((warp%{yb}+YDelta)&{yb-1});
''']
    for field,name in enumerate(FIELDS):
        slots=t.providers['slots'][name]
        if not slots:continue
        rank=':'.join(f'S=={s}?{i}' for i,s in enumerate(slots))+':-1'
        index=f'rank*{t.threads}+provider*32+lane' if t.slot_major else f'(provider*{len(slots)}+rank)*32+lane'
        source.append(f' if constexpr(Field=={field})'+'{'+f'constexpr int rank={rank};static_assert(rank>=0,"Provider outside the exact traced cut");return {t.providers["offsets"][name]}+{index};'+'}')
    source+=[' return -1;','}']
    if volatile_addresses:
        source.append('''// Immutable warp IDs are staged before the first CTA barrier. Volatile
// reads bound address lifetimes; they never touch field or physics storage.
struct BoxAddressRegion{int warp,version;};
__device__ __forceinline__ BoxAddressRegion BoxAddressScope(int warp,int version){
#ifdef __CUDA_ARCH__
 extern __shared__ float frame[];
 volatile int*ids=reinterpret_cast<volatile int*>(frame+BoxProviderWords+19);
 return {ids[warp],version};
#else
 return {warp,version};
#endif
}
''')
    elif late_addresses:
        source.append('''// Bit-preserving address-region lifetime boundary. No memory or FP effect.
struct BoxAddressRegion{int warp,version;};
__device__ __forceinline__ BoxAddressRegion BoxAddressScope(int warp,int version){
 BoxAddressRegion r;
#ifdef __CUDA_ARCH__
 asm volatile("mov.b32 %0,%2; mov.b32 %1,%3;":"=&r"(r.warp),"=&r"(r.version):"r"(warp),"r"(version));
#else
 r={warp,version};
#endif
 return r;
}
''')
    source.append('''template<int Depth,int Layout,bool CachedProfiles,class Env>
__device__ __forceinline__ void RunBoxWarp(const Parameters&p,const int32_t*tiles,Env env){
 extern __shared__ float frame[];
 size_t*bases=reinterpret_cast<size_t*>(frame+BoxProviderWords);
 volatile int*window=reinterpret_cast<volatile int*>(bases+8);
 // Box address-lifetime metadata initialization is inserted here if enabled.
 if constexpr(CachedProfiles)StageIndependentCpmlProfiles(p,frame+SharedBytes/sizeof(float));
 const int tile=tiles[blockIdx.x],bz=tile/(p.nx*p.ny),by=(tile/p.nx)%p.ny,bx=tile%p.nx;
 const int oz=bz*BZ,oy=by*BY,ox=bx*BX,rank=bz+by+bx,warp=threadIdx.x>>5,lane=threadIdx.x&31;
 const uint64_t epoch=uint64_t(*p.epoch)<<32;
 if(!threadIdx.x){const int owner=(rank&3)*p.frontier+p.slots[tile];p.owner_tags[owner]=epoch|uint32_t(tile);}
 __syncthreads();
 if(!threadIdx.x){
  if constexpr(Env::SupportSafe){window[0]=1;window[1]=Depth;}
  else {const int mz=p.physical_n[0]+1,my=p.physical_n[1]+1,mx=p.physical_n[2]+1;
   window[0]=max(1,max(oz-mz+1,max(oy-my+1,ox-mx+1)));
   window[1]=min(Depth,min(oz+BZ-1,min(oy+BY-1,ox+BX-1)));
  }
 }
 __syncthreads();if(window[0]>window[1])return;
 if(threadIdx.x<8){const int bits=threadIdx.x,dz=bits&1,dy=(bits>>1)&1,dx=(bits>>2)&1;size_t base=size_t(-1);
  if(bz>=dz&&by>=dy&&bx>=dx){const int producer=tile-dz*p.nx*p.ny-dy*p.nx-dx;
   const int owner=((rank-dz-dy-dx)&3)*p.frontier+p.slots[producer];
   if(bits&&p.owner_tags[owner]!=(epoch|uint32_t(producer)))atomicOr(p.bad,1);else base=size_t(owner)*Depth*6*SurfacePoints;
  }bases[bits]=base;
 }
 __syncthreads();const int first=window[0],last=window[1];
 float h[3][BoxSlots],e[3][BoxSlots];
''')
    def coords(slot,version):
        if late_addresses:
            region_warp=f'address_region.warp&{t.threads//32-1}' if bounded_warp else 'address_region.warp'
            return f'''const auto address_region=BoxAddressScope(threadIdx.x>>5,{version});
   const int warp={region_warp},lane=threadIdx.x&31,q=BoxQ<{slot}>((warp<<5)|lane),zs=q/(CY*CX),ys=(q/CX)&(CY-1),low=-address_region.version-1;
   const int dz=(zs-low)&(CZ-1),dy=(ys-low)&(CY-1),dx=(lane-low)&(CX-1),z=oz+low+dz,y=oy+low+dy,x=ox+low+dx;
   const bool owned=dz&&dy&&dx;'''
        return f'''const int q=BoxQ<{slot}>(threadIdx.x),zs=q/(CY*CX),ys=(q/CX)&(CY-1),low=-({version})-1;
   const int dz=(zs-low)&(CZ-1),dy=(ys-low)&(CY-1),dx=(lane-low)&(CX-1),z=oz+low+dz,y=oy+low+dy,x=ox+low+dx;
   const bool owned=dz&&dy&&dx;'''
    def providers(field,slot,indent):
        if t.provider_home:return []
        if slot in t.providers['slots'][FIELDS[field]]:
            return [f'{indent}frame[BoxProviderIndex<{field},{slot}>(warp,lane)]={target(t,field,slot)};']
        return []
    for slot in range(t.slots):
        source+=[' { '+coords(slot,'first-1'),'  FieldPacket v{};if(first==1){',
                 '   v={Initial(p,0,z,y,x),Initial(p,1,z,y,x),Initial(p,2,z,y,x),Initial(p,3,z,y,x),Initial(p,4,z,y,x),Initial(p,5,z,y,x)};',
                 '  }else if(!owned)v=ImportPacket<Depth,false,(Layout==1)>(p,bases,first-1,oz,oy,ox,dz,dy,dx);']
        for field,member in enumerate(('hx','hy','hz','ex','ey','ez')):source.append(f'  {target(t,field,slot)}=v.{member};')
        for field in range(6):source+=providers(field,slot,'  ')
        source.append(' }')
    source+=[' __syncthreads();',' #pragma unroll 1',' for(int n=first;n<=last;++n){']
    for phase in (0,1):
        for slot in range(t.slots):
            source.append('  { '+coords(slot,'n'))
            source.append('   if(!owned){const auto v=ImportPacket<Depth,Env::SupportSafe,(Layout==1)>(p,bases,n,oz,oy,ox,dz,dy,dx);')
            for field in range(phase*3,phase*3+3):source.append(f'    {target(t,field,slot)}=v.{("hx","hy","hz","ex","ey","ez")[field]};')
            source.append('   }')
            binding=BoxReadBinding(t,slot);blocks=generate_native_cell(read_binding=binding)['point_blocks']
            used='\n'.join(blocks[FIELDS[phase*3+c]] for c in range(3))
            for name,(value,dx) in binding.shuffles.items():
                if name in used:source.append(f'   const float {name}=__shfl_sync(0xffffffffu,{value},(lane+({dx}))&31);')
            source.append('   if(owned){const int surface=Surface(dz-1,dy-1,dx-1);')
            publish='surface>=0' if terminal_epilogue and phase==0 else 'surface>=0&&(Layout!=1||n<Depth)'
            for field in range(phase*3,phase*3+3):
                source+=['    {float value=0.f;',f'     if(env.Valid({field},z,y,x))'+'{',blocks[FIELDS[field]],'      value=result;}',f'     {target(t,field,slot)}=value;',
                         *([] if group_publications else [f'     if({publish})p.trails[bases[0]+((n-1)*6+{field})*SurfacePoints+surface]=value;',
                         *([] if terminal_epilogue else [f'     if(n==Depth){{const auto&f=FinalField(p,{field});if(Contains(f,z,y,x))f.data[z*f.plane+y*f.row+x]=value;}}'])]),'    }']
            if group_publications:
                source.append(f'    if({publish})'+'{float*address=p.trails+bases[0]+(n-1)*6*SurfacePoints+surface;')
                for field in range(phase*3,phase*3+3):source.append(f'     address[{field}*SurfacePoints]={target(t,field,slot)};')
                source.append('    }')
                if not terminal_epilogue:
                    source.append('    if(n==Depth){const unsigned final_mask=Env::SupportSafe?63u:NativeSupportMask(p,z,y,x);')
                    for field in range(phase*3,phase*3+3):source.append(f'     {{const auto&f=FinalField(p,{field});if(final_mask&{1<<field}u)f.data[z*f.plane+y*f.row+x]={target(t,field,slot)};}}')
                    source.append('    }')
            source.append('   }')
            for field in range(phase*3,phase*3+3):source+=providers(field,slot,'   ')
            source.append('  }')
        source.append('  __syncthreads(); // Separate family banks publish current values and retire opposite readers.')
    source.append(' }')
    if terminal_epilogue:
        source.append(' if(last==Depth){')
        for slot in range(t.slots):
            source+=['  { '+coords(slot,'Depth'),'   if(owned){const unsigned final_mask=Env::SupportSafe?63u:NativeSupportMask(p,z,y,x);']
            for field in range(6):source.append(f'    {{const auto&f=FinalField(p,{field});if(final_mask&{1<<field}u)f.data[z*f.plane+y*f.row+x]={target(t,field,slot)};}}')
            source+=['   }','  }']
        source.append(' }')
    source+=['}','']
    text='\n'.join(source)
    if volatile_addresses:
        text=text.replace('// Box address-lifetime metadata initialization is inserted here if enabled.',
                          f'if(threadIdx.x<{t.threads//32})reinterpret_cast<volatile int*>(frame+BoxProviderWords+19)[threadIdx.x]=threadIdx.x;')
    proof.update(source_sha256=hashlib.sha256(text.encode()).hexdigest(),barriers_per_step=2,late_addresses=late_addresses,volatile_addresses=volatile_addresses,
                 grouped_publications=group_publications,bounded_warp=bounded_warp,terminal_epilogue=terminal_epilogue,
                 schedule='H ghosts, H component SSA and publication; join; E ghosts, E component SSA and publication; join',
                 side_effect_order='Component-local CPML and Finish stay canonical; global H publication moves earlier inside its complete rank task only',CUDA_executed=False)
    return text,proof
