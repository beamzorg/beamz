"""Split the paired resident tile DAG into H(n) and E(n) publications.

H imports read only H after ready2n-1; current E ghosts are delayed until
ready2n. All per-cell arithmetic and paired E retirement order are retained.
"""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path

try:
    from scripts.patch_cuda_native_persistent import sha,once
except ModuleNotFoundError:
    from patch_cuda_native_persistent import sha,once


H_PUBLISH=r'''  // Current owned H is complete; only ghost H imports changed other slots.
  #pragma unroll
  for(int gy=0;gy<YGroups;++gy){const int ys=sy+gy*(T/CX),dy=(ys-hy)&(CY-1),dx=(sx-hx)&31,y=oy+low+dy,x=ox+low+dx;
   #pragma unroll
   for(int j=0;j<LocalZ;++j){const int zs=zbase+j,si=gy*LocalZ+j,dz=(zs-hz)&(CZ-1),z=oz+low+dz;
    if(zs!=hz&&ys!=hy&&sx!=hx){const int surface=Surface(dz-1,dy-1,dx-1);
     #pragma unroll
     for(int c=0;c<3;++c){
      if(surface>=0)p.trails[bases[0]+((n-1)*6+c)*SurfacePoints+surface]=h[c][si];
      if(n==Depth){const auto&hf=FinalField(p,c);if(Contains(hf,z,y,x))hf.data[z*hf.plane+y*hf.row+x]=h[c][si];}
     }
    }
   }
  }
  __threadfence(); // Every H/psiH/observationH/trailH writer publishes before R0.
'''

E_IMPORT=r'''  GateImports(2*n);if(C().abort_local)return;
  #pragma unroll
  for(int gy=0;gy<YGroups;++gy){const int ys=sy+gy*(T/CX),dy=(ys-hy)&(CY-1),dx=(sx-hx)&31;
   #pragma unroll
   for(int j=0;j<LocalZ;++j){const int zs=zbase+j,si=gy*LocalZ+j,dz=(zs-hz)&(CZ-1);
    if(zs==hz||ys==hy||sx==hx){const auto v=ImportPhasePacket<Depth,SupportSafe,1>(p,bases,n,oz,oy,ox,dz,dy,dx);
     frame[Slot(zs,ys,sx)]=v.ex;frame[PAD+Slot(zs,ys,sx)]=v.ez;set_ey(zs,ys,si,v.ey);
    }
   }
  }
'''


def phase_packet(original):
    a=original.index('template<int Depth,bool Fast>\n__device__ __forceinline__ FieldPacket ImportPacket(')
    b=original.index('__global__ void BeginEpoch',a)
    packet=original[a:b]
    packet=once(packet,'template<int Depth,bool Fast>','template<int Depth,bool Fast,int Family>')
    packet=once(packet,'FieldPacket ImportPacket(', 'FieldPacket ImportPhasePacket(')
    packet=once(packet,' unsigned mask=63;', ' static_assert(Family==0||Family==1,"H/E phase family only");\n unsigned mask=63;')
    packet=once(packet,' if(!mask)return {};',' mask&=Family==0?7u:56u;\n if(!mask)return {};')
    # Compile-time family selection suppresses every unready opposite-family
    # load, including both scalar values and ownership-guard bad returns.
    for c in range(6):
        packet=packet.replace(f'mask&{1<<c}?',f'(Family=={c//3}&&(mask&{1<<c}))?')
    return packet


def phase_ready(code,original):
    if 'PersistentUniformBody' in code:raise ValueError('First phase lowering requires P5-only paired source')
    if '// TRACE_STARTUP_PREVIOUS_VERSION' not in code:raise ValueError('Startup-prev H-overlap parent required')
    code=once(code,'constexpr int PublicationSteps=4;','constexpr int PublicationSteps=1;')
    code=once(code,'if(ShouldPublish(step,last))Release(C().ready+C().owner,uint32_t(step));',
        'Release(C().ready+C().owner,uint32_t(step));')
    code=once(code,'WaitUpstream(window[0]-1);','WaitUpstream(2*(window[0]-1));')
    code=once(code,'GateImports(n);if(C().abort_local)return;',
        'GateImports(2*n-1);if(C().abort_local)return;')
    code=once(code,'C().depth=Depth;', 'C().depth=2*Depth;')
    code=once(code,'Release(C().ready+C().owner,uint32_t(Depth));',
        'Release(C().ready+C().owner,uint32_t(2*Depth));')
    marker='// TRACE_STARTUP_PREVIOUS_VERSION'
    helper=phase_packet(original)
    a=original.index('template<int Depth,bool Safe,int T>\n__device__ __forceinline__ void StageSourcePendingH(')
    b=original.index('// Two-plane publication groups;',a)
    stage=original[a:b].replace('void StageSourcePendingH(', 'void StagePhaseH(')
    stage=once(stage,'ImportPacket<Depth,Safe>', 'ImportPhasePacket<Depth,Safe,0>')
    code=once(code,marker,helper+stage+'\n// TRACE_PHASE_READY_COUNTER:2*n-1=H,2*n=E; G1; completion2Depth.\n'+marker)
    # Current H ghost import, and its low-X staging, must not fetch E(n).
    code=once(code,'const auto v=ImportPacket<Depth,SupportSafe>(p,bases,n,oz,oy,ox,(zs-hz)&(CZ-1),dy,dx);h[0][si]=v.hx;',
        'const auto v=ImportPhasePacket<Depth,SupportSafe,0>(p,bases,n,oz,oy,ox,(zs-hz)&(CZ-1),dy,dx);h[0][si]=v.hx;')
    code=once(code,'StageSourcePendingH<Depth,SupportSafe,T>(p,low_stage,bases,n,oz,oy,ox);',
        'StagePhaseH<Depth,SupportSafe,T>(p,low_stage,bases,n,oz,oy,ox);')
    barrier='  __syncthreads(); // R0: retire all old E readers; publish low-X H imports.'
    code=once(code,barrier,H_PUBLISH+barrier+'\n  if(!threadIdx.x)PublishAndWait(2*n-1,2*window[1]);')
    old_ghost='    if(zs==hz||ys==hy||sx==hx){const auto v=ImportPacket<Depth,SupportSafe>(p,bases,n,oz,oy,ox,dz,dy,dx);frame[Slot(zs,ys,sx)]=v.ex;frame[PAD+Slot(zs,ys,sx)]=v.ez;set_ey(zs,ys,si,v.ey);}\n'
    code=once(code,old_ghost,'')
    hstore='p.trails[bases[0]+((n-1)*6+c)*SurfacePoints+surface]=h[c][si];'
    # H_PUBLISH contains one copy; the two old paired E blocks contain two.
    if code.count(hstore)!=3:raise ValueError('Expected two paired H publication sites')
    before,after=code.split(H_PUBLISH,1)
    if after.count(hstore)!=2:raise ValueError('Unexpected H pub order')
    after=after.replace(hstore,'')
    old_final='const auto&hf=FinalField(p,c);const auto&ef=FinalField(p,c+3);if(Contains(hf,z,y,x))hf.data[z*hf.plane+y*hf.row+x]=h[c][si];'
    if after.count(old_final)!=2:raise ValueError('Expected paired final-H stores')
    after=after.replace(old_final,'const auto&ef=FinalField(p,c+3);')
    code=before+H_PUBLISH+after
    code=once(code,'if(ShouldPublish(n,window[1]))__threadfence();','__threadfence();')
    rl='  __syncthreads(); // RLpair retires BOTH final planes\' H XY readers.'
    code=once(code,rl,rl+'\n  if(!threadIdx.x)PublishAndWait(2*n,2*window[1]);')
    code=once(code,'  if(!threadIdx.x){*phase_step=n+1;PublishAndWait(n,window[1]);}',
        E_IMPORT+'  if(!threadIdx.x)*phase_step=n+1;')
    return code,helper


def patch(source,output):
    source,output=source.resolve(),output.resolve()
    if output.exists() or source==output or source in output.parents or output in source.parents:
        raise ValueError('New disjoint private source required')
    old=(source/'src/native_persistent.cuh').read_text();original=(source/'src/asymmetric_native.cuh').read_text()
    new,packet=phase_ready(old,original);shutil.copytree(source,output);(output/'src/native_persistent.cuh').write_text(new)
    proof={'parent_persistent_consumer_sha256':sha(old.encode()),'persistent_consumer_sha256':sha(new.encode()),
        'patcher_sha256':sha(Path(__file__).read_bytes()),'phase_packet_sha256':sha(packet.encode()),
        'workspace_abi_changed':False,'expected_entries':65,'phase_counters':'H2n-1/E2n; startupE2(first-1); completion2Depth',
        'publication':'ownedH+psiH fenced before existingR0, releaseHafterR0; E publications remain per-pair, fenceRL, releaseEafterRL before waitParentE/importEghost/RE',
        'imports':'currentH family0 only beforeR0; currentE family1 only afterlocalE publication/parentEwait, beforeRE; initial late previouspacket remains full',
        'barriers_unchanged':old.count('__syncthreads()')==new.count('__syncthreads()'),
        'requires':['phaseDAG proof','exact field/publication/source ordering proof','actual phase controller proof','resources','all30 exact','sanitizers','paired timings'],'gpu_executed':False}
    (output.parent/'persistent-phase-ready-proof.json').write_text(json.dumps(proof,indent=2)+'\n');return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();print(json.dumps(patch(a.source,a.output)))
