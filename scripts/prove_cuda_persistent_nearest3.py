"""CPU accumulated-visibility and actual-controller proof for nearest3 waits."""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

try:
    from scripts.cuda_persistent_tile_pipeline import prepare as plan,windows
    from scripts.cuda_persistent_wavefront_dag import rank_slots
    from scripts.prove_cuda_persistent_control import prepare as control_prepare
except ModuleNotFoundError:
    from cuda_persistent_tile_pipeline import prepare as plan,windows
    from cuda_persistent_wavefront_dag import rank_slots
    from prove_cuda_persistent_control import prepare as control_prepare


def any_support(lo,hi,extent):
    lows=[max(a,0) for a in lo];highs=[min(a,n) for a,n in zip(hi,extent)]
    return all(a<=b for a,b in zip(lows,highs)) and any(a<n for a,n in zip(lows,extent))


def prove_geometry_visibility(extent,depth):
    sizes=(7,15,31);counts=tuple((n+1+depth+b-1)//b for n,b in zip(extent,sizes));slots,frontier=rank_slots(counts)
    d=plan(counts,slots,frontier);spans=windows(counts,extent,depth);ny,nx=counts[1:]
    pubs=[[] for _ in slots];field_checks=mutable_checks=boxes=0
    def origin(t):return (t//(ny*nx)*7,(t//nx)%ny*15,t%nx*31)
    def active(t,n):return spans[t][0]<=n<=spans[t][1]
    def publication(t,n):
        for counter,visibility in pubs[t]:
            if counter>=n:return visibility
        raise AssertionError(('Missing fullstep publication',t,n))
    def event(t,n):return 1<<(t*depth+n-1)
    for t in d['jobs']:
        first,last=spans[t];o=origin(t);history=0;own=0
        if first<=last:
            for n in range(first,last+1):
                deps=d['raw_predecessors'][t] if n==first else {p for p in d['raw_predecessors'][t] if d['ranks'][t]-d['ranks'][p]==1}
                for p in deps:history|=publication(p,n)
                low=tuple(a-n for a in o);high=tuple(a+b-1 for a,b in zip(low,sizes))
                for mask in range(1,8):
                    bits=(mask&1,(mask>>1)&1,(mask>>2)&1)
                    if any(q and a==0 for q,a in zip(bits,(t//(ny*nx),(t//nx)%ny,t%nx))):continue
                    p=t-bits[0]*ny*nx-bits[1]*nx-bits[2]
                    # Current packet coordinates: a zero local coordinate is
                    # the current low ghost; all free axes span owned volume.
                    gl=tuple(a-1 if q else a for a,q in zip(low,bits));gh=tuple(a-1 if q else b for a,b,q in zip(low,high,bits))
                    if any_support(gl,gh,extent):
                        if not active(p,n) or not(history&event(p,n)):raise AssertionError(('Unready supported field packet',extent,depth,t,p,n,mask))
                        field_checks+=1
                        # Every intermediary coordinate chooses current or
                        # diagonal producer origin; separable overlap holds.
                        for sub in range(1,8):
                            if sub&~mask:continue
                            mid=t-(sub&1)*ny*nx-((sub>>1)&1)*nx-((sub>>2)&1)
                            if not active(mid,n):raise AssertionError(('Inactive current-payload intermediary',t,p,mid,n))
                            boxes+=1
                    if n>first and n>1:
                        # A newly-owned point is a valid prior low ghost. Its
                        # prior-owner write is already in accumulated history.
                        fl=tuple(a if q else a+1 for a,q in zip(low,bits));fh=tuple(a if q else b for a,b,q in zip(low,high,bits))
                        if any_support(fl,fh,extent):
                            if not active(p,n-1) or not(history&event(p,n-1)):raise AssertionError(('Unready prior mutable owner',extent,depth,t,p,n,mask))
                            mutable_checks+=1
                own|=event(t,n)
                if n%4==0 or n==last:pubs[t].append((n,history|own))
        # Cumulative completion joins all earlier acquisitions/writes; skipped
        # unsupported versions need no payload. This is not an early phase pub.
        pubs[t].append((depth,history|own))
        if any(a[0]>b[0] or a[1]&~b[1] for a,b in zip(pubs[t],pubs[t][1:])):raise AssertionError('Visibility not monotone')
    return {'extent':extent,'depth':depth,'tiles':len(slots),'supported_field_packets':field_checks,'intermediate_boxes':boxes,'prior_mutable_handoffs':mutable_checks,'pass':True}


STARTUP=r'''
static void run_nearest_startup_regression(){
 using namespace beamz::cuda::moving_persistent;
 for(uint32_t first:{1u,4u,13u}){
  int32_t slots[4]={0,0,1,0},abort=0,bad=0;std::vector<uint64_t>tags(8,0);std::vector<uint32_t>ready(8,0);
  auto&c=local_cold;c={};c.slots=slots;c.abort_global=&abort;c.bad=&bad;c.tags=tags.data();c.ready=ready.data();
  c.timeout_ns=1000000000ULL;c.nz=2;c.ny=2;c.nx=1;c.frontier=2;c.epoch=1;c.tile=3;c.depth=16;c.upstream_floor=0;
  tags[2]=(uint64_t(1)<<32)|1;ready[2]=first;tags[3]=(uint64_t(1)<<32)|2;ready[3]=first;
  // Nearest faces are ready. A delayed DIAGONAL remains mandatory at startup,
  // including positive late-first versions; this is a metadata-only fixture.
  std::thread delayed([&]{std::this_thread::sleep_for(std::chrono::milliseconds(10));Release(tags.data(),uint64_t(1)<<32);Release(ready.data(),first);});
  WaitUpstream<false>(first);
  const bool missing=Acquire(tags.data())!=(uint64_t(1)<<32)||Acquire(ready.data())<first;
  const bool failed=abort||bad||c.abort_local;delayed.join();
  if(missing||failed)throw std::logic_error("ALL7 startup omitted delayed diagonal tag/version");
 }
}
'''


def prepare_actual(source,output,negative_startup=False):
    proof=control_prepare(source,output);h=output/'actual-persistent-control.cc';text=h.read_text()
    inc=output/'actual_persistent_control.inc';control=inc.read_text()
    needle='const uint32_t seen=WaitProducer(producer,owner,step);'
    if control.count(needle)!=1:raise ValueError('Actual nearest query site changed')
    inc.write_text(control.replace(needle,'if constexpr(NearestOnly)NearestVisit();'+needle))
    # CPU instrumentation only; relaxed counters add no payload HB edge.
    text=text.replace('#include "actual_persistent_control.inc"',
        'static std::atomic<size_t>actual_nearest_queries{0};\nstatic void NearestVisit(){actual_nearest_queries.fetch_add(1,std::memory_order_relaxed);}\n#include "actual_persistent_control.inc"')
    startup_check=r'''WaitUpstream(1);if(c.abort_local)break;
    {const int t=c.tile,z=t/(counts[1]*counts[2]),y=(t/counts[2])%counts[1],x=t%counts[2];
     for(int bits=1;bits<8;++bits){const int dz=bits&1,dy=(bits>>1)&1,dx=(bits>>2)&1;if(z<dz||y<dy||x<dx)continue;
      int p=t-dz*counts[1]*counts[2]-dy*counts[2]-dx;
      if(Acquire(tags.data()+owner(p))!=((uint64_t(epoch)<<32)|uint32_t(p)))throw std::logic_error("Startup base tag not validated");
     }}'''
    text=text.replace('WaitUpstream(1);if(c.abort_local)break;',startup_check)
    fixture=STARTUP.replace('WaitUpstream<false>(first);','WaitUpstream<true>(first);') if negative_startup else STARTUP
    text=text.replace('int main(){',fixture+'\nint main(){').replace(' try{\n  for(auto counts:', ' try{\n  run_nearest_startup_regression();\n  for(auto counts:')
    text=text.replace('if(TestedPublicationSteps<16&&overlaps==0)',
        'if(actual_nearest_queries==0)throw std::logic_error("nearest3 later path was not exercised");\n  if(TestedPublicationSteps<16&&overlaps==0)')
    text=text.replace('<<completed_cases<<','<<completed_cases<<",\\\"actual_nearest_queries\\\":"<<actual_nearest_queries.load()<<')
    h.write_text(text);proof.update({'harness_sha256':hashlib.sha256(h.read_bytes()).hexdigest(),'control_extract_sha256':hashlib.sha256(inc.read_bytes()).hexdigest(),'ALL7_startup_versions':[1,4,13],'negative_nearest_startup':negative_startup,'all7_payload_checks_retained':True,'CPU_instrumentation':'relaxed count only at actual NearestOnly query; positive count mandatory'})
    (output/'nearest-controller-plan.json').write_text(json.dumps(proof,indent=2)+'\n');return proof


def execute_actual(proof,output,expect_reject=False):
    r=subprocess.run(proof['compiler_command'],capture_output=True,text=True);(output/'compile.log').write_text(r.stdout+r.stderr)
    if r.returncode:raise RuntimeError('Nearest controller CPU compile failed')
    r=subprocess.run([proof['compiler_command'][-1]],capture_output=True,text=True,timeout=120);(output/'run.log').write_text(r.stdout+r.stderr)
    if expect_reject:
        if not r.returncode or 'ALL7 startup omitted delayed diagonal' not in r.stderr:raise RuntimeError('Missing-startup negative was not rejected')
        proof.update(executed=True,expected_negative_rejected=True)
    else:
        if r.returncode:raise RuntimeError('Nearest actual controller proof rejected')
        proof.update(executed=True,positive_result=json.loads(r.stdout))
    (output/'nearest-controller-proof.json').write_text(json.dumps(proof,indent=2)+'\n');return proof


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--execute',action='store_true');p.add_argument('--negative-startup',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    geometry=[prove_geometry_visibility(n,d) for n in [(25,25,25),(25,31,47),(64,96,128),(96,96,96)] for d in (4,16)]
    (a.output/'geometry-visibility-proof.json').write_text(json.dumps({'cases':geometry,'all_passed':True},indent=2)+'\n')
    proof=prepare_actual(a.source,a.output,a.negative_startup)
    if a.execute:proof=execute_actual(proof,a.output,a.negative_startup)
    print(json.dumps({'executed':proof.get('executed',False),'negative_rejected':proof.get('expected_negative_rejected'),'actual':proof.get('positive_result'),'geometry_cases':len(geometry),'field_packets':sum(x['supported_field_packets'] for x in geometry),'prior_mutable_handoffs':sum(x['prior_mutable_handoffs'] for x in geometry)}))


if __name__=='__main__':main()
