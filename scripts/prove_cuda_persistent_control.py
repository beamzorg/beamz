"""Prepare/execute actual persistent-control functions with CPU atomic_ref.

The generated header's protocol functions are extracted bytewise. Only the
shared-state accessor and GPU timer are replaced by thread_local/steady_clock;
CUDA atomics map to std::atomic_ref. No GPU or FDTD arithmetic is emulated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


STARTUP_ZERO_REGRESSION=r'''
static void run_startup_zero_regression(){
 using namespace beamz::cuda::moving_persistent;
 int32_t slots[4]={0,0,1,0},abort=0,bad=0;
 std::vector<uint64_t>tags(8,0);std::vector<uint32_t>ready(8,0);
 auto&c=local_cold;c={};c.slots=slots;c.abort_global=&abort;c.bad=&bad;
 c.tags=tags.data();c.ready=ready.data();c.timeout_ns=1000000000ULL;
 c.nz=2;c.ny=2;c.nx=1;c.frontier=2;c.epoch=1;c.tile=3;c.depth=32;c.upstream_floor=0;
 tags[2]=(uint64_t(1)<<32)|1; // FIRST checked producer is claimed, ready0.
 std::thread delayed([&]{
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  Release(tags.data()+3,(uint64_t(1)<<32)|2);
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  Release(tags.data()+0,uint64_t(1)<<32);
 });
 WaitUpstream(0);
 const bool missing=Acquire(tags.data()+3)!=((uint64_t(1)<<32)|2)||Acquire(tags.data()+0)!=(uint64_t(1)<<32);
 const bool failed=bad||abort||c.abort_local||c.upstream_floor!=0;
 delayed.join();
 if(missing||failed)throw std::logic_error("Startup ready0 returned before ALL delayed parent tags were acquired");
}
'''


def prepare(source, output):
    source,output=source.resolve(),output.resolve();output.mkdir(parents=True,exist_ok=True)
    path=source/"src/native_persistent.cuh";text=path.read_text()
    start=text.index("struct Cold {")
    marker="template<bool Fast,int Policy,bool UniformE>\nstruct PersistentPairedEnvironment:" if 'struct PersistentPairedEnvironment:' in text else "struct PersistentEnvironment:"
    end=text.index(marker,start)
    control=text[start:end]
    typed_global='// TRACE_TYPED_GLOBAL_ATOMICS_BEGIN' in control
    if typed_global:
        try:
            from scripts.patch_cuda_native_persistent_global_atomics import shim_global_helpers
        except ModuleNotFoundError:
            from patch_cuda_native_persistent_global_atomics import shim_global_helpers
        control=shim_global_helpers(control)
    a=control.index("constexpr size_t ColdOffset=");b=control.index("template<class T>",a)
    control=control[:a]+"volatile Cold&C();\n"+control[b:]
    timer='__device__ __forceinline__ uint64_t Timer(){uint64_t t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t));return t;}'
    replacement='__device__ __forceinline__ uint64_t Timer(){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}'
    if control.count(timer)!=1:raise ValueError("Actual GPU timer changed")
    control=control.replace(timer,replacement,1)
    (output/"actual_persistent_control.inc").write_text("namespace beamz::cuda::moving_persistent {\n"+control+"}\n")
    harness=output/"actual-persistent-control.cc"
    phase_ready='// TRACE_PHASE_READY_COUNTER:' in text
    harness_template=Path(__file__).with_name('prove_cuda_persistent_phase_control.cc') if phase_ready else Path(__file__).with_suffix('.cc')
    harness_text=harness_template.read_text()
    h_overlap='// TRACE_IMPORT_GATE_BEGIN' in text
    if h_overlap and not phase_ready:
        # This harness has no H arithmetic; its first operation is reading
        # current trails, so exercise the actual relocated wait at that point.
        harness_text=harness_text.replace('for(int step=1;step<=depth;++step){', 'for(int step=1;step<=depth;++step){\n     WaitUpstream(step);if(c.abort_local)break;')
        if '// TRACE_STARTUP_PREVIOUS_VERSION' in text:
            harness_text=harness_text.replace('WaitUpstream(1);','WaitUpstream(0);')
    startup_check=r'''WaitUpstream(0);if(c.abort_local)break;
    // Exact GPU initialization reads every upstream owner tag BEFORE any
    // current-phase wait. Ready0 success must not skip remaining parents.
    {const int t=c.tile,z=t/(counts[1]*counts[2]),y=(t/counts[2])%counts[1],x=t%counts[2];
     for(int bits=1;bits<8;++bits){const int dz=bits&1,dy=(bits>>1)&1,dx=(bits>>2)&1;if(z<dz||y<dy||x<dx)continue;
      const int parent=t-dz*counts[1]*counts[2]-dy*counts[2]-dx;
      if(Acquire(tags.data()+owner(parent))!=((uint64_t(epoch)<<32)|uint32_t(parent)))throw std::logic_error("Startup skipped an unclaimed parent base/tag");
     }
    }'''
    if h_overlap:
        harness_text=harness_text.replace('WaitUpstream(0);if(c.abort_local)break;',startup_check)
        harness_text=harness_text.replace('int main(){',STARTUP_ZERO_REGRESSION+'\nint main(){')
        harness_text=harness_text.replace(' try{\n  for(auto counts:', ' try{\n  run_startup_zero_regression();\n  for(auto counts:')
    early_h_join='// TRACE_H_PUBLICATION_JOIN_BEGIN' in text
    if early_h_join:
        harness_text=harness_text.replace('     WaitUpstream(2*step-1);if(c.abort_local)break;\n','')
        harness_text=harness_text.replace('     verify(0);\n','')
        harness_text=harness_text.replace('     trail[size_t(c.owner)*2*depth+2*(step-1)]=symbol;',
            '     std::this_thread::sleep_for(std::chrono::microseconds((t+step*3)%5));\n     trail[size_t(c.owner)*2*depth+2*(step-1)]=symbol;')
        harness_text=harness_text.replace('PublishAndWait(2*step-1,2*depth);if(c.abort_local)break;',
            'PublishAndWait(2*step-1,2*depth);if(c.abort_local)break;\n     WaitUpstream(2*step-1);if(c.abort_local)break;verify(0);')
    e_group=re.search(r'constexpr int EPublicationSteps=(1|2|4);',text)
    e_group=int(e_group[1]) if e_group else 1
    if e_group>1:
        harness_text=harness_text.replace('PublishAndWait(2*step,2*depth);',f'if(step%{e_group}==0||step==depth)PublishAndWait(2*step,2*depth);')
    group=re.search(r'constexpr int PublicationSteps=(1|2|4|16);',text)
    group=int(group[1]) if group else 1
    if 'upstream_floor' in control:
        harness_text=harness_text.replace('c.abort_local=0;',f'c.abort_local=0;c.depth={"2*depth" if phase_ready else "depth"};c.upstream_floor=0;')
        harness_text=harness_text.replace('static size_t completed_cases=0;', 'static std::atomic<size_t>verified_floor_skips{0};\nstatic size_t completed_cases=0;')
        harness_text=harness_text.replace('PublishAndWait(step,depth);', 'if(step<depth&&uint32_t(step+1)<=c.upstream_floor)++verified_floor_skips;PublishAndWait(step,depth);')
        if phase_ready:
            harness_text=harness_text.replace('PublishAndWait(2*step-1,2*depth);','if(uint32_t(2*step)<=c.upstream_floor)++verified_floor_skips;PublishAndWait(2*step-1,2*depth);')
        harness_text=harness_text.replace('if(TestedPublicationSteps<16&&overlaps==0)', 'if(verified_floor_skips==0)throw std::logic_error("actual acquired floor never authorized a later step");\n  if(TestedPublicationSteps<16&&overlaps==0)')
        harness_text=harness_text.replace('<<completed_cases<<', '<<completed_cases<<",\\\"verified_floor_skips\\\":"<<verified_floor_skips.load()<<')
    harness_text=harness_text.replace('static size_t completed_cases=0;',f'static size_t completed_cases=0;\nconstexpr int TestedPublicationSteps={group};')
    harness.write_text(harness_text)
    command=["c++","-std=c++20","-O2","-pthread","-I"+str(output),"-I"+str(source/"src"),str(harness),"-o",str(output/"actual-persistent-control")]
    proof={"source":str(source),"generated_consumer_sha256":sha(path),"workspace_header_sha256":sha(source/"src/native_persistent.h"),
           "control_extract_sha256":sha(output/"actual_persistent_control.inc"),"harness_sha256":sha(harness),"compiler_command":command,
           "publication_steps":group,"E_publication_steps":e_group,"typed_global_atomic_shim":typed_global,"H_overlap_import_wait":h_overlap,"packed_H_E_phase_ready":phase_ready,"early_H_publication_join":early_h_join,"forced_startup_ready0_delayed_tags":h_overlap,"scope":"Actual control protocol only; no GPU/FDTD/performance claim","executed":False}
    (output/"proof-plan.json").write_text(json.dumps(proof,indent=2)+"\n");return proof


def execute(proof,output):
    output=output.resolve();command=proof["compiler_command"]
    r=subprocess.run(command,capture_output=True,text=True);(output/"compile.log").write_text(r.stdout+r.stderr)
    if r.returncode:raise RuntimeError("CPU control proof compile failed; see compile.log")
    r=subprocess.run([command[-1]],capture_output=True,text=True,timeout=120)
    (output/"run.log").write_text(r.stdout+r.stderr)
    if r.returncode:raise RuntimeError("Actual control proof rejected; see run.log")
    proof.update({"executed":True,"positive_result":json.loads(r.stdout)})
    (output/"proof.json").write_text(json.dumps(proof,indent=2)+"\n");return proof


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--source",required=True,type=Path);p.add_argument("--output",required=True,type=Path);p.add_argument("--execute",action="store_true")
    a=p.parse_args();proof=prepare(a.source,a.output)
    if a.execute:proof=execute(proof,a.output)
    print(json.dumps({"executed":proof["executed"],"output":str(a.output),"positive_result":proof.get("positive_result")}))


if __name__=="__main__":main()
