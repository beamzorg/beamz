"""Audit actual fused-controller, planner/geometry/physics proofs and host capture."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from scripts.build_cuda_paired_handoff import WORKER
from scripts.cuda_trace_phase_cycle import block


def emulate(header):
    replacements={
        'volatile Cold& C()':'volatile Cold& C(){static thread_local Cold cold[2];return cold[Group()];}',
        'void Join(':'void Join(){std::atomic_thread_fence(std::memory_order_seq_cst);}',
        'float* Frame(':'float* Frame(){static thread_local float storage[101376/4];return storage+Group()*GroupStride/4;}',
        'float* CommonProfiles(':'float* CommonProfiles(){static thread_local float profiles[576];return profiles;}',
        'uint64_t Timer()':'uint64_t Timer(){return std::chrono::steady_clock::now().time_since_epoch().count();}',
        'int Acquire(':'int Acquire(const int32_t*pointer){return std::atomic_ref<int>(*const_cast<int*>(pointer)).load(std::memory_order_acquire);}',
        'void Release(':'void Release(int32_t*pointer,int value){std::atomic_ref<int>(*pointer).store(value,std::memory_order_release);}',
        'int CasAcqRel(':'int CasAcqRel(int32_t*pointer,int expected,int desired){std::atomic_ref<int>(*pointer).compare_exchange_strong(expected,desired,std::memory_order_acq_rel,std::memory_order_acquire);return expected;}',
        'int AddAcqRel(':'int AddAcqRel(int32_t*pointer,int increment){return std::atomic_ref<int>(*pointer).fetch_add(increment,std::memory_order_acq_rel);}',
        'int AddRelaxed(':'int AddRelaxed(int32_t*pointer,int increment){return std::atomic_ref<int>(*pointer).fetch_add(increment,std::memory_order_relaxed);}',
        'int AcquireBlock(':'int AcquireBlock(const volatile int*pointer){return std::atomic_ref<int>(*const_cast<int*>(pointer)).load(std::memory_order_acquire);}',
        'void ReleaseBlock(':'void ReleaseBlock(volatile int*pointer,int value){std::atomic_ref<int>(*const_cast<int*>(pointer)).store(value,std::memory_order_release);}',
    }
    for signature,replacement in replacements.items():
        a,b,_=block(header,signature);header=header[:a]+replacement+header[b:]
    return header


def validate(candidate,parent,planner_proof,model):
    candidate=Path(candidate).resolve();src=candidate/'source/src';parent=Path(parent).resolve();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    proof=json.loads((candidate/'paired-handoff-compiler-proof.json').read_text());c=(src/'asymmetric_native.cuh').read_text();extension=(src/'native_extension.cc').read_text();kernel=(src/'native_closed_pipeline_kernel.cuh').read_text()
    if not proof.get('paired_handoff') or proof.get('paired_packet_epochs')!=5 or sha(src/'asymmetric_native.cuh')!=proof['consumer_sha256'] or sha(src/'native_extension.cc')!=proof['extension_sha256']:raise ValueError('Bound paired compiler/host required')
    for name,value in proof['helpers'].items():
        if sha(src/name)!=value:raise ValueError('Changed helper:'+name)
    for name in ('native_moving.cc','native_moving.h','asymmetric_native_cell_generated.cuh','yee_cell_generated.cuh'):
        if (src/name).read_bytes()!=(parent/'src'/name).read_bytes():raise ValueError('Frozen numerical/certificate source changed:'+name)
    for signature in ('float Finish(','FieldPacket ImportPacket(','void Observe(','int PsiOffset(','float Difference(','float Coefficient('):
        if block(c,signature)[2]!=block((parent/'src/asymmetric_native.cuh').read_text(),signature)[2]:raise ValueError('Canonical helper changed:'+signature)
    expected_worker=WORKER.replace('PhaseRebindingBody<Depth,CubeN,512>','PhaseRebindingBody<Depth,CubeN,256>') if proof.get('phase_threads')==256 else WORKER
    if block(kernel,'void AsymmetricClosedStep(')[2]!=expected_worker:raise ValueError('Actual full-CTA reservation/START/COMPLETE binding changed')
    for required in ('moving_closed_cpml_uniform_=certificate.supported;',
                     'if((policy==1||policy==5)&&(!certificate.supported||!certificate.admitted[t]))moving_closed_cpml_uniform_=false;',
                     'closed.cpml_uniform=moving_closed_cpml_uniform_;','moving_closed_active_=moving_closed_cpml_uniform_&&',
                     'append_bytes(&closed.cpml_uniform,sizeof(int));append_bytes(&closed.original_count,sizeof(int));append_bytes(&closed.node_plane,sizeof(int));',
                     'closed.ready=static_cast<int32_t*>(upload(nullptr,moving_plan_.tile_ids.size()*sizeof(int32_t)));',
                     'closed.finish_children=static_cast<int32_t*>(upload(nullptr,task_bytes*16));'):
        if extension.count(required)!=1:raise ValueError('Actual captured allocation/admission fact required:'+required)
    phase=json.loads((candidate/'host-phase-rebinding-proof/result.json').read_text());storage=json.loads((candidate/'host-paired-storage-proof/result.json').read_text());model=json.loads(Path(model).read_text());planner=json.loads(Path(planner_proof).read_text())
    if not model.get('passed') or model.get('source_sha256')!=sha(Path(__file__).with_name('prove_cuda_paired_phase_handoff.py')) or model.get('supernode_source_sha256')!=sha(Path(__file__).with_name('cuda_paired_supernodes.py')):raise ValueError('Current canonical paired-phase model required')
    for value in (phase,storage):
        if not value.get('passed') or value['consumer_sha256']!=proof['consumer_sha256'] or value['helpers']!=proof['helpers']:raise ValueError('Actual geometry/physics proof binding required')
    if phase.get('paired_phase_model_sha256')!=model['source_sha256'] or not phase.get('canonical_region_inverse') or not storage.get('actual_local_geometry_and_guards'):raise ValueError('Canonical arithmetic inverse and actual local shared provider guards required')
    if not planner.get('passed') or planner.get('paired_planner_sha256')!=sha(src/'native_paired_cohorts.h') or planner.get('native_planner_sha256')!=sha(src/'native_moving.cc') or planner.get('native_header_sha256')!=sha(src/'native_moving.h') or planner.get('closed_planner_sha256')!=sha(src/'native_closed_cohorts.h'):raise ValueError('Exact actual paired planner/metadata proof required')
    out=candidate/'host-closed-pipeline-proof';out.mkdir(exist_ok=False);folder=out/'actual';folder.mkdir()
    (folder/'control.h').write_text(emulate((src/'native_closed_pipeline_control.cuh').read_text()));cc=folder/'proof.cc';cc.write_bytes(Path(__file__).with_name('prove_cuda_paired_control_host.cc').read_bytes());binary=folder/'proof'
    command=['g++','-std=c++20','-O2','-pthread','-fsanitize=address,undefined','-DBEAMZ_GROUP_THREADS='+str(proof['phase_threads']),'-I'+str(folder),'-I'+str(src),str(cc),str(src/'native_moving.cc'),'-o',str(binary)]
    built=subprocess.run(command,capture_output=True,text=True);(folder/'build.log').write_text(built.stdout+built.stderr)
    if built.returncode:raise RuntimeError('Actual paired controller compile failed:'+str(folder/'build.log'))
    run=subprocess.run([str(binary)],capture_output=True,text=True,timeout=180);(folder/'run.log').write_text(run.stdout+run.stderr)
    if run.returncode:raise RuntimeError('Actual paired control failed:'+run.stdout+run.stderr)
    result=dict(passed=True,paired_handoff=True,consumer_sha256=proof['consumer_sha256'],helpers=proof['helpers'],actual_controller=json.loads(run.stdout),planner_proof_sha256=sha(planner_proof),paired_phase_model_sha256=model['source_sha256'],canonical_region_inverse=True,actual_local_geometry_and_guards=True,
                scope='Actual typed supernode queue and per-original readiness with CPUatomic emulation; independent5rank/all7retirement planner and metadata bounds; actual shared providers/guards, canonical physics methods/body inverse, allPolicy1/5 admission and original/node array+key bindings. CUDA visibility/1024thread behavior/full30 FP32 and speed remain separate.')
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);p.add_argument('--planner-proof',type=Path,required=True);p.add_argument('--model',type=Path,required=True);a=p.parse_args();print(json.dumps(validate(a.candidate,a.parent,a.planner_proof,a.model)))
