"""Audit staged closed-pipeline numerical ancestry and actual planner/queue."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from scripts.cuda_trace_phase_cycle import block


def validate(candidate,parent):
    candidate,parent=Path(candidate).resolve(),Path(parent).resolve();src=candidate/'source/src'
    proof=json.loads((candidate/'closed-pipeline-compiler-proof.json').read_text());c=(src/'asymmetric_native.cuh').read_text();original=(parent/'src/asymmetric_native.cuh').read_text()
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if sha(src/'asymmetric_native.cuh')!=proof['consumer_sha256'] or sha(parent/'src/asymmetric_native.cuh')!=proof['parent_consumer_sha256']:raise ValueError('Frozen ancestry changed')
    for name,digest in proof['helpers'].items():
        if sha(src/name)!=digest:raise ValueError('Bound helper changed:'+name)
    for name in ('asymmetric_native_cell_generated.cuh','yee_cell_generated.cuh'):
        if (src/name).read_bytes()!=(parent/'src'/name).read_bytes():raise ValueError('Canonical arithmetic changed')
    for name in ('native_moving.cc','native_moving.h'):
        if (src/name).read_bytes()!=(parent/'src'/name).read_bytes():raise ValueError('Frozen material certificate changed')
    extension=(src/'native_extension.cc').read_text()
    if sha(src/'native_extension.cc')!=proof['extension_sha256']:raise ValueError('Captured extension binding changed')
    native_body_count=3
    if proof.get('effect_regions'):
        region=json.loads((candidate/'host-effect-regions-proof/result.json').read_text())
        if not region.get('passed') or not region.get('field_body_exact_inverse') or region['consumer_sha256']!=proof['consumer_sha256'] or region['helpers']!=proof['helpers']:raise ValueError('Actual region body/method/fact proof required')
        native_body_count+=1
    if proof.get('phase_rebinding'):
        phase=json.loads((candidate/'host-phase-rebinding-proof/result.json').read_text())
        if not phase.get('passed') or not phase.get('canonical_region_inverse') or phase['consumer_sha256']!=proof['consumer_sha256'] or phase['helpers']!=proof['helpers'] or phase['model_sha256']!=sha(Path(__file__).with_name('prove_cuda_phase_rebinding.py')):raise ValueError('Actual single-bank providers/methods/model proof required')
    if proof.get('double_shared'):
        double=json.loads((candidate/'host-double-shared-proof/result.json').read_text())
        if not double.get('passed') or not double.get('canonical_region_inverse') or not double.get('same_family_center_only_model') or double['consumer_sha256']!=proof['consumer_sha256'] or double['helpers']!=proof['helpers'] or double['model_sha256']!=sha(Path(__file__).with_name('prove_cuda_double_shared.py')):raise ValueError('Actual shared in-place dependency/provider/profile/physics proof required')
    if proof.get('late_startup_neighbors')==7:
        if proof.get('front_startup'):
            kernel=(src/'native_closed_pipeline_kernel.cuh').read_text()
            if kernel.count('if(first>1&&first<=last)WaitParents(p,first-1,true);')!=1 or c.count('closed_pipeline::WaitParents(p,prev);')!=native_body_count:raise ValueError('Hoisted all7startup must precede native register bodies')
        elif c.count('closed_pipeline::WaitParents(p,prev,true);')!=3:raise ValueError('All3native bodies require all7late-startup readiness')
    if proof.get('program_fact'):
        for required in ('moving_closed_cpml_uniform_=certificate.supported;',
                         'if((policy==1||policy==5)&&(!certificate.supported||!certificate.admitted[t]))moving_closed_cpml_uniform_=false;',
                         'closed.cpml_uniform=moving_closed_cpml_uniform_;',
                         'moving_closed_active_=moving_closed_cpml_uniform_&&',
                         'append_bytes(&closed.cpml_uniform,sizeof(int));'):
            if extension.count(required)!=1:raise ValueError('Explicit allPolicy1/5 captured certificate guard required:'+required)
    for signature in ('void UniformBody(','void FixedCubeCachedBody(','void SourcePendingPairedBody('):
        body=block(c,signature)[2]
        if proof.get('distal_disposal'):
            if 'Paired' in signature:
                ack='\n   if constexpr(Pipeline){if(pair==0)closed_pipeline::RetireDistalPacket(p,n);}\n'
                if body.count(ack)!=1 or '// Ppair' not in body[:body.index(ack)].splitlines()[-1]:raise ValueError('Distal source retirement must follow first Ppair barrier')
            else:
                ack='\n  if constexpr(Pipeline)closed_pipeline::RetireDistalPacket(p,n);'
                if body.count(ack)!=1 or '// B1' not in body[:body.index(ack)].splitlines()[-1]:raise ValueError('Distal retirement must follow B1 import join')
            body=body.replace(ack,'')
        if proof.get('dead_face_discard'):
            # Check exact placement before removing the liveness bindings.
            init=' if constexpr(Pipeline)closed_pipeline::InitializePackets(p);\n'
            if body.count(init)!=1 or body.index(init)>body.index('p.owner_tags[owner]='):raise ValueError('Packet counters must initialize before owner publication')
            body=body.replace(init,'')
            if 'Paired' in signature:
                ack='\n   if constexpr(Pipeline){if(pair==0)closed_pipeline::AckPacketsUpTo(p,n);}\n'
                if body.count(ack)!=1 or '// Ppair' not in body[:body.index(ack)].splitlines()[-1]:raise ValueError('Source ACK must follow first Ppair barrier')
            else:
                ack='\n  if constexpr(Pipeline)closed_pipeline::AckPacketsUpTo(p,n);'
                if body.count(ack)!=1 or '// B1' not in body[:body.index(ack)].splitlines()[-1]:raise ValueError('Uniform ACK must follow B1 import join')
            body=body.replace(ack,'').replace('closed_pipeline::PublishPacket(p,n);','closed_pipeline::Publish(n);')
        body=body.replace('const int tile=Pipeline?closed_pipeline::Tile():tiles[blockIdx.x],','const int tile=tiles[blockIdx.x],')
        body=body.replace('const int owner=Pipeline?closed_pipeline::Owner(p,tile):(rank&3)*p.frontier+p.slots[tile];','const int owner=(rank&3)*p.frontier+p.slots[tile];')
        body=body.replace('const int owner=Pipeline?closed_pipeline::Owner(p,producer):((rank-dz-dy-dx)&3)*p.frontier+p.slots[producer];','const int owner=((rank-dz-dy-dx)&3)*p.frontier+p.slots[producer];')
        body=body.replace('(Pipeline?__ldcg(p.owner_tags+owner):p.owner_tags[owner])','p.owner_tags[owner]')
        body=body.replace('\n if constexpr(Pipeline){if(!threadIdx.x)closed_pipeline::Started(p);}','')
        body=body.replace('\n  if constexpr(Pipeline){closed_pipeline::Skipped(prev);closed_pipeline::WaitParents(p,prev);if(closed_pipeline::Aborted())return;}','')
        body=body.replace('\n  if constexpr(Pipeline){closed_pipeline::Skipped(prev);closed_pipeline::WaitParents(p,prev,true);if(closed_pipeline::Aborted())return;}','')
        body=body.replace('  if constexpr(Pipeline){closed_pipeline::WaitParents(p,n);if(closed_pipeline::Aborted())return;}\n','')
        body=body.replace(' if constexpr(Pipeline)closed_pipeline::Publish(n);\n ','')
        body=body.replace('ActiveAxes,CachedProfiles,Pipeline>env','ActiveAxes,CachedProfiles>env').replace('FixedCubeCachedEnvironment<N,ActiveAxes,Pipeline>env','FixedCubeCachedEnvironment<N,ActiveAxes>env')
        body=body.replace('constexpr int T=BlockT?BlockT:((Fast||(Layout==1&&(Policy==1||Policy==2)&&IsotropicH&&UniformE))?1024:512),','constexpr int T=(Fast||(Layout==1&&(Policy==1||Policy==2)&&IsotropicH&&UniformE))?1024:512,')
        if body!=block(original,signature)[2]:raise ValueError('Exact arithmetic loop inverse failed:'+signature)
    for signature in ('float Finish(','FieldPacket ImportPacket(','void Observe(','int PsiOffset(','float Difference(','float Coefficient('):
        if block(c,signature)[2]!=block(original,signature)[2]:raise ValueError('Arithmetic/import helper changed:'+signature)
    directory=candidate/'host-closed-pipeline-proof';directory.mkdir(exist_ok=False)
    header=(src/'native_closed_pipeline_control.cuh').read_text()
    a,b,_=block(header,'volatile Cold& C()');header=header[:a]+'volatile Cold& C(){static thread_local Cold cold;return cold;}'+header[b:]
    a,b,_=block(header,'uint64_t Timer()');header=header[:a]+'uint64_t Timer(){return std::chrono::steady_clock::now().time_since_epoch().count();}'+header[b:]
    if proof.get('phase_grouped_worker'):
        for signature,replacement in {
            'void Join(':'void Join(){std::atomic_thread_fence(std::memory_order_seq_cst);}',
            'float* Frame(':'float* Frame(){static thread_local float storage[2*GroupStride/sizeof(float)];return storage+Group()*GroupStride/sizeof(float);}',
            'float* CommonProfiles(':'float* CommonProfiles(){static thread_local float profiles[576];return profiles;}',
        }.items():
            a,b,_=block(header,signature);header=header[:a]+replacement+header[b:]
    if 'int CasAcqRel(' in header:
        replacements={
            'int Acquire(':'int Acquire(const int32_t*pointer){return std::atomic_ref<int>(*const_cast<int*>(pointer)).load(std::memory_order_acquire);}',
            'void Release(':'void Release(int32_t*pointer,int value){std::atomic_ref<int>(*pointer).store(value,std::memory_order_release);}',
            'int CasAcqRel(':'int CasAcqRel(int32_t*pointer,int expected,int desired){std::atomic_ref<int>(*pointer).compare_exchange_strong(expected,desired,std::memory_order_acq_rel,std::memory_order_acquire);return expected;}',
            'int AddAcqRel(':'int AddAcqRel(int32_t*pointer,int increment){return std::atomic_ref<int>(*pointer).fetch_add(increment,std::memory_order_acq_rel);}',
            'int AddRelaxed(':'int AddRelaxed(int32_t*pointer,int increment){return std::atomic_ref<int>(*pointer).fetch_add(increment,std::memory_order_relaxed);}',
            'int AcquireBlock(':'int AcquireBlock(const volatile int*pointer){return std::atomic_ref<int>(*const_cast<int*>(pointer)).load(std::memory_order_acquire);}',
            'void ReleaseBlock(':'void ReleaseBlock(volatile int*pointer,int value){std::atomic_ref<int>(*const_cast<int*>(pointer)).store(value,std::memory_order_release);}'
        }
        for signature,replacement in replacements.items():
            a,b,_=block(header,signature);header=header[:a]+replacement+header[b:]
    rows=[];planner=(src/'native_closed_cohorts.h').read_text();driver=Path(__file__).with_name('prove_cuda_closed_pipeline_host.cc').read_text()
    for name in ('actual','missing-old-reader','premature-export-internal','missing-completion-count'):
        folder=directory/name;folder.mkdir();h=planner
        if name=='missing-old-reader':h=h.replace('retire.insert(old+dz*ny*nx+dy*nx+dx);','(void)old;')
        if name=='premature-export-internal':h=h.replace('if(last_reader[t]==cohort)internal_frontier','last_reader[t]=cohort;if(last_reader[t]==cohort)internal_frontier')
        if name=='missing-completion-count':h=h.replace('++q.initial_pending[t];++q.finish_edges;finish_children[parent].push_back(t);','++q.finish_edges;finish_children[parent].push_back(t);')
        (folder/'native_closed_cohorts.h').write_text(h);(folder/'control.h').write_text(header);cc=folder/'proof.cc';cc.write_text(driver);binary=folder/'proof'
        defines=['-DBEAMZ_READY_FIFO=1'] if proof.get('queue_fifo') else []
        r=subprocess.run(['g++','-std=c++20','-O2','-pthread','-fsanitize=address,undefined',*defines,'-I'+str(folder),'-I'+str(src),str(cc),str(src/'native_moving.cc'),'-o',str(binary)],capture_output=True,text=True)
        if r.returncode:raise RuntimeError('Host compile failed:'+name+'\n'+r.stderr)
        run=subprocess.run([str(binary)],capture_output=True,text=True,timeout=300)
        if (run.returncode!=0 if name=='actual' else run.returncode==0):raise RuntimeError('Unexpected host result:'+name+'\n'+run.stdout+run.stderr)
        rows.append(dict(name=name,returncode=run.returncode,stdout=run.stdout,stderr=run.stderr[-2000:]))
    result=dict(passed=True,consumer_sha256=proof['consumer_sha256'],helpers=proof['helpers'],rows=rows,canonical_headers_byte_identical=True,exact_native_loop_inverse=True,unchanged_arithmetic_import_helpers=True,
                scope='Actual staged schedule-first allocator/planner, all7reader typed completion CSR, material compaction invariance, exactGPUmetadata bytes, actual unique-nodeLIFO queue/control under concurrentCPUworkers; explicit PTX atomic helpers emulated with matching C++ acquire/release scopes; ASan/UBSan and3negativecontrols. Frozen material-certificate admission bound to every Policy1/5 trajectory. PTX/SASS/CUDAmemoryvisibility/1024threadbehavior/FP/performance remain separate gates.')
    (directory/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);a=p.parse_args();print(json.dumps(validate(a.candidate,a.parent)))
