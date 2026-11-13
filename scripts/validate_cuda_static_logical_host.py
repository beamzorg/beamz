"""Bind fixed logical route/field-profile allocation to emitted CUDA helpers."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from scripts.build_cuda_static_logical import lower_region
from scripts.cuda_space_time_ir import FIELDS,yee_operator
from scripts.cuda_trace_phase_cycle import block


def validate(candidate,parent,model,allocation):
    candidate=Path(candidate).resolve();src=candidate/'source/src';parent=Path(parent).resolve();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    proof=json.loads((candidate/'static-logical-compiler-proof.json').read_text());c=(src/'asymmetric_native.cuh').read_text();body=(src/'native_static_logical.cuh').read_text();old=(parent/'src/asymmetric_native.cuh').read_text()
    if not proof.get('static_logical') or sha(src/'asymmetric_native.cuh')!=proof['consumer_sha256'] or sha(src/'native_extension.cc')!=proof['extension_sha256']:raise ValueError('Bound static logical compiler/host required')
    for name,value in proof['helpers'].items():
        if sha(src/name)!=value:raise ValueError('Changed static helper:'+name)
    for name in ('native_moving.cc','native_moving.h','asymmetric_native_cell_generated.cuh','yee_cell_generated.cuh','yee_primitives.cuh'):
        if (src/name).read_bytes()!=(parent/'src'/name).read_bytes():raise ValueError('Changed original numerical/owner/certificate source:'+name)
    for signature in ('struct Environment {','struct FixedCubeCachedEnvironment:','float Finish(','FieldPacket ImportPacket(','void Observe('):
        if block(c,signature)[2]!=block(old,signature)[2]:raise ValueError('Changed original environment/import/effect helper:'+signature)
    model=json.loads(Path(model).read_text());allocation=json.loads(Path(allocation).read_text())
    if not model.get('passed') or model.get('source_sha256')!=sha(Path(__file__).with_name('prove_cuda_static_logical_cells.py')) or len(model.get('negative_controls',{}))!=5:raise ValueError('Current fixed-cell version/WAR model required')
    if not allocation.get('passed') or allocation.get('source_sha256')!=sha(Path(__file__).with_name('prove_cuda_static_profile_allocation.py')) or allocation.get('parent_source_sha256')!=model['source_sha256'] or len(allocation.get('negative_controls',{}))!=3:raise ValueError('Current dead-plane/independent-profile poison proof required')
    native=block(old,'void UniformBody(')[2];position=0
    for phase,binding in enumerate(proof['region_bindings']):
        marker='{int step;asm volatile("mov.b32 %0,%1;":';a=native.index(marker,position);_,b,region=block(native[a:],marker);position=a+b
        if region!=binding['original'] or lower_region(region,phase)!=binding['emitted'] or body.count(binding['emitted'])!=1:raise ValueError('Canonical scalar arithmetic and exact storage-provider transform required')
    for wrapper in proof['wrapper_bindings']:
        if block(old,wrapper['signature'])[2]!=wrapper['original'] or block(c,wrapper['signature'])[2]!=wrapper['emitted']:raise ValueError('Original policy/thread/material/source dispatch signature binding required')
    for publication in proof['publication_bindings']:
        if body.count(publication['emitted'])!=1 or 'const float value=result[c][j];StaticStoreField(frame,f,zs,sy,sx,value);' not in publication['emitted']:raise ValueError('Every field export must read the private phase output, including reserved Eplanes')
    loop=block(body,'for(int n=first;n<=last;++n)')[2]
    if loop.count('__syncthreads();')!=4 or body.count('float result[3][Slots];')!=1:raise ValueError('One temporary bank and four reader-retirement/publication joins required')
    out=candidate/'host-static-logical-proof';out.mkdir(exist_ok=False)
    constants='\n'.join(line for line in c.splitlines() if line.startswith('constexpr int BZ=') or line.startswith('constexpr int PAD='))
    functions=[]
    for signature in ('int StaticSlot(','bool StaticFieldLive(','int StaticProfileSlot(','void StaticStoreField(','int StaticReadSlot(','void StaticStageProfiles('):
        functions.append(('template<int Phase>\n' if 'ReadSlot' in signature else 'template<int T>\n' if 'StageProfiles' in signature else '')+block(body,signature)[2])
    records=[]
    for u in yee_operator(cpml_axes=(0,1,2)).updates:
        if u.output.field not in FIELDS:continue
        for r in sorted(u.reads):
            if r.field in FIELDS:records.append((FIELDS.index(u.output.field)//3,FIELDS.index(u.output.field),FIELDS.index(r.field),r.version,*r.offset))
    reads='static const Read reads[]={'+','.join('{'+','.join(map(str,row))+'}' for row in records)+'};'
    code=Path(__file__).with_name('prove_cuda_static_storage_host.cc').read_text().replace('// ACTUAL_CONSTANTS_AND_HELPERS',constants+'\n'+'\n'.join(functions)).replace('// CANONICAL_READS',reads)
    rows=[]
    for name in ('actual','wrong-previous-shift','clobber-profile-slots','profile-word-alias'):
        folder=out/name;folder.mkdir();text=code
        if name=='wrong-previous-shift':text=text.replace('const int shift=Phase==0||field>=3?1:0;','const int shift=0;')
        if name=='clobber-profile-slots':a,b,_=block(text,'bool StaticFieldLive(');text=text[:a]+'bool StaticFieldLive(int,int,int,int){return true;}'+text[b:]
        if name=='profile-word-alias':text=text.replace('int StaticProfileSlot(int i){','int StaticProfileSlot(int i){i%=768;')
        cc=folder/'proof.cc';cc.write_text(text);binary=folder/'proof';command=['g++','-std=c++20','-O2','-fsanitize=address,undefined',str(cc),'-o',str(binary)]
        built=subprocess.run(command,capture_output=True,text=True);(folder/'build.log').write_text(built.stdout+built.stderr)
        if built.returncode:raise RuntimeError('Static route/profile proof compile failed:'+str(folder/'build.log'))
        run=subprocess.run([str(binary)],capture_output=True,text=True,timeout=180);(folder/'run.log').write_text(run.stdout+run.stderr)
        if (run.returncode!=0 if name=='actual' else run.returncode==0):raise RuntimeError('Unexpected static route/profile proof:'+name+run.stdout+run.stderr)
        rows.append(dict(name=name,returncode=run.returncode,stdout=run.stdout,stderr=run.stderr[-1500:]))
    result=dict(passed=True,consumer_sha256=proof['consumer_sha256'],helpers=proof['helpers'],canonical_region_inverse=True,frozen_rank_owner_and_material_source=True,actual_field_profile_helpers=rows,model_sha256=model['source_sha256'],allocation_model_sha256=allocation['source_sha256'],scope='Actual fixed field routes/shared store guards/full864profile stage+lookup allocation; canonical scalar arithmetic inverse, original policies/thread/source/material/owner ABI and four shared WAR joins. Physics environment methods/CUDA shared visibility/full30 FP32/runtime are separate.')
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--allocation',type=Path,required=True);a=p.parse_args();print(json.dumps(validate(a.candidate,a.parent,a.model,a.allocation)))
