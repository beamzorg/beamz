"""Audit native phase/psi lowering, then execute emitted body in host fibers."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from scripts.build_cuda_native_noio_private_psi import region
from scripts.build_cuda_native_noio_psi512 import region512
from scripts.build_cuda_native_noio_shared_psi import shared_region
from scripts.build_cuda_native_noio_partial_shared_psi import partial_region
from scripts.build_cuda_native_noio_owned_psi import owned_region
from scripts.build_cuda_native_noio_rotating_psi import rotating_region,ROTATE
from scripts.build_cuda_native_noio_stationary_owned_psi import stationary_region
from scripts.build_cuda_stationary_source_psi import source_function
from scripts.cuda_native_phase_state_fact import generate_guard
from scripts.cuda_space_time_ir import FIELDS,yee_operator
from scripts.cuda_trace_phase_cycle import block
from scripts.patch_cuda_traced_lifetime_private_psi import POINT


def validate(candidate,parent,*,suffix=''):
    candidate,parent=Path(candidate).resolve(),Path(parent).resolve();src=candidate/'source/src';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    proof=json.loads((candidate/'native-noio-private-psi-compiler-proof.json').read_text());c=(src/'asymmetric_native.cuh').read_text();old=(parent/'src/asymmetric_native.cuh').read_text();body=(src/'native_noio_private_psi.cuh').read_text()
    if sha(src/'asymmetric_native.cuh')!=proof['consumer_sha256'] or sha(src/'native_extension.cc')!=proof['extension_sha256']:raise ValueError('Changed phase/psi compiler binding')
    for name,value in proof['helpers'].items():
        if sha(src/name)!=value:raise ValueError('Changed emitted helper:'+name)
    if (src/'runtime_private_psi_state_fact.h').read_text()!=generate_guard()['source'] or body.count(POINT)!=1:raise ValueError('Exact proven independent byte guard/private state adapter required')
    for name in ('native_moving.cc','native_moving.h','asymmetric_native_cell_generated.cuh','yee_cell_generated.cuh','yee_primitives.cuh'):
        if (src/name).read_bytes()!=(parent/'src'/name).read_bytes():raise ValueError('Changed canonical math/owner ABI:'+name)
    native512=proof.get('threads')==512;shared_psi=proof.get('shared_psi',False);owned_psi=proof.get('owned_shared_psi',False);partial_psi=proof.get('partial_shared_psi',False);rotating=proof.get('rotating_point_code',False);stationary=proof.get('stationary_owned_psi',False);source_only=proof.get('source_only',False);separate_entries=native512 or shared_psi or owned_psi
    for signature in ('struct Environment {','struct FixedCubeCachedEnvironment:','FieldPacket ImportPacket(','void SourcePendingPairedBody(','inline cudaError_t LaunchRanks('):
        if block(c,signature)[2]!=block(old,signature)[2]:raise ValueError('Changed frozen physics/policy/rank scheduling:'+signature)
    source_launcher=block(c,'inline cudaError_t LaunchSourcePending(')[2]
    if source_only:
        if source_launcher!=proof['source_launcher_emitted'] or proof['source_launcher_original']!=block(old,'inline cudaError_t LaunchSourcePending(')[2]:raise ValueError('Source launch fact scope changed')
        native=block(body,'void NativeNoioPrivatePsiBody(')[2];source_body=(src/'native_stationary_source_psi.cuh').read_text()
        if native!=proof['source_function_original'] or source_function(native)!=proof['source_function_emitted'] or block(source_body,'void NativeStationarySourcePsiBody(')[2]!=proof['source_function_emitted']:raise ValueError('Source phase effect motion inverse failed')
    elif source_launcher!=block(old,'inline cudaError_t LaunchSourcePending(')[2]:raise ValueError('Changed frozen source dispatcher')
    if separate_entries:
        for signature in ('void UniformBody(','void FixedCubeCachedBody('):
            if block(c,signature)[2]!=block(old,signature)[2]:raise ValueError('Original68 body changed:'+signature)
        launch=block(c,'inline cudaError_t LaunchNoIO(')[2]
        if launch!=proof['noio_launcher_emitted'] or proof['noio_launcher_original']!=block(old,'inline cudaError_t LaunchNoIO(')[2]:raise ValueError('512 dispatch scope changed')
    elif block(c,'inline cudaError_t LaunchNoIO(')[2]!=block(old,'inline cudaError_t LaunchNoIO(')[2]:raise ValueError('Changed original1024 dispatcher')
    for wrapper in proof['wrapper_bindings']:
        signature=wrapper['signature'];legacy='void NativeNoioLegacyUniformBody(' if 'Uniform' in signature else 'void NativeNoioLegacyFixedCubeBody('
        if block(c,legacy)[2].replace(legacy,signature,1)!=block(old,signature)[2] or block(c,signature)[2]!=wrapper['emitted']:raise ValueError('Fallback inverse/new scope mismatch:'+signature)
    actual_prepare=block(c,'inline bool Prepare(')[2].replace(proof['prepare_guard_emitted'],proof['prepare_guard_original'])
    if actual_prepare!=block(old,'inline bool Prepare(')[2]:raise ValueError('Prepare must change only independent state fact')
    for binding in proof['region_bindings']:
        emitted=stationary_region(binding['phase'],binding['original']) if stationary else rotating_region(binding['phase'],binding['original']) if rotating else owned_region(binding['phase'],binding['original']) if owned_psi else partial_region(binding['phase'],binding['original']) if partial_psi else shared_region(binding['phase'],binding['original']) if shared_psi else region512(binding['phase'],binding['original']) if native512 else region(binding['phase'],binding['original'],proof.get('phase_mask',3))
        if emitted!=binding['emitted'] or body.count(binding['emitted'])!=1:raise ValueError('Canonical phase/provider/private-psi lowering mismatch')
    out=candidate/('host-native-noio-psi-proof'+suffix);out.mkdir(exist_ok=False)
    helpers=['using MstepCompactField=Field;']
    for signature in ('int WrapZ(','int Slot(','int Surface(','bool Contains(','const MstepCompactField& InitialField(','const MstepCompactField& FinalField(','float Initial(','unsigned NativeSupportMask('):helpers.append(block(c,signature)[2])
    helpers.append('struct FieldPacket{float hx,hy,hz,ex,ey,ez;};')
    helpers.append('template<int Depth,bool Fast,bool ReuseTerminal=true>\n'+block(c,'FieldPacket ImportPacket(')[2]);helpers.append(block(c,'void StageIndependentCpmlProfiles(')[2])
    reads=[]
    for u in yee_operator(cpml_axes=()).updates:
        if u.output.field not in FIELDS:continue
        f=FIELDS.index(u.output.field)
        for r in sorted(u.reads):
            if r.field in FIELDS:reads.append((f//3,f,FIELDS.index(r.field),r.version,*r.offset))
    records='static const Read reads[]={'+','.join('{'+','.join(map(str,r))+'}' for r in reads)+'};'
    dispatch=''
    for depth in (4,16):
        dispatch+=f'if(active_depth=={depth}){{switch(active_axis){{'
        target='NativeStationarySourcePsiBody' if source_only else 'NativeNoioPrivatePsiBody'
        for axis in (1,2,4):dispatch+=f'case {axis}:{target}<{depth},0,{axis}>(*active_p,active_tiles);break;'
        if native512:dispatch+=f'default:NativeNoioPrivatePsiBody<{depth},0,-1>(*active_p,active_tiles);break;'
        dispatch+='}}'
    cpu_body=body.replace('asm volatile("mov.b32 %0,%1;":"=r"(step):"r"(n));','step=n;')
    if source_only:cpu_body+='\n'+source_body.replace('asm volatile("mov.b32 %0,%1;":"=r"(step):"r"(n));','step=n;')
    if stationary:cpu_body=cpu_body.replace('#include "native_noio_stationary_owned_psi.cuh"',(src/'native_noio_stationary_owned_psi.cuh').read_text())
    if owned_psi:cpu_body=cpu_body.replace('#include "native_noio_owned_psi.cuh"',(src/'native_noio_owned_psi.cuh').read_text())
    if 'asm volatile' in cpu_body:raise ValueError('Only explicit scalar clock alias may be adapted for host')
    code=Path(__file__).with_name('prove_cuda_native_noio_fibers.cc').read_text().replace('// CANONICAL_READS',records).replace('// ACTUAL_NATIVE_HELPERS','\n'.join(helpers)).replace('// ACTUAL_PHASE_BODY',cpu_body).replace('// ACTUAL_BODY_DISPATCH',dispatch)
    if stationary:
        geometry=r'''
static void stationary_geometry(){
 for(int n=1;n<=64;++n){std::set<int>seen;
  for(int dz=1;dz<=BZ;++dz)for(int dy=1;dy<=BY;++dy)for(int dx=1;dx<=BX;++dx){
   const int d[3]={dz,dy,dx},b[3]={BZ,BY,BX},q[3]={3,4,5};int p[3],r[3],previous[3];bool entering=false;
   for(int a=0;a<3;++a){p[a]=q[a]*b[a]-n-1+d[a];r[a]=(p[a]%b[a]+b[a])%b[a];previous[a]=d[a]==1?b[a]:d[a]-1;entering|=d[a]==1;}
   const int index=NativeStationaryOwnedIndex(dz,dy,dx,NativeStationaryOwnedShift<BZ>(n),NativeStationaryOwnedShift<BY>(n),NativeStationaryOwnedShift<BX>(n));
   require(index==(r[0]*BY+r[1])*BX+r[2]&&index>=0&&index<NativeOwnedPsiPoints&&seen.insert(index).second,"Stationary owned modulo physical identity/bijection");
   const int old_index=NativeStationaryOwnedIndex(previous[0],previous[1],previous[2],NativeStationaryOwnedShift<BZ>(n-1),NativeStationaryOwnedShift<BY>(n-1),NativeStationaryOwnedShift<BX>(n-1));
   require(old_index==index,"Retained point or outgoing highpoint changes stationary slot");
   bool old_in_current=true,old_same=true;
   for(int a=0;a<3;++a){const int old_p=q[a]*b[a]-(n-1)-1+previous[a];old_same&=old_p==p[a];old_in_current&=old_p>=q[a]*b[a]-n&&old_p<q[a]*b[a]-n+b[a];}
   require(entering?(!old_same&&!old_in_current):(old_same&&old_in_current),"Entering cell clobbers a still-current old point");
  }require(seen.size()==NativeOwnedPsiPoints,"Incomplete stationary owned map");
  for(int dz=1;dz<=BZ;++dz)for(int dy=1;dy<=BY;++dy){std::set<int>banks;for(int dx=1;dx<=BX;++dx)banks.insert(NativeStationaryOwnedIndex(dz,dy,dx,NativeStationaryOwnedShift<BZ>(n),NativeStationaryOwnedShift<BY>(n),NativeStationaryOwnedShift<BX>(n))%32);require(banks.size()==31,"Stationary psi row has shared bank collisions");}
 }
}
'''
        code=code.replace('int main(){',geometry+'\nint main(){stationary_geometry();')
    rows=[]
    names=['actual','omit-low-refresh','omit-cache-publication','skip-H-ghost-rebind']
    if stationary:names+=['wrong-stationary-shift','aliased-owned-colors']
    elif owned_psi:names+=['wrong-old-owned-index','premature-owned-publication','aliased-owned-colors']
    elif shared_psi:names+=['aliased-shared-colors']
    if rotating:names+=['omit-H-register-rotation','omit-ghost-register-rotation']
    if source_only:names+=['omit-source-H-finish','omit-source-E-finish','duplicate-source-finish','unguarded-source-finish']
    for name in names:
        folder=out/name;folder.mkdir();text=code
        if name=='omit-low-refresh':text=text.replace('n==first||dz_owned==1||dy_owned==1||dx_owned==1','n==first')
        if name=='omit-cache-publication':
            if owned_psi:text=text.replace('n==Depth||dz_owned==BZ||dy_owned==BY||dx_owned==BX);','false);')
            else:
                prefix='h' if proof.get('phase_mask')==1 else 'e';before=f'history.{prefix}a[j],history.{prefix}b[j]' if not shared_psi else 'psi_cache[PAD+Slot(zs,sy,sx)],psi_cache[2*PAD+Slot(zs,sy,sx)]';text=text.replace(before+',n==first||dz_owned==1||dy_owned==1||dx_owned==1,n==Depth||dz_owned==BZ||dy_owned==BY||dx_owned==BX',before+',n==first||dz_owned==1||dy_owned==1||dx_owned==1,false')
        if name=='skip-H-ghost-rebind':text=text.replace('state[c][j]=frame[c*PAD+Slot(zs,sy,sx)];','state[c][j]=0.f;')
        if name=='aliased-shared-colors':text=text.replace('psi_cache[2*PAD+Slot(zs,sy,sx)]','psi_cache[PAD+Slot(zs,sy,sx)]')
        if name=='wrong-old-owned-index':text=text.replace('NativeOwnedPsiIndex(dz_owned-1,dy_owned-1,dx_owned-1)','NativeOwnedPsiIndex(dz_owned,dy_owned,dx_owned)')
        if name=='premature-owned-publication':text=text.replace('if(Color(slot))b=value;else a=value;','if(Color(slot))b=value;else a=value;const_cast<float*>(previous)[Color(slot)*NativeOwnedPsiPoints+index+BY*BX+BX+1]=value;')
        if name=='aliased-owned-colors':text=text.replace('Color(slot)*NativeOwnedPsiPoints+index','0*NativeOwnedPsiPoints+index')
        if name=='wrong-stationary-shift':text=text.replace('return r?N-r:0;','return r;')
        if name=='omit-H-register-rotation':text=text.replace(ROTATE,'',1)
        if name=='omit-ghost-register-rotation':text=text.replace(ROTATE,'if(zs!=hz&&sy!=hy&&sx!=hx){'+ROTATE+'}',1)
        if name=='omit-source-H-finish':text=text.replace('effects.Finish(c,n,z,y,x,state[c][j])','state[c][j]')
        if name=='omit-source-E-finish':text=text.replace('effects.Finish(c+3,n,z,y,x,state[c][j])','state[c][j]')
        if name=='duplicate-source-finish':text=text.replace('effects.Finish(c,n,z,y,x,state[c][j])','effects.Finish(c,n,z,y,x,effects.Finish(c,n,z,y,x,state[c][j]))')
        if name=='unguarded-source-finish':text=text.replace('env.Valid(c,z,y,x)?effects.Finish','true?effects.Finish')
        cc=folder/'proof.cc';cc.write_text(text);binary=folder/'proof';built=subprocess.run(['g++','-std=c++20','-O2','-fsanitize=undefined','-I'+str(src),f'-DBEAMZ_PSI_FIBER_SOURCE={1 if source_only else 0}',f'-DBEAMZ_PSI_FIBER_THREADS={512 if native512 else 1024}',f'-DBEAMZ_PSI_FIBER_SHARED_BYTES={101376 if owned_psi else 99584 if shared_psi else 55204}',str(cc),'-o',str(binary)],capture_output=True,text=True);(folder/'build.log').write_text(built.stdout+built.stderr)
        if built.returncode:raise RuntimeError('Native emitted body host compile failed:'+str(folder/'build.log'))
        run=subprocess.run([str(binary)],capture_output=True,text=True,timeout=240);(folder/'run.log').write_text(run.stdout+run.stderr)
        if (run.returncode!=0 if name=='actual' else run.returncode==0):raise RuntimeError('Unexpected native phase fiber proof:'+name+run.stdout+run.stderr)
        rows.append(dict(name=name,returncode=run.returncode,stdout=run.stdout,stderr=run.stderr[-2000:]))
    result=dict(passed=True,consumer_sha256=proof['consumer_sha256'],helpers=proof['helpers'],canonical_phase_inverse=True,frozen_fallback_inverse=True,actual_cooperative_fiber_cases=rows,stationary_owned_geometry_checks=64*7*15*31 if stationary else 0,stationary_owned_warp_bank_sets=64*7*15 if stationary else 0,fiber_harness_source_sha256=sha(Path(__file__).with_name('prove_cuda_native_noio_fibers.cc')),actual_positive_cc_sha256=sha(out/'actual/proof.cc'),threads=512 if native512 else 1024,scope='Actual emitted native phase body/POINT/ImportPacket/Slot/profile staging, independently ordered host fibers per CTA barrier, canonical field/version tokens, independent physical CPML recurrences and terminal outputs. Full FP32 arithmetic/CUDA visibility are separate GPU checks. Host ucontext uses UBSan; no claim of ASan support for context switches.')
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);p.add_argument('--suffix',default='');a=p.parse_args();print(json.dumps(validate(a.candidate,a.parent,suffix=a.suffix)))
