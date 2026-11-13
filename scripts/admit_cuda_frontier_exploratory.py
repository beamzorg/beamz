"""Admit a bounded-resource physical-frontier trial without weakening strict gates.

Architecture exploration may measure a small spill cost rather than infer its
effect from resource counts. This explicitly labeled trial permits at most64B
stack and512B statically reported spill loads/stores per changed moving kernel.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

from scripts.admit_cuda_diamond_exploratory import CUOBJDUMP
from scripts.check_cuda_moving_build import is_moving_kernel,ninja_contract


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def admit(candidate,output,*,box=False,native_macro=False,closed_pipeline=False,effect_regions=False,phase_rebinding=False,legality=None,visibility_proof=None):
    candidate,output=Path(candidate).resolve(),Path(output).resolve()
    if output.exists():raise ValueError("New research admission directory required")
    if sum((box,native_macro,closed_pipeline,effect_regions,phase_rebinding))>1:raise ValueError('One explicit architectural admission mode required')
    compiler=json.loads((candidate/("closed-pipeline-compiler-proof.json" if closed_pipeline or effect_regions or phase_rebinding else "macro-pipeline-compiler-proof.json" if native_macro else "box-warp-compiler-proof.json" if box else "physical-frontier-compiler-proof.json")).read_text())
    stack_budget,spill_budget=(8,64) if phase_rebinding else (16,128) if effect_regions else (8,128) if closed_pipeline else (32,128) if native_macro else (256,8192) if box else (64,512)
    if phase_rebinding:
        phase=json.loads((candidate/'host-phase-rebinding-proof/result.json').read_text())
        variant=(compiler.get('phase_threads'),compiler.get('phase_resident_target'),compiler.get('phase_shared_bytes'))
        if compiler.get('paired_handoff'):
            if variant!=(512,2,101376) or compiler.get('phase_group_count')!=2 or compiler.get('phase_group_stride')!=49536 or compiler.get('paired_packet_epochs')!=5 or compiler.get('phase_named_barriers')!=[1,2]:raise ValueError('Exact paired phase handoff architecture required')
            if not visibility_proof:raise ValueError('Actual paired GPU memory/ACK/barrier visibility proofs required')
            folder=Path(visibility_proof);visibility=json.loads((folder/'manifest.json').read_text())
            if visibility['helpers']!=compiler['helpers'] or digest(visibility['binary'])!=visibility['sha256'] or digest(folder/'stress.cu')!=visibility['source_sha256']:raise ValueError('Actual immutable paired GPU stress binding required')
            for sanitizer in ('memcheck','racecheck','synccheck'):
                checked=json.loads((folder/('execution-'+sanitizer+'.json')).read_text())
                if checked.get('returncode')!=0 or not checked.get('GPU_executed') or not checked.get('result',{}).get('passed') or checked.get('sanitizer')!=sanitizer or checked.get('binary_sha256')!=visibility['sha256'] or checked.get('helpers')!=compiler['helpers']:raise ValueError('Passing actual paired '+sanitizer+' result required')
            paired=json.loads((candidate/'host-closed-pipeline-proof/result.json').read_text())
            if not paired.get('paired_handoff') or not paired.get('actual_local_geometry_and_guards') or paired.get('paired_phase_model_sha256')!=digest(Path(__file__).with_name('prove_cuda_paired_phase_handoff.py')):raise ValueError('Actual paired controller/geometry/canonical model proof required')
            stack_budget,spill_budget=96,2048
        elif compiler.get('phase_full_profile_cache'):
            if variant!=(1024,1,52960) or compiler.get('phase_profile_words')!=864:raise ValueError('Exact full-profile single-bank architecture required')
            stack_budget,spill_budget=48,64
        elif compiler.get('phase_grouped_worker'):
            if variant!=(512,2,101312) or compiler.get('phase_group_count')!=2 or compiler.get('phase_group_stride')!=49504 or compiler.get('phase_named_barriers')!=[1,2]:raise ValueError('Exact isolated two-group architecture required')
            stack_budget,spill_budget=64,1024
        elif variant!=(512,2,49504):raise ValueError('Exact original two-CTA phase architecture required')
        if not compiler.get('phase_single_register_bank') or not phase.get('passed') or not phase.get('canonical_region_inverse') or phase['consumer_sha256']!=compiler['consumer_sha256'] or phase['helpers']!=compiler['helpers'] or phase['model_sha256']!=digest(Path(__file__).with_name('prove_cuda_phase_rebinding.py')):raise ValueError('Actual single-bank phase/provider/physics/canonical model proof required')
    if effect_regions:
        region=json.loads((candidate/'host-effect-regions-proof/result.json').read_text())
        if not compiler.get('effect_regions') or not compiler.get('effect_program_specialization') or not compiler.get('single_kernel') or not region.get('passed') or not region.get('field_body_exact_inverse') or region['consumer_sha256']!=compiler['consumer_sha256'] or region['helpers']!=compiler['helpers']:raise ValueError('Actual program-specialized region proof required')
    if closed_pipeline or effect_regions or phase_rebinding:
        host=json.loads((candidate/'host-closed-pipeline-proof/result.json').read_text());startup=json.loads((candidate/'startup-controller-proof-v1/result.json').read_text())
        model=json.loads(Path(legality).read_text()) if legality else {}
        if not host.get('passed') or host.get('consumer_sha256')!=compiler['consumer_sha256'] or host.get('helpers')!=compiler['helpers'] or not startup.get('passed') or startup.get('control_sha256')!=compiler['helpers']['native_closed_pipeline_control.cuh']:
            raise ValueError('Actual closed planner/queue ANDall7startup controller proof required')
        if not model.get('passed') or not model.get('negative') or model.get('positive',{}).get('ready_mode')!='startup7' or model.get('model_sha256')!=digest(Path(__file__).with_name('prove_cuda_partial_plane_pipeline.py')):
            raise ValueError('Exact clippedcorner negative/fixedcanonical-version proof required')
    if native_macro:
        host=json.loads((candidate/'host-macro-pipeline-proof/result.json').read_text())
        model=json.loads(Path(legality).read_text()) if legality else {}
        if not compiler.get('native_banked') or compiler.get('macro_count') not in (2,4,8) or not host.get('passed') or host.get('consumer_sha256')!=compiler['consumer_sha256']:
            raise ValueError('Actual native macro compiler and CPU binding proof required')
        if not model.get('passed') or model.get('schema')!='beamz.native-banked-macro-pipeline.v1' or len(model.get('negative_controls',{}))!=3 or model.get('source_sha256')!=digest(Path(__file__).with_name('prove_cuda_native_macro_pipeline.py')):
            raise ValueError('Exact native banks+compact-ring version proof required')
    if box:
        if not compiler.get("late_addresses") or compiler.get("volatile_addresses") or compiler.get("cached_shared_bytes",0)>101376:
            raise ValueError("Explicit MOV-region box trial within shared-memory capacity required")
    expected=compiler.get("expected_kernels",68)
    if expected not in ((75,) if effect_regions or phase_rebinding else (81,85) if closed_pipeline else (68,69)):raise ValueError("Explicit architectural kernel inventory required")
    if not box and not native_macro and not closed_pipeline and not effect_regions and not phase_rebinding and compiler["trail_payload_accesses"]!=0:raise ValueError("Physical-frontier compiler identity required")
    if digest(candidate/"source/src/asymmetric_native.cuh")!=compiler["consumer_sha256"]:raise ValueError("Compiler binding changed")
    if box and digest(candidate/"source/src/trace_tile_generated.cuh")!=compiler["source_sha256"]:raise ValueError("Generated box binding changed")
    staged=json.loads((candidate/"staged-source.json").read_text())
    if any(digest(candidate/"source"/name)!=sha for name,sha in staged["source_hashes"].items()):raise ValueError("Compiled source changed")
    build=candidate/"build";module=build/"_cuda_native.cpython-311-x86_64-linux-gnu.so"
    contract=ninja_contract((build/"build.ninja").read_text(),(build/"CMakeFiles/rules.ninja").read_text())
    usage=subprocess.run([CUOBJDUMP,"--dump-resource-usage",str(module)],capture_output=True,text=True,check=True).stdout
    rows=[]
    def allowed(name):return bool(re.search(r"moving_native\d+AsymmetricClosedStepILi(?:4|16)ELi6E" if effect_regions or phase_rebinding else r"moving_native\d+AsymmetricClosedStepILi(?:4|16)ELi1E" if closed_pipeline else r"moving_native\d+Asymmetric" if box or native_macro else r"moving_native\d+Asymmetric(?:IL|NoIOUniform1024|NoIOFixedCube)",name))
    for name,fields in re.findall(r" Function ([^\n]+):\n([^\n]+)",usage):
        if not is_moving_kernel(name):continue
        values={k:int(v) for k,v in re.findall(r"\b(REG|STACK|SHARED|LOCAL):(\d+)",fields)}
        if set(values)!={"REG","STACK","SHARED","LOCAL"} or values["LOCAL"] or values["STACK"]>(stack_budget if allowed(name) else 0):raise ValueError("Resource budget exceeded:"+name)
        rows.append(dict(name=name,**values))
    ptxas=[];name=""
    for line in (candidate/"build.log").read_text().splitlines():
        if "Compiling entry function" in line:name=line.split("'")[1]
        match=re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads",line)
        if match and is_moving_kernel(name):
            values=list(map(int,match.groups()))
            if values[0]>(stack_budget if allowed(name) else 0) or any(v>(spill_budget if allowed(name) else 0) for v in values[1:]):raise ValueError("Static spill budget exceeded:"+name)
            ptxas.append(dict(name=name,stack=values[0],spill_stores=values[1],spill_loads=values[2]))
    if any(len(r)!=expected or len({x["name"] for x in r})!=expected for r in (rows,ptxas)):raise ValueError("Exact physical-frontier kernel inventory required")
    if not any(r["STACK"] for r in rows):raise ValueError("Use strict admission for zero-stack builds")
    output.mkdir();sha=digest(module);frozen=output.parent/"frozen-execution"/output.name/sha/module.name;frozen.parent.mkdir(parents=True,exist_ok=False);shutil.copy2(module,frozen);frozen.chmod(0o444)
    if digest(frozen)!=sha:raise ValueError("Frozen module changed")
    report=dict(schema="beamz.frontier-exploratory-admission.v1",module=str(frozen),sha256=sha,kernel_count=expected,
                all_moving_zero_stack_local_spill=False,moving_exploratory_stack_budget_bytes=stack_budget,static_spill_load_store_budget_bytes=spill_budget,
                scope=(f"Phase single-register-bank, globalworkers only{stack_budget}Bstack/{spill_budget}Bstaticspillbudget" if phase_rebinding else "Program-specialized effect regions, globalworkers only16Bstack/128Bstaticspillbudget" if effect_regions else "Closed pipeline, NoIOonly8Bstack/128Bstaticspillbudget" if closed_pipeline else "Native-banked macro pipeline" if native_macro else "MOV-region box" if box else "Physical-frontier")+" architectural research only; strict rejection retained; full coupled physics required",strict_rejection_source=str(candidate),
                gpu_executed=False,resources=rows,ptxas=ptxas,compile_contract=contract,compiler=compiler,
                staged_source_manifest_sha256=digest(candidate/"staged-source.json"),producer_sha256=digest(__file__))
    (output/"exploratory-admission.json").write_text(json.dumps(report,indent=2)+"\n")
    manifest={k:report[k] for k in ("module","sha256","kernel_count","all_moving_zero_stack_local_spill","moving_exploratory_stack_budget_bytes","gpu_executed","scope")}
    (output/"frozen-execution-manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return manifest


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--candidate",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--box",action="store_true");p.add_argument('--native-macro',action='store_true');p.add_argument('--closed-pipeline',action='store_true');p.add_argument('--effect-regions',action='store_true');p.add_argument('--phase-rebinding',action='store_true');p.add_argument('--legality',type=Path);p.add_argument('--visibility-proof',type=Path);a=p.parse_args();print(json.dumps(admit(a.candidate,a.output,box=a.box,native_macro=a.native_macro,closed_pipeline=a.closed_pipeline,effect_regions=a.effect_regions,phase_rebinding=a.phase_rebinding,legality=a.legality,visibility_proof=a.visibility_proof)))
