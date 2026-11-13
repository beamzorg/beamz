"""Verify fact caching leaves the actual native arithmetic/effect body intact."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re

from scripts.cuda_trace_phase_cycle import block
from scripts.cuda_trace_point_facts import verify_lifetime


def strip_facts(body):
    body=body.replace('FixedCubeCachedEnvironment<N,ActiveAxes,true>env','FixedCubeCachedEnvironment<N,ActiveAxes>env')
    body=body.replace(',CachedProfiles,(Layout==1)>env',',CachedProfiles>env')
    for pattern in (r'^\s*constexpr bool UsePointFacts=.*?;\n',r'^\s*uint32_t point_facts\[.*?;\n',r'^\s*const bool point_facts_initial=.*?;\n'):
        body,count=re.subn(pattern,'\n',body,flags=re.MULTILINE)
        if count!=1:raise ValueError('One field-lifetime declaration expected')
    begin,end,_=block(body,'if constexpr(UsePointFacts){');body=body[:begin]+body[end:]
    body=re.sub(r'if constexpr\(UsePointFacts\)env\.point_tag=point_facts\[(?:j|si)\];','',body)
    if 'point_facts' in body or 'point_tag' in body:raise ValueError('Unreviewed point-fact use escaped inverse')
    return body


def normalize(text):return re.sub(r'\s+','',text)


def validate(source,candidate):
    source,candidate=Path(source).resolve(),Path(candidate).resolve()
    proof=json.loads((candidate/'point-facts-compiler-proof.json').read_text())
    original=(source/'src/asymmetric_native.cuh').read_text();text=(candidate/'source/src/asymmetric_native.cuh').read_text()
    if hashlib.sha256(original.encode()).hexdigest()!=proof['parent_consumer_sha256'] or hashlib.sha256(text.encode()).hexdigest()!=proof['consumer_sha256']:raise ValueError('Native fact source identities changed')
    bodies=[]
    for signature in ('void UniformBody(','void FixedCubeCachedBody(','void SourcePendingPairedBody('):
        old=block(original,signature)[2];new=strip_facts(block(text,signature)[2])
        if normalize(old)!=normalize(new):raise ValueError('Arithmetic/effect/window/barrier body changed:'+signature)
        bodies.append(signature)
    unchanged=[]
    for cls in ('struct Environment {','struct FixedCubeCachedEnvironment:'):
        old=block(original,cls)[2];new=block(text,cls)[2]
        for signature in ('int PsiOffset(','float LoadLocal(','void StoreLocal(','float Coefficient('):
            if block(old,signature)[2]!=block(new,signature)[2]:raise ValueError('Nonpredicate point effect changed:'+signature)
            unchanged.append(cls+signature)
        if cls=='struct Environment {' and block(old,'float Finish(')[2]!=block(new,'float Finish(')[2]:raise ValueError('Source/observation finish changed')
    for signature in ('void Observe(','FieldPacket ImportPacket(','inline bool Prepare(','inline cudaError_t LaunchRanks(','int BeamzEnqueueNativeAsymmetric('):
        if block(original,signature)[2]!=block(text,signature)[2]:raise ValueError('Global memory/dispatch effect changed:'+signature)
        unchanged.append(signature)
    for filename in ('asymmetric_native_cell_generated.cuh','native_moving.h'):
        if (source/'src'/filename).read_bytes()!=(candidate/'source/src'/filename).read_bytes():raise ValueError('Canonical physics/geometry header changed:'+filename)
    extension=(candidate/'source/src/native_extension.cc').read_text().replace(' module.attr("moving_cached_point_facts")=true;\n','')
    if extension!=(source/'src/native_extension.cc').read_text():raise ValueError('Native memory/clock runtime changed')
    lifetimes=[verify_lifetime(cache=(8,16,32),owned=(7,15,31),shape=(64,96,128),origin=origin,first=first,steps=35)
               for first in (1,4,13) for origin in ((0,0,0),(45,75,93),(60,90,122))]
    result=dict(passed=True,consumer_sha256=proof['consumer_sha256'],inverse_native_bodies=bodies,unchanged_effect_helpers=unchanged,
                native_runtime_unchanged_except_capability_marker=True,canonical_math_and_host_geometry_headers_unchanged=True,lifetimes=lifetimes,
                scope='Exact whitespace-normalized inverse of fact declarations/bindings; all native numerical operations, source/raw/psi effects, global publications, windows, shared barriers and runtime match frozen control. Independent8x16x32 physical-point lifetime checks; GPU results separate.')
    (candidate/'point-facts-binding-proof.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);a=p.parse_args();print(json.dumps(validate(a.source,a.candidate)))
