"""Compose checked finite-horizon/source passes on an immutable native parent."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil

from scripts.cuda_trace_phase_cycle import ROOT,block,sha
from scripts.cuda_trace_cache_exit import prove_exit,transform_body
from scripts.cuda_trace_terminal_reuse import terminal_cut,transform as terminal_transform
from scripts.cuda_trace_bounded_plane_source import transform as bounded_transform


def compose(source,output,expected,passes):
    source,output=Path(source).resolve(),Path(output).resolve()
    if (output.exists() or output==ROOT/'cuda' or ROOT/'cuda' in output.parents
            or source in output.parents or output in source.parents):raise ValueError('New disjoint copied source required')
    if not passes or len(set(passes))!=len(passes):raise ValueError('Each requested pass must occur once')
    original=(source/'src/asymmetric_native.cuh').read_text()
    if sha(original.encode())!=expected:raise ValueError('Explicit immutable parent hash required')
    text=original;records=[];count=66
    parent_manifest=source.parent/'manifest.json'
    if parent_manifest.exists():count=json.loads(parent_manifest.read_text())['expected_moving_kernels']
    for name in passes:
        before=sha(text.encode())
        if name=='cache-exit':
            for signature in ('void UniformBody(','void FixedCubeCachedBody('):
                start,end,body=block(text,signature)
                text=text[:start]+transform_body(body)+text[end:]
            evidence=prove_exit()
        elif name=='terminal-reuse':
            text,sites=terminal_transform(text)
            evidence=dict(horizons=[terminal_cut(d) for d in (4,16)],sites=sites)
        elif name=='bounded-source':
            text=bounded_transform(text);count+=2
            evidence=dict(admission='Existing SourcePlanePairFact; retain transverse bounds in Policy5',
                          source_additions='Original ordered split-round two-source additions')
        else:raise ValueError('Unknown pass: '+name)
        records.append(dict(name=name,input_sha256=before,output_sha256=sha(text.encode()),evidence=evidence))
    shutil.copytree(source,output)
    (output/'src/asymmetric_native.cuh').write_text(text)
    result=dict(schema='beamz.systolic-trace-composition.v1',parent_source=str(source),
                parent_consumer_sha256=expected,consumer_sha256=sha(text.encode()),
                passes=records,expected_kernels=count,producer_sha256=sha(Path(__file__).read_bytes()),
                numerical_cell_header_unchanged=True,source_and_monitor_physics_retained=True,GPU_executed=False)
    (output.parent/'composition-proof.json').write_text(json.dumps(result,indent=2)+'\n')
    if 'bounded-source' in passes:
        bounded=dict(result,schema='beamz.trace-bounded-source-plane.v1')
        (output.parent/'bounded-source-proof.json').write_text(json.dumps(bounded,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--expected-consumer-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--passes',nargs='+',choices=('cache-exit','terminal-reuse','bounded-source'),required=True)
    a=p.parse_args()
    print(json.dumps(compose(a.source,a.output,a.expected_consumer_sha256,a.passes)))
