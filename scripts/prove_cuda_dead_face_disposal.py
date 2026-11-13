"""Check trace-driven destructive packet disposal against canonical version reads."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from scripts.prove_cuda_partial_plane_pipeline import simulate


def report():
    cases=[]
    for minimum,owned,depth in [((11,13,15),(3,5,7),4),((13,17,21),(7,15,31),16)]:
        for workers in (1,4,82):
            for seed in (13,37):cases.append(simulate(minimum=minimum,owned=owned,depth=depth,workers=workers,seed=seed,disposal=True))
    cases.append(simulate(workers=82,schedule='producer-fast',disposal=True))
    negatives={}
    for name in ('early-packet-disposal','premature-late-ack'):
        try:simulate(negative=name,workers=82,disposal=True)
        except AssertionError as e:negatives[name]=str(e)
        else:raise AssertionError('Ineffective disposal negative:'+name)
    # Only complete cache lines wholly inside one live packet may be destroyed.
    packet_bytes=6*735*4;alignment_checks=0;preserved_edge_bytes=set()
    for owner in range(4):
        for version in range(1,16):
            begin=(owner*16+version-1)*packet_bytes;end=begin+packet_bytes
            first=(begin+127)&~127;last=end&~127
            for address in range(first,last,128):
                assert address%128==0 and begin<=address and address+128<=end;alignment_checks+=1
            preserved_edge_bytes.add((first-begin)+(end-last))
    return dict(schema='beamz.trace-driven-dead-face-disposal-model.v1',passed=True,cases=cases,negative_controls=negatives,
                packet_alignment_checks=alignment_checks,preserved_edge_bytes=sorted(preserved_edge_bytes),
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                pipeline_model_sha256=hashlib.sha256(Path(__file__).with_name('prove_cuda_partial_plane_pipeline.py').read_bytes()).hexdigest(),
                scope='All7nominal reader ACKs, late initialization/skipped/trailing/empty windows, producer READY-bit race and poison-after-disposal canonical reads. Not actual CUDA/SASS or FP/memory-ordering validation.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=report();a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(passed=True,cases=len(r['cases']),disposal_checks=sum(c['disposal_checks'] for c in r['cases']),disposed_packets=sum(c['disposed_packets'] for c in r['cases']))))
