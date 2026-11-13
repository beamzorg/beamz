"""Join bounded CUPTI software instruction counts to an identical kernel SASS.

Instruction shares are not execution-time shares. Global sectors are requests
at the measured interface, not DRAM bytes. Actual/ideal shared wavefronts expose
extra shared transactions, not their critical-path contribution.
"""
from __future__ import annotations
import argparse,collections,hashlib,json,re
from pathlib import Path


def instructions(text):
    result={}
    for line in text.splitlines():
        match=re.match(r'\s*/\*([0-9a-f]+)\*/\s+(.+?)\s*;',line)
        if not match:continue
        words=match[2].split()
        if words[0].startswith('@'):words=words[1:]
        result[int(match[1],16)]=dict(opcode=words[0],text=match[2])
    if not result:raise ValueError('Nonempty selected-kernel SASS required')
    return result


def category(opcode):
    base=opcode.split('.')[0]
    if base in ('LDG','STG','LDGSTS','ATOM','ATOMG','RED','REDG'):return 'global memory'
    if base in ('LDS','STS','LDSM','LDASYNC'):return 'shared memory'
    if base in ('LDL','STL'):return 'local memory'
    if base in ('LDC','ULDC'):return 'constant memory'
    if base.startswith(('FADD','FMUL','FFMA','FSET','DADD','DMUL','DFMA')):return 'floating point'
    if base.startswith(('BRA','BRX','JMP','CALL','RET','EXIT','BSSY','BSYNC','BAR','WARPSYNC','NANOSLEEP')):return 'control / synchronization'
    if base in ('NOP',):return 'other'
    return 'integer / address / predicate / move'


def analyze(metrics,sass):
    data=json.loads(Path(metrics).read_text());assembly=instructions(Path(sass).read_text())
    if data['dropped_records'] or not data['rows']:raise ValueError('Nonempty zero-drop collection required')
    functions={row['function'] for row in data['rows']}
    if len(functions)!=1:raise ValueError('Exactly one selected kernel expected')
    totals=collections.Counter();opcodes=collections.Counter();groups=collections.Counter();joined=[];pcs=set()
    for row in data['rows']:
        pc=row['pc']
        if pc not in assembly:raise ValueError('Collected PC absent from identical kernel: '+str(pc))
        pcs.add(pc);ins=assembly[pc];m=row['metrics'];count=m.get('smsp__sass_inst_executed',0)
        totals.update(m);opcodes[ins['opcode']]+=count;groups[category(ins['opcode'])]+=count
        joined.append(dict(pc=pc,**ins,**m))
    warp=totals['smsp__sass_inst_executed'];thread=totals['smsp__sass_thread_inst_executed_pred_on']
    def ratio(a,b):return totals[a]/totals[b] if totals[b] else None
    return dict(schema='beamz.software-SASS-analysis.v1',metrics_sha256=hashlib.sha256(Path(metrics).read_bytes()).hexdigest(),
                sass_sha256=hashlib.sha256(Path(sass).read_bytes()).hexdigest(),function=next(iter(functions)),
                rows=data['records'],unique_collected_pcs=len(pcs),dropped_records=0,totals=dict(totals),
                dynamic_warp_instruction_categories=[dict(category=k,instructions=v,fraction=v/warp) for k,v in groups.most_common()],
                dynamic_opcodes=[dict(opcode=k,instructions=v,fraction=v/warp) for k,v in opcodes.most_common()],
                active_predicated_lane_fraction=thread/(warp*32) if warp else None,
                global_requested_to_ideal_sectors=ratio('smsp__sass_sectors_mem_global','smsp__sass_sectors_mem_global_ideal'),
                shared_actual_to_ideal_wavefronts=ratio('smsp__sass_l1tex_pipe_lsu_wavefronts_mem_shared','smsp__sass_l1tex_pipe_lsu_wavefronts_mem_shared_ideal'),
                instructions=joined,limitations=['Instruction shares are not timing or stall shares','Global sectors are not DRAM bytes','One kernel launch is not the whole workload'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--metrics',type=Path,required=True);p.add_argument('--sass',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=analyze(a.metrics,a.sass);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ('instructions','dynamic_opcodes','totals')}))
