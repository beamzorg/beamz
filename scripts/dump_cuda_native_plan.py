"""Extract actual source/monitor geometry into the frozen CPU native planner."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

from scripts.cuda_fixture_store import ArrayReader,drop_pages


def dump(fixture,source,output,depth=16):
    fixture,source,output=map(lambda p:Path(p).resolve(),(fixture,source,output))
    output.mkdir(exist_ok=False);f=json.loads(fixture.read_text());reader=ArrayReader(f['data_directory']);inputs=f['inputs']
    packet=output/'geometry.bin'
    with packet.open('wb') as stream:
        def put(values):np.asarray(values,dtype='<i4').tofile(stream)
        put([0x425A504C,depth])
        for row in inputs[:6]:put(row['shape'])
        for g in range(9):
            put(inputs[86+3*g]['shape']);put(reader[inputs[88+3*g]['key']].reshape(-1))
        put(inputs[113]['shape']);obs=reader[inputs[113]['key']];obs.tofile(stream);drop_pages(obs)
    binary=output/'planner';helper=Path(__file__).with_suffix('.cc')
    subprocess.run(['g++','-std=c++17','-O2','-I'+str(source/'src'),str(helper),str(source/'src/native_moving.cc'),str(source/'src/native_deep.cc'),'-o',str(binary)],capture_output=True,text=True,check=True)
    result=subprocess.run([str(binary),str(packet)],capture_output=True,text=True,check=True);plan=json.loads(result.stdout)
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    plan['fixture_sha256']=sha(fixture);plan['geometry_packet_sha256']=sha(packet)
    plan['source_hashes']={name:sha(source/'src'/name) for name in ('native_moving.cc','native_deep.cc','native_moving.h','native_deep.h')}
    plan['producer_sha256']=sha(__file__);plan['cpp_helper_sha256']=sha(helper)
    (output/'plan.json').write_text(json.dumps(plan)+'\n');return {k:plan[k] for k in ('counts','depth','max_frontier','observation_keys','scope')}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fixture',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--depth',type=int,default=16);a=p.parse_args();print(json.dumps(dump(a.fixture,a.source,a.output,a.depth)))
