"""Derive a short fixture using an independent ordinary native Program solve.

All original physical input files are aliased byte-for-byte. Only the executed
step count and30reference leaves change. Does not import JAX or BeamZ Python.
"""
from __future__ import annotations
import argparse,copy,hashlib,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.cuda_fixture_store import ArrayReader,digest_array,drop_pages,host_chunks
from scripts.export_fdtdx_comparison import file_digest


def export(fixture,module,output,steps):
    if output.exists():raise ValueError('New immutable fixture output required')
    f=copy.deepcopy(json.loads(fixture.read_text()))
    if f['data_format']!='npy_directory_v1' or not 0<steps<f['attributes']['nsteps']:raise ValueError('A proper prefix of an NPY fixture required')
    store=ArrayReader(fixture.parent/f['data_directory'])
    if file_digest(fixture.parent/f['data_file'])!=f['data_sha256']:raise ValueError('Source manifest changed')
    inputs=[store[x['key']] for x in f['inputs']]
    for item,value in zip(f['inputs'],inputs):
        if digest_array(value)!=item['sha256']:raise ValueError('Physical input changed:'+item['name'])
        drop_pages(value)
    spec=importlib.util.spec_from_file_location('_cuda_native',module);native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
    attrs=dict(f['attributes']);attrs['nsteps']=steps
    session=native.NativeSession(inputs,attrs,preserve_inputs=False,kind='program')
    output.parent.mkdir(parents=True,exist_ok=True);directory=output.parent/(output.stem+'.arrays');directory.mkdir()
    arrays={}
    for key,value in store.metadata.items():
        arrays[key]=dict(value,file=str((store.directory/value['file']).resolve()))
    start=time.monotonic()
    try:
        result=session.run_schedule([min(256,steps)]*(steps//256)+([steps%256] if steps%256 else []))
        clocks=session.clocks();maxima={}
        for leaf in f['state_leaves']:
            key=leaf['reference_key'];m=leaf['mapping'];raw=None
            if m['kind']=='field':raw=(18 if result['output_bank'] else 0)+m['component']
            elif m['kind']=='output':raw=m['index']
            path=directory/(key+'.npy')
            value=np.lib.format.open_memmap(path,mode='w+',dtype=np.dtype(leaf['dtype']),shape=tuple(leaf['shape']))
            if raw is not None:session.download_into(raw,value)
            elif m['kind']=='clock':value[...]=clocks['time' if m['name']=='t' else 'current_step']
            else:value[...]=store[leaf['initial_key']]
            peak=0.0
            for _,chunk in host_chunks(value):
                if np.dtype(value.dtype).kind=='f' and not np.isfinite(chunk).all():raise ValueError('Nonfinite reference:'+leaf['name'])
                if chunk.size:peak=max(peak,float(np.max(np.abs(chunk))))
            value.flush();arrays[key]=dict(file=path.name,shape=list(value.shape),dtype=value.dtype.str,sha256=digest_array(value),nbytes=value.nbytes,max_abs=peak,finite=True)
            maxima[leaf['name']]=peak;drop_pages(value);del value
        manifest=directory/'manifest.json';manifest.write_text(json.dumps(dict(schema='beamz.npy-array-directory.v1',arrays=arrays),indent=2)+'\n')
        f.update(attributes=attrs,data_directory=directory.name,data_file=str(manifest.relative_to(output.parent)),data_sha256=file_digest(manifest),
                 reference_backend='Independent ordinary native Program; no temporal moving kernel',reference_schedule=[steps],reference_max_abs=maxima,
                 prefix_derivation=dict(source_fixture=str(fixture.resolve()),source_fixture_sha256=file_digest(fixture),
                                        original_waveform_horizon_steps=f['attributes']['nsteps'],executed_steps=steps,
                                        physical_inputs_byte_identical=True,ordinary_module_sha256=file_digest(module),
                                        elapsed_s=time.monotonic()-start,worker_sha256=file_digest(__file__)))
        output.write_text(json.dumps(f,indent=2)+'\n')
        return f['prefix_derivation']
    finally:session.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fixture',type=Path,required=True);p.add_argument('--module',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=256);a=p.parse_args();print(json.dumps(export(a.fixture,a.module,a.output,a.steps)))
