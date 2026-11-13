"""CPU-only private worker copy reducing quiet-oracle peak host storage."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path

PARENT_SHA="dba51164d0e7d6453a1aefdc32bb28f4f9341fc99ac216515f331063de32732a"
HELPER='''
def numeric_sensitivity_fingerprint(value, chunk_elements=131072):
    """Finite float32/64 numeric equality, normalized signedzero, <=2MiB buffers."""
    import hashlib
    array = np.asarray(value)
    if array.dtype.kind != "f" or array.dtype.itemsize > 8:
        raise ValueError("Sensitivity fields/DFT require finite float32/64")
    digest = hashlib.sha256(repr(tuple(array.shape)).encode("ascii"))
    for part in np.nditer(array, flags=["external_loop", "buffered", "zerosize_ok"],
                          op_flags=["readonly"], op_dtypes=[np.dtype("<f8")],
                          order="C", buffersize=chunk_elements):
        canonical = np.array(part, dtype="<f8", copy=True)
        if not np.isfinite(canonical).all():
            raise ValueError("Nonfinite sensitivity state")
        canonical[canonical == 0] = 0.0
        digest.update(memoryview(canonical).cast("B"))
    return digest.hexdigest()

'''
OLD='''        expected = ordinary(module, fixture, data, changed, attrs, args.steps)
        quiet = list(changed)
        for group in (5, 7):
            quiet[86 + 3 * group] = np.zeros_like(changed[86 + 3 * group])
        no_source = ordinary(module, fixture, data, quiet, attrs, args.steps)
        field_delta = any(not np.array_equal(expected[n], no_source[n])
                          for n in ("ex", "ey", "ez", "hx", "hy", "hz"))
        dft_delta = any(not np.array_equal(expected[n], no_source[n])
                        for n in expected if "dft" in n)
'''
NEW='''        # Quiet sensitivity storage is retired before retaining the full oracle.
        quiet = list(changed)
        for group in (5, 7):
            quiet[86 + 3 * group] = np.zeros_like(changed[86 + 3 * group])
        no_source = ordinary(module, fixture, data, quiet, attrs, args.steps)
        quiet_fingerprints = {n: numeric_sensitivity_fingerprint(v)
                              for n, v in no_source.items()
                              if n in ("ex", "ey", "ez", "hx", "hy", "hz") or "dft" in n}
        del no_source, quiet
        import gc
        gc.collect()
        expected = ordinary(module, fixture, data, changed, attrs, args.steps)
        field_delta = any(numeric_sensitivity_fingerprint(expected[n]) != quiet_fingerprints[n]
                          for n in ("ex", "ey", "ez", "hx", "hy", "hz"))
        dft_delta = any(numeric_sensitivity_fingerprint(expected[n]) != quiet_fingerprints[n]
                        for n in expected if "dft" in n)
'''

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def transform(text):
 if hashlib.sha256(text.encode()).hexdigest()!=PARENT_SHA:raise ValueError("Exact frozen parent worker required")
 if text.count(OLD)!=1:raise ValueError("Exact original sensitivity storage block required")
 marker="def probe_cases(module, fixture, data, inputs, args):"
 new=text.replace(OLD,NEW,1).replace(marker,HELPER+marker,1)
 if new.replace(HELPER, "",1).replace(NEW,OLD,1)!=text:raise ValueError("Sensitivity-only inverse failed")
 return new
def prepare(area,output):
 area,output=Path(area).resolve(),Path(output).resolve()
 if output.exists():raise ValueError("New private output required")
 oldplan=json.loads((area/'fixed-cube-post-validation/plan.json').read_text());binding=json.loads((area/'fixed-cube-post-validation/binding.json').read_text());parent=Path(binding['worker'])
 output.mkdir(parents=True);worker=output/'validate_plane_memory_bounded.py';worker.write_text(transform(parent.read_text()))
 b=dict(binding);b['worker']=str(worker);b['immutable_files']=dict(binding['immutable_files']);b['immutable_files'][str(worker)]=sha(worker)
 bp=output/'binding.json';bp.write_text(json.dumps(b,indent=2)+'\n')
 plan=dict(oldplan);plan['schema']='beamz.fixed-cube-memory-bounded-probe-plan.v1';plan['rows']=[];plan['worker_sha256']=sha(worker);plan['binding_sha256']=sha(bp);plan['parent_worker_sha256']=PARENT_SHA
 for row in oldplan['rows']:
  if row['N']!=564:continue
  row=json.loads(json.dumps(row));oldroot=str(area/'fixed-cube-post-validation');newroot=str(output)
  def convert(cmd):
   result=[]
   for arg in cmd:
    if arg==str(parent):arg=str(worker)
    elif arg==str(area/'fixed-cube-post-validation/binding.json'):arg=str(bp)
    elif arg.startswith(oldroot) and not arg.endswith('/cupti_adapter.py'):arg=arg.replace(oldroot,newroot,1)
    result.append(arg)
   return result
  row['cupti_command']=convert(row['cupti_command']);row['parity_output']=row['parity_output'].replace(oldroot,newroot,1)
  for s in row['sanitizers_low_corner']:s['command']=convert(s['command'])
  plan['rows'].append(row)
 proof={'parent_worker':str(parent),'parent_worker_sha256':PARENT_SHA,'new_worker':str(worker),'new_worker_sha256':sha(worker),'exact_source_inverse':True,'only_change':'quiet Program first; finite field/DFT sensitivity hashes then del+gc; original full expected Program and literal all30 moving comparer unchanged','signedzero_normalized':True,'chunk_elements':131072,'candidate_module_sha256':binding['candidate_module_sha256'],'GPU_executed':False}
 plan['memory_change']=proof;plan['purpose']='N564 actual gates after prior worker host OOM; unchanged fixture/source/Program math and all30 comparer';(output/'proof.json').write_text(json.dumps(proof,indent=2)+'\n');(output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');return proof
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--area',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();print(json.dumps(prepare(a.area,a.output),indent=2))
