"""CPU-only preparation of exact-shape Program-oracle/CUPTI/sanitizer commands.

No engine or fixture copies are edited. Execution belongs to the GPU coordinator.
The CUPTI adapter returns native results unchanged to the existing frozen worker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


WRAPPER = r'''
"""Run the unchanged frozen plane oracle, collecting only moving calls."""
import argparse,ctypes,hashlib,json,math,re,runpy,sys
from pathlib import Path

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--binding',required=True,type=Path);p.add_argument('--collector',required=True,type=Path);p.add_argument('--trace-prefix',required=True,type=Path)
a,remaining=p.parse_known_args();binding=json.loads(a.binding.read_text())
for path,expected in binding['immutable_files'].items():
 if sha(path)!=expected:raise RuntimeError('Frozen validation dependency changed: '+path)
def arg(name):return remaining[remaining.index(name)+1]
module=Path(arg('--module'));fixture=Path(arg('--fixture'));depth=int(arg('--depth'));report=Path(arg('--output'))
if sha(module)!=binding['candidate_module_sha256']:raise RuntimeError('Candidate identity differs')
f=json.loads(fixture.read_text());shape=tuple(min(item['shape'][axis] for item in f['inputs'][:6]) for axis in range(3))
if len(set(shape))!=1 or shape[0] not in (512,564):raise RuntimeError('Actual FixedCube shape required')
cube=shape[0];library=ctypes.CDLL(str(a.collector.resolve()));library.beamz_profile_start.argtypes=[ctypes.c_char_p];library.beamz_profile_error.restype=ctypes.c_char_p
program=runpy.run_path(binding['worker']);g=program['main'].__globals__;original_module_at=g['module_at'];witnesses=[]
class SessionProxy:
 def __init__(self,native):self.native=native
 def __getattr__(self,name):return getattr(self.native,name)
 def run_moving(self,*args,**kwargs):
  trace=a.trace_prefix.with_name(a.trace_prefix.name+'.run'+str(len(witnesses))+'.activity.jsonl');trace.parent.mkdir(parents=True,exist_ok=True)
  if trace.exists():raise RuntimeError('New activity path required')
  error=library.beamz_profile_start(str(trace.resolve()).encode())
  if error:raise RuntimeError('CUPTI start failed: '+str(library.beamz_profile_error()))
  try:result=self.native.run_moving(*args,**kwargs)
  finally:
   error=library.beamz_profile_stop()
   if error:raise RuntimeError('CUPTI stop failed: '+str(library.beamz_profile_error()))
  rows=[json.loads(line) for line in trace.read_text().splitlines()];collectors=[r for r in rows if r.get('kind')=='collector'];kernels=[r for r in rows if r.get('kind')=='kernel']
  if not collectors or any(r.get('dropped_records')!=0 for r in collectors):raise RuntimeError('Dropped/missing CUPTI records')
  fixed=sorted([r for r in kernels if 'AsymmetricNoIOFixedCube' in r['name']],key=lambda r:r['start_ns'])
  names=sorted(set(r['name'] for r in fixed));wanted='FixedCubeILi'+str(depth)+'ELi'+str(cube)+'E'
  if len(names)!=1 or wanted not in names[0]:raise RuntimeError('Actual shape/depth kernel witness missing')
  if any('AsymmetricNoIOUniform1024' in r['name'] for r in kernels):raise RuntimeError('Eligible cached NoIO used old entry')
  meta=self.native.metadata();expected=int(meta['moving_uniform_noio_tiles'])*int(result['moving_macro_count']);observed=sum(math.prod(r['grid']) for r in fixed)
  if observed!=expected or expected<=0:raise RuntimeError('Certified CTA coverage differs')
  if any(r['block']!=[1024,1,1] or r['registers']!=64 or r['dynamic_shared_bytes']!=55204 or r['local_bytes_per_thread']!=0 for r in fixed):raise RuntimeError('Actual fixed-kernel resources differ')
  epochs=sorted(r['start_ns'] for r in kernels if '10BeginEpoch' in r['name']);first_end=epochs[1] if len(epochs)>1 else max(r['end_ns'] for r in kernels)+1
  first_macro=[r for r in fixed if r['start_ns']<first_end]
  witness={'trace':str(trace),'trace_sha256':sha(trace),'module_sha256':sha(module),'fixture_sha256':sha(fixture),'N':cube,'depth':depth,'kernel_names':names,'launches':len(fixed),'CTAs':observed,'expected_certified_CTAs':expected,'dropped_records':0,'first_macro_matching_launches':len(first_macro),'low_corner_matching_index':0,'high_corner_matching_index':len(first_macro)-1,'low_corner_grid':first_macro[0]['grid'],'high_corner_grid':first_macro[-1]['grid'],'resources':{'block':[1024,1,1],'registers':64,'dynamic_shared_bytes':55204,'local_bytes_per_thread':0}}
  witnesses.append(witness);a.trace_prefix.with_name(a.trace_prefix.name+'.witness.json').write_text(json.dumps({'schema':'beamz.exact-shape-Program-oracle-CUPTI-witness.v1','worker':binding['worker'],'worker_sha256':sha(binding['worker']),'wrapper_sha256':sha(__file__),'candidate_module_sha256':sha(module),'calls':witnesses,'parity_output':str(report),'scope':'Only actual moving calls collected; native results returned unchanged to frozen all30 Program oracle; no timing claim'},indent=2)+'\n')
  return result
class ModuleProxy:
 def __init__(self,native):self.native=native
 def __getattr__(self,name):return getattr(self.native,name)
 def NativeSession(self,*args,**kwargs):return SessionProxy(self.native.NativeSession(*args,**kwargs))
g['module_at']=lambda path:ModuleProxy(original_module_at(path));sys.argv=[binding['worker']]+remaining
code=program['main']()
if code==0:
 r=json.loads(report.read_text())
 if not r.get('passed') or not r.get('positive_noio_branch') or len(witnesses)<2:raise RuntimeError('Missing complete repeat/Program oracle witness')
for path,expected in binding['immutable_files'].items():
 if sha(path)!=expected:raise RuntimeError('Frozen dependency changed during validation')
raise SystemExit(code)
'''


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(area, output):
    area, output = Path(area).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("New private plan directory required")
    campaign = json.loads((area / "fixed-cube-full512/plan.json").read_text())
    manifest = json.loads((area / "noio-fixed-cubes-cached-v1/manifest.json").read_text())
    frozen = json.loads((area / "noio-fixed-cubes-cached-v1/frozen-execution-manifest.json").read_text())
    module_candidates = list((area / "frozen-execution/noio-fixed-cubes-cached-v1" / manifest["module_sha256"]).glob("*.so"))
    if len(module_candidates) != 1:
        raise ValueError("One exact frozen candidate module required")
    module = module_candidates[0]
    if sha(module) != manifest["module_sha256"]:
        raise ValueError("Frozen module identity differs")
    worker_row = next(row for row in campaign["workers"] if row["frozen"].endswith("validate_cuda_plane_retirement.py"))
    worker = Path(worker_row["frozen"])
    immutable = {str(worker): worker_row["frozen_sha256"], str(module): manifest["module_sha256"], **campaign["dependencies"]}
    fixtures = {512: Path(".cache/main-vs-temporal-2026-11-12/fixture-prefix-stock-default.json").resolve(),
                564: Path(".cache/cuda-five-hour-2026-11-13/mie564/mie564-full30fs.json").resolve()}
    immutable.update({str(p): sha(p) for p in fixtures.values()})
    collectors = [Path(".cache/cuda-engine-autoresearch-2026-11-11-2032/cuda_activity_profile_v2.so").resolve(), Path(".cache/cuda-engine-autoresearch-2026-11-11-2032/cuda_activity_profile.so").resolve()]
    profile = json.loads((area / "aot-native-profile").read_text())
    collector = next(p for p in collectors if p.exists() and sha(p) == profile["collector_sha256"])
    immutable[str(collector)] = sha(collector)
    for path, expected in immutable.items():
        if sha(path) != expected:
            raise ValueError("Existing frozen campaign dependency changed:" + path)
    # Resolving this leaf symlink changes sys.prefix to the base interpreter
    # and loses NumPy. Preserve the absolute virtual-environment executable.
    python = Path(__file__).resolve().parents[1] / ".cache/cuda-native-autoresearch-2026-11-11-1607/native-only/bin/python"
    interpreter = json.loads(subprocess.run([str(python), "-c", "import importlib.util,json,sys;print(json.dumps({'executable':sys.executable,'prefix':sys.prefix,'numpy_available':importlib.util.find_spec('numpy') is not None,'jax_available':importlib.util.find_spec('jax') is not None}))"], check=True, capture_output=True, text=True).stdout)
    if not interpreter["numpy_available"] or interpreter["jax_available"] or Path(interpreter["prefix"]) != python.parent.parent:
        raise ValueError("NumPy-only virtual-environment executable required")
    sanitizer = Path("/home/quentinwach/.cache/beamz-cuda-toolkit-13.3/root/opt/cuda/compute-sanitizer/compute-sanitizer")
    output.mkdir(parents=True)
    adapter, binding_path = output / "cupti_adapter.py", output / "binding.json"
    adapter.write_text(WRAPPER)
    binding = {"worker": str(worker), "candidate_module_sha256": manifest["module_sha256"], "immutable_files": immutable}
    binding_path.write_text(json.dumps(binding, indent=2) + "\n")
    rows = []
    for n, fixture in fixtures.items():
        for depth in (4, 16):
            label = f"N{n}-D{depth}"
            oracle = output / (label + ".json")
            base = [str(worker), "--module", str(module), "--fixture", str(fixture), "--mode", "probes", "--case-index", "1", "--steps", "16", "--depth", str(depth), "--repeats", "2", "--require-noio-branch", "--output", str(oracle)]
            cupti = [str(python), str(adapter), "--binding", str(binding_path), "--collector", str(collector), "--trace-prefix", str(output / (label + ".cupti")), *base[1:]]
            tools = []
            for tool in ("memcheck", "synccheck", "racecheck"):
                sanitizer_args = list(base)
                sanitizer_args[sanitizer_args.index("--repeats") + 1] = "1"
                sanitizer_args[sanitizer_args.index("--output") + 1] = str(output / (label + "." + tool + ".low.json"))
                command = [str(sanitizer), "--tool", tool, "--error-exitcode", "86", "--print-session-details", "--kernel-name", "kns=AsymmetricNoIOFixedCube", "--launch-skip", "0", "--launch-count", "1", str(python), *sanitizer_args]
                tools.append({"tool": tool, "window": "first matching new-path launch", "command": command,
                              "scope": "Bounded instrumentation only; completes ordinary Program oracle and full16-step new-path run, but sanitizer instruments one launch"})
            rows.append({"label": label, "N": n, "depth": depth, "steps": 16,
                         "fixture_bytes_unchanged": True, "stored_full_reference_used": False,
                         "oracle": "Fresh ordinary CUDA Program for the changed point-source/DFT packet; all30 and two reset/repeats",
                         "cupti_command": cupti, "parity_output": str(oracle), "sanitizers_low_corner": tools,
                         "high_corner_followup": "After this case CUPTI witness exists, use its high_corner_matching_index in --launch-skip and a new high output/log; do not assume512 index127 applies to564 orD4",
                         "optional_middle_followup": "Use first_macro_matching_launches//2 for a single middle-face memcheck/synccheck if budget permits"})
    report = {"schema": "beamz.fixed-cube-post-ABBA-validation-plan.v1", "candidate_module": str(module),
              "candidate_module_sha256": manifest["module_sha256"], "worker_sha256": sha(worker),
              "adapter_sha256": sha(adapter), "binding_sha256": sha(binding_path), "interpreter_validation": interpreter, "rows": rows,
              "GPU_execution_authority": "Coordinator only, after existing full ABBA queue completes", "GPU_executed": False,
              "completion_requirements": ["unchanged frozen module/worker/dependencies", "actual matching shape/depth FixedCube CUPTI kernel and exact certified CTA count with dropped0", "frozen worker passed all30 and reset/replay; positive NoIO macro and source/DFT sensitivity", "each requested sanitizer completes exit0 and explicit0 errors/hazards; timeout, OOM or missing summary is incomplete", "bounded windows are documented, never called whole-grid sanitizer coverage"],
              "memory_note": "Probe worker uses preserve_inputs=False and attrs.nsteps16; geometry512/564 is retained and still requires large fields. Do not run alongside ABBA/GPU jobs.",
              "purpose": "Actual new-path reliability witness; no performance measurements or engine/fixture source changes"}
    (output / "plan.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--area", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    result = prepare(a.area, a.output)
    print(json.dumps({"candidate_module_sha256": result["candidate_module_sha256"], "rows": [r["label"] for r in result["rows"]], "GPU_executed": False}))
