"""CUPTI adapter preserves native results and rejects vacuous/dropped traces."""

import ctypes
import hashlib
import json
import runpy
import sys
from pathlib import Path

import pytest

from scripts.prepare_cuda_fixed_cube_validation_plan import WRAPPER


WORKER = r'''
import json,sys
from pathlib import Path
RESULT={'moving_macro_count':1,'unchanged_native_token':'yes'}
class Session:
 def metadata(self):return {'moving_uniform_noio_tiles':1}
 def run_moving(self,*a,**k):return RESULT
class Module:
 def NativeSession(self,*a,**k):return Session()
def module_at(path):return Module()
def main():
 s=module_at(None).NativeSession([],{})
 assert s.run_moving([16],depth=16) is RESULT
 assert s.run_moving([16],depth=16) is RESULT
 output=Path(sys.argv[sys.argv.index('--output')+1]);output.write_text(json.dumps({'passed':True,'positive_noio_branch':True}))
 return 0
'''


class Function:
    def __init__(self, callback):
        self.callback = callback

    def __call__(self, *args):
        return self.callback(*args)


def invoke(tmp_path, monkeypatch, *, dropped=0, missing_fixed=False):
    module, worker, adapter, fixture = [tmp_path / n for n in ("native.so", "worker.py", "adapter.py", "fixture.json")]
    module.write_bytes(b"CPU-only dummy module; never imported")
    worker.write_text(WORKER)
    adapter.write_text(WRAPPER)
    fixture.write_text(json.dumps({"inputs": [{"shape": [512, 512, 512]} for _ in range(6)]}))
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    binding = tmp_path / "binding.json"
    binding.write_text(json.dumps({"worker": str(worker), "candidate_module_sha256": digest(module), "immutable_files": {str(p): digest(p) for p in (module, worker, fixture)}}))

    class Library:
        def __init__(self):
            self.beamz_profile_start = Function(self.start)
            self.beamz_profile_stop = Function(self.stop)
            self.beamz_profile_error = Function(lambda: b"CPU test")

        def start(self, path):
            self.path = Path(path.decode())
            return 0

        def stop(self):
            rows = [{"kind": "kernel", "name": "_ZN5beamz4cuda13moving_native10BeginEpochEPjPi", "start_ns": 1, "end_ns": 2, "grid": [1, 1, 1]}]
            if not missing_fixed:
                rows.append({"kind": "kernel", "name": "_ZN5beamz4cuda13moving_native23AsymmetricNoIOFixedCubeILi16ELi512EEEvNS1_10ParametersEPKi", "start_ns": 3, "end_ns": 4, "graph_node_id": 1, "grid": [1, 1, 1], "block": [1024, 1, 1], "registers": 64, "dynamic_shared_bytes": 55204, "local_bytes_per_thread": 0})
            rows.append({"kind": "collector", "dropped_records": dropped})
            self.path.write_text("\n".join(json.dumps(x) for x in rows))
            return 0

    monkeypatch.setattr(ctypes, "CDLL", lambda path: Library())
    monkeypatch.setattr(sys, "argv", [str(adapter), "--binding", str(binding), "--collector", "CPU-test-stub", "--trace-prefix", str(tmp_path / "trace"), "--module", str(module), "--fixture", str(fixture), "--depth", "16", "--output", str(tmp_path / "parity.json")])
    return adapter


def test_adapter_returns_exact_native_objects_and_records_both_replays(tmp_path, monkeypatch):
    adapter = invoke(tmp_path, monkeypatch)
    with pytest.raises(SystemExit) as stop:
        runpy.run_path(str(adapter), run_name="__main__")
    assert stop.value.code == 0
    report = json.loads((tmp_path / "trace.witness.json").read_text())
    assert len(report["calls"]) == 2
    assert all(row["CTAs"] == row["expected_certified_CTAs"] == 1 for row in report["calls"])


@pytest.mark.parametrize("mutation", ("dropped", "missing"))
def test_adapter_rejects_dropped_or_missing_actual_new_path_witness(tmp_path, monkeypatch, mutation):
    adapter = invoke(tmp_path, monkeypatch, dropped=int(mutation == "dropped"), missing_fixed=mutation == "missing")
    with pytest.raises(RuntimeError, match="Dropped|witness missing"):
        runpy.run_path(str(adapter), run_name="__main__")
