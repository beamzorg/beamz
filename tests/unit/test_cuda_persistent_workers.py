"""Persistent adapters preserve truthful dispatch, allocation and oracle checks."""

import ast
import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.prepare_cuda_persistent_workers import AREA, benchmark_adapter, validation_adapter, verify


DIRECTORY = AREA / "persistent-workers"


def worker(name):
    path = DIRECTORY / name
    if not path.exists():
        pytest.skip("Prepared frozen campaign required")
    spec = importlib.util.spec_from_file_location("persistent_worker_test_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def result(chunks=(35,), depth=16, order="locality"):
    return {"algorithm": "persistent_tile_step", "persistent_tile_step": True, "persistent_order": order,
            "moving_macro_count": sum(n // depth for n in chunks), "ordinary_tail_steps": sum(n % depth for n in chunks)}


def metadata(order="locality", workers=8):
    return {"persistent_tile_step_ready": True, "persistent_tile_step_order": order,
            "persistent_tile_step_workers": workers, "persistent_tile_step_jobs": 100}


def test_frozen_sources_and_dependency_contract_are_unchanged():
    if not DIRECTORY.exists():
        pytest.skip("Prepared frozen campaign required")
    contract = verify(DIRECTORY)
    original = AREA / "validation-template/workers"
    assert (DIRECTORY / "benchmark_baseline.py").read_bytes() == (original / "benchmark_native.py").read_bytes()
    assert (DIRECTORY / "cuda_research_compare.py").read_bytes() == (original / "cuda_research_compare.py").read_bytes()
    assert contract["normal_baseline_and_strict_comparator_byte_identical"]


def test_validation_adapter_invokes_explicit_API_with_order_workers_and_positive_body_witness():
    module = worker("validate_persistent.py")
    calls = []

    class FakeSession:
        def run_moving_persistent(self, chunks, **kw):
            calls.append((chunks, kw))
            return result(chunks, kw["depth"], kw["order"])

        def metadata(self):
            return metadata()

    output = module.persistent_run(FakeSession(), [17, 18], 16, "locality", 8)
    assert output["moving_macro_count"] == 2
    assert calls == [([17, 18], {"depth": 16, "order": "locality", "workers": 8})]
    assert module.PERSISTENT_METHOD_WITNESSES[-1]["persistent_body_positive"]
    source = (DIRECTORY / "validate_persistent.py").read_text()
    assert "session.run_moving(" not in source
    assert "result = session.run_schedule(chunks)" in source
    assert 'clocks["elapsed_steps"] - clocks["current_step"]' in source
    assert '"host_inputs_unchanged": immutable' in source


@pytest.mark.parametrize("mutation", ["normal-algorithm", "wrong-order", "missing-ready", "wrong-workers", "false-API"])
def test_wrong_method_or_queue_identity_cannot_pass_as_persistent(mutation):
    module = worker("benchmark_persistent.py")
    r, m = result(), metadata()
    if mutation == "normal-algorithm":
        r["algorithm"] = "asymmetric_moving"
    elif mutation == "wrong-order":
        r["persistent_order"] = "rank"
    elif mutation == "missing-ready":
        m["persistent_tile_step_ready"] = False
    elif mutation == "wrong-workers":
        m["persistent_tile_step_workers"] = 7
    else:
        r["persistent_tile_step"] = False
    with pytest.raises(RuntimeError):
        module.persistent_witness(r, m, [35], 16, "locality", 8)


def test_tail_only_calls_do_not_claim_persistent_body_coverage():
    module = worker("benchmark_persistent.py")
    r = result([1], 16)
    witness = module.persistent_witness(r, {}, [1], 16, "locality", 0)
    assert not witness["persistent_body_positive"]
    assert witness["actual_moving_macros"] == 0


def test_benchmark_dispatch_contains_no_extra_metadata_inside_timed_native_call():
    if not DIRECTORY.exists():
        pytest.skip("Prepared frozen campaign required")
    tree = ast.parse((DIRECTORY / "benchmark_persistent.py").read_text())
    native = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "native")
    first_if = next(n for n in native.body if isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
                    and ast.unparse(n.test).startswith("args.method == 'moving_persistent'"))
    run = next(n for n in first_if.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    assert len(run.body) == 1 and isinstance(run.body[0], ast.Return)
    call = run.body[0].value
    assert ast.unparse(call.func) == "session.run_moving_persistent"
    assert {k.arg for k in call.keywords} == {"depth", "order", "workers"}


def test_preinit_queue_bytes_match_bound_CPP_formula_without_GPU(monkeypatch):
    module = worker("benchmark_persistent.py")
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="24576\n"))
    monkeypatch.setattr(module, "available_host_bytes", lambda: 64 << 30)
    f = {"inputs": [{"index": i, "key": str(i)} for i in range(134)]}

    class Store:
        metadata = {str(i): {"nbytes": 4} for i in range(134)}

        def __getitem__(self, key):
            return np.zeros(1, np.float32)

    native = SimpleNamespace(moving_resource_plan=lambda *a, **kw: {
        "tile_count": 17, "epochs": 4, "max_frontier": 5, "extra_gpu_bytes": 1000})
    args = SimpleNamespace(ownership="preserve", backend="native", method="moving_persistent", moving_depth=16)
    admission = module.exact_admission(f, Store(), args, native)
    assert admission["moving"]["persistent_queue_expected_bytes"] == 17 * 16 + 4 * 5 * 4 + 8
    assert admission["GPU_payload_expected_bytes"] == 134 * 4 + admission["persistent_bytes"] + 4 + 1000 + 360


def test_unrecognized_frozen_worker_source_fails_closed():
    with pytest.raises(ValueError):
        validation_adapter("different worker")
    with pytest.raises(ValueError):
        benchmark_adapter("different worker")
