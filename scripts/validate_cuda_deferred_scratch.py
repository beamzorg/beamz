"""Exercise lazy scratch downloads, method switches and CUDA graph replay.

Run under the research GPU lock with the NumPy-only interpreter. This verifies
private input capture, exact physical state, and allocation after every switch.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.benchmark_cuda_native_runtime import logical_native_state
from scripts.export_fdtdx_comparison import file_digest
from scripts.validate_cuda_native_moving import exact, load_fixture


def validate(module_path, fixture_path, output, *, require_lazy_plan=False):
    assert importlib.util.find_spec("jax") is None
    spec = importlib.util.spec_from_file_location("_cuda_native", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fixture, data, originals = load_fixture(fixture_path)
    records = []

    def state(session, result):
        return logical_native_state(
            fixture, data, session.download(), session.clocks(), result["output_bank"]
        )

    def reference(count):
        session = module.NativeSession(
            originals, fixture["attributes"], preserve_inputs=False, kind="program"
        )
        try:
            return state(session, session.run_schedule([count]))
        finally:
            session.close()

    for preserve in (False, True):
        inputs = list(originals)
        for c in range(6):
            inputs[80 + c] = np.full_like(originals[80 + c], np.float32(17 + c))
        if require_lazy_plan:
            for index in range(129, 134):
                inputs[index] = np.array(originals[index], copy=True)
        session = module.NativeSession(
            inputs, fixture["attributes"], preserve_inputs=preserve, kind="mstep"
        )
        try:
            expected_scratch = [x.copy() for x in inputs[80:86]]
            for x in inputs[80:86]:
                x.fill(np.float32(-41))
            assert session.metadata()["allocated_scratch_bytes"] == 0
            downloaded = session.download()
            for c, x in enumerate(expected_scratch):
                assert np.array_equal(downloaded[24 + c], x)
                destination = np.empty_like(x)
                session.download_into(24 + c, destination)
                assert np.array_equal(destination, x)
            del downloaded
            total = 0
            if require_lazy_plan:
                deferred_plan = sum(value.nbytes for value in inputs[129:134])
                assert (
                    deferred_plan > 0
                    and session.metadata()["deferred_plan_bytes"] == deferred_plan
                )
                # A later ordinary call must use privately captured plan bytes,
                # even when the caller destroys the original plan in between.
                for value in inputs[129:134]:
                    value.fill(0)
                result = session.run_moving([35], 4)
                total = 35
                parity = exact(state(session, result), reference(total))
                assert parity["all30_bitwise"]
                assert session.metadata()["deferred_plan_bytes"] == deferred_plan
                records.append(
                    dict(
                        preserve=preserve,
                        method="cold_moving",
                        steps=35,
                        total_steps=total,
                        all30_bitwise=True,
                        metadata=session.metadata(),
                    )
                )
            # The repeated ordinary chunks must also remain correct if CUDA
            # reuses a freed address and its cached executable graph.
            for method, steps in [
                ("mstep", 4),
                ("moving", 35),
                ("mstep", 4),
                ("moving", 35),
                ("mstep", 4),
                ("moving", 35),
            ]:
                result = (
                    session.run_schedule([steps])
                    if method == "mstep"
                    else session.run_moving([steps], 4)
                )
                total = steps if preserve else total + steps
                parity = exact(state(session, result), reference(total))
                assert parity["all30_bitwise"], (preserve, method, total, parity)
                metadata = session.metadata()
                assert (metadata["allocated_scratch_bytes"] == 0) == (
                    method == "moving"
                )
                if method == "mstep":
                    downloaded = session.download()
                    expected_scratch = [x.copy() for x in downloaded[24:30]]
                    del downloaded
                else:
                    for c, x in enumerate(expected_scratch):
                        destination = np.empty_like(x)
                        session.download_into(24 + c, destination)
                        assert np.array_equal(destination, x), (
                            "Method switch changed idle scratch"
                        )
                records.append(
                    dict(
                        preserve=preserve,
                        method=method,
                        steps=steps,
                        total_steps=total,
                        all30_bitwise=parity["all30_bitwise"],
                        metadata=metadata,
                    )
                )
        finally:
            session.close()
    result = dict(
        passed=True,
        module=str(module_path.resolve()),
        module_sha256=file_digest(module_path),
        private_scratch_capture=True,
        download_and_download_into=True,
        records=records,
    )
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--module", type=Path, required=True)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--require-lazy-plan", action="store_true")
    a = p.parse_args()
    result = validate(
        a.module, a.fixture, a.output, require_lazy_plan=a.require_lazy_plan
    )
    print(json.dumps(dict(passed=result["passed"], cases=len(result["records"]))))
