"""Bind persistent scheduler hooks to the exact paired-cell source family."""

from pathlib import Path

import pytest

from scripts.patch_cuda_native_persistent import clone_body


def parent():
    p = Path(".cache/cuda-two-hour-2026-11-13-0819/source-plane-axis-cg-v1/source/src/asymmetric_native.cuh")
    if not p.exists():
        pytest.skip("Private frozen research parent unavailable")
    return p.read_text()


def test_per_step_global_publish_uses_existing_barriers_and_uniform_abort():
    body = clone_body(parent())
    assert body.count("__threadfence();") == 1
    assert body.index("__threadfence();") < body.index("// RLpair")
    assert body.index("PublishAndWait(n,window[1])") < body.index("// RE:")
    assert body.count("if(C().abort_local)return;") == 2
    assert "__syncthreads();if(C().abort_local||window[0]>window[1])return;" in body


def test_startup_wait_precedes_base_tag_and_prev_version_imports():
    body = clone_body(parent())
    assert body.index("WaitUpstream(window[0])") < body.index("Acquire(p.owner_tags+owner)")
    assert body.index("WaitUpstream(window[0])") < body.index("if(first==1)")
    assert "tiles[blockIdx.x]" not in body
    assert "p.owner_tags[owner]=" not in body


def test_import_packet_keeps_all_six_coherent_CG_loads():
    code = parent()
    a = code.index("__device__ __forceinline__ FieldPacket ImportPacket")
    b = code.index("__global__ void BeginEpoch", a)
    import_body = code[a:b]
    assert import_body.count("__ldcg(point+") == 6
    assert "__ldg(" not in import_body
