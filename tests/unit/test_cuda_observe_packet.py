"""Raw observation retiming preserves arbitrary FP32 payloads and sample IDs."""

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "observe_packet_proof", ROOT / "scripts/prove_cuda_observe_packet.py"
)
PROOF = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROOF)


@pytest.mark.parametrize("depth", [4, 16, 33])
def test_retained_h_and_current_e_have_identical_raw_write_sets(depth):
    matched, stores = PROOF.token_writes(depth)
    assert matched
    assert stores == depth * 4 * 6


@pytest.mark.parametrize("bad", ["wrong_version", "stale_h", "high_only"])
def test_wrong_sample_clock_stale_h_and_surface_only_publication_fail(bad):
    matched, _ = PROOF.token_writes(16, **{bad: True})
    assert not matched


def test_compiled_actual_packet_and_native_sparse_map_preserve_payloads(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    source = tmp_path / "proof.cc"
    text = PROOF.CPP.replace("@SUPPORT@", PROOF.SUPPORT_HELPER).replace(
        "@PACKET@", PROOF.PATCHER.PACKET
    )
    source.write_text("#include <array>\n" + text)
    binary = tmp_path / "proof"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-I",
            str(ROOT / "cuda/src"),
            str(source),
            str(ROOT / "cuda/src/native_moving.cc"),
            "-o",
            str(binary),
        ],
        check=True,
    )
    result = json.loads(
        subprocess.run([str(binary)], check=True, capture_output=True, text=True).stdout
    )
    assert result["pass"]
    assert result["actual_packet_calls"] == 2_814_112
    assert result["matched_raw_stores"] == 283_016
    assert result["guard_words_checked"] == 192
