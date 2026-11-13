"""Native traced arithmetic/control-flow and canonical local-term ABI gates."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.cuda_space_time_ir import yee_operator
from scripts.generate_cuda_native_cell import generate_native_cell
from scripts.prove_cuda_packet_warp_retirement import prove

ROOT = Path(__file__).resolve().parents[2]


def test_native_functor_ids_callbacks_and_guarded_cpml_are_deterministic():
    result = generate_native_cell()
    assert result == generate_native_cell()
    assert result["field_ids"] == {f + c: 3 * p + k for p, f in enumerate("HE") for k, c in enumerate("xyz")}
    assert result["local_term_ids"] == {f"psi_{f}{t}": 6 * p + t for p, f in enumerate("HE") for t in range(6)}
    assert result["source"].count("if(env.LocalActive(") == 12
    assert "__syncthreads" not in result["source"]
    assert "read(5,(version-1)," in result["source"]
    assert "env.Difference(" in result["source"]
    sparse = generate_native_cell(yee_operator(cpml_axes=(1,)))
    assert sparse["local_term_ids"] == {"psi_H0": 0, "psi_H5": 5, "psi_E0": 6, "psi_E5": 11}


@pytest.mark.parametrize("options", (
    (), ("--cpml", "--bf16", "--pec-mask", "63"),
    ("--cpml", "--bf16", "--zero-fields", "--inactive-cpml"),
    ("--cpml", "--scalar-h-class", "--pec-mask", "63"),
))
def test_actual_native_functor_matches_independent_scalar_oracle(tmp_path, options):
    if not shutil.which("c++"):
        pytest.skip("C++20 compiler required")
    result = subprocess.run([sys.executable, str(ROOT / "scripts/validate_cuda_streamed_host.py"),
                             "--native-cell", "--output-dir", str(tmp_path), *options],
                            cwd=ROOT, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"errors":0' in result.stdout


def test_sparse_packet_reuse_requires_real_warp_fence_and_cta_retirement():
    assert prove(steps=32)["stage_bytes"] == 600
    for option in ("omit_ghost_lane", "compact_high_key", "high_read_before_b3", "next_import_before_b1"):
        with pytest.raises(AssertionError):
            prove(**{option: True})
    with pytest.raises(AssertionError):
        prove(warp_fence=False)
