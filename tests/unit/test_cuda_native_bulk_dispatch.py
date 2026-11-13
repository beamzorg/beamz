"""The real planner's queue/slot maps survive the private bulk refinement."""

import shutil
from pathlib import Path

import pytest

from scripts import patch_cuda_native_bulk_uniform_dispatch as patch
from scripts.prove_cuda_native_bulk_dispatch import prove

ROOT = Path(__file__).resolve().parents[2]


def test_actual_compiled_bulk_partition_queue_ring_and_guards(tmp_path):
    if shutil.which("g++") is None:
        pytest.skip("g++ unavailable")
    # The canonical old research planner predates material fact bit64. Reproduce
    # the independently admitted Policy mask in a private fixture, leaving the
    # actual planner implementation untouched. Real candidate proofs separately
    # compile its frozen complete header with no such adaptation.
    fixture = tmp_path / "source" / "src"
    fixture.mkdir(parents=True)
    for header in (ROOT / "cuda" / "src").glob("*.h"):
        shutil.copy2(header, fixture / header.name)
    shutil.copy2(ROOT / "cuda/src/deep_temporal_observation.cuh",
                 fixture / "deep_temporal_observation.cuh")
    shutil.copy2(ROOT / "cuda/src/native_moving.cc", fixture / "native_moving.cc")
    header = (fixture / "native_moving.h").read_text()
    if "  flags&=63;" not in header:
        header = patch.once(header, "inline constexpr int Policy(uint8_t flags){\n",
                            "inline constexpr int Policy(uint8_t flags){\n  flags&=63;\n")
        (fixture / "native_moving.h").write_text(header)
    report = prove(fixture.parent, tmp_path / "proof")
    assert report["cases"] == 144
    assert report["producer_links"] > 0
    assert report["mixed_ranges"] > 0
    assert report["negative_checks"] == 2


def test_copy_patcher_and_anchors_fail_closed():
    with pytest.raises(ValueError, match="Exact partial"):
        patch.transformed(dict.fromkeys(patch.PARENT_HASHES, b"changed"))
    with pytest.raises(ValueError, match="one exact"):
        patch.once("twice twice", "twice", "changed")
