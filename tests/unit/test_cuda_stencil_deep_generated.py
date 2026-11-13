"""Reproducibility and independent dependency gates for experimental deep D4."""

import hashlib
from pathlib import Path

import pytest

from scripts.cuda_deep_temporal_warp import generate, generate_bundle
from scripts.prove_cuda_deep_shared_races import prove as prove_shared_races
from scripts.prove_cuda_deep_warp import prove as prove_versions


def test_checked_in_deep_header_reproduces_explicit_typed_single_frame_flags():
    source, _ = generate_bundle(typed_mie=True)
    header = Path(__file__).resolve().parents[2] / "cuda/src/deep_temporal_warp_generated.cuh"
    assert source == header.read_text()
    assert hashlib.sha256(source.encode()).hexdigest() == "a3ec109b0f04f10711afdff31f100a83e4ccef5f6b014b7d03241815a6a6ba21"


def test_generic_single_frame_reproduces_frozen_v6_control():
    source, _ = generate_bundle()
    assert "int BeamzDeepD4AdmissionKind() { return 0; }\n" in source
    source = source.replace("int BeamzDeepD4AdmissionKind() { return 0; }\n", "")
    assert hashlib.sha256(source.encode()).hexdigest() == "47110f5a9914fafd030dd557a661861f71dc56d7c3240a3777273a2247efe8c8"


@pytest.mark.parametrize("mask", (0, 1, 2, 4))
def test_double_frame_version_dependencies(mask):
    result = prove_versions((2, 2, 5), 4, mask, field_frames=2)
    assert result["versioned_reads_checked"] > 0


def test_wrong_frame_age_is_rejected():
    with pytest.raises(AssertionError):
        prove_versions((2, 2, 5), 4, 0, field_frames=2, bad_frame=True)


def test_double_frame_keeps_required_publication_barrier():
    prove_shared_races(4, 2, pre_barrier=False)
    with pytest.raises(AssertionError):
        prove_shared_races(4, 2, pre_barrier=False, post_barrier=False)
    with pytest.raises(AssertionError):
        prove_shared_races(4, 1, pre_barrier=False)


def test_source_signature_and_unchecked_cpml_lowering_fail_closed():
    with pytest.raises(ValueError):
        generate(source_signature=64)
    with pytest.raises(ValueError):
        generate(mask=1, interior=True)
