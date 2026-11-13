"""Point-local CG/CA routing preserves original tile facts and cell graph."""

import hashlib
import shutil
from pathlib import Path

import pytest

from scripts.patch_cuda_persistent_mixed_psi import BOUND, HELPER, MIXED_LOAD, SITE_ASSIGNMENTS, transformed
from scripts.prove_cuda_persistent_mixed_psi import validate


ROOT = Path(".cache/cuda-five-hour-2026-11-13-1059")
NAMES = {"p5": "persistent-p5-g4-floor-global-v1", "specialized": "persistent-specialized-g4-floor-global-v1"}


def parent(variant):
    source = ROOT / NAMES[variant] / "source"
    if not source.exists():
        pytest.skip("Reviewed persistent floor/global source required")
    return source


@pytest.mark.parametrize("variant", tuple(NAMES))
def test_exact_site_count_low7_and_explicit_coherent_loads_keep_barriers_and_graph(variant):
    source = parent(variant)
    old = (source / "src/native_persistent.cuh").read_text()
    new = transformed(old, variant)
    assert new.count(HELPER) == 1
    assert new.count("env.flags=PersistentMixedPsiFlags(") == (3 if variant == "p5" else 5)
    assert new.count(MIXED_LOAD) == (1 if variant == "p5" else 2)
    assert "(facts&127u)|((n==first||dz==1||dy==1||dx==1)?128u:0u)" in new
    assert "n,window[0]," in SITE_ASSIGNMENTS["paired_H"]
    assert "n,window[0]," in SITE_ASSIGNMENTS["paired_E0_E1"]
    assert "if constexpr(!Fast)" in SITE_ASSIGNMENTS["uniform_H"]
    assert new.count("__syncthreads()") == old.count("__syncthreads()")
    assert new.count("__threadfence()") == old.count("__threadfence()")
    for name, expected in BOUND.items():
        assert hashlib.sha256((source / "src" / name).read_bytes()).hexdigest() == expected


def test_unknown_source_cannot_inherit_cache_handoff_proof():
    with pytest.raises(ValueError, match="Exact reviewed"):
        transformed("wrong", "p5")


@pytest.mark.parametrize("variant", tuple(NAMES))
def test_actual_CXX_point_sites_preserve_first_handoff_and_low7_facts(tmp_path, variant):
    if not shutil.which("c++"):
        pytest.skip("Host C++ required")
    original = parent(variant)
    source = tmp_path / "source" / "src"
    source.mkdir(parents=True)
    for name in BOUND:
        (source / name).write_bytes((original / "src" / name).read_bytes())
    (source / "native_persistent.cuh").write_text(transformed((original / "src/native_persistent.cuh").read_text(), variant))
    report = validate(original, source.parent, tmp_path / "oracle", variant)
    counts = report["actual_CXX"]
    assert counts["errors"] == 0
    assert report["actual_point_assignment_sites_tested"] == (3 if variant == "p5" else 5)
    assert counts["actual_point_site_checks"] == (1425690 if variant == "p5" else 2376150)
    assert counts["same_point_prior_store_checks"] == 317520
    assert counts["all_low7_fact_checks"] == 6144
    assert counts["CG_route_checks"] == counts["CA_route_checks"] == 6144
