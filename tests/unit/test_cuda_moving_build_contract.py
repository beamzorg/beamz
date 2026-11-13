"""Reject misleading resource probes and incomplete private moving builds."""

import pytest

from scripts.check_cuda_moving_build import (
    REQUIRED_DEFINES,
    moving_resources,
    ninja_contract,
)


def ninja_fixture():
    defs = " ".join("-D" + value for value in REQUIRED_DEFINES)
    return (
        "build CMakeFiles/beamz_cuda_moving.dir/src/native_moving_kernel.cu.o: moving source.cu\n"
        f"  DEFINES = {defs}\n  FLAGS = --extended-lambda\n\n"
        "build CMakeFiles/beamz_cuda_core.dir/src/update.cu.o: core source.cu\n\n"
        "build CMakeFiles/_cuda_native.dir/src/native_runtime.cu.o: runtime source.cu\n\n",
        "rule moving\n  command = nvcc $DEFINES $FLAGS -c source.cu\n\n"
        "rule core\n  command = nvcc -rdc=true -c source.cu\n\n"
        "rule runtime\n  command = nvcc -rdc=true -c source.cu\n\n",
    )


def test_correct_mixed_compilation_contract():
    assert ninja_contract(*ninja_fixture())["verified"]


@pytest.mark.parametrize(
    "problem", ["source-definitions", "lambda", "rdc", "core-rdc", "contradiction"]
)
def test_real_command_failures_are_rejected(problem):
    ninja, rules = ninja_fixture()
    if problem == "source-definitions":
        ninja = ninja.replace("-DBEAMZ_CUDA_NATIVE_MOVING=1", "")
    elif problem == "lambda":
        ninja = ninja.replace("--extended-lambda", "")
    elif problem == "rdc":
        rules = rules.replace(
            "rule moving\n  command = nvcc", "rule moving\n  command = nvcc -rdc=true"
        )
    elif problem == "core-rdc":
        rules = rules.replace("-rdc=true", "", 1)
    else:
        ninja = ninja.replace(
            "--extended-lambda", "--extended-lambda -DBEAMZ_CUDA_NATIVE_MOVING=0"
        )
    with pytest.raises(ValueError):
        ninja_contract(ninja, rules)


RESOURCE = """ Function _ZN5beamz4cuda13moving_native25AsymmetricNoIOUniform1024ILi16EEEv:
  REG:64 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:1024
 Function _ZN5beamz4cuda13moving_native10AsymmetricILi16EEEv:
  REG:128 STACK:0 SHARED:0 LOCAL:0
 Function unrelated_ordinary_DFT:
  REG:40 STACK:32 SHARED:0 LOCAL:0
"""


def test_only_expected_moving_entry_points_define_the_resource_scope():
    assert len(moving_resources(RESOURCE, 2)) == 2


def test_persistent_entries_cannot_evade_resource_gate():
    extra = " Function _ZN5beamz4cuda17moving_persistent18PersistentTileStepILi16EEEv:\n  REG:64 STACK:0 SHARED:0 LOCAL:0\n"
    extra += " Function _ZN5beamz4cuda17moving_persistent15ResetPersistentEv:\n  REG:8 STACK:0 SHARED:0 LOCAL:0\n"
    assert len(moving_resources(RESOURCE + extra, 4)) == 4
    with pytest.raises(ValueError):
        moving_resources(RESOURCE + extra.replace("STACK:0", "STACK:8", 1), 4)


def test_bounded_exploration_cannot_admit_ordinary_or_unbounded_spills():
    extra = " Function _ZN5beamz4cuda17moving_persistent18PersistentTileStepILi16EEEv:\n  REG:64 STACK:8 SHARED:0 LOCAL:0\n"
    assert len(moving_resources(RESOURCE + extra, 3, 8)) == 3
    for bad in (RESOURCE.replace("STACK:0", "STACK:8", 1) + extra,
                RESOURCE + extra.replace("STACK:8", "STACK:24"),
                RESOURCE + extra.replace("LOCAL:0", "LOCAL:8")):
        with pytest.raises(ValueError):
            moving_resources(bad, 3, 16)


@pytest.mark.parametrize(
    "output",
    [
        "",
        RESOURCE.replace("STACK:0", "STACK:4", 1),
        RESOURCE.replace("LOCAL:0", "LOCAL:8", 1),
        RESOURCE + RESOURCE,
        RESOURCE.replace(" REG:64", " REG_MISSING:64"),
    ],
)
def test_empty_duplicate_missing_or_spilled_build_cannot_pass(output):
    with pytest.raises(ValueError):
        moving_resources(output, 2)
