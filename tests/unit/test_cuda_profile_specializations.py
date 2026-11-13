"""Specialized kernels must retain their policy in timing reports."""
from scripts.profile_cuda_native_phases import summarize


def test_specialized_policy_names_preserve_duration_and_overlap_accounting():
    names = [
        "_ZN5beamz4cuda13moving_native25AsymmetricNoIOUniform1024ILi16ELi2EEEv",
        "_ZN5beamz4cuda13moving_native24AsymmetricNoIORectCachedILi16ELi2EEEv",
        "_ZN5beamz4cuda13moving_native29AsymmetricObserverUniform1024ILi16ELi2EEEv",
        "_ZN5beamz4cuda13moving_native25AsymmetricModeSource54SixILi16ELi2ELi5EEEv",
        "_ZN5beamz4cuda13moving_native15TemporalSourcesILi16ELi2ELb1ELi4EEEv",
    ]
    result = summarize([dict(kind="kernel", name=name, start_ns=i, end_ns=i+10)
                        for i, name in enumerate(names)])
    groups = {row["group"]: row for row in result["groups"]}
    assert len(groups) == 4
    assert groups["moving policy 1 NoIO CPML/boundary (uniform)"]["count"] == 2
    assert groups["moving policy 2 interior observation (uniform)"]["duration_ns"] == 10
    assert groups["moving policy 5 full general fallback (traced mode source)"]["duration_ns"] == 10
    assert groups["moving policy 4 interior source+observation (shared source/effects)"]["duration_ns"] == 10
    assert result["kernel_total_s"] == 50 / 1e9
    assert result["kernel_interval_union_s"] == 14 / 1e9
    assert result["maximum_simultaneous_kernels"] == 5
