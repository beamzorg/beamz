from scripts.profile_cuda_native_phases import summarize


def test_overlapping_kernels_have_summed_and_union_scopes_and_policy_labels():
    records = []
    for policy, start, end in [(1, 0, 10), (2, 5, 15), (0, 20, 25)]:
        records.append(
            {
                "kind": "kernel",
                "name": f"AsymmetricILi16ELi1ELb0ELb1ELi{policy}E",
                "start_ns": start,
                "end_ns": end,
            }
        )
    result = summarize(records)
    assert result["kernel_total_s"] == 25e-9
    assert result["kernel_interval_union_s"] == 20e-9
    assert result["summed_over_union_concurrency_factor"] == 1.25
    assert result["maximum_simultaneous_kernels"] == 2
    assert any("NoIO CPML/boundary" in row["group"] for row in result["groups"])
    assert any("interior observation" in row["group"] for row in result["groups"])
