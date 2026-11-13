"""Independent actual paired-read poisoning and runtime bridge binding."""

from pathlib import Path
import shutil

import pytest

from scripts.review_cuda_traced_exit_elision import CONTEXT, paired, poison_oracle, source_review


@pytest.mark.parametrize("depth", [4, 16])
def test_dead_current_electric_ghosts_can_be_poisoned_through_real_paired_reads(depth):
    result = poison_oracle()(paired.Geometry(8, 16, 32), version=depth, seed=29)
    assert result["checked_reads"] > 100000
    assert result["cross_chunk_snapshot_reads"] > 0


@pytest.mark.parametrize("mutation", ["erase-old-E-ghosts", "erase-current-H-ghosts", "admit-ghost-E-center", "reuse-old-cache-for-next-macro"])
def test_live_actual_consumers_reject_corresponding_poison(mutation):
    with pytest.raises(paired.Hazard):
        poison_oracle(mutation)(paired.Geometry(8, 16, 32), version=16, seed=13)


def test_all_import_sites_and_actual_host_macro_tail_bridge_are_bound():
    source = Path(".cache/cuda-two-hour-2026-11-13-0819/source-plane-axis-cg-v1/source")
    if not source.is_dir():
        pytest.skip("Frozen research source required")
    result = source_review(source)
    assert len(result["seven_actual_ImportPacket_sites"]) == 7
    assert result["next_macro_tail_DFT_and_graph_replay_field_bank_bridge_bound"]


def test_changed_next_macro_bridge_fails_closed_even_with_original_consumer(tmp_path):
    source = Path(".cache/cuda-two-hour-2026-11-13-0819/source-plane-axis-cg-v1/source")
    if not source.is_dir():
        pytest.skip("Frozen research source required")
    output = tmp_path / "source" / "src"
    output.mkdir(parents=True)
    for name in CONTEXT:
        shutil.copy2(source / "src" / name, output / name)
    host = output / "native_extension.cc"
    host.write_text(host.read_text().replace("            std::swap(current_,other_);", "            /* broken next macro field bank */", 1))
    with pytest.raises(ValueError, match="runtime bridge changed"):
        source_review(output.parent)
