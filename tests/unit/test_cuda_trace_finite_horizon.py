"""Finite-horizon optimization must preserve future and observer consumers."""
import pytest

from scripts.cuda_trace_cache_exit import prove_exit
from scripts.cuda_trace_terminal_reuse import terminal_cut
from scripts.cuda_trace_systolic_compiler import ThreadMap,prove_deduplicated_z
from scripts.cuda_trace_source_wave_cache import prove_wave_lifetime
from scripts.cuda_phase_schedule_ir import ScheduleHazard


@pytest.mark.parametrize('depth',[4,16])
def test_terminal_cut_has_only_current_H_evolving_consumers(depth):
    plan=terminal_cut(depth)
    for field,consumers in plan['terminal_consumers'].items():
        assert bool(consumers)==field.startswith('H')
    assert 'FinalField H' in plan['terminal_H_provider']


@pytest.mark.parametrize('depth',[4,16])
def test_next_macro_cannot_use_the_elided_trails(depth):
    with pytest.raises(ValueError,match='Terminal E'):
        terminal_cut(depth,consumer_horizon=depth+1)


def test_shared_observer_prevents_exit_barrier_elision():
    with pytest.raises(ValueError,match='observer'):
        prove_exit(shared_observer=True)
    assert prove_exit()['last_shared_E_has_no_in_CTA_reader']


@pytest.mark.parametrize('mapping',[ThreadMap(threads=1024,z_groups=2),
    ThreadMap(cache=(16,8,32),threads=1024,z_groups=4),
    ThreadMap(cache=(8,8,32),threads=512,z_groups=2)])
def test_frontier_deduplication_reuses_real_Y_publications(mapping):
    proof=prove_deduplicated_z(mapping,steps=4)
    assert proof['reused_y_publications']>0
    assert proof['allocated_z_words_per_family']<proof['prior_z_words_per_family']


def test_wave_cache_uses_existing_reader_retirement_and_publication():
    assert prove_wave_lifetime()['result']=='pass'
    with pytest.raises(ScheduleHazard):
        prove_wave_lifetime(early_overwrite=True)
    with pytest.raises(ScheduleHazard):
        prove_wave_lifetime(omit_publication=True)
