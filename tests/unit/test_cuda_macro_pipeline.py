"""Cross-macro reuse must preserve old readers and exact physical versions."""
import pytest

from scripts.prove_cuda_macro_pipeline import macro_parent_theorem,token_pipeline
from scripts.prove_cuda_native_macro_pipeline import token_pipeline as native_pipeline


def test_previous_macro_upper_corner_dominates_all_old_readers():
    proof=macro_parent_theorem()
    assert proof['passed']
    assert sum(c['checks'] for c in proof['axis_cases'])==1186830


@pytest.mark.parametrize('macros',[2,4])
def test_native_banks_and_compact_histories_support_actual_macro_overlap(macros):
    result=native_pipeline(minimum=(5,7,9),owned=(3,5,7),depth=4,macros=macros,seed=13)
    assert result['history_epochs']==4
    assert result['early_macro_starts']>0
    assert result['traced_reads']>10000


@pytest.mark.parametrize('negative',['short-macro-stride','short-history-ring','wrong-initial-bank'])
def test_native_bank_or_history_lifetime_violations_are_detected(negative):
    with pytest.raises(AssertionError):native_pipeline(negative=negative)


def test_physical_frontier_requires_conservative_previous_macro_parent():
    with pytest.raises(AssertionError):token_pipeline(negative='floor-macro-parent')
