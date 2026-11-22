from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Mapping, Protocol

import numpy as np

from beamz.simulation import _cuda_abi as abi

_FIELDS = ("hx", "hy", "hz", "ex", "ey", "ez")
_DFT = ("dft_vec_re", "dft_vec_im", "dft_weight_sum")
_PHASES = ("pre_e", "h", "e")
# One descriptor layout covers the existing CPML and no-CPML native packets.
@dataclass(frozen=True, slots=True)
class _Layout:
    source: int = 0
    monitor: int = 0
    input_count: int = 0
    output_count: int = 0
    inputs: Mapping[str, int] = field(default_factory=lambda: MappingProxyType({}))
    outputs: Mapping[str, int] = field(default_factory=lambda: MappingProxyType({}))
    deferred: frozenset[int] = frozenset()
    zeroable: frozenset[int] = frozenset()


def _layout(cpml: bool = True) -> _Layout:
    graph = abi.CPML_GRAPH_INPUT_COUNT if cpml else abi.YEE_GRAPH_INPUT_COUNT
    state = abi.CPML_GRAPH_OUTPUT_COUNT if cpml else abi.YEE_GRAPH_OUTPUT_COUNT
    source = graph + 2 * abi.FIELD_COUNT
    monitor = source + abi.SOURCE_GROUP_COUNT * abi.SOURCE_GROUP_BUFFER_COUNT
    count = monitor + abi.MONITOR_INPUT_COUNT + abi.MSTEP_PLAN_BUFFER_COUNT
    electric = abi.CPML_E_INPUT_OFFSET if cpml else 2 * abi.FIELD_COUNT
    metrics = (abi.CPML_H_METRIC_OFFSET, abi.CPML_E_METRIC_OFFSET) if cpml else (18,21)
    indices = {**dict(zip(_FIELDS, range(abi.FIELD_COUNT), strict=True)),
        **{f"{p}_psi_{i}":o+i for p,o in (("h",abi.CPML_H_PSI_INPUT_OFFSET),("e",abi.CPML_E_PSI_INPUT_OFFSET)) for i in range(abi.CPML_TERM_COUNT) if cpml},
        **{f"e_{kind}_{axis}":electric+k*3+i for k,kind in enumerate(("decay","source")) for i,axis in enumerate("xyz")},
        **{f"{p}_metric_{axis}":o+i for p,o in zip(("h","e"),metrics,strict=True) for i,axis in enumerate("zyx")},
        **{f"source_{p}_{axis}_coefficients":source+(k*3+i)*abi.SOURCE_GROUP_BUFFER_COUNT for k,p in enumerate(_PHASES) for i,axis in enumerate("xyz")},
        **{n:monitor+abi.MONITOR_DFT_RE_INPUT+i for i,n in enumerate(_DFT)},
        **{n:monitor+i for n,i in (("monitor_frequencies",abi.MONITOR_FREQUENCIES_INPUT),("time",abi.MONITOR_TIME_INPUT),("current_step",abi.MONITOR_CURRENT_STEP_INPUT),("elapsed_steps",abi.MONITOR_ELAPSED_STEPS_INPUT))},
        "e_term_metadata":electric+abi.FIELD_COUNT, "workspace_hx":graph, "workspace_psi_0":graph+abi.FIELD_COUNT}
    deferred = frozenset((*range(graph,source), *range(monitor+abi.MONITOR_INPUT_COUNT,count)))
    zeroable = deferred | frozenset((*range(abi.FIELD_COUNT),*(indices[n] for n in _DFT),*(indices[f"{p}_psi_{i}"] for p in ("h","e") for i in range(abi.CPML_TERM_COUNT) if cpml)))
    outputs = MappingProxyType({**dict(zip(_FIELDS,range(abi.FIELD_COUNT),strict=True)),
        **{f"{phase}_psi_{i}": abi.FIELD_COUNT + phase_index * abi.CPML_TERM_COUNT + i
            for phase_index, phase in enumerate(("h", "e"))
            for i in range(abi.CPML_TERM_COUNT) if cpml},
        **{n:state+2*abi.FIELD_COUNT+i for i,n in enumerate(_DFT)}})
    return _Layout(source,monitor,count,state+2*abi.FIELD_COUNT+2*len(_DFT),MappingProxyType(indices),outputs,deferred,zeroable)


_CPML, _YEE = _layout(), _layout(False)
_SOURCE, _MONITOR, _INPUT_COUNT, _OUTPUT_COUNT = _CPML.source, _CPML.monitor, _CPML.input_count, _CPML.output_count
_IN, _OUT, _DEFERRED, _ZEROABLE = _CPML.inputs, _CPML.outputs, _CPML.deferred, _CPML.zeroable
_H_PSI = slice(abi.FIELD_COUNT, abi.FIELD_COUNT + abi.CPML_TERM_COUNT)
_E_PSI = slice(_H_PSI.stop, abi.CPML_GRAPH_OUTPUT_COUNT)


class _Buffer(Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def dtype(self) -> np.dtype[Any]: ...
    @property
    def nbytes(self) -> int: ...


class _ResourcePlanner(Protocol):
    compiled_program_sha256: str

    def moving_resource_plan(
        self, inputs: tuple[_Buffer, ...], *, depth: int = 16, raw_slots: int = 3,
        scatters: tuple = (), recorders: tuple = (), tensor: bool = False,
        budget: int = ..., minimum_overlap: int = 0, scheduling_sms: int = 0, macro_slab_count: int = 16,
        boundary_code: int = 0, rolling_band: int = 0,
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class _Packet:
    arguments: tuple[_Buffer, ...] = ()
    attributes: Mapping[str, int | float] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if len(self.arguments) not in (_CPML.input_count, _YEE.input_count): raise ValueError("Unexpected CUDA buffer contract")
        object.__setattr__(self, "attributes", MappingProxyType(dict(self.attributes)))

    @property
    def _layout(self) -> _Layout:
        return _CPML if len(self.arguments) == _CPML.input_count else _YEE

    def _buffer(self, name: str = "") -> _Buffer:
        return self.arguments[self._layout.inputs[name]]

    def _state_inputs(self) -> dict[str, Any]:
        return {**{n:self._buffer(n) for n in (*_FIELDS,*_DFT)},
            "cpml_psi_h_terms":tuple(self._buffer(f"h_psi_{i}") for i in range(abi.CPML_TERM_COUNT) if self._layout is _CPML),
            "cpml_psi_e_terms":tuple(self._buffer(f"e_psi_{i}") for i in range(abi.CPML_TERM_COUNT) if self._layout is _CPML)}


@dataclass(frozen=True, slots=True)
class _StateContract:
    total_steps: int = 0
    initial_step: int = 0
    steps: int = 0
    ownership: Literal["private", "preserved", "donated"] = "private"

    def __post_init__(self) -> None:
        if not 0 <= self.initial_step <= self.total_steps: raise ValueError("state.current_step is outside the simulation time grid.")
        if not 0 < self.steps <= self.total_steps-self.initial_step:
            raise ValueError(f"num_steps must be in [1, {self.total_steps-self.initial_step}], got {self.steps}.")
        if self.ownership not in ("private", "preserved", "donated"): raise ValueError("Unknown CUDA state ownership")

    @property
    def _stop(self) -> int:
        return self.initial_step + self.steps
