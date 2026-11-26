"""Memory-bounded reverse mode for a fixed simulation and material map."""

from dataclasses import replace
from math import isqrt
from operator import index

import jax
import jax.numpy as jnp
from jax.core import Tracer

from beamz.simulation.differentiation import build_differentiable_scan


def _positive_count(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        count = index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if count < 1:
        raise ValueError(f"{name} must be a positive integer")
    return count


def _nested_scan(program, steps, levels):
    if levels == 0 or steps <= 1:
        segment = replace(program, config=replace(program.config, num_steps=steps))
        return build_differentiable_scan(segment)
    target = max(2, round(steps ** (1 / (levels + 2))))
    block_steps = (steps + target - 1) // target
    groups, tail = divmod(steps, block_steps)
    advance = jax.checkpoint(_nested_scan(program, block_steps, levels - 1))
    advance_tail = (
        jax.checkpoint(_nested_scan(program, tail, levels - 1)) if tail else None
    )

    def run(state, coefficients):
        def chunk(carry, _):
            carry = carry._replace(t=jax.lax.stop_gradient(carry.t))
            return advance(carry, coefficients), None

        state = jax.lax.scan(chunk, state, xs=None, length=groups)[0]
        if advance_tail is not None:
            state = state._replace(t=jax.lax.stop_gradient(state.t))
            state = advance_tail(state, coefficients)
        return state

    return run


def build_host_value_and_grad(
    solver, material_map, objective, segment_steps, segments_per_group, has_aux
):
    """Build a host-driven value-and-gradient function for material parameters."""
    segment_steps = _positive_count(segment_steps, "segment_steps")
    count, tail = divmod(solver.program.config.num_steps, segment_steps)
    sizes = (segment_steps,) * count + ((tail,) if tail else ())
    per_group = (
        max(1, isqrt(len(sizes)))
        if segments_per_group is None
        else _positive_count(segments_per_group, "segments_per_group")
    )
    groups = [sizes[i : i + per_group] for i in range(0, len(sizes), per_group)]
    forward, reverse = {}, {}
    for count in set(sizes):
        scan = _nested_scan(solver.program, count, 2)

        def advance(state, parameters, scan=scan):
            state = state._replace(t=jax.lax.stop_gradient(state.t))
            return scan(state, solver.coefficients(material_map(parameters)))

        def transpose(state, parameters, seed, advance=advance):
            _, pullback = jax.vjp(advance, state, parameters)
            return pullback(seed)

        forward[count] = jax.jit(advance)
        reverse[count] = jax.jit(transpose)

    terminal = jax.jit(jax.value_and_grad(objective, has_aux=has_aux, allow_int=True))

    def value_and_grad(parameters, *, initial_state=None):
        if any(isinstance(leaf, Tracer) for leaf in jax.tree.leaves(parameters)):
            raise TypeError("Host checkpointing must run outside JAX transformations.")
        state = solver.initial_state
        sharding = state.ex.sharding
        if initial_state is not None:
            state = jax.device_put(initial_state, sharding)
        outer = []
        for group in groups:
            outer.append(jax.device_get(state))
            for count in group:
                state = forward[count](state, parameters)

        value, seed = terminal(state)
        del state
        gradient = jax.tree.map(jnp.zeros_like, parameters)
        for group in reversed(groups):
            initial = outer.pop()
            cursor = jax.device_put(initial, sharding)
            inner = [initial]
            for count in group[:-1]:
                cursor = forward[count](cursor, parameters)
                inner.append(jax.device_get(cursor))
            del cursor
            for count in reversed(group):
                start = jax.device_put(inner.pop(), sharding)
                seed, contribution = reverse[count](start, parameters, seed)
                gradient = jax.tree.map(jnp.add, gradient, contribution)
                jax.block_until_ready(gradient)
                del start
        return value, gradient

    return value_and_grad
