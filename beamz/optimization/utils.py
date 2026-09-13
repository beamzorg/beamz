"""Convenience functions for JAX post-processing callbacks."""

import jax.numpy as jnp


def get_amps(sim_data, monitor_name, **sel_kwargs):
    return sim_data[monitor_name].amps.sel(**sel_kwargs).values


def sum_abs_squared(array):
    return jnp.sum(jnp.abs(jnp.asarray(array)) ** 2)
