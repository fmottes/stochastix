"""Timestamp normalization utilities for analysis APIs."""

from __future__ import annotations

import warnings

import jax.numpy as jnp


def normalize_time_scalar(t, *, arg_name: str) -> jnp.ndarray:
    """Normalize a scalar timestamp to float dtype without concretization."""
    t_arr = jnp.asarray(t)
    if t_arr.ndim != 0:
        raise ValueError(f'{arg_name} must be a scalar numeric time value.')
    if not jnp.issubdtype(t_arr.dtype, jnp.number):
        raise ValueError(f'{arg_name} must be a scalar numeric time value.')
    if jnp.issubdtype(t_arr.dtype, jnp.integer):
        warnings.warn(
            f'{arg_name} was provided as an integral scalar; it is interpreted as '
            'physical time (not index). This compatibility path is temporary and '
            'may become an error in a future release.',
            FutureWarning,
            stacklevel=2,
        )
    return t_arr.astype(jnp.result_type(float))
