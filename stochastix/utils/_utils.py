"""Miscellaneous utility functions."""

from __future__ import annotations

import jax.numpy as jnp


def entr_safe(p):
    """Like jax.scipy.special.entr, but with zero handling.

    Args:
        p: The probability distribution.

    Returns:
        -p * log2(p) if p > 0, 0 otherwise.
    """
    tiny = jnp.finfo(p.dtype).tiny
    p_clip = jnp.clip(p, a_min=tiny, a_max=1.0)

    return -p * jnp.log2(p_clip)


def algebraic_sigmoid(x: jnp.ndarray):
    """Compute an algebraic sigmoid function.

    This function is a with much more slowly decaying gradients in the tails
    with respect to the standard sigmoid function.
    It is defined as:

    .. math::
        f(x) = 0.5 + (x / (2 * sqrt(1 + x^2)))

    Args:
        x: The input array.

    Returns:
        The algebraic sigmoid of the input array.

    """
    return 0.5 + (x / (2 * jnp.sqrt(1 + x**2)))


def entropy(p: jnp.ndarray, base: float = 2) -> jnp.ndarray:
    """MLE estimator for the entropy of a probability distribution.

    Default calculation is in bits. No correction for finite sample size is applied.

    Args:
        p: The probability distribution.
        base: The base of the logarithm. Default is 2 (bits).

    Returns:
        The entropy of the probability distribution.
    """
    # entr calculates -plogp
    h = jnp.sum(entr_safe(p))
    return h / jnp.log2(base)
