"""Miscellaneous utility functions."""

from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.special import entr


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
        base: The base of the logarithm. Default is 2.

    Returns:
        The entropy of the probability distribution.
    """
    # entr calculates -plogp
    h = jnp.sum(entr(p))
    return h / jnp.log(base)
