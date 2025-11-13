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
    p_clip = jnp.clip(p, min=tiny, max=1.0)

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


def rate_constant_conc_to_count(
    rate_constant: jnp.ndarray | float,
    reaction_order: float | int | jnp.floating | jnp.integer,
    volume: float | jnp.floating,
    use_molar_units: bool = True,
    avogadro_number: float | jnp.floating | None = None,
    return_log: bool = False,
) -> jnp.ndarray:
    """Convert a concentration-based rate constant to a count-based constant.

    This converts the macroscopic (deterministic ODE) rate constant ``k`` to the
    mesoscopic (stochastic) propensity constant ``c`` used with molecule counts.

    General relation for an m-th order reaction:
      c = k * (V)^(1 - m)                [number-density units]
      c = k * (N_A * V)^(1 - m)          [molar units]

    Computation is performed in log10 space for numerical stability and interpretability.

    Args:
        rate_constant: Macroscopic rate constant ``k`` (in concentration units).
        reaction_order: Total order ``m`` of the reaction (sum of reactant stoichiometries). May be non-integer for effective kinetics.
        volume: System volume. Liters if ``use_molar_units`` is True.
        use_molar_units: Whether ``k`` is in molar units (e.g., M^(1-m)/s).
        avogadro_number: Optional Avogadro's number override. Defaults to 6.02214076e23.
        return_log: If True, return log10(c) instead of c.

    Returns:
        The propensity constant ``c`` (or its log10 if ``return_log`` is True).
    """
    # Normalize dtypes to a common floating dtype for stable numeric ops
    k = jnp.asarray(rate_constant)
    vol = jnp.asarray(volume)
    common_dtype = jnp.result_type(k, vol, jnp.array(1.0))
    k = k.astype(common_dtype)
    vol = vol.astype(common_dtype)

    if jnp.any(vol <= 0):
        raise ValueError(
            'volume must be positive to compute a logarithm-based conversion.'
        )
    if reaction_order is None:
        raise ValueError('reaction_order must be provided.')

    m = jnp.asarray(reaction_order, dtype=common_dtype)
    one_minus_m = 1 - m

    # log10(k)
    tiny = jnp.finfo(k.dtype).tiny
    # Use maximum to avoid dtype surprises from clip with mixed dtypes
    log10_k = jnp.log10(jnp.maximum(k, tiny))

    # log10 of scaling base
    if use_molar_units:
        NA = 6.02214076e23 if avogadro_number is None else avogadro_number
        NA = jnp.asarray(NA, dtype=common_dtype)
        if jnp.any(NA <= 0):
            raise ValueError('avogadro_number must be positive when provided.')
        log10_base = jnp.log10(vol) + jnp.log10(NA)
    else:
        log10_base = jnp.log10(vol)

    log10_c = log10_k + one_minus_m * log10_base

    if return_log:
        return jnp.where(k <= 0, -jnp.inf, log10_c)

    c = jnp.power(10.0, log10_c)
    return jnp.where(k <= 0, 0.0, c)
