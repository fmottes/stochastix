"""Thermodynamic bounds and efficiency diagnostics."""

from __future__ import annotations

import jax


def tur_bound(mean_current: jax.Array, var_current: jax.Array) -> jax.Array:
    r"""Thermodynamic Uncertainty Relation lower bound on entropy production.

    For a time-homogeneous Markov jump process in a nonequilibrium steady state,
    the standard TUR states that an integrated current ``J`` accumulated over a
    fixed time window satisfies ``Var(J) / <J>^2 >= 2 / Sigma``, where ``Sigma``
    is the mean total entropy production (in units of ``k_B``) over that window.
    Rearranged, this lower-bounds entropy production from current fluctuations:

    ``Sigma >= 2 <J>^2 / Var(J)``

    This form should not be applied unmodified to transient, time-dependent, or
    non-Markovian dynamics; those settings require their corresponding TUR
    generalizations.

    Args:
        mean_current: The ensemble mean of an integrated current ``<J>``.
        var_current: The ensemble variance of that current ``Var(J)``.

    Returns:
        The TUR lower bound on the total entropy production (nats).
    """
    return 2.0 * mean_current**2 / var_current


def thermodynamic_efficiency(
    mean_current: jax.Array,
    var_current: jax.Array,
    entropy_production: jax.Array,
) -> jax.Array:
    r"""TUR quality factor of a current.

    The ratio of the TUR bound to the mean entropy production,

    ``Q = 2 <J>^2 / ( Var(J) * Sigma ) <= 1``,

    measures how close the current comes to saturating the steady-state
    uncertainty relation (``Q = 1`` is saturation). The bound ``Q <= 1`` assumes
    the same steady-state Markov conditions as `tur_bound` and exact ensemble
    moments; finite-sample estimates can exceed one. Useful as an optimization
    target.

    Args:
        mean_current: The ensemble mean of an integrated current ``<J>``.
        var_current: The ensemble variance of that current ``Var(J)``.
        entropy_production: The mean total entropy production ``Sigma`` (nats)
            over the same window.

    Returns:
        The dimensionless TUR quality factor ``Q``.
    """
    return 2.0 * mean_current**2 / (var_current * entropy_production)
