"""Mass-action and constant kinetics for elementary reactions."""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.lax import lgamma

from ._base import AbstractKinetics


class Constant(AbstractKinetics):
    """Constant kinetics for zeroth-order reactions.

    This class implements constant kinetics where the reaction rate is independent
    of species concentrations, representing a zeroth-order reaction.

    Attributes:
        k: The rate constant.
        transform: A function to transform the rate constant `k` before it is
            used, e.g., `jnp.exp` for log-rates.
        _requires_species: Tuple of species names required by this kinetics.

    Methods:
        propensity_fn: Computes the constant propensity.
        ode_rate_fn: Computes the constant deterministic rate.
    """

    k: float | jnp.ndarray
    transform: Callable = eqx.field(static=True)
    _requires_species: tuple[str, ...] = eqx.field(static=True)

    def __init__(self, k: float, transform: Callable = None):
        """Initializes the Constant kinetics.

        Args:
            k: The rate constant.
            transform: A function to transform the rate constant `k` before it's
                used, e.g., `jnp.exp` for log-rates. Defaults to the identity function.
        """
        self.k = k
        self.transform = (lambda val: val) if transform is None else transform
        self._requires_species = ()

    def propensity_fn(self, x, reactants, t=None, volume=1.0):
        """Computes the constant propensity.

        For zeroth-order reactions, the propensity is independent of species
        counts and volume, and is equal to the rate constant `k`.

        Args:
            x: The current state vector (unused).
            reactants: The reactant stoichiometry (unused).
            t: The current time (unused).
            volume: The system volume (unused).

        Returns:
            The rate constant `k` as a float.
        """
        return self.transform(self.k)

    def ode_rate_fn(self, x, reactants, t=None, volume=1.0):
        """Computes the constant deterministic rate.

        For a zeroth-order reaction, the deterministic rate is equal to the
        rate constant `k`.

        Args:
            x: The current state vector (unused).
            reactants: The reactant stoichiometry (unused).
            t: The current time (unused).
            volume: The system volume (unused).

        Returns:
            The rate constant `k` as a float.
        """
        return self.transform(self.k)


class MassAction(AbstractKinetics):
    """Mass-action kinetics for elementary reactions.

    The propensity is computed as `k * product_i C(x_i, s_i)`, where `C` is the
    binomial coefficient, `x_i` is the count of species `i`, and `s_i` is its
    stoichiometric coefficient. This corresponds to the standard stochastic
    formulation of mass-action kinetics.

    The deterministic rate is computed as `k * product_i (x_i/V)^s_i`, where `V`
    is the volume.

    Attributes:
        k: The rate constant.
        transform: A function to transform the rate constant `k` before it's
            used, e.g., `jnp.exp` for log-rates.
        _requires_species: Tuple of species names required by this kinetics.

    Methods:
        propensity_fn: Computes the stochastic mass-action propensity.
        ode_rate_fn: Computes the deterministic mass-action rate.
    """

    k: float | jnp.ndarray
    transform: Callable = eqx.field(static=True)
    _requires_species: tuple[str, ...] = eqx.field(static=True)

    def __init__(self, k: float, transform: Callable = None):
        """Initializes the MassAction kinetics.

        Args:
            k: The rate constant. The units depend on the order of the reaction.
            transform: A function to transform the rate constant `k` before it's
                used, e.g., `jnp.exp` for log-rates. Defaults to the identity function.
        """
        self.k = k
        self.transform = (lambda val: val) if transform is None else transform
        self._requires_species = ()

    @staticmethod
    def _binom(n, k):
        """Computes the binomial coefficient C(n, k)."""
        # Use jax.lax.cond to avoid evaluating the lgamma branch when k > n,
        # which would produce NaNs. `cond` provides true short-circuiting.
        return jax.lax.cond(
            k > n,
            lambda: 0.0,
            lambda: jnp.exp(lgamma(n + 1.0) - lgamma(k + 1.0) - lgamma(n - k + 1.0)),
        )

    def propensity_fn(self, x, reactants, t=None, volume=1.0):
        """Computes the stochastic mass-action propensity.

        The propensity is calculated as `k_corr * product_i C(x_i, s_i)`, where
        `k_corr` is the rate constant corrected for the system volume.

        Args:
            x: The current state vector (species counts).
            reactants: The reactant stoichiometry vector.
            t: The current time (unused).
            volume: The system volume.

        Returns:
            The stochastic propensity value.
        """
        k = self.transform(self.k)
        order = jnp.sum(reactants)
        # Apply volume correction for macroscopic rate constants
        k_corrected = k * jnp.power(volume, 1 - order)
        return k_corrected * jnp.prod(jax.vmap(self._binom)(x, reactants))

    def ode_rate_fn(self, x, reactants, t=None, volume=1.0):
        """Computes the deterministic mass-action rate.

        The rate is calculated as `k * product_i (x_i/V)^s_i`, where `V` is the
        system volume. The final result is returned in units of molecules/time.

        Args:
            x: The current state vector (species counts).
            reactants: The reactant stoichiometry vector.
            t: The current time (unused).
            volume: The system volume.

        Returns:
            The deterministic rate in molecules per unit time.
        """
        k = self.transform(self.k)
        # Convert species counts to concentrations for the rate law
        concentrations = x / volume
        # The rate law is k * product(concentrations^reactants)
        # This gives rate in concentration/time. We multiply by volume
        # to get rate in molecules/time, consistent with the S-matrix.
        rate_in_conc_units = k * jnp.prod(concentrations**reactants)
        return rate_in_conc_units * volume


### Mass Action explicit functions utilities


def mass_action_propensity(
    rate_constant: float | jnp.floating,
    x: jnp.ndarray,
    reactants: jnp.ndarray,
    volume: float | jnp.floating = 1.0,
) -> jnp.ndarray:
    """Compute the mass action propensity.

    Args:
        rate_constant: The rate constant.
        x: The current state vector.
        reactants: The reactant stoichiometry vector.
        volume: The system volume. Defaults to 1.0.
    """
    order = jnp.sum(reactants)
    # Apply volume correction for macroscopic rate constants
    k_corrected = rate_constant * jnp.power(volume, 1 - order)
    return k_corrected * jnp.prod(jax.vmap(MassAction._binom)(x, reactants))


def mass_action_ode_rate(
    rate_constant: float | jnp.floating,
    x: jnp.ndarray,
    reactants: jnp.ndarray,
    volume: float | jnp.floating = 1.0,
) -> jnp.ndarray:
    """Compute the mass action deterministic rate.

    Args:
        rate_constant: The rate constant.
        x: The current state vector.
        reactants: The reactant stoichiometry vector.
        volume: The system volume. Defaults to 1.0.
    """
    # Convert species counts to concentrations for the rate law
    concentrations = x / volume
    # The rate law is k * product(concentrations^reactants)
    # This gives rate in concentration/time. We multiply by volume
    # to get rate in molecules/time, consistent with the S-matrix.
    rate_in_conc_units = rate_constant * jnp.prod(concentrations**reactants)
    return rate_in_conc_units * volume
