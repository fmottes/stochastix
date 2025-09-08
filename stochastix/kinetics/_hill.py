"""Hill kinetics for single-regulator gene expression models."""

from __future__ import annotations

import warnings
from collections.abc import Callable

import equinox as eqx
import jax.nn as nn
import jax.numpy as jnp

from ._base import AbstractKinetics


class HillActivator(AbstractKinetics):
    """Hill kinetics for gene activation by a single regulator.

    This class implements Hill kinetics for modeling gene activation, where the
    expression rate increases with regulator concentration following a sigmoidal
    response curve.

    Attributes:
        regulator: The name of the regulator species that activates the reaction.
        regulator_idx: The index of the regulator species in the state vector.
        v: The limiting rate, in concentration/time.
        K: The half-saturation constant, in concentration units.
        n: The Hill coefficient. Must be positive.
        v0: The leaky (basal) expression rate, in concentration/time.
        transform_v: A function to transform the `v` parameter.
        transform_K: A function to transform the `K` parameter.
        transform_n: A function to transform the `n` parameter.
        transform_v0: A function to transform the `v0` parameter.
        _requires_species: Tuple of species names required by this kinetics.

    Methods:
        propensity_fn: Calculate the Hill activator propensity.
        ode_rate_fn: Calculate the Hill activator ODE rate.
    """

    regulator: str = eqx.field(static=True)
    regulator_idx: int
    v: float | jnp.ndarray
    transform_v: Callable = eqx.field(static=True)
    v0: float | jnp.ndarray
    transform_v0: Callable = eqx.field(static=True)
    K: float | jnp.ndarray
    transform_K: Callable = eqx.field(static=True)
    n: float | jnp.ndarray
    transform_n: Callable = eqx.field(static=True)
    _requires_species: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        regulator: str,
        v: float | jnp.ndarray,
        K: float | jnp.ndarray,
        n: float | jnp.ndarray = 1.0,
        v0: float | jnp.ndarray = 0.0,
        transform_v: Callable = None,
        transform_K: Callable = None,
        transform_n: Callable = None,
        transform_v0: Callable = None,
        *,
        _regulator_idx: int = -1,
    ):
        """Initializes the HillActivator kinetics.

        Args:
            regulator: The name of the regulator species.
            v: The limiting rate (concentration/time).
            K: The half-saturation constant (concentration).
            n: The Hill coefficient (must be positive). Defaults to 1.0.
            v0: The leaky expression rate (concentration/time). Defaults to 0.0.
            transform_v: A transform for `v`.
            transform_K: A transform for `K`.
            transform_n: A transform for `n`.
            transform_v0: A transform for `v0`.
            _regulator_idx: Internal use: index of the regulator species.
        """
        if v < 0:
            if transform_v is None:
                raise ValueError('`v` must be positive for HillActivator.')
            else:
                warnings.warn(
                    '`v` is negative. Please ensure the provided `transform_v` maps it to a positive value.'
                )
        if v0 < 0:
            if transform_v0 is None:
                raise ValueError('`v0` must be positive for HillActivator.')
            else:
                warnings.warn(
                    '`v0` is negative. Please ensure the provided `transform_v0` maps it to a positive value.'
                )

        if K < 0:
            if transform_K is None:
                raise ValueError('`K` must be positive for HillActivator.')
            else:
                warnings.warn(
                    '`K` is negative. Please ensure the provided `transform_K` maps it to a positive value.'
                )

        if n < 0:
            if transform_n is None:
                raise ValueError('`n` must be positive for HillActivator.')
            else:
                warnings.warn(
                    '`n` is negative. Please ensure the provided `transform_n` maps it to a positive value.'
                )

        self.regulator = regulator
        self.v = v
        self.transform_v = (lambda val: val) if transform_v is None else transform_v
        self.v0 = v0
        self.transform_v0 = (lambda val: val) if transform_v0 is None else transform_v0
        self.K = K
        self.transform_K = (lambda val: val) if transform_K is None else transform_K
        self.n = n
        self.transform_n = (lambda val: val) if transform_n is None else transform_n
        self._requires_species = (regulator,)
        self.regulator_idx = _regulator_idx

    def _bind_to_network(self, species_map: dict[str, int]):
        """Binds the kinetics to a network by setting the regulator index."""
        idx = species_map[self.regulator]
        return HillActivator(
            regulator=self.regulator,
            v=self.v,
            K=self.K,
            n=self.n,
            v0=self.v0,
            transform_v=self.transform_v,
            transform_K=self.transform_K,
            transform_n=self.transform_n,
            transform_v0=self.transform_v0,
            _regulator_idx=idx,
        )

    def propensity_fn(self, x, reactants, t=None, volume=1.0):
        """Calculate the Hill activator propensity.

        The propensity is given by `v0 + v * X^n / (K^n + X^n)`.

        Args:
            x: The current state of the system (species counts).
            reactants: Stoichiometry of the reactants. Unused for Hill kinetics.
            t: The current time. Unused for Hill kinetics.
            volume: The volume of the system.

        Returns:
            The propensity of the reaction.
        """
        X = x[self.regulator_idx]
        v = self.transform_v(self.v)
        v0 = self.transform_v0(self.v0)
        K = self.transform_K(self.K)
        n = self.transform_n(self.n)

        K_in_counts = K * volume
        safe_K = jnp.maximum(K_in_counts, jnp.finfo(jnp.float32).eps)
        ratio = X / safe_K

        # The expression `ratio**n / (1 + ratio**n)` is equivalent to sigmoid(n * log(ratio))
        # and is more numerically stable.
        eps = jnp.finfo(jnp.float32).eps
        safe_ratio = jnp.maximum(ratio, eps)
        log_arg = n * jnp.log(safe_ratio)
        prop = nn.sigmoid(log_arg)

        return (v0 * volume) + (v * volume) * prop

    def ode_rate_fn(self, x, reactants, t=None, volume=1.0):
        """Calculate the Hill activator ODE rate.

        The rate is given by `v0 + v * X_conc^n / (K^n + X_conc^n)`, where
        `X_conc` is the concentration of the regulator. The result is returned
        in units of molecules/time.

        Args:
            x: The current state of the system (species counts).
            reactants: Stoichiometry of the reactants. Unused for Hill kinetics.
            t: The current time. Unused for Hill kinetics.
            volume: The volume of the system.

        Returns:
            The rate of change of the species (molecules/time).
        """
        X = x[self.regulator_idx]
        X_conc = X / volume
        v = self.transform_v(self.v)
        v0 = self.transform_v0(self.v0)
        K = self.transform_K(self.K)
        n = self.transform_n(self.n)

        safe_K = jnp.maximum(K, jnp.finfo(jnp.float32).eps)
        ratio = X_conc / safe_K

        # The expression `ratio**n / (1 + ratio**n)` is equivalent to sigmoid(n * log(ratio))
        # and is more numerically stable.
        eps = jnp.finfo(jnp.float32).eps
        safe_ratio = jnp.maximum(ratio, eps)
        log_arg = n * jnp.log(safe_ratio)
        rate_in_conc_units = v * nn.sigmoid(log_arg)

        return (v0 + rate_in_conc_units) * volume


class HillRepressor(AbstractKinetics):
    """Hill kinetics for gene repression by a single regulator.

    This class implements Hill kinetics for modeling gene repression, where the
    expression rate decreases with regulator concentration following an inverted
    sigmoidal response curve.

    Attributes:
        regulator: The name of the regulator species that represses the reaction.
        regulator_idx: The index of the regulator species in the state vector.
        v: The limiting rate, in concentration/time.
        K: The half-saturation constant, in concentration units.
        n: The Hill coefficient. Must be positive.
        v0: The leaky (basal) expression rate, in concentration/time.
        transform_v: A function to transform the `v` parameter.
        transform_K: A function to transform the `K` parameter.
        transform_n: A function to transform the `n` parameter.
        transform_v0: A function to transform the `v0` parameter.
        _requires_species: Tuple of species names required by this kinetics.

    Methods:
        propensity_fn: Calculate the Hill repressor propensity.
        ode_rate_fn: Calculate the Hill repressor ODE rate.
    """

    regulator: str = eqx.field(static=True)
    regulator_idx: int
    v: float | jnp.ndarray
    transform_v: Callable = eqx.field(static=True)
    v0: float | jnp.ndarray
    transform_v0: Callable = eqx.field(static=True)
    K: float | jnp.ndarray
    transform_K: Callable = eqx.field(static=True)
    n: float | jnp.ndarray
    transform_n: Callable = eqx.field(static=True)
    _requires_species: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        regulator: str,
        v: float | jnp.ndarray,
        K: float | jnp.ndarray,
        n: float | jnp.ndarray = 1.0,
        v0: float | jnp.ndarray = 0.0,
        transform_v: Callable = None,
        transform_K: Callable = None,
        transform_n: Callable = None,
        transform_v0: Callable = None,
        *,
        _regulator_idx: int = -1,
    ):
        """Initializes the HillRepressor kinetics.

        Args:
            regulator: The name of the regulator species.
            v: The limiting rate (concentration/time).
            K: The half-saturation constant (concentration).
            n: The Hill coefficient (must be positive). Defaults to 1.0.
            v0: The leaky expression rate (concentration/time). Defaults to 0.0.
            transform_v: A transform for `v`.
            transform_K: A transform for `K`.
            transform_n: A transform for `n`.
            transform_v0: A transform for `v0`.
            _regulator_idx: Internal use: index of the regulator species.
        """
        if v < 0:
            if transform_v is None:
                raise ValueError('`v` must be positive for HillRepressor.')
            else:
                warnings.warn(
                    '`v` is negative. Please ensure the provided `transform_v` maps it to a positive value.'
                )
        if v0 < 0:
            if transform_v0 is None:
                raise ValueError('`v0` must be positive for HillRepressor.')
            else:
                warnings.warn(
                    '`v0` is negative. Please ensure the provided `transform_v0` maps it to a positive value.'
                )

        if K < 0:
            if transform_K is None:
                raise ValueError('`K` must be positive for HillRepressor.')
            else:
                warnings.warn(
                    '`K` is negative. Please ensure the provided `transform_K` maps it to a positive value.'
                )

        if n < 0:
            if transform_n is None:
                raise ValueError('`n` must be positive for HillRepressor.')
            else:
                warnings.warn(
                    '`n` is negative. Please ensure the provided `transform_n` maps it to a positive value.'
                )

        self.regulator = regulator
        self.v = v
        self.transform_v = (lambda val: val) if transform_v is None else transform_v
        self.v0 = v0
        self.transform_v0 = (lambda val: val) if transform_v0 is None else transform_v0
        self.K = K
        self.transform_K = (lambda val: val) if transform_K is None else transform_K
        self.n = n
        self.transform_n = (lambda val: val) if transform_n is None else transform_n
        self._requires_species = (regulator,)
        self.regulator_idx = _regulator_idx

    def _bind_to_network(self, species_map: dict[str, int]):
        """Binds the kinetics to a network by setting the regulator index."""
        idx = species_map[self.regulator]
        return HillRepressor(
            regulator=self.regulator,
            v=self.v,
            K=self.K,
            n=self.n,
            v0=self.v0,
            transform_v=self.transform_v,
            transform_K=self.transform_K,
            transform_n=self.transform_n,
            transform_v0=self.transform_v0,
            _regulator_idx=idx,
        )

    def propensity_fn(self, x, reactants, t=None, volume=1.0):
        """Calculate the Hill repressor propensity.

        The propensity is given by `v0 + v * K^n / (K^n + X^n)`, which is
        equivalent to `v0 + v / (1 + (X/K)^n)`. The result is returned in
        units of molecules/time.

        Args:
            x: The current state of the system (species counts).
            reactants: Stoichiometry of the reactants. Unused for Hill kinetics.
            t: The current time. Unused for Hill kinetics.
            volume: The volume of the system.

        Returns:
            The propensity of the reaction (molecules/time).
        """
        X = x[self.regulator_idx]
        v = self.transform_v(self.v)
        v0 = self.transform_v0(self.v0)
        K = self.transform_K(self.K)
        n = self.transform_n(self.n)

        K_in_counts = K * volume
        safe_K = jnp.maximum(K_in_counts, jnp.finfo(jnp.float32).eps)
        ratio = X / safe_K

        # The expression `1 / (1 + ratio**n)` is equivalent to sigmoid(-n * log(ratio))
        # and is more numerically stable.
        eps = jnp.finfo(jnp.float32).eps
        safe_ratio = jnp.maximum(ratio, eps)
        log_arg = -n * jnp.log(safe_ratio)
        prop = nn.sigmoid(log_arg)

        return (v0 * volume) + (v * volume) * prop

    def ode_rate_fn(self, x, reactants, t=None, volume=1.0):
        """Calculate the Hill repressor ODE rate.

        The rate is given by `v0 + v * K^n / (K^n + X_conc^n)`. The result is
        returned in units of molecules/time.

        Args:
            x: The current state of the system (species counts).
            reactants: Stoichiometry of the reactants. Unused for Hill kinetics.
            t: The current time. Unused for Hill kinetics.
            volume: The volume of the system.

        Returns:
            The rate of change of the species (molecules/time).
        """
        X = x[self.regulator_idx]
        X_conc = X / volume
        v = self.transform_v(self.v)
        v0 = self.transform_v0(self.v0)
        K = self.transform_K(self.K)
        n = self.transform_n(self.n)

        safe_K = jnp.maximum(K, jnp.finfo(jnp.float32).eps)
        ratio = X_conc / safe_K

        # The expression `1 / (1 + ratio**n)` is equivalent to sigmoid(-n * log(ratio))
        # and is more numerically stable.
        eps = jnp.finfo(jnp.float32).eps
        safe_ratio = jnp.maximum(ratio, eps)
        log_arg = -n * jnp.log(safe_ratio)
        rate_in_conc_units = v * nn.sigmoid(log_arg)

        return (v0 + rate_in_conc_units) * volume


class HillSingleRegulator(AbstractKinetics):
    """Hill kinetics for a single regulator with flexible activation/repression.

    This class provides a general form of Hill kinetics that can model both
    activation and repression using a single formulation. The sign of the Hill
    coefficient `n` determines the regulatory behavior.

    Attributes:
        regulator: The name of the regulator species.
        regulator_idx: The index of the regulator species in the state vector.
        v: The limiting rate, in concentration/time.
        K: The half-saturation constant, in concentration units.
        n: The Hill coefficient. Positive for activation, negative for repression.
        v0: The leaky (basal) expression rate, in concentration/time.
        transform_v: A transform for the `v` parameter.
        transform_K: A transform for the `K` parameter.
        transform_n: A transform for the `n` parameter.
        transform_v0: A transform for the `v0` parameter.
        _requires_species: Tuple of species names required by this kinetics.

    Methods:
        propensity_fn: Calculate the Hill propensity.
        ode_rate_fn: Calculate the Hill ODE rate.
    """

    regulator: str = eqx.field(static=True)
    regulator_idx: int
    v: float | jnp.ndarray
    transform_v: Callable = eqx.field(static=True)
    v0: float | jnp.ndarray
    transform_v0: Callable = eqx.field(static=True)
    K: float | jnp.ndarray
    transform_K: Callable = eqx.field(static=True)
    n: float | jnp.ndarray
    transform_n: Callable = eqx.field(static=True)
    _requires_species: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        regulator: str,
        v: float | jnp.ndarray,
        K: float | jnp.ndarray,
        n: float | jnp.ndarray,
        v0: float | jnp.ndarray = 0.0,
        transform_v: Callable = None,
        transform_K: Callable = None,
        transform_n: Callable = None,
        transform_v0: Callable = None,
        *,
        _regulator_idx: int = -1,
    ):
        """Initializes the HillSingleRegulator kinetics.

        Args:
            regulator: The name of the regulator species.
            v: The limiting rate (concentration/time).
            K: The half-saturation constant (concentration).
            n: The Hill coefficient (positive for activation, negative for repression).
            v0: The leaky expression rate (concentration/time). Defaults to 0.0.
            transform_v: A transform for `v`.
            transform_K: A transform for `K`.
            transform_n: A transform for `n`.
            transform_v0: A transform for `v0`.
            _regulator_idx: Internal use: index of the regulator species.
        """
        if v < 0:
            if transform_v is None:
                raise ValueError('`v` must be positive for HillSingleRegulator.')
            else:
                warnings.warn(
                    '`v` is negative. Please ensure the provided `transform_v` maps it to a positive value.'
                )
        if v0 < 0:
            if transform_v0 is None:
                raise ValueError('`v0` must be positive for HillSingleRegulator.')
            else:
                warnings.warn(
                    '`v0` is negative. Please ensure the provided `transform_v0` maps it to a positive value.'
                )
        if K < 0:
            if transform_K is None:
                raise ValueError('`K` must be positive for HillSingleRegulator.')
            else:
                warnings.warn(
                    '`K` is negative. Please ensure the provided `transform_K` maps it to a positive value.'
                )
        self.regulator = regulator
        self.v = v
        self.transform_v = (lambda val: val) if transform_v is None else transform_v
        self.v0 = v0
        self.transform_v0 = (lambda val: val) if transform_v0 is None else transform_v0
        self.K = K
        self.transform_K = (lambda val: val) if transform_K is None else transform_K
        self.n = n
        self.transform_n = (lambda val: val) if transform_n is None else transform_n
        self._requires_species = (regulator,)
        self.regulator_idx = _regulator_idx

    def _bind_to_network(self, species_map: dict[str, int]):
        """Binds the kinetics to a network by setting the regulator index."""
        idx = species_map[self.regulator]
        return HillSingleRegulator(
            regulator=self.regulator,
            v=self.v,
            K=self.K,
            n=self.n,
            v0=self.v0,
            transform_v=self.transform_v,
            transform_K=self.transform_K,
            transform_n=self.transform_n,
            transform_v0=self.transform_v0,
            _regulator_idx=idx,
        )

    def propensity_fn(self, x, reactants, t=None, volume=1.0):
        """Calculate the Hill propensity.

        The propensity is calculated using a numerically stable formulation. The
        parameter `v` is assumed to be in units of concentration/time and `K` in
        concentration units. The result is returned in molecules/time.

        Args:
            x: The current state of the system (species counts).
            reactants: Stoichiometry of the reactants. Unused for Hill kinetics.
            t: The current time. Unused for Hill kinetics.
            volume: The volume of the system.

        Returns:
            The propensity of the reaction in molecules/time.
        """
        X = x[self.regulator_idx]
        v = self.transform_v(self.v)
        v0 = self.transform_v0(self.v0)
        K = self.transform_K(self.K)
        n = self.transform_n(self.n)
        # v is in conc/time, needs to be scaled by volume to get molecules/time.
        # K is in concentration, X is in counts. Convert K to counts.
        K_in_counts = K * volume
        # Numerically stable formulation to avoid overflow with large n
        # This is equivalent to v * X^n / (K^n + X^n)
        # Add small epsilon to prevent division by zero when K is very small
        safe_K = jnp.maximum(K_in_counts, jnp.finfo(jnp.float32).eps)
        ratio = X / safe_K

        # The expression `ratio**n / (1 + ratio**n)` is equivalent to sigmoid(n * log(ratio))
        # and is more numerically stable. This single formulation works for both
        # activation (n > 0) and repression (n < 0).
        eps = jnp.finfo(jnp.float32).eps
        safe_ratio = jnp.maximum(ratio, eps)
        log_arg = n * jnp.log(safe_ratio)
        prop = nn.sigmoid(log_arg)

        return (v0 * volume) + (v * volume) * prop

    def ode_rate_fn(self, x, reactants, t=None, volume=1.0):
        """Calculate the Hill ODE rate.

        The rate is calculated for a deterministic ODE model. The final result
        is returned in units of molecules/time.

        Args:
            x: The current state of the system (species counts).
            reactants: Stoichiometry of the reactants. Unused for Hill kinetics.
            t: The current time. Unused for Hill kinetics.
            volume: The volume of the system.

        Returns:
            The rate of change of the species in molecules/time.
        """
        X = x[self.regulator_idx]
        X_conc = X / volume
        v = self.transform_v(self.v)
        v0 = self.transform_v0(self.v0)
        K = self.transform_K(self.K)
        n = self.transform_n(self.n)
        # Numerically stable formulation to avoid overflow with large n
        safe_K = jnp.maximum(K, jnp.finfo(jnp.float32).eps)
        ratio = X_conc / safe_K

        # The expression `ratio**n / (1 + ratio**n)` is equivalent to sigmoid(n * log(ratio))
        # and is more numerically stable. This single formulation works for both
        # activation (n > 0) and repression (n < 0).
        eps = jnp.finfo(jnp.float32).eps
        safe_ratio = jnp.maximum(ratio, eps)
        log_arg = n * jnp.log(safe_ratio)
        rate_in_conc_units = v * nn.sigmoid(log_arg)

        return (v0 + rate_in_conc_units) * volume
