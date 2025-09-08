"""Two-dimensional Hill kinetics for dual-regulator gene expression."""

from __future__ import annotations

import warnings
from collections.abc import Callable

import equinox as eqx
import jax.nn as nn
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from ._base import AbstractKinetics


def _check_params(v, transform_v, v0, transform_v0, K, transform_K, n, transform_n):
    if v < 0:
        if transform_v is None:
            raise ValueError('`v` must be positive.')
        else:
            warnings.warn(
                '`v` is negative. Please ensure the provided `transform_v` maps it to a positive value.'
            )
    if v0 < 0:
        if transform_v0 is None:
            raise ValueError('`v0` must be positive.')
        else:
            warnings.warn(
                '`v0` is negative. Please ensure the provided `transform_v0` maps it to a positive value.'
            )

    if K < 0:
        if transform_K is None:
            raise ValueError('`K` must be positive.')
        else:
            warnings.warn(
                '`K` is negative. Please ensure the provided `transform_K` maps it to a positive value.'
            )
    if n is not None and n < 0:
        if transform_n is None:
            raise ValueError('`n` must be positive for activators/repressors.')
        else:
            warnings.warn(
                '`n` is negative. Please ensure the provided `transform_n` maps it to a positive value.'
            )


def _hill_aa_fn(x, y, n1, n2, logic, competitive_binding):
    """Dimensionless production rate for two activators."""
    eps = jnp.finfo(jnp.float32).eps
    log_x = jnp.log(jnp.maximum(x, eps))
    log_y = jnp.log(jnp.maximum(y, eps))

    if logic == 'and':
        # competitive_binding=False
        # P(A1 bound AND A2 bound) = P(A1 bound) * P(A2 bound)
        p1 = nn.sigmoid(n1 * log_x)
        p2 = nn.sigmoid(n2 * log_y)
        return p1 * p2

    # or logic
    log_xn1 = n1 * log_x
    log_yn2 = n2 * log_y
    if competitive_binding:
        # P(bound) = (X^n1 + Y^n2) / (1 + X^n1 + Y^n2)
        # Use logsumexp for stability
        logits = jnp.array([log_xn1, log_yn2])
        log_num = logsumexp(logits)
        log_den = logsumexp(jnp.array([0.0, log_xn1, log_yn2]))
        return jnp.exp(log_num - log_den)

    # independent binding
    # P(A1 bound OR A2 bound) = 1 - P(A1 unbound) * P(A2 unbound)
    p1_unbound = nn.sigmoid(-n1 * log_x)
    p2_unbound = nn.sigmoid(-n2 * log_y)
    return 1.0 - p1_unbound * p2_unbound


def _hill_rr_fn(x, y, n1, n2, logic, competitive_binding):
    """Dimensionless production rate for two repressors."""
    eps = jnp.finfo(jnp.float32).eps
    log_x = jnp.log(jnp.maximum(x, eps))
    log_y = jnp.log(jnp.maximum(y, eps))

    if logic == 'and':
        # P(R1 unbound AND R2 unbound)
        log_xn1 = n1 * log_x
        log_yn2 = n2 * log_y
        if competitive_binding:
            # 1 / (1 + X^n1 + Y^n2)
            log_den = logsumexp(jnp.array([0.0, log_xn1, log_yn2]))
            return jnp.exp(-log_den)

        # independent binding
        # P(R1 unbound) * P(R2 unbound)
        p1_unbound = nn.sigmoid(-log_xn1)
        p2_unbound = nn.sigmoid(-log_yn2)
        return p1_unbound * p2_unbound

    # or logic, competitive_binding=False
    # P(R1 unbound OR R2 unbound) = 1 - P(R1 bound) * P(R2 bound)
    p1_bound = nn.sigmoid(n1 * log_x)
    p2_bound = nn.sigmoid(n2 * log_y)
    return 1.0 - p1_bound * p2_bound


def _hill_ar_fn(x, y, n1, n2, logic, competitive_binding):
    """Dimensionless production rate for one activator and one repressor."""
    eps = jnp.finfo(jnp.float32).eps
    log_x = jnp.log(jnp.maximum(x, eps))  # activator
    log_y = jnp.log(jnp.maximum(y, eps))  # repressor

    log_xn1 = n1 * log_x
    log_yn2 = n2 * log_y

    if logic == 'and':
        # P(A bound AND R unbound)
        if competitive_binding:
            # X^n1 / (1 + X^n1 + Y^n2)
            log_num = log_xn1
            log_den = logsumexp(jnp.array([0.0, log_xn1, log_yn2]))
            return jnp.exp(log_num - log_den)

        # independent binding
        # P(A bound) * P(R unbound)
        p_a_bound = nn.sigmoid(log_xn1)
        p_r_unbound = nn.sigmoid(-log_yn2)
        return p_a_bound * p_r_unbound

    # or logic
    # P(A bound OR R unbound) = 1 - P(A unbound) * P(R bound)
    if competitive_binding:
        # (1 + X^n1) / (1 + X^n1 + Y^n2)
        log_num = logsumexp(jnp.array([0.0, log_xn1]))
        log_den = logsumexp(jnp.array([0.0, log_xn1, log_yn2]))
        return jnp.exp(log_num - log_den)

    # independent binding
    p_a_unbound = nn.sigmoid(-log_xn1)
    p_r_bound = nn.sigmoid(log_yn2)
    return 1.0 - p_a_unbound * p_r_bound


class HillAA(AbstractKinetics):
    """Hill kinetics for gene regulation by two activators.

    This class implements Hill kinetics for modeling gene expression regulated
    by two activator species, supporting different logical combinations (AND/OR)
    and binding modes (competitive/independent).

    Attributes:
        activator1: The name of the first activator species.
        activator2: The name of the second activator species.
        activator1_idx: The index of the first activator in the state vector.
        activator2_idx: The index of the second activator in the state vector.
        logic: The logic for gene expression, either 'and' or 'or'.
        competitive_binding: Whether the activators bind competitively.
        v: The limiting rate, in concentration/time.
        v0: The leaky (basal) expression rate, in concentration/time.
        K1: The half-saturation constant for the first activator, in concentration.
        K2: The half-saturation constant for the second activator, in concentration.
        n1: The Hill coefficient for the first activator (must be positive).
        n2: The Hill coefficient for the second activator (must be positive).
        transform_v: A function to transform the `v` parameter.
        transform_K1: A function to transform the `K1` parameter.
        transform_K2: A function to transform the `K2` parameter.
        transform_n1: A function to transform the `n1` parameter.
        transform_n2: A function to transform the `n2` parameter.
        transform_v0: A function to transform the `v0` parameter.
        _requires_species: Tuple of species names required by this kinetics.

    Methods:
        propensity_fn: Calculate the Hill activator-activator propensity.
        ode_rate_fn: Calculates the deterministic ODE rate for two activators.
    """

    activator1: str = eqx.field(static=True)
    activator2: str = eqx.field(static=True)
    activator1_idx: int
    activator2_idx: int
    logic: str = eqx.field(static=True)
    competitive_binding: bool = eqx.field(static=True)

    v: float | jnp.ndarray
    transform_v: Callable = eqx.field(static=True)
    v0: float | jnp.ndarray
    transform_v0: Callable = eqx.field(static=True)
    K1: float | jnp.ndarray
    transform_K1: Callable = eqx.field(static=True)
    K2: float | jnp.ndarray
    transform_K2: Callable = eqx.field(static=True)
    n1: float | jnp.ndarray
    transform_n1: Callable = eqx.field(static=True)
    n2: float | jnp.ndarray
    transform_n2: Callable = eqx.field(static=True)
    _requires_species: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        activator1: str,
        activator2: str,
        v: float | jnp.ndarray,
        K1: float | jnp.ndarray,
        K2: float | jnp.ndarray,
        n1: float | jnp.ndarray,
        n2: float | jnp.ndarray,
        logic: str,
        competitive_binding: bool = False,
        v0: float | jnp.ndarray = 0.0,
        transform_v: Callable = None,
        transform_K1: Callable = None,
        transform_K2: Callable = None,
        transform_n1: Callable = None,
        transform_n2: Callable = None,
        transform_v0: Callable = None,
        *,
        _activator1_idx: int = -1,
        _activator2_idx: int = -1,
    ):
        """Initializes the HillAA kinetics.

        Args:
            activator1: The name of the first activator species.
            activator2: The name of the second activator species.
            v: The limiting rate, in concentration/time.
            K1: The half-saturation constant for activator1, in concentration.
            K2: The half-saturation constant for activator2, in concentration.
            n1: The Hill coefficient for activator1 (must be positive).
            n2: The Hill coefficient for activator2 (must be positive).
            logic: The interaction logic, either 'and' or 'or'.
            competitive_binding: Whether binding is competitive. Defaults to False.
            v0: The leaky (basal) rate, in concentration/time. Defaults to 0.0.
            transform_v: A transform for `v`.
            transform_K1: A transform for `K1`.
            transform_K2: A transform for `K2`.
            transform_n1: A transform for `n1`.
            transform_n2: A transform for `n2`.
            transform_v0: A transform for `v0`.
            _activator1_idx: Internal use: index of the first activator.
            _activator2_idx: Internal use: index of the second activator.

        Raises:
            ValueError: If `logic` is not 'and' or 'or', or if `logic` is 'and'
                and `competitive_binding` is True.
        """
        if logic not in ['and', 'or']:
            raise ValueError("`logic` must be either 'and' or 'or'.")
        if logic == 'and' and competitive_binding:
            raise ValueError(
                '`and` logic is not compatible with `competitive_binding=True` for HillAA.'
            )

        self.activator1 = activator1
        self.activator2 = activator2
        self.logic = logic
        self.competitive_binding = competitive_binding

        _check_params(
            v, transform_v, v0, transform_v0, K1, transform_K1, n1, transform_n1
        )
        _check_params(
            v, transform_v, v0, transform_v0, K2, transform_K2, n2, transform_n2
        )

        self.v = v
        self.transform_v = (lambda val: val) if transform_v is None else transform_v
        self.v0 = v0
        self.transform_v0 = (lambda val: val) if transform_v0 is None else transform_v0
        self.K1 = K1
        self.transform_K1 = (lambda val: val) if transform_K1 is None else transform_K1
        self.K2 = K2
        self.transform_K2 = (lambda val: val) if transform_K2 is None else transform_K2
        self.n1 = n1
        self.transform_n1 = (lambda val: val) if transform_n1 is None else transform_n1
        self.n2 = n2
        self.transform_n2 = (lambda val: val) if transform_n2 is None else transform_n2

        self._requires_species = (activator1, activator2)
        self.activator1_idx = _activator1_idx
        self.activator2_idx = _activator2_idx

    def _bind_to_network(self, species_map: dict[str, int]):
        """Binds the kinetics to a network by setting the regulator indices."""
        idx1 = species_map[self.activator1]
        idx2 = species_map[self.activator2]
        return HillAA(
            activator1=self.activator1,
            activator2=self.activator2,
            v=self.v,
            K1=self.K1,
            K2=self.K2,
            n1=self.n1,
            n2=self.n2,
            logic=self.logic,
            competitive_binding=self.competitive_binding,
            v0=self.v0,
            transform_v=self.transform_v,
            transform_K1=self.transform_K1,
            transform_K2=self.transform_K2,
            transform_n1=self.transform_n1,
            transform_n2=self.transform_n2,
            transform_v0=self.transform_v0,
            _activator1_idx=idx1,
            _activator2_idx=idx2,
        )

    def _get_rate_in_conc(self, X1_conc, X2_conc, v, v0, K1, K2, n1, n2):
        safe_K1 = jnp.maximum(K1, jnp.finfo(jnp.float32).eps)
        safe_K2 = jnp.maximum(K2, jnp.finfo(jnp.float32).eps)

        x = X1_conc / safe_K1
        y = X2_conc / safe_K2

        prop = _hill_aa_fn(x, y, n1, n2, self.logic, self.competitive_binding)
        return v0 + v * prop

    def propensity_fn(self, x, reactants, t=None, volume=1.0):
        """Calculate the Hill activator-activator propensity."""
        X1 = x[self.activator1_idx]
        X2 = x[self.activator2_idx]
        X1_conc = X1 / volume
        X2_conc = X2 / volume

        v = self.transform_v(self.v)
        v0 = self.transform_v0(self.v0)
        K1 = self.transform_K1(self.K1)
        K2 = self.transform_K2(self.K2)
        n1 = self.transform_n1(self.n1)
        n2 = self.transform_n2(self.n2)

        rate_in_conc = self._get_rate_in_conc(X1_conc, X2_conc, v, v0, K1, K2, n1, n2)

        return rate_in_conc * volume

    def ode_rate_fn(self, x, reactants, t=None, volume=1.0):
        """Calculates the deterministic ODE rate for two activators.

        The rate is calculated based on the concentrations of the activators and
        the specified logic ('and'/'or') and binding mode (competitive/independent).
        The result is returned in units of concentration/time.

        Args:
            x: The current state vector (species counts).
            reactants: The reactant stoichiometry vector (unused).
            t: The current time (unused).
            volume: The system volume.

        Returns:
            The deterministic rate in concentration per unit time.
        """
        return self.propensity_fn(x, reactants, t, volume)


class HillRR(AbstractKinetics):
    """Hill kinetics for gene regulation by two repressors.

    This class implements Hill kinetics for modeling gene expression regulated
    by two repressor species, supporting different logical combinations (AND/OR)
    and binding modes (competitive/independent).

    Attributes:
        repressor1: The name of the first repressor species.
        repressor2: The name of the second repressor species.
        repressor1_idx: The index of the first repressor in the state vector.
        repressor2_idx: The index of the second repressor in the state vector.
        logic: The logic for gene expression. Use 'and' for (NOT R1 AND NOT R2) and 'or' for (NOT R1 OR NOT R2).
        competitive_binding: Whether the repressors bind competitively.
        v: The limiting rate, in concentration/time.
        v0: The leaky (basal) expression rate, in concentration/time.
        K1: The half-saturation constant for the first repressor, in concentration.
        K2: The half-saturation constant for the second repressor, in concentration.
        n1: The Hill coefficient for the first repressor (must be positive).
        n2: The Hill coefficient for the second repressor (must be positive).
        transform_v: A function to transform the `v` parameter.
        transform_K1: A function to transform the `K1` parameter.
        transform_K2: A function to transform the `K2` parameter.
        transform_n1: A function to transform the `n1` parameter.
        transform_n2: A function to transform the `n2` parameter.
        transform_v0: A function to transform the `v0` parameter.
        _requires_species: Tuple of species names required by this kinetics.

    Methods:
        propensity_fn: Calculate the Hill repressor-repressor propensity.
        ode_rate_fn: Calculates the deterministic ODE rate for two repressors.
    """

    repressor1: str = eqx.field(static=True)
    repressor2: str = eqx.field(static=True)
    repressor1_idx: int
    repressor2_idx: int
    logic: str = eqx.field(static=True)
    competitive_binding: bool = eqx.field(static=True)

    v: float | jnp.ndarray
    transform_v: Callable = eqx.field(static=True)
    v0: float | jnp.ndarray
    transform_v0: Callable = eqx.field(static=True)
    K1: float | jnp.ndarray
    transform_K1: Callable = eqx.field(static=True)
    K2: float | jnp.ndarray
    transform_K2: Callable = eqx.field(static=True)
    n1: float | jnp.ndarray
    transform_n1: Callable = eqx.field(static=True)
    n2: float | jnp.ndarray
    transform_n2: Callable = eqx.field(static=True)
    _requires_species: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        repressor1: str,
        repressor2: str,
        v: float | jnp.ndarray,
        K1: float | jnp.ndarray,
        K2: float | jnp.ndarray,
        n1: float | jnp.ndarray,
        n2: float | jnp.ndarray,
        logic: str,
        competitive_binding: bool = False,
        v0: float | jnp.ndarray = 0.0,
        transform_v: Callable = None,
        transform_K1: Callable = None,
        transform_K2: Callable = None,
        transform_n1: Callable = None,
        transform_n2: Callable = None,
        transform_v0: Callable = None,
        *,
        _repressor1_idx: int = -1,
        _repressor2_idx: int = -1,
    ):
        """Initializes the HillRR kinetics.

        Args:
            repressor1: The name of the first repressor species.
            repressor2: The name of the second repressor species.
            v: The limiting rate, in concentration/time.
            K1: The half-saturation constant for repressor1, in concentration.
            K2: The half-saturation constant for repressor2, in concentration.
            n1: The Hill coefficient for repressor1 (must be positive).
            n2: The Hill coefficient for repressor2 (must be positive).
            logic: The interaction logic, either 'and' or 'or'.
            competitive_binding: Whether binding is competitive. Defaults to False.
            v0: The leaky (basal) rate, in concentration/time. Defaults to 0.0.
            transform_v: A transform for `v`.
            transform_K1: A transform for `K1`.
            transform_K2: A transform for `K2`.
            transform_n1: A transform for `n1`.
            transform_n2: A transform for `n2`.
            transform_v0: A transform for `v0`.
            _repressor1_idx: Internal use: index of the first repressor.
            _repressor2_idx: Internal use: index of the second repressor.

        Raises:
            ValueError: If `logic` is not 'and' or 'or', or if `logic` is 'and'
                and `competitive_binding` is True.
        """
        if logic not in ['and', 'or']:
            raise ValueError("`logic` must be either 'and' or 'or'.")
        if logic == 'and' and competitive_binding:
            raise ValueError(
                '`and` logic is not compatible with `competitive_binding=True` for HillRR.'
            )

        self.repressor1 = repressor1
        self.repressor2 = repressor2
        self.logic = logic
        self.competitive_binding = competitive_binding

        _check_params(
            v, transform_v, v0, transform_v0, K1, transform_K1, n1, transform_n1
        )
        _check_params(
            v, transform_v, v0, transform_v0, K2, transform_K2, n2, transform_n2
        )

        self.v = v
        self.transform_v = (lambda val: val) if transform_v is None else transform_v
        self.v0 = v0
        self.transform_v0 = (lambda val: val) if transform_v0 is None else transform_v0
        self.K1 = K1
        self.transform_K1 = (lambda val: val) if transform_K1 is None else transform_K1
        self.K2 = K2
        self.transform_K2 = (lambda val: val) if transform_K2 is None else transform_K2
        self.n1 = n1
        self.transform_n1 = (lambda val: val) if transform_n1 is None else transform_n1
        self.n2 = n2
        self.transform_n2 = (lambda val: val) if transform_n2 is None else transform_n2

        self._requires_species = (repressor1, repressor2)
        self.repressor1_idx = _repressor1_idx
        self.repressor2_idx = _repressor2_idx

    def _bind_to_network(self, species_map: dict[str, int]):
        """Binds the kinetics to a network by setting the regulator indices."""
        idx1 = species_map[self.repressor1]
        idx2 = species_map[self.repressor2]
        return HillRR(
            repressor1=self.repressor1,
            repressor2=self.repressor2,
            v=self.v,
            K1=self.K1,
            K2=self.K2,
            n1=self.n1,
            n2=self.n2,
            logic=self.logic,
            competitive_binding=self.competitive_binding,
            v0=self.v0,
            transform_v=self.transform_v,
            transform_K1=self.transform_K1,
            transform_K2=self.transform_K2,
            transform_n1=self.transform_n1,
            transform_n2=self.transform_n2,
            transform_v0=self.transform_v0,
            _repressor1_idx=idx1,
            _repressor2_idx=idx2,
        )

    def _get_rate_in_conc(self, X1_conc, X2_conc, v, v0, K1, K2, n1, n2):
        safe_K1 = jnp.maximum(K1, jnp.finfo(jnp.float32).eps)
        safe_K2 = jnp.maximum(K2, jnp.finfo(jnp.float32).eps)

        x = X1_conc / safe_K1
        y = X2_conc / safe_K2

        prop = _hill_rr_fn(x, y, n1, n2, self.logic, self.competitive_binding)
        return v0 + v * prop

    def propensity_fn(self, x, reactants, t=None, volume=1.0):
        """Calculate the Hill repressor-repressor propensity."""
        X1 = x[self.repressor1_idx]
        X2 = x[self.repressor2_idx]
        X1_conc = X1 / volume
        X2_conc = X2 / volume

        v = self.transform_v(self.v)
        v0 = self.transform_v0(self.v0)
        K1 = self.transform_K1(self.K1)
        K2 = self.transform_K2(self.K2)
        n1 = self.transform_n1(self.n1)
        n2 = self.transform_n2(self.n2)

        rate_in_conc = self._get_rate_in_conc(X1_conc, X2_conc, v, v0, K1, K2, n1, n2)

        return rate_in_conc * volume

    def ode_rate_fn(self, x, reactants, t=None, volume=1.0):
        """Calculates the deterministic ODE rate for two repressors.

        The rate is calculated based on the concentrations of the repressors and
        the specified logic ('and'/'or') and binding mode (competitive/independent).
        The result is returned in units of concentration/time.

        Args:
            x: The current state vector (species counts).
            reactants: The reactant stoichiometry vector (unused).
            t: The current time (unused).
            volume: The system volume.

        Returns:
            The deterministic rate in concentration per unit time.
        """
        return self.propensity_fn(x, reactants, t, volume)


class HillAR(AbstractKinetics):
    """Hill kinetics for gene regulation by one activator and one repressor.

    This class implements Hill kinetics for modeling gene expression regulated
    by both an activator and a repressor species, supporting different logical
    combinations (AND/OR) and binding modes (competitive/independent).

    Attributes:
        activator: The name of the activator species.
        repressor: The name of the repressor species.
        activator_idx: The index of the activator in the state vector.
        repressor_idx: The index of the repressor in the state vector.
        logic: The logic for gene expression. Use 'and' for (A AND NOT R) and 'or' for (A OR NOT R).
        competitive_binding: Whether the regulators bind competitively.
        v: The limiting rate, in concentration/time.
        v0: The leaky (basal) expression rate, in concentration/time.
        Ka: The half-saturation constant for the activator, in concentration.
        Kr: The half-saturation constant for the repressor, in concentration.
        na: The Hill coefficient for the activator (must be positive).
        nr: The Hill coefficient for the repressor (must be positive).
        transform_v: A function to transform the `v` parameter.
        transform_Ka: A function to transform the `Ka` parameter.
        transform_Kr: A function to transform the `Kr` parameter.
        transform_na: A function to transform the `na` parameter.
        transform_nr: A function to transform the `nr` parameter.
        transform_v0: A function to transform the `v0` parameter.
        _requires_species: Tuple of species names required by this kinetics.

    Methods:
        propensity_fn: Calculate the Hill activator-repressor propensity.
        ode_rate_fn: Calculates the deterministic ODE rate for one activator and one repressor.
    """

    activator: str = eqx.field(static=True)
    repressor: str = eqx.field(static=True)
    activator_idx: int
    repressor_idx: int
    logic: str = eqx.field(static=True)
    competitive_binding: bool = eqx.field(static=True)

    v: float | jnp.ndarray
    transform_v: Callable = eqx.field(static=True)
    v0: float | jnp.ndarray
    transform_v0: Callable = eqx.field(static=True)
    Ka: float | jnp.ndarray
    transform_Ka: Callable = eqx.field(static=True)
    Kr: float | jnp.ndarray
    transform_Kr: Callable = eqx.field(static=True)
    na: float | jnp.ndarray
    transform_na: Callable = eqx.field(static=True)
    nr: float | jnp.ndarray
    transform_nr: Callable = eqx.field(static=True)
    _requires_species: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        activator: str,
        repressor: str,
        v: float | jnp.ndarray,
        Ka: float | jnp.ndarray,
        Kr: float | jnp.ndarray,
        na: float | jnp.ndarray,
        nr: float | jnp.ndarray,
        logic: str,
        competitive_binding: bool = False,
        v0: float | jnp.ndarray = 0.0,
        transform_v: Callable = None,
        transform_Ka: Callable = None,
        transform_Kr: Callable = None,
        transform_na: Callable = None,
        transform_nr: Callable = None,
        transform_v0: Callable = None,
        *,
        _activator_idx: int = -1,
        _repressor_idx: int = -1,
    ):
        """Initializes the HillAR kinetics.

        Args:
            activator: The name of the activator species.
            repressor: The name of the repressor species.
            v: The limiting rate, in concentration/time.
            Ka: The half-saturation constant for the activator, in concentration.
            Kr: The half-saturation constant for the repressor, in concentration.
            na: The Hill coefficient for the activator (must be positive).
            nr: The Hill coefficient for the repressor (must be positive).
            logic: The interaction logic, either 'and' or 'or'.
            competitive_binding: Whether binding is competitive. Defaults to False.
            v0: The leaky (basal) rate, in concentration/time. Defaults to 0.0.
            transform_v: A transform for `v`.
            transform_Ka: A transform for `Ka`.
            transform_Kr: A transform for `Kr`.
            transform_na: A transform for `na`.
            transform_nr: A transform for `nr`.
            transform_v0: A transform for `v0`.
            _activator_idx: Internal use: index of the activator.
            _repressor_idx: Internal use: index of the repressor.

        Raises:
            ValueError: If `logic` is not 'and' or 'or'.
        """
        if logic not in ['and', 'or']:
            raise ValueError("`logic` must be either 'and' or 'or'.")

        self.activator = activator
        self.repressor = repressor
        self.logic = logic
        self.competitive_binding = competitive_binding

        _check_params(
            v, transform_v, v0, transform_v0, Ka, transform_Ka, na, transform_na
        )
        _check_params(
            v, transform_v, v0, transform_v0, Kr, transform_Kr, nr, transform_nr
        )

        self.v = v
        self.transform_v = (lambda val: val) if transform_v is None else transform_v
        self.v0 = v0
        self.transform_v0 = (lambda val: val) if transform_v0 is None else transform_v0
        self.Ka = Ka
        self.transform_Ka = (lambda val: val) if transform_Ka is None else transform_Ka
        self.Kr = Kr
        self.transform_Kr = (lambda val: val) if transform_Kr is None else transform_Kr
        self.na = na
        self.transform_na = (lambda val: val) if transform_na is None else transform_na
        self.nr = nr
        self.transform_nr = (lambda val: val) if transform_nr is None else transform_nr

        self._requires_species = (activator, repressor)
        self.activator_idx = _activator_idx
        self.repressor_idx = _repressor_idx

    def _bind_to_network(self, species_map: dict[str, int]):
        """Binds the kinetics to a network by setting the regulator indices."""
        idx_a = species_map[self.activator]
        idx_r = species_map[self.repressor]
        return HillAR(
            activator=self.activator,
            repressor=self.repressor,
            v=self.v,
            Ka=self.Ka,
            Kr=self.Kr,
            na=self.na,
            nr=self.nr,
            logic=self.logic,
            competitive_binding=self.competitive_binding,
            v0=self.v0,
            transform_v=self.transform_v,
            transform_Ka=self.transform_Ka,
            transform_Kr=self.transform_Kr,
            transform_na=self.transform_na,
            transform_nr=self.transform_nr,
            transform_v0=self.transform_v0,
            _activator_idx=idx_a,
            _repressor_idx=idx_r,
        )

    def _get_rate_in_conc(self, Xa_conc, Xr_conc, v, v0, Ka, Kr, na, nr):
        safe_Ka = jnp.maximum(Ka, jnp.finfo(jnp.float32).eps)
        safe_Kr = jnp.maximum(Kr, jnp.finfo(jnp.float32).eps)

        x = Xa_conc / safe_Ka  # activator
        y = Xr_conc / safe_Kr  # repressor

        prop = _hill_ar_fn(x, y, na, nr, self.logic, self.competitive_binding)
        return v0 + v * prop

    def propensity_fn(self, x, reactants, t=None, volume=1.0):
        """Calculate the Hill activator-repressor propensity."""
        Xa = x[self.activator_idx]
        Xr = x[self.repressor_idx]
        Xa_conc = Xa / volume
        Xr_conc = Xr / volume

        v = self.transform_v(self.v)
        v0 = self.transform_v0(self.v0)
        Ka = self.transform_Ka(self.Ka)
        Kr = self.transform_Kr(self.Kr)
        na = self.transform_na(self.na)
        nr = self.transform_nr(self.nr)

        rate_in_conc = self._get_rate_in_conc(Xa_conc, Xr_conc, v, v0, Ka, Kr, na, nr)

        return rate_in_conc * volume

    def ode_rate_fn(self, x, reactants, t=None, volume=1.0):
        """Calculates the deterministic ODE rate for one activator and one repressor.

        The rate is calculated based on the concentrations of the regulators and
        the specified logic ('and'/'or') and binding mode (competitive/independent).
        The result is returned in units of concentration/time.

        Args:
            x: The current state vector (species counts).
            reactants: The reactant stoichiometry vector (unused).
            t: The current time (unused).
            volume: The system volume.

        Returns:
            The deterministic rate in concentration per unit time.
        """
        return self.propensity_fn(x, reactants, t, volume)
