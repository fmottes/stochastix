"""Michaelis-Menten kinetics for enzyme-catalyzed reactions."""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp

from ._base import AbstractKinetics


class MichaelisMenten(AbstractKinetics):
    """Michaelis-Menten kinetics for enzyme-catalyzed reactions.

    The substrate is inferred from the reactants of the reaction, assuming it has
    a single reactant species. The rate is given by `v_max * S / (k_m + S)`,
    where `v_max = k_cat * E`. The enzyme `E` can be a fixed abundance or a
    dynamic species.

    Attributes:
        enzyme: The enzyme, which can be specified as a species name or a fixed abundance.
        enzyme_idx: The index of the enzyme species in the state vector.
        k_cat: The turnover number (units of 1/time).
        k_m: The Michaelis constant (units of concentration).
        transform_k_cat: A function to transform the `k_cat` parameter.
        transform_k_m: A function to transform the `k_m` parameter.
        _requires_species: Tuple of species names required by this kinetics.

    Methods:
        propensity_fn: Calculates the Michaelis-Menten propensity.
        ode_rate_fn: Calculates the Michaelis-Menten deterministic rate.
    """

    enzyme: str | float = eqx.field(static=True)
    enzyme_idx: int
    k_cat: float | jnp.ndarray
    transform_k_cat: Callable = eqx.field(static=True)
    k_m: float | jnp.ndarray
    transform_k_m: Callable = eqx.field(static=True)
    _requires_species: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        enzyme: str | float,
        k_cat: float,
        k_m: float,
        transform_k_cat: Callable = None,
        transform_k_m: Callable = None,
        *,
        _enzyme_idx: int = -1,
    ):
        """Initializes the MichaelisMenten kinetics.

        Args:
            enzyme: The enzyme, either as a species name or a fixed
                abundance (in number of molecules).
            k_cat: The turnover number (in 1/time).
            k_m: The Michaelis constant (in concentration).
            transform_k_cat: A transform for `k_cat`.
            transform_k_m: A transform for `k_m`.
            _enzyme_idx: Internal use: index of the enzyme species.
        """
        self.enzyme = enzyme
        self.k_cat = k_cat
        self.transform_k_cat = (
            (lambda val: val) if transform_k_cat is None else transform_k_cat
        )
        self.k_m = k_m
        self.transform_k_m = (
            (lambda val: val) if transform_k_m is None else transform_k_m
        )

        if isinstance(enzyme, str):
            self._requires_species = (enzyme,)
        else:
            self._requires_species = ()
        self.enzyme_idx = _enzyme_idx

    def _bind_to_network(self, species_map: dict[str, int]):
        """Binds the kinetics by embedding the enzyme index."""
        if isinstance(self.enzyme, str):
            idx = species_map[self.enzyme]
            return MichaelisMenten(
                enzyme=self.enzyme,
                k_cat=self.k_cat,
                k_m=self.k_m,
                transform_k_cat=self.transform_k_cat,
                transform_k_m=self.transform_k_m,
                _enzyme_idx=idx,
            )
        return self

    def propensity_fn(self, x, reactants, t=None, volume=1.0):
        """Calculates the Michaelis-Menten propensity.

        This function assumes that `k_cat` is in units of 1/time and `k_m` is in
        concentration units. The reaction is assumed to have a single reactant
        species, which is treated as the substrate.

        Args:
            x: The current state vector (species counts).
            reactants: The reactant stoichiometry vector.
            t: The current time (unused).
            volume: The system volume.

        Returns:
            The stochastic propensity of the reaction.
        """
        S_idx = jnp.argmax(reactants)
        S = x[S_idx]

        k_cat = self.transform_k_cat(self.k_cat)
        k_m = self.transform_k_m(self.k_m)

        if isinstance(self.enzyme, str):
            E = x[self.enzyme_idx]
        else:
            E = self.enzyme

        v_max = k_cat * E  # in molecules/time
        return v_max * S / (k_m * volume + S)

    def ode_rate_fn(self, x, reactants, t=None, volume=1.0):
        """Calculates the Michaelis-Menten deterministic rate.

        This function assumes that the reaction has a single reactant species
        (the substrate). The rate is returned in units of molecules/time.

        Args:
            x: The current state vector (species counts).
            reactants: The reactant stoichiometry vector.
            t: The current time (unused).
            volume: The system volume.

        Returns:
            The deterministic rate in molecules per unit time.
        """
        S_idx = jnp.argmax(reactants)
        S_conc = x[S_idx] / volume

        k_cat = self.transform_k_cat(self.k_cat)
        k_m = self.transform_k_m(self.k_m)

        if isinstance(self.enzyme, str):
            E_conc = x[self.enzyme_idx] / volume
        else:
            E_conc = self.enzyme / volume

        v_max_conc = k_cat * E_conc  # in concentration/time
        rate_in_conc_units = v_max_conc * S_conc / (k_m + S_conc)
        return rate_in_conc_units * volume
