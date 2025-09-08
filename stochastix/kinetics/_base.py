"""Abstract base class for kinetic law implementations."""

from __future__ import annotations

import abc

import equinox as eqx


class AbstractKinetics(eqx.Module):
    """Abstract base class for kinetic laws.

    This class defines the interface for all kinetic law implementations used
    in chemical reaction networks. Concrete subclasses must implement both
    stochastic propensity and deterministic ODE rate calculations.

    Attributes:
        _requires_species: Tuple of species names required by this kinetics.

    Methods:
        propensity_fn: Computes the propensity for a reaction.
        ode_rate_fn: Computes the rate for ODE integration.
    """

    _requires_species: tuple[str, ...] = eqx.field(static=True)

    @abc.abstractmethod
    def propensity_fn(self, x, reactants, t=None, volume=1.0):
        """Computes the propensity for a reaction.

        This method must be implemented by all concrete kinetics classes. It
        calculates the stochastic reaction propensity, typically denoted as `a(x)`.

        Args:
            x: The current state vector (species counts).
            reactants: The stoichiometry of the reactants for this reaction.
            t: The current time (optional).
            volume: The volume of the system (optional).

        Returns:
            The computed propensity for the reaction (in units of 1/time).
        """
        pass

    @abc.abstractmethod
    def ode_rate_fn(self, x, reactants, t=None, volume=1.0):
        """Computes the rate for ODE integration.

        This method must be implemented by all concrete kinetics classes. It
        calculates the deterministic rate of the reaction for use in an ODE model.

        Args:
            x: The current state vector (species counts).
            reactants: The stoichiometry of the reactants for this reaction.
            t: The current time (optional).
            volume: The volume of the system (optional).

        Returns:
            The computed deterministic rate (in units of molecules/time).
        """
        pass

    def _bind_to_network(self, species_map: dict[str, int]) -> AbstractKinetics:
        """Binds the kinetics object to a network's species map.

        This internal method is called when a `ReactionNetwork` is created. It
        allows the kinetics object to store any necessary information derived
        from the network structure, such as the indices of regulator species.

        Note:
            This method should always return a deep copy of the kinetics object
            to avoid side effects between different network instances.

        Args:
            species_map: A dictionary mapping species names to their integer indices.

        Returns:
            A new, bound `AbstractKinetics` object.
        """
        # default for kinetics that don't need to be bound to a network explicitly
        return self
