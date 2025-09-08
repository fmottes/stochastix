"""Abstract base classes for stochastic simulation algorithm solvers."""

from __future__ import annotations

import abc
import typing

import equinox as eqx
import jax.numpy as jnp

if typing.TYPE_CHECKING:
    from ..reaction import ReactionNetwork


class SimulationStep(typing.NamedTuple):
    """Result of a single solver step.

    Attributes:
        x_new: The new state of the system.
        dt: The time step taken.
        reaction_idx: Index of the reaction that fired or array of reaction counts.
        propensities: The propensities at the beginning of the step.
    """

    x_new: jnp.ndarray
    dt: jnp.ndarray
    reaction_idx: jnp.ndarray
    propensities: jnp.ndarray


class AbstractStochasticSolver(eqx.Module):
    """Abstract base class for stochastic solvers.

    Attributes:
        is_exact_solver: Boolean flag indicating whether the solver is exact.
        is_pathwise_differentiable: Boolean flag indicating whether the solver
            is pathwise differentiable.

    Methods:
        init: Initialize the solver's state.
        propensities: Compute the reaction propensities.
        step: Perform a single step of the solver.
    """

    is_exact_solver: bool = eqx.field(static=True)
    is_pathwise_differentiable: bool = eqx.field(static=True)

    def init(
        self,
        network: ReactionNetwork,
        t: jnp.floating,
        x: jnp.ndarray,
        a: jnp.ndarray,
        *,
        key: jnp.ndarray,
    ):
        """Initialize the solver's state.

        This method is called once at the beginning of a simulation.

        Args:
            network: The reaction network.
            t: The initial time.
            x: The initial state.
            a: The initial propensities.
            key: A JAX random key.

        Returns:
            The initial state of the solver.
        """
        return None

    @abc.abstractmethod
    def propensities(
        self,
        network: ReactionNetwork,
        x: jnp.ndarray,
        t: jnp.floating,
    ) -> jnp.ndarray:
        """Compute the reaction propensities.

        Args:
            network: The reaction network.
            x: The current state of the system.
            t: The current time.

        Returns:
            Array of propensities for each reaction.
        """
        pass

    @abc.abstractmethod
    def step(
        self,
        network: ReactionNetwork,
        t: jnp.floating,
        x: jnp.ndarray,
        a: jnp.ndarray,
        state,
        *,
        key: jnp.ndarray,
    ) -> tuple[SimulationStep, ...]:
        """Perform a single step of the solver.

        Args:
            network: The reaction network.
            t: The current time.
            x: The current state.
            a: The current propensities.
            state: The current state of the solver.
            key: A JAX random key.

        Returns:
            Tuple containing the SimulationStep result and the new solver state.
        """
        pass

    @abc.abstractmethod
    def _delta_x(
        self,
        stoichiometry_matrix: jnp.ndarray | list[list[int | float]],
        reaction_selection: jnp.ndarray,
    ) -> jnp.ndarray:
        """Calculate the change in state based on the selected reaction(s).

        Args:
            stoichiometry_matrix: The stoichiometry matrix of the network.
            reaction_selection: For exact solvers, the index of the reaction
                that fired. For approximate solvers, this may be an array of
                reaction counts or a soft index.

        Returns:
            The change in the state vector.
        """
        pass
