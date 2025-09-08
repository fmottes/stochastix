"""Exact stochastic simulation algorithms."""

from __future__ import annotations

import typing

import jax
import jax.numpy as jnp
import jax.random as rng

from ._base import AbstractStochasticSolver, SimulationStep

if typing.TYPE_CHECKING:
    from ..reaction import ReactionNetwork

#####################################################################################
# Direct method (Gillespie 1977)
#####################################################################################


class DirectMethod(AbstractStochasticSolver):
    """Exact stochastic simulation using the Direct Method.

    The Direct Method is one of the two original exact algorithms proposed by
    Gillespie (1977). It samples the next reaction time from an exponential
    distribution and selects the reaction to fire based on their relative
    propensities.

    This implementation is JAX-compatible and designed for efficient compilation
    and automatic differentiation.

    Attributes:
        is_exact_solver: Boolean flag indicating that this is an exact solver.
        is_pathwise_differentiable: Boolean flag indicating that this solver
            is not pathwise differentiable.

    Methods:
        propensities: Calculate reaction propensities for the current state.
        step: Execute one step of the Direct Method algorithm.

    References:
        Gillespie, D. T. (1977). Exact stochastic simulation of coupled chemical
        reactions. The Journal of Physical Chemistry, 81(25), 2340-2361.
    """

    def __init__(self):
        """Initialize the DirectMethod solver."""
        super().__init__(is_exact_solver=True, is_pathwise_differentiable=False)

    def propensities(self, network, x, t):
        """Calculate reaction propensities for the current state.

        Args:
            network: The reaction network containing the propensity function.
            x: The current state vector (species counts).
            t: The current time.

        Returns:
            Array of propensities for each reaction.
        """
        return network.propensity_fn(x, t)

    def _delta_x(self, stoichiometry_matrix, reaction_idx):
        """Calculate the change in state for a single reaction event."""
        stoichiometry_matrix = jnp.asarray(stoichiometry_matrix)
        return stoichiometry_matrix[:, reaction_idx]

    def step(
        self,
        network: ReactionNetwork,
        t: jnp.floating,
        x: jnp.ndarray,
        a: jnp.ndarray,
        state,
        *,
        key: jax.random.PRNGKey,
    ) -> tuple[SimulationStep, None]:
        """Execute one step of the Direct Method algorithm.

        This method implements the core Direct Method algorithm:
        1. Calculate the total propensity.
        2. Sample the next reaction time from an exponential distribution.
        3. Select the reaction to fire based on relative propensities.
        4. Update the system state.

        Args:
            network: The reaction network.
            t: The current time.
            x: The current state vector.
            a: The current propensity vector.
            state: The current solver state (unused).
            key: A JAX random key for sampling.

        Returns:
            Tuple containing the SimulationStep result and the new solver state (None since this is a stateless solver).
        """
        key_dt, key_r = jax.random.split(key)

        a0 = jnp.sum(a)

        # sample time
        dt = -jnp.log(rng.uniform(key_dt)) / a0

        # sample reaction
        p = a / a0
        r = rng.choice(key_r, jnp.arange(len(a)), p=p)

        # update state
        delta_x = self._delta_x(network.stoichiometry_matrix, r)
        x_new = x + delta_x

        step_result = SimulationStep(
            x_new=x_new,
            dt=dt,
            reaction_idx=r,
            propensities=a,
        )

        return step_result, None


#####################################################################################
# First reaction method (Gillespie 1977)
#####################################################################################


class FirstReactionMethod(AbstractStochasticSolver):
    """Exact stochastic simulation using the First Reaction Method.

    The First Reaction Method is the second exact algorithm proposed by Gillespie (1977).
    It generates a candidate firing time for each reaction independently and selects
    the reaction with the smallest firing time.

    While mathematically equivalent to the Direct Method, this approach can be
    computationally less efficient for large numbers of reactions due to the need
    to generate multiple random numbers per step.

    Attributes:
        is_exact_solver: Boolean flag indicating that this is an exact solver.
        is_pathwise_differentiable: Boolean flag indicating that this solver
            is not pathwise differentiable.

    Methods:
        propensities: Calculate reaction propensities for the current state.
        step: Execute one step of the First Reaction Method algorithm.

    References:
        Gillespie, D. T. (1977). Exact stochastic simulation of coupled chemical
        reactions. The Journal of Physical Chemistry, 81(25), 2340-2361.
    """

    def __init__(self):
        """Initialize the FirstReactionMethod solver."""
        super().__init__(is_exact_solver=True, is_pathwise_differentiable=False)

    def propensities(self, network, x, t):
        """Calculate reaction propensities for the current state.

        Args:
            network: The reaction network containing the propensity function.
            x: The current state vector (species counts).
            t: The current time.

        Returns:
            Array of propensities for each reaction.
        """
        return network.propensity_fn(x, t)

    def _delta_x(self, stoichiometry_matrix, reaction_idx):
        """Calculate the change in state for a single reaction event."""
        stoichiometry_matrix = jnp.asarray(stoichiometry_matrix)
        return stoichiometry_matrix[:, reaction_idx]

    def step(
        self,
        network: ReactionNetwork,
        t: jnp.floating,
        x: jnp.ndarray,
        a: jnp.ndarray,
        state,
        *,
        key: jax.random.PRNGKey,
    ) -> tuple[SimulationStep, None]:
        """Execute one step of the First Reaction Method algorithm.

        This method implements the core First Reaction Method algorithm:
        1. Generate a candidate firing time for each reaction independently.
        2. Select the reaction with the minimum firing time.
        3. Update the system state with the selected reaction.

        Args:
            network: The reaction network.
            t: The current time.
            x: The current state vector.
            a: The current propensity vector.
            state: The current solver state (unused).
            key: A JAX random key for sampling.

        Returns:
            Tuple containing the SimulationStep result and the new solver state.
        """
        dt_vec = -jnp.log(rng.uniform(key, shape=a.shape)) / a

        r = jnp.argmin(dt_vec)
        dt = dt_vec[r]

        # update state
        delta_x = self._delta_x(network.stoichiometry_matrix, r)
        x_new = x + delta_x

        return SimulationStep(x_new=x_new, dt=dt, reaction_idx=r, propensities=a), None
