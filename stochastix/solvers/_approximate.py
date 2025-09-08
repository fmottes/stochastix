"""Approximate stochastic simulation algorithms."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as rng

from ..reaction import ReactionNetwork
from ._base import AbstractStochasticSolver, SimulationStep

#####################################################################################
# Tau-leaping method (Cao et al. 2006)
#####################################################################################


class TauLeaping(AbstractStochasticSolver):
    """Approximate stochastic simulation using the tau-leaping method.

    This solver implements the tau-leaping method described in Cao et al. (2006), which
    approximates the exact stochastic simulation algorithm (SSA) by allowing multiple
    reactions to fire in a single time step tau. This can significantly accelerate
    simulations when reaction propensities are large.

    The method selects a time step tau such that the propensities do not change
    significantly during the leap, maintaining accuracy while improving computational
    efficiency. The leap size is determined by balancing mean and variance constraints
    to control the approximation error.

    Attributes:
        is_exact_solver: Boolean flag indicating this is an approximate method.
        epsilon: Error control parameter. Smaller values give
            higher accuracy but smaller time steps.
        tau_min: Minimum allowed time step to prevent simulation stalling.

    Methods:
        propensities: Compute reaction propensities at the current state.
        step: Perform a single step of the tau-leaping algorithm.

    References:
        Cao, Y., Gillespie, D. T., & Petzold, L. R. (2006). Efficient step size
        selection for the tau-leaping simulation method. Journal of Chemical Physics,
        124(4), 044109.
    """

    epsilon: float  # error control parameter
    tau_min: float  # minimum allowed timestep

    def __init__(self, epsilon: float = 0.03, tau_min: float = 1e-10):
        """Initialize the tau-leaping solver.

        Args:
            epsilon: The error control parameter, between 0 and 1. Smaller
                values provide higher accuracy but require smaller time steps.
                Defaults to 0.03 (3% relative error tolerance).
            tau_min: The minimum allowed time step. Must be positive. Prevents
                the simulation from taking excessively small steps. Defaults to 1e-10.

        Raises:
            ValueError: If epsilon is not between 0 and 1, or if tau_min is not positive.
        """
        # check epsilon
        if epsilon <= 0 or epsilon >= 1:
            raise ValueError('epsilon must be between 0 and 1')

        # check tau_min
        if tau_min <= 0:
            raise ValueError('tau_min must be positive')

        super().__init__(is_exact_solver=False, is_pathwise_differentiable=False)

        self.epsilon = epsilon
        self.tau_min = tau_min

    def propensities(self, network: ReactionNetwork, x: jnp.ndarray, t: float):
        """Compute reaction propensities at the current state.

        Args:
            network: Chemical reaction network model with propensities method.
            x: Current species populations.
            t: Current time.

        Returns:
            Reaction propensities.
        """
        return network.propensity_fn(x, t)

    def _delta_x(
        self,
        stoichiometry_matrix: jnp.ndarray | list[list[int | float]],
        reaction_counts: jnp.ndarray,
    ):
        """Update the state based on a vector of reaction counts.

        For tau-leaping, multiple reactions can fire during a single step. This
        method updates the state based on a vector of reaction counts rather
        than a single reaction index.

        Args:
            stoichiometry_matrix: The stoichiometry matrix of the network.
            reaction_counts: A vector where each element is the number of times
                the corresponding reaction fired in the time step.

        Returns:
            The change in the state vector.
        """
        stoichiometry_matrix = jnp.asarray(stoichiometry_matrix)
        return jnp.dot(stoichiometry_matrix, reaction_counts)

    def _compute_tau(self, network: ReactionNetwork, x: jnp.ndarray, a: jnp.ndarray):
        """Compute leap size tau using the Cao et al. (2006) algorithm.

        The algorithm selects tau to satisfy both mean and variance constraints:
        - Mean constraint: tau ≤ N_crit_i / |μ_i| for each species i
        - Variance constraint: tau ≤ (N_crit_i)² / σ²_i for each species i

        where N_crit_i = max(1, (ε/g_i) * x_i) is the critical number that controls
        the maximum allowed change for species i, μ_i is the expected change rate,
        σ²_i is the variance of the change rate, g_i is the highest reaction order
        affecting species i, and ε is the error control parameter.

        Args:
            network: Chemical reaction network model with stoichiometry_matrix attribute.
            x: Current species populations.
            a: Current reaction propensities.

        Returns:
            Computed leap size tau. Returns jnp.inf if no reactions are possible.
        """
        stoichiometry_matrix = jnp.asarray(network.stoichiometry_matrix)

        # Handle case where all propensities are zero
        a_sum = jnp.sum(a)

        # Compute highest order reaction for each species (g_i in the paper)
        # This is the maximum absolute stoichiometry coefficient for each species
        g_i = jnp.max(jnp.abs(stoichiometry_matrix), axis=1)
        g_i = jnp.maximum(g_i, 1.0)  # Ensure g_i >= 1

        # Compute mean and variance of state change per unit time
        mu = jnp.dot(stoichiometry_matrix, a)  # E[dX_i]
        sigma_squared = jnp.dot(stoichiometry_matrix**2, a)  # Var[dX_i]

        # Calculate N_crit_i = max(1, (epsilon/g_i) * x_i) for each species i.
        # This term is crucial for accuracy and stability when species counts (x_i) are small (e.g., 0, 1, 2),
        # ensuring that tau is chosen such that not too many reactions occur.
        # See Cao et al. (2006), J. Chem. Phys. 124, 044109, Eq. 12 and surrounding discussion.
        # We use x_safe to ensure non-negativity, though x should already be >= 0 from the end of step.
        x_safe = jnp.maximum(x, 0.0)
        epsilon_prime_x = (self.epsilon / g_i) * x_safe
        N_crit = jnp.maximum(1.0, epsilon_prime_x)

        # Compute tau candidates from mean bound: tau <= N_crit_i / |mu_i|
        mu_nonzero = jnp.abs(mu) > 1e-12  # Avoid division by zero
        tau_mu = jnp.where(mu_nonzero, N_crit / jnp.abs(mu), jnp.inf)

        # Compute tau candidates from variance bound: tau <= N_crit_i^2 / sigma_i^2
        sigma_nonzero = sigma_squared > 1e-12  # Avoid division by zero
        tau_sigma = jnp.where(
            sigma_nonzero,
            N_crit**2
            / sigma_squared,  # N_crit is squared here, as per Cao et al. Eq. 12
            jnp.inf,
        )

        # Take the minimum over all species and both bounds
        tau = jnp.minimum(jnp.min(tau_mu), jnp.min(tau_sigma))

        # Ensure tau is not too small and handle numerical issues
        tau = jnp.maximum(tau, self.tau_min)

        # Additional safety: if tau is too large compared to characteristic time scales,
        # limit it to prevent excessive jumping
        characteristic_time = jnp.where(a_sum > 0, 1.0 / a_sum, jnp.inf)
        tau = jnp.where(
            jnp.logical_and(jnp.isfinite(tau), a_sum > 0),
            jnp.minimum(tau, 10.0 * characteristic_time),
            tau,
        )

        # Return infinity if no reactions are possible
        tau = jnp.where(a_sum == 0, jnp.inf, tau)

        return tau

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
        """Perform a single step of the tau-leaping algorithm.

        This method computes the time step `tau`, samples the number of times
        each reaction fires from a Poisson distribution, and updates the state.

        Args:
            network: The reaction network.
            t: The current time.
            x: The current state.
            a: The current propensities.
            state: The current solver state (unused).
            key: A JAX random key.

        Returns:
            Tuple containing the SimulationStep result and the new solver
            state (None for this stateless solver).
        """
        # Compute tau
        dt = self._compute_tau(network, x, a)

        # Sample reaction counts from Poisson distribution
        means = a * dt
        # Clip means to prevent numerical issues with very large values
        means = jnp.clip(means, 0.0, 1000.0)

        # Handle case where dt is infinite (no reactions possible)
        r = jnp.where(
            jnp.isfinite(dt),
            rng.poisson(key, means).astype(jnp.float32),
            jnp.zeros_like(a),
        )

        # Update state
        delta_x = self._delta_x(network.stoichiometry_matrix, r)
        x_new = x + delta_x

        # Ensure non-negative populations
        x_new = jnp.maximum(x_new, 0.0)

        # Return appropriate dt and reaction counts
        dt = jnp.where(jnp.isfinite(dt), dt, jnp.inf)

        return SimulationStep(x_new=x_new, dt=dt, reaction_idx=r, propensities=a), None
