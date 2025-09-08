"""Pathwise differentiable stochastic simulation algorithms for gradient-based optimization."""

from __future__ import annotations

import typing
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as rng

from ._base import AbstractStochasticSolver, SimulationStep

if typing.TYPE_CHECKING:
    from ..reaction import ReactionNetwork

#####################################################################################
# Differentiable Gillespie Algorithm (DGA)
#####################################################################################


class DGA(AbstractStochasticSolver):
    """Approximate simulation using the Differentiable Gillespie Algorithm.

    This solver is highly unstable and sensitive to hyperparameter choices. In
    most cases, reparameterized versions of exact solvers should be preferred.

    The method has two key approximations:
    1. **Differentiable reaction selection**: The discrete choice of which
        reaction fires is replaced by a "soft" selection using sigmoid functions,
        controlled by the steepness parameter `a`.
    2. **Differentiable state update**: The state is updated using a
        Gaussian-based weighting scheme centered around a soft reaction index,
        controlled by the sharpness parameter `sigma`.

    Attributes:
        is_exact_solver: Boolean flag indicating this is an approximate method.
        is_pathwise_differentiable: Boolean flag indicating differentiability.
        a: Steepness parameter for the sigmoid function.
        sigma: Width parameter (standard deviation) for the Gaussian function.

    Methods:
        propensities: Compute reaction propensities at the current state.
        step: Perform a single step of the Differentiable Gillespie Algorithm.

    References:
        Rijal, K., & Mehta, P. (2025). A differentiable Gillespie algorithm for
        simulating chemical kinetics, parameter estimation, and designing synthetic
        biological circuits. eLife, 14, RP103877.
        https://doi.org/10.7554/eLife.103877.2
    """

    a: float  # sigmoid steepness parameter
    sigma: float  # Gaussian width parameter

    def __init__(self, a: float = 0.005, sigma: float = 0.1581):
        """Initialize the Differentiable Gillespie Algorithm (DGA) solver.

        Args:
            a: Steepness parameter for the sigmoid function. Must be positive.
                Smaller values make the sigmoid steeper, approaching a true
                Heaviside function. Defaults to 0.005.
            sigma: Standard deviation of the Gaussian for smoothing the state
                update. Must be positive. Smaller values make the Gaussian
                sharper, approaching a true discrete update. Defaults to ~0.1581.

        Raises:
            ValueError: If `a` or `sigma` are not positive.
        """
        if a <= 0:
            raise ValueError('a must be positive')
        if sigma <= 0:
            raise ValueError('sigma must be positive')

        super().__init__(is_exact_solver=False, is_pathwise_differentiable=True)

        self.a = a
        self.sigma = sigma

    def propensities(self, network, x, t):
        """Compute reaction propensities at the current state.

        Args:
            network: Chemical reaction network model with propensities method.
            x: Current species populations.
            t: Current time.

        Returns:
            Reaction propensities.
        """
        return network.propensity_fn(x, t)

    def _delta_x(self, stoichiometry_matrix, i_prime):
        """Compute the smoothed, differentiable state update.

        This method approximates the standard discrete state update by applying a
        Gaussian function centered at the soft reaction index `i_prime`. The
        update is a weighted sum of all possible reaction state changes.

        Args:
            stoichiometry_matrix: The stoichiometry matrix of the network.
            i_prime: The soft reaction index from the differentiable selection.

        Returns:
            The computed change in state.
        """
        stoichiometry_matrix = jnp.asarray(stoichiometry_matrix)
        num_reactions = stoichiometry_matrix.shape[1]
        i_values = jnp.arange(num_reactions, dtype=i_prime.dtype)

        # Approximate Kronecker delta with a Gaussian function
        weights = jnp.exp(-((i_prime - i_values) ** 2) / (2 * self.sigma**2))

        delta_x = jnp.dot(stoichiometry_matrix, weights)
        return delta_x

    def step(
        self,
        network: ReactionNetwork,
        t: jnp.floating,
        x: jnp.ndarray,
        a: jnp.ndarray,
        state: Any,
        *,
        key: jnp.ndarray,
    ) -> tuple[SimulationStep, None]:
        """Perform a single step of the Differentiable Gillespie Algorithm.

        Args:
            network: The reaction network.
            t: The current simulation time.
            x: The current species populations.
            a: The current propensities.
            state: The solver state (not used).
            key: A JAX random key.

        Returns:
            Tuple containing the SimulationStep result and the new solver
            state (None for this stateless solver).
        """
        key_dt, key_r = rng.split(key)
        a0 = jnp.sum(a)

        # 1. Sample time step dt, same as in exact Gillespie
        dt = -jnp.log(rng.uniform(key_dt)) / a0
        dt = jnp.where(a0 > 0, dt, jnp.inf)

        # 2. Differentiable reaction selection
        p = a / a0
        c = jnp.cumsum(p)

        # Sample uniform random number for selection
        u = rng.uniform(key_r)

        # The soft index is the sum of sigmoids, approximating the sum of Heaviside
        # functions.
        i_prime = jnp.sum(jax.nn.sigmoid((u - c[:-1]) / self.a))

        # 3. Differentiable state update
        delta_x = self._delta_x(network.stoichiometry_matrix, i_prime)
        x_new = x + delta_x

        # Ensure non-negative populations
        x_new = jnp.maximum(x_new, 0.0)

        # Handle case where no reactions are possible
        i_prime = jnp.where(a0 > 0, i_prime, -1.0)
        x_new = jnp.where(a0 > 0, x_new, x)

        step_result = SimulationStep(
            x_new=x_new,
            dt=dt,
            reaction_idx=jnp.full_like(a, i_prime),
            propensities=a,
        )

        return step_result, None


#####################################################################################
# Differentiable Direct method (Gumbel-max trick)
#####################################################################################


class DifferentiableDirect(AbstractStochasticSolver):
    """Pathwise differentiable Direct Method using the Straight-Through Gumbel-Softmax Gradient Estimator.

    This solver implements the Direct Method in a form that is pathwise
    differentiable with respect to reaction parameters. Differentiation is
    enabled by the Gumbel-max trick, which reparameterizes the discrete choice
    of which reaction fires next.

    The forward pass can be either exact or approximate (relaxed):
    - If `exact_fwd=True` (default), it uses a straight-through estimator,
        making the forward pass of the simulation exact.
    - If `exact_fwd=False`, it uses a continuous relaxation of the reaction
        choice, making the forward pass approximate.

    In both cases, the backward pass uses a continuous relaxation (softmax) to
    enable gradient propagation, allowing for end-to-end training of stochastic
    models with gradient-based optimization.

    Attributes:
        logits_scale: The temperature for the softmax relaxation.
        is_exact_solver: Boolean flag indicating exactness based on exact_fwd.
        is_pathwise_differentiable: Boolean flag indicating differentiability.

    Methods:
        propensities: Calculate reaction propensities for the current state.
        step: Execute one step of the differentiable Direct Method.
    """

    logits_scale: float

    def __init__(self, logits_scale: float = 1.0, exact_fwd: bool = True):
        """Initialize the DifferentiableDirect solver.

        Args:
            logits_scale: The temperature parameter for the Gumbel-softmax
                relaxation. Smaller values give a better approximation but may
                have higher variance gradients. Defaults to 1.0.
            exact_fwd: If True, the forward pass is exact, using a straight-
                through gradient estimator. If False, the forward pass is relaxed and approximate.
        """
        super().__init__(is_exact_solver=exact_fwd, is_pathwise_differentiable=True)
        self.logits_scale = logits_scale

    def propensities(self, network, x, t):
        """Calculate reaction propensities for the current state.

        Args:
            network: The reaction network containing the propensity function.
            x: Current state vector (species concentrations/counts).
            t: Current time.

        Returns:
            Array of propensities for each reaction.
        """
        return network.propensity_fn(x, t)

    def _delta_x(self, stoichiometry_matrix, reaction_idx):
        """Calculate the change in state for a single reaction event."""
        stoichiometry_matrix = jnp.asarray(stoichiometry_matrix)
        return stoichiometry_matrix[:, reaction_idx]

    def _delta_x_soft(self, stoichiometry_matrix, p_gumbel):
        """Calculate the change in state for a single reaction event."""
        stoichiometry_matrix = jnp.asarray(stoichiometry_matrix)
        return stoichiometry_matrix @ p_gumbel

    def step(
        self,
        network: ReactionNetwork,
        t: jnp.floating,
        x: jnp.ndarray,
        a: jnp.ndarray,
        state,
        *,
        key: jnp.ndarray,
    ) -> tuple[SimulationStep, None]:
        """Execute one step of the differentiable Direct Method.

        This method uses the Gumbel-max trick to make the reaction selection
        differentiable.

        1. The next reaction time is sampled from an exponential distribution.
        2. A reaction is sampled using Gumbel-perturbed propensities.
        3. For the backward pass, a soft reaction selection is computed via softmax.
        4. If `exact_fwd=True`, the state is updated using the exact selection,
            but gradients are propagated through the soft selection via a
            straight-through estimator.

        Args:
            network: The reaction network.
            t: The current time.
            x: The current state vector.
            a: The current propensity vector.
            state: The current solver state (unused).
            key: A JAX random key.

        Returns:
            Tuple containing the SimulationStep result and the new solver
            state (None for this stateless solver).
        """
        key_dt, key_gumbel = jax.random.split(key)

        stoichiometry_matrix = jnp.asarray(network.stoichiometry_matrix)

        a0 = jnp.sum(a)

        # sample time
        dt = -jnp.log(rng.uniform(key_dt)) / a0

        gumbels = rng.gumbel(key_gumbel, shape=a.shape)
        logits = jnp.log(a + jnp.finfo(a.dtype).eps)
        perturbed_logits = logits + gumbels

        p_gumbel = jax.nn.softmax(perturbed_logits / self.logits_scale)

        delta_x_soft = self._delta_x_soft(stoichiometry_matrix, p_gumbel)

        if self.is_exact_solver:
            # Exact forward pass with straight-through estimator
            # sample reaction via Gumbel-max trick
            r_hard = jnp.argmax(perturbed_logits)
            reaction_idx = r_hard

            # compute delta_x for the exact forward pass
            delta_x_hard = self._delta_x(stoichiometry_matrix, r_hard)
            delta_x = jax.lax.stop_gradient(delta_x_hard - delta_x_soft) + delta_x_soft

            x_new = x + delta_x
        else:
            # Soft forward pass (inexact)
            x_new = x + delta_x_soft
            reaction_idx = p_gumbel

        step_result = SimulationStep(
            x_new=x_new,
            dt=dt,
            reaction_idx=reaction_idx,
            propensities=a,
        )

        return step_result, None


#####################################################################################
# Differentiable First Reaction method
#####################################################################################


class DifferentiableFirstReaction(AbstractStochasticSolver):
    """Pathwise differentiable First Reaction Method using the Straight-Through Gumbel-Softmax Gradient Estimator.

    This solver implements the First Reaction Method in a form that is
    pathwise differentiable. It is the differentiable counterpart to FirstReactionMethod.

    The forward pass can be either exact or approximate (relaxed), controlled
    by `exact_fwd`:
    - If `exact_fwd=True` (default), it uses a straight-through estimator for
        both the reaction choice and the time step `dt`, ensuring an exact
        forward pass.
    - If `exact_fwd=False`, it uses a continuous relaxation for both the
        state update and `dt`, making the forward pass approximate.

    The backward pass uses a continuous relaxation (soft-argmin) of the
    reaction choice and a relaxed `dt` to enable gradient propagation.

    Attributes:
        logits_scale: The temperature for the softmax relaxation.
        is_exact_solver: Boolean flag indicating exactness based on exact_fwd.
        is_pathwise_differentiable: Boolean flag indicating differentiability.

    Methods:
        propensities: Calculate reaction propensities for the current state.
        step: Execute one step of the differentiable First Reaction Method.
    """

    logits_scale: float

    def __init__(self, logits_scale: float = 1.0, exact_fwd: bool = True):
        """Initialize the DifferentiableFirstReaction solver.

        Args:
            logits_scale: The temperature for the soft-argmin relaxation.
                Smaller values give a better approximation but may have higher
                variance gradients. Defaults to 1.0.
            exact_fwd: If True, the forward pass is exact, using a straight-
                through estimator. If False, the forward pass is relaxed and
                approximate.
        """
        super().__init__(is_exact_solver=exact_fwd, is_pathwise_differentiable=True)
        self.logits_scale = logits_scale

    def propensities(self, network, x, t):
        """Calculate reaction propensities for the current state.

        Args:
            network: The reaction network containing the propensity function.
            x: Current state vector (species concentrations/counts).
            t: Current time.

        Returns:
            Array of propensities for each reaction.
        """
        return network.propensity_fn(x, t)

    def _delta_x(self, stoichiometry_matrix, reaction_idx):
        """Calculate the change in state for a single reaction event."""
        stoichiometry_matrix = jnp.asarray(stoichiometry_matrix)
        return stoichiometry_matrix[:, reaction_idx]

    def _delta_x_soft(self, stoichiometry_matrix, p_gumbel):
        """Calculate the change in state for a single reaction event."""
        stoichiometry_matrix = jnp.asarray(stoichiometry_matrix)
        return stoichiometry_matrix @ p_gumbel

    def step(
        self,
        network: ReactionNetwork,
        t: jnp.floating,
        x: jnp.ndarray,
        a: jnp.ndarray,
        state,
        *,
        key: jnp.ndarray,
    ) -> tuple[SimulationStep, None]:
        """Execute one step of the differentiable First Reaction Method.

        1. Samples candidate firing times for all reactions.
        2. For the backward pass, computes a soft reaction selection
            (`soft-argmin`) and a soft `dt` using a softmax relaxation.
        3. If `exact_fwd=True`, the forward pass selects the reaction with the
            minimum firing time (`argmin`) and its `dt`, while gradients are
            propagated through the soft values via a straight-through estimator.

        Args:
            network: The reaction network.
            t: The current time.
            x: The current state vector.
            a: The current propensity vector.
            state: The current solver state (unused).
            key: A JAX random key.

        Returns:
            Tuple containing the SimulationStep result and the new solver
            state (None for this stateless solver).
        """
        stoichiometry_matrix = jnp.asarray(network.stoichiometry_matrix)

        # Sample candidate firing times
        dt_vec = -jnp.log(rng.uniform(key, shape=a.shape)) / (
            a + jnp.finfo(a.dtype).eps
        )

        # Differentiable backward pass using softmax relaxation
        # soft-argmin is softmax of negative values:
        # The argmin of Exp(1)/a_j is equivalent to argmax(log(a_j) + g_j)
        p_gumbel = jax.nn.softmax(-dt_vec / self.logits_scale)

        delta_x_soft = self._delta_x_soft(stoichiometry_matrix, p_gumbel)
        dt_soft = jnp.sum(p_gumbel * dt_vec)

        if self.is_exact_solver:
            # Exact forward pass with straight-through estimator
            r_hard = jnp.argmin(dt_vec)
            reaction_idx = r_hard

            # compute delta_x for the exact forward pass
            delta_x_hard = self._delta_x(stoichiometry_matrix, r_hard)
            delta_x = jax.lax.stop_gradient(delta_x_hard - delta_x_soft) + delta_x_soft
            x_new = x + delta_x

            # compute dt for the exact forward pass
            dt_hard = dt_vec[r_hard]
            dt = dt_soft + jax.lax.stop_gradient(dt_hard - dt_soft)

        else:
            # Soft forward pass (inexact)
            x_new = x + delta_x_soft
            dt = dt_soft
            reaction_idx = p_gumbel

        step_result = SimulationStep(
            x_new=x_new,
            dt=dt,
            reaction_idx=reaction_idx,
            propensities=a,
        )
        return step_result, None
