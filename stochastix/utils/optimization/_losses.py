"""Loss functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from ..._simulation_results import SimulationResults
    from ..._systems import StochasticModel
    from ...reaction import ReactionNetwork


def reinforce_loss(stop_returns_grad: bool = True):
    """Create a loss function for the REINFORCE algorithm.

    Args:
        stop_returns_grad: If True, gradients will not be computed for the
            returns. This is the standard formulation of REINFORCE.

    Returns:
        A loss function that computes the REINFORCE loss.

    The returned loss function has the signature:
    `loss(model, ssa_results, returns, baseline=0.0)`

    Args:
            model: The model being trained. It must have a `log_prob` method
                (e.g., `ReactionNetwork`) or a `.network` attribute with a
                `log_prob` method (e.g., `StochasticModel`).
            ssa_results: The output from a stochastic simulation.
            returns: The returns for each step of the trajectory, typically
                discounted cumulative rewards.
            baseline: A baseline to subtract from the returns to reduce
                variance. It should not have gradients with respect to the
                policy parameters.

    Returns:
            The computed REINFORCE loss as a scalar.
    """

    def _loss(
        model: ReactionNetwork | StochasticModel,
        ssa_results: SimulationResults,
        returns: jnp.ndarray,
        baseline: float = 0.0,
    ):
        """Compute the REINFORCE loss.

        Args:
            model: The model being trained, which must have a `log_prob` method
                or a `.network.log_prob` attribute.
            ssa_results: The results from a stochastic simulation.
            returns: The returns for each step of the trajectory.
            baseline: A variance-reduction baseline.

        Returns:
            The computed REINFORCE loss.
        """
        if stop_returns_grad:
            returns = jax.lax.stop_gradient(returns)

        # Duck typing to avoid circular imports

        # If `log_probabilities` is exposed then it's a ReactionNetwork
        if hasattr(model, 'log_prob'):
            log_prob_fn = model.log_prob
        # If `network` is present then it's a system (StochasticModel, MeanFieldModel)
        elif hasattr(model, 'network') and hasattr(model.network, 'log_prob'):
            log_prob_fn = model.network.log_prob
        else:
            raise TypeError(
                '`model` must implement `log_prob` either directly or through a `.network` attribute. Usually in this case `model` is a `ReactionNetwork` or a `StochasticModel`.'
            )

        log_ps = log_prob_fn(ssa_results)

        advantages = returns - jax.lax.stop_gradient(baseline)

        loss = -jnp.sum(log_ps * advantages)

        return loss

    return _loss
