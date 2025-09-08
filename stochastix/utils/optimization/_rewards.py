"""Reward functions for reinforcement learning with chemical reaction networks."""

from __future__ import annotations

import jax.numpy as jnp

from ..._simulation_results import SimulationResults


def steady_state_distance(from_t=0.0, species=0):
    """Compute the distance from a target steady state.

    Args:
        from_t: The time from which to start computing the distance.
        species: The index or name of the species to track.

    Returns:
        A function that takes SimulationResults and a target steady-state
        value and returns the absolute distance from the target.
    """
    # check if species is an integer or string
    if not isinstance(species, int | str):
        raise ValueError('species must be an integer or a string')

    def _loss(ssa_results: SimulationResults, target_ss: jnp.ndarray):
        # if species is a string, get its index
        if isinstance(species, str):
            _species = ssa_results.species.index(species)
        else:
            _species = species

        return jnp.where(
            ssa_results.t < from_t, 0, jnp.abs(ssa_results.x[:, _species] - target_ss)
        )

    return _loss


def neg_final_state_distance(species=0, distance='L1'):
    """Compute the negative distance from a target final state.

    This is useful for creating a reward function where the goal is to minimize
    the distance to a target state at the end of the simulation.

    Args:
        species: The index or name of the species to track.
        distance: The distance metric, either 'L1' or 'L2'.

    Returns:
        A function that takes SimulationResults and a target state value
        and returns a reward vector. The reward is non-zero only at the final
        step of the simulation.
    """
    # check if species is an integer or string
    if not isinstance(species, int | str):
        raise ValueError('species must be an integer or a string')

    if distance == 'L1':
        dist_fn = lambda x, y: jnp.abs(x - y)
    elif distance == 'L2':
        dist_fn = lambda x, y: (x - y) ** 2
    else:
        raise ValueError("distance must be 'L1' or 'L2'")

    def rewards_fn(ssa_results: SimulationResults, target_ss: jnp.ndarray):
        i = jnp.where(ssa_results.reactions < 0, size=1, fill_value=-2)[0]

        # if the simulation ends with a valid reaction, we need to consider the last state
        i = jnp.maximum(i, -1)

        # if species is a string, get its index
        if isinstance(species, str):
            _species = ssa_results.species.index(species)
        else:
            _species = species

        d = dist_fn(ssa_results.x[i, _species], target_ss)

        rew = (
            jnp.zeros_like(ssa_results.reactions, dtype=ssa_results.x.dtype)
            .at[i]
            .set(-d)
        )

        return rew

    return rewards_fn


def rewards_from_state_metric(metric_fn, metric_type='cost'):
    """Generate a reward function from the differences of a state metric.

    This is a general-purpose utility to create reward functions. It takes a
    metric function that computes a value at each time step and returns the
    difference between consecutive values as the reward.

    Args:
        metric_fn: A function that takes SimulationResults and returns a
            vector of metric values, one for each time step.
        metric_type: Either 'cost' or 'reward'. If 'cost', the reward is the
            negative difference of the metric. If 'reward', it is the positive
            difference.

    Returns:
        A function that takes SimulationResults and computes a reward vector.
    """

    def _rewards(ssa_results: SimulationResults, **metric_kwargs):
        state_metrics = metric_fn(ssa_results, **metric_kwargs)

        if metric_type == 'reward':
            rewards = jnp.diff(state_metrics)
        elif metric_type == 'cost':
            rewards = -jnp.diff(state_metrics)

        return rewards

    return _rewards
