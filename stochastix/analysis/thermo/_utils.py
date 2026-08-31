"""Shared internals for stochastic-thermodynamics observables."""

from __future__ import annotations

import typing

import jax
import jax.numpy as jnp

from ..._state_utils import pytree_to_state

if typing.TYPE_CHECKING:
    from ..._simulation_results import SimulationResults
    from ...reaction import ReactionNetwork


def _require_reaction_indices(results: SimulationResults) -> None:
    """Raise if the per-step fired-reaction record is missing.

    Pathwise thermodynamic observables need to know which reaction fired at
    each step. That record is only available from an exact-solver trajectory
    saved with ``save_trajectory=True``.
    """
    if results.reactions is None:
        raise ValueError(
            'This observable needs the per-step fired-reaction record, but '
            '`results.reactions` is None. Produce the trajectory with '
            '`stochsimsolve(..., save_trajectory=True)` using an exact solver '
            '(e.g. `DirectMethod` or `DifferentiableDirect`).'
        )


def _validate_reversible(network: ReactionNetwork, require_reversible: bool) -> None:
    """Raise if `require_reversible` and some reaction has no reverse channel.

    A structural reverse channel is necessary, but not sufficient, for finite
    entropy production and affinities; its propensity must also be positive on
    every traversed reverse edge.
    """
    if require_reversible and not network.has_reverse_channels:
        missing = network._reactions_missing_reverse()
        raise ValueError(
            'Entropy production requires every reaction to have a reverse '
            'channel, but the following reactions have none: '
            f'{missing}. Add the missing reverse reactions, or pass '
            '`require_reversible=False` to let those steps contribute `+inf`.'
        )


def _propensity_trajectory(
    network: ReactionNetwork,
    results: SimulationResults,
    *,
    use_stored_propensities: bool = True,
) -> jax.Array:
    """Return the propensity vector at every state of a trajectory.

    Produces ``a(x[k])`` for all states ``k = 0 .. N`` (shape
    ``(N + 1, n_reactions)``), the single quantity from which every
    thermodynamic observable is derived.

    Args:
        network: The reaction network.
        results: A single (non-batched) trajectory `SimulationResults`.
        use_stored_propensities: If ``True`` (default), reuse
            `results.propensities` (``a(x[0 .. N-1])``) and recompute only the
            terminal state ``a(x[N])``; requires ``save_propensities=True``. If
            ``False``, recompute the propensities at every state.

    Returns:
        The propensity trajectory, shape ``(N + 1, n_reactions)``.
    """
    x = pytree_to_state(results.x, network.species)
    t = results.t

    if use_stored_propensities:
        if results.propensities is None:
            raise ValueError(
                '`use_stored_propensities=True` requires results produced with '
                '`save_propensities=True`, but `results.propensities` is None.'
            )
        # Stored propensities cover pre-jump states x[0 .. N-1], but are zero on
        # padded (post-termination) steps and omit the terminal state x[N]. The
        # frozen final-state propensity is recomputed once and used both for the
        # terminal slot and to patch the first padded slot, which is the
        # post-jump state that the last valid jump's reverse term reads.
        terminal = network.propensity_fn(x[-1], t[-1])
        a_all = jnp.concatenate([results.propensities, terminal[None, :]], axis=0)
        n_valid = jnp.sum(results.reactions >= 0)
        return a_all.at[n_valid].set(terminal)

    return jax.vmap(network.propensity_fn)(x, t)


def _total_time(results: SimulationResults) -> jax.Array:
    """Return the observation duration represented by a trajectory.

    For an uncleaned trajectory that reaches the requested terminal time, the
    final timestamp includes the censored dwell after the last reaction and is
    repeated through the trailing padding. Endpoint subtraction therefore gives
    the full observation window rather than merely the time to the final event.
    For a cleaned or max-step-limited trajectory, it gives the duration actually
    retained in that trajectory. Time is treated as data (gradient-stopped).
    """
    return jax.lax.stop_gradient(results.t[-1] - results.t[0])
