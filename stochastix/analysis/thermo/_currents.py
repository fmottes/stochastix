"""Reaction currents and thermodynamic affinities."""

from __future__ import annotations

import typing

import jax
import jax.numpy as jnp

from ..._state_utils import pytree_to_state
from ._utils import (
    _propensity_trajectory,
    _require_reaction_indices,
    _total_time,
    _validate_reversible,
)

if typing.TYPE_CHECKING:
    from ..._simulation_results import SimulationResults
    from ...reaction import ReactionNetwork

_AFFINITY_REDUCTIONS = ('none', 'mean', 'time_average')


def affinities(
    network: ReactionNetwork,
    results: SimulationResults,
    *,
    reduce: str = 'mean',
    require_reversible: bool = True,
    use_stored_propensities: bool = True,
) -> jax.Array:
    r"""Compute stochastic edge affinities along a trajectory.

    For reaction ``rho`` with state change ``nu_rho``, the affinity of the
    stochastic transition edge leaving state ``x`` is

    ``A_rho(x) = log( a_rho(x) / a_rho_bar(x + nu_rho) )``

    Thus the reverse propensity is evaluated at the state on the other end of
    the same Markov-jump edge, not at ``x``. This finite-copy correction is what
    distinguishes the stochastic affinity from the macroscopic same-state
    reaction-coordinate convention. Channels that cannot fire at a state are
    assigned zero affinity because no outgoing edge exists there.

    Args:
        network: The reaction network that generated the trajectory.
        results: A single (non-batched) trajectory `SimulationResults`.
        reduce: How to reduce over the visited states. One of ``'none'`` (return
            the per-step affinity of every channel, shape ``(n_steps,
            n_reactions)``), ``'mean'`` (uniform average over valid steps, shape
            ``(n_reactions,)``), or ``'time_average'`` (dwell-time-weighted
            average over valid steps, shape ``(n_reactions,)``).
        require_reversible: If ``True`` (default), raise if any reaction lacks a
            reverse channel. If ``False``, channels without a reverse have
            affinity ``+inf``.
        use_stored_propensities: If ``True`` (default), reuse
            `results.propensities` for the forward propensities (requires
            ``save_propensities=True``); if ``False``, recompute them. Reverse
            propensities at successor states are always computed.

    Returns:
        The reaction affinities, reduced as requested (see ``reduce``).
    """
    if reduce not in _AFFINITY_REDUCTIONS:
        raise ValueError(
            f'reduce must be one of {list(_AFFINITY_REDUCTIONS)}, got {reduce!r}.'
        )
    _require_reaction_indices(results)
    _validate_reversible(network, require_reversible)

    a_all = _propensity_trajectory(
        network, results, use_stored_propensities=use_stored_propensities
    )
    a_pre = a_all[:-1]
    x = pytree_to_state(results.x, network.species)
    x_pre = x[:-1]
    t_post = results.t[1:]
    stoichiometry = jnp.asarray(network.stoichiometry_matrix).T
    reactants = jnp.asarray(network.reactant_matrix)

    # Evaluate each reverse channel at the successor state on the same edge.
    # The Python loop is over static network structure and remains JIT-traceable.
    reverse_propensities = []
    for rho, reverse_rho in enumerate(network._reverse_reaction_index):
        enabled = a_pre[:, rho] > 0
        successor = x_pre + stoichiometry[rho]
        # An unavailable reaction can have a formally negative successor. Its
        # affinity is masked below, so evaluate at x instead to keep arbitrary
        # user kinetics away from invalid states.
        safe_successor = jnp.where(enabled[:, None], successor, x_pre)

        if reverse_rho < 0:
            a_reverse_rho = jnp.zeros_like(a_pre[:, rho])
        else:
            reverse_reaction = network.reactions[reverse_rho]
            reverse_reactants = reactants[:, reverse_rho]

            def reverse_propensity(
                state,
                t,
                reaction=reverse_reaction,
                reactant_counts=reverse_reactants,
            ):
                return reaction.kinetics.propensity_fn(
                    state, reactant_counts, t, volume=network.volume
                )

            a_reverse_rho = jax.vmap(reverse_propensity)(safe_successor, t_post)
        reverse_propensities.append(a_reverse_rho)

    a_reverse = (
        jnp.stack(reverse_propensities, axis=1)
        if reverse_propensities
        else jnp.zeros_like(a_pre)
    )
    enabled = a_pre > 0
    safe_forward = jnp.where(enabled, a_pre, 1.0)
    safe_reverse = jnp.where(enabled, a_reverse, 1.0)
    affinity = jnp.where(enabled, jnp.log(safe_forward) - jnp.log(safe_reverse), 0.0)

    has_reverse = network.reverse_reaction_index >= 0
    affinity = jnp.where(enabled & ~has_reverse[None, :], jnp.inf, affinity)

    mask = results.reactions >= 0
    if reduce == 'none':
        return jnp.where(mask[:, None], affinity, 0.0)

    if reduce == 'mean':
        weights = mask.astype(a_all.dtype)
        reduction_mask = mask
    else:  # 'time_average'
        dt = jax.lax.stop_gradient(jnp.diff(results.t))
        weights = dt
        # Include the censored dwell from the final reaction to the observation
        # endpoint even though no reaction fires at the end of that interval.
        reduction_mask = dt > 0

    weights = weights[:, None]
    tiny = jnp.finfo(a_all.dtype).tiny
    denom = jnp.maximum(jnp.sum(weights), tiny)
    return (
        jnp.sum(jnp.where(reduction_mask[:, None], affinity * weights, 0.0), axis=0)
        / denom
    )


def reaction_currents(
    network: ReactionNetwork,
    results: SimulationResults,
    *,
    normalize_by_time: bool = True,
) -> jax.Array:
    r"""Compute the net current through each reaction channel.

    The net current of channel ``rho`` counts its firings minus those of its
    reverse channel ``rho_bar``:

    ``J_rho = (#rho - #rho_bar) / T``

    (the ``/ T`` applied only when ``normalize_by_time=True``). Irreversible
    channels have no reverse to subtract, so their current is just their firing
    count.

    Note: Differentiability
        Firing counts are discrete, so this quantity is piecewise-constant in
        the rate parameters. Gradients require a pathwise-differentiable solver
        (e.g. `DifferentiableDirect`) or a score-function estimator built from
        `ReactionNetwork.log_prob`. `affinities` and `entropy_production`, by
        contrast, are smooth in the rate parameters.

    Args:
        network: The reaction network that generated the trajectory.
        results: A single (non-batched) trajectory `SimulationResults`.
        normalize_by_time: If ``True`` (default), divide counts by the total
            simulated time to give a rate; otherwise return net integer counts.

    Returns:
        The per-channel net current, shape ``(n_reactions,)``.
    """
    _require_reaction_indices(results)

    reactions = results.reactions
    mask = reactions >= 0
    channels = jnp.arange(network.n_reactions)

    # Firing count per channel over valid steps.
    fired = (reactions[:, None] == channels[None, :]) & mask[:, None]
    counts = jnp.sum(fired, axis=0)

    reverse_index = network.reverse_reaction_index
    safe_reverse = jnp.maximum(reverse_index, 0)
    reverse_counts = jnp.where(reverse_index >= 0, counts[safe_reverse], 0)

    net = counts - reverse_counts
    if normalize_by_time:
        return net / _total_time(results)
    return net


def schnakenberg_epr(
    network: ReactionNetwork,
    results: SimulationResults,
    *,
    require_reversible: bool = True,
    use_stored_propensities: bool = True,
) -> jax.Array:
    r"""Estimate the steady-state Schnakenberg entropy production rate.

    Schnakenberg's decomposition is a state-edge-resolved current-affinity
    contraction. For a stationary trajectory, its empirical pathwise estimator
    is the time-averaged sum of the realized jump affinities,

    ``sigma_dot = (1 / T) sum_i log(a_rho_i(x_i) /
    a_rho_i_bar(x_{i+1}))``.

    This equals the medium entropy production rate trajectory by trajectory and
    converges to the total Schnakenberg entropy production rate at steady state,
    where the mean system-entropy change vanishes.

    Note: State resolution
        It is generally incorrect to multiply a channel-aggregated net firing
        current by a separately time-averaged same-state affinity: propensity
        ratios depend on the state, and that aggregation discards their
        correlation with individual jumps. This function therefore uses the
        post-jump reverse propensity for every realized transition.

    Args:
        network: The reaction network that generated the trajectory.
        results: A single (non-batched) trajectory `SimulationResults`.
        require_reversible: See `affinities`.
        use_stored_propensities: See `affinities`.

    Returns:
        The steady-state trajectory estimator of the Schnakenberg entropy
        production rate (scalar, nats per unit time).
    """
    edge_affinity = affinities(
        network,
        results,
        reduce='none',
        require_reversible=require_reversible,
        use_stored_propensities=use_stored_propensities,
    )

    reactions = results.reactions
    mask = reactions >= 0
    rows = jnp.arange(reactions.shape[0])
    safe_reactions = jnp.maximum(reactions, 0)
    realized_affinity = edge_affinity[rows, safe_reactions]
    return jnp.sum(jnp.where(mask, realized_affinity, 0.0)) / _total_time(results)
