"""Entropy production observables for stochastic trajectories."""

from __future__ import annotations

import math
import typing

import equinox as eqx
import jax
import jax.numpy as jnp

from .._utils import normalize_time_scalar
from ._utils import (
    _propensity_trajectory,
    _require_reaction_indices,
    _total_time,
    _validate_reversible,
)

if typing.TYPE_CHECKING:
    from ..._simulation_results import SimulationResults
    from ...reaction import ReactionNetwork


def entropy_production(
    network: ReactionNetwork,
    results: SimulationResults,
    *,
    require_reversible: bool = True,
    per_step: bool = False,
    use_stored_propensities: bool = True,
) -> jax.Array:
    r"""Compute the medium (environment) entropy production of a trajectory.

    For a Markov-jump trajectory that jumps ``x[i] -> x[i+1]`` via reaction
    ``rho``, the entropy produced in the environment is

    ``Delta s_env = sum_i log( a_rho(x[i]) / a_rho_bar(x[i+1]) )``

    where ``a_rho`` is the propensity of the fired reaction and ``a_rho_bar`` is
    the propensity of its reverse reaction evaluated at the post-jump state.
    This log path-rate ratio is the medium entropy flow of the Markov jump
    process. When the rates obey local detailed balance, it can additionally be
    identified with dissipated heat divided by ``k_B T``. Arbitrary user-defined
    kinetics need not admit that physical heat interpretation. The quantity is
    differentiable with respect to the rate parameters.

    Args:
        network: The reaction network that generated the trajectory.
        results: A single (non-batched) trajectory `SimulationResults`, produced
            by an exact solver with ``save_trajectory=True``.
        require_reversible: If ``True`` (default), raise if any reaction lacks a
            reverse channel (entropy production would diverge). If ``False``,
            steps firing an irreversible reaction contribute ``+inf``.
        per_step: If ``True``, return the per-step increments (shape
            ``(n_steps,)``) instead of their sum.
        use_stored_propensities: If ``True`` (default), reuse
            `results.propensities` (requires ``save_propensities=True``); if
            ``False``, recompute them.

    Returns:
        The total medium entropy production (scalar, in nats), or the per-step
        increments if ``per_step=True``.
    """
    _require_reaction_indices(results)
    _validate_reversible(network, require_reversible)

    a_all = _propensity_trajectory(
        network, results, use_stored_propensities=use_stored_propensities
    )
    reactions = results.reactions
    reverse_index = network.reverse_reaction_index

    mask = reactions >= 0
    n_steps = reactions.shape[0]
    rows = jnp.arange(n_steps)
    safe = jnp.maximum(reactions, 0)

    # Forward propensity a_rho(x[i]) from the pre-jump states a(x[0 .. N-1]).
    a_forward = a_all[:-1][rows, safe]

    # Reverse channel of the fired reaction, and its propensity a_rho_bar(x[i+1])
    # from the post-jump states a(x[1 .. N]).
    reverse_channel = reverse_index[safe]
    safe_reverse = jnp.maximum(reverse_channel, 0)
    a_reverse = a_all[1:][rows, safe_reverse]

    # Make padded gathers safe without regularizing physical zero rates. A
    # fired edge with zero reverse propensity must retain its +inf entropy cost.
    safe_forward = jnp.where(mask, a_forward, 1.0)
    safe_reverse = jnp.where(mask, a_reverse, 1.0)
    log_ratio = jnp.log(safe_forward) - jnp.log(safe_reverse)

    # A fired reaction with no reverse channel produces infinite entropy.
    increments = jnp.where(reverse_channel >= 0, log_ratio, jnp.inf)
    # Padded/invalid steps do not contribute.
    increments = jnp.where(mask, increments, 0.0)

    if per_step:
        return increments
    return jnp.sum(increments)


def entropy_production_rate(
    network: ReactionNetwork,
    results: SimulationResults,
    *,
    require_reversible: bool = True,
    use_stored_propensities: bool = True,
) -> jax.Array:
    """Compute the average medium entropy production rate of a trajectory.

    This is `entropy_production` divided by the observation duration represented
    by the trajectory endpoints. The elapsed time is treated as data
    (gradient-stopped), so gradients flow only through the propensities.

    Args:
        network: The reaction network that generated the trajectory.
        results: A single (non-batched) trajectory `SimulationResults`.
        require_reversible: See `entropy_production`.
        use_stored_propensities: See `entropy_production`.

    Returns:
        The entropy production rate (scalar, nats per unit time).
    """
    ep = entropy_production(
        network,
        results,
        require_reversible=require_reversible,
        use_stored_propensities=use_stored_propensities,
    )
    return ep / _total_time(results)


def system_entropy(
    results: SimulationResults,
    species_at_t: tuple[str, typing.Any],
    n_grid_points: int | None = None,
    min_max_vals: tuple[float, float] | None = None,
    *,
    kde_type: str = 'wendland_c2',
    bw_multiplier: float = 1.0,
    dirichlet_alpha: float | None = 0.1,
    dirichlet_kappa: float | None = None,
) -> jax.Array:
    """Compute the system (Shannon) entropy of a species marginal, in nats.

    In the stochastic-thermodynamics formulation the system entropy is the
    Shannon entropy ``S_sys = -sum p log p`` of the state distribution. This is
    a thin wrapper around `analysis.state_entropy` fixed to natural units
    (nats), estimating the entropy of a single species' marginal from a batched
    ensemble at a chosen physical time.

    Note: Dimensionality
        The KDE estimator supports one species (this function) or two species
        (`analysis.mutual_information` / `kde_2d`). A rigorous joint entropy over
        more than two species is not yet available; use per-species marginals or
        await an n-dimensional KDE.

    Args:
        results: Batched `SimulationResults` from an ensemble of simulations.
        species_at_t: Tuple ``(species_name, t)`` with ``t`` a scalar physical
            time.
        n_grid_points: KDE grid points. If ``None``, inferred from data (not
            JIT-able).
        min_max_vals: KDE grid range. If ``None``, inferred from data.
        kde_type: KDE kernel. Default ``'wendland_c2'``.
        bw_multiplier: KDE bandwidth multiplier.
        dirichlet_alpha: Per-bin pseudo-count for Dirichlet smoothing.
        dirichlet_kappa: Total pseudo-count for Dirichlet smoothing (takes
            priority over ``dirichlet_alpha``).

    Returns:
        The system entropy of the selected species at the requested time, in
        nats.
    """
    # Imported here to keep the analysis.entropy dependency local and avoid any
    # import-order coupling within the analysis package.
    from ..entropy import state_entropy

    return state_entropy(
        results,
        species_at_t,
        n_grid_points=n_grid_points,
        min_max_vals=min_max_vals,
        base=math.e,
        kde_type=kde_type,
        bw_multiplier=bw_multiplier,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_kappa=dirichlet_kappa,
    )


def total_entropy_production(
    network: ReactionNetwork,
    results: SimulationResults,
    species: str,
    t_initial: typing.Any,
    t_final: typing.Any,
    *,
    require_reversible: bool = True,
    n_grid_points: int | None = None,
    min_max_vals: tuple[float, float] | None = None,
    kde_type: str = 'wendland_c2',
    bw_multiplier: float = 1.0,
    dirichlet_alpha: float | None = 0.1,
    dirichlet_kappa: float | None = None,
) -> jax.Array:
    r"""Compute the ensemble total entropy production between two times.

    The total entropy production decomposes as

    ``Delta S_tot = Delta S_sys + <Delta s_env>``

    where ``<Delta s_env>`` is the ensemble-mean medium entropy production and
    ``Delta S_sys = S_sys(t_final) - S_sys(t_initial)`` is the change in Shannon
    system entropy, estimated for the given species' marginal via KDE. A jump is
    included in the medium term when its event time lies in
    ``(t_initial, t_final]``.

    Note: System-entropy scope
        ``Delta S_sys`` here is the marginal Shannon entropy change of a single
        ``species`` (exact when the system state is that species; a marginal
        approximation otherwise, since a joint entropy over many species is not
        yet available). The medium term is always exact.

    Args:
        network: The reaction network that generated the trajectories.
        results: A **batched** ensemble `SimulationResults` (leading batch axis),
            produced by an exact solver with ``save_trajectory=True``.
        species: The species whose marginal system entropy is used.
        t_initial: Initial physical time for the observation interval.
        t_final: Final physical time for the observation interval.
        require_reversible: See `entropy_production`.
        n_grid_points: KDE grid points for ``S_sys``.
        min_max_vals: KDE grid range for ``S_sys``.
        kde_type: KDE kernel for ``S_sys``.
        bw_multiplier: KDE bandwidth multiplier for ``S_sys``.
        dirichlet_alpha: Dirichlet smoothing per-bin pseudo-count for ``S_sys``.
        dirichlet_kappa: Dirichlet smoothing total pseudo-count for ``S_sys``.

    Returns:
        The total entropy production over ``[t_initial, t_final]`` (scalar, nats).
    """
    t_initial = normalize_time_scalar(t_initial, arg_name='t_initial')
    t_final = normalize_time_scalar(t_final, arg_name='t_final')

    def _medium(single_results: SimulationResults) -> jax.Array:
        increments = entropy_production(
            network,
            single_results,
            require_reversible=require_reversible,
            per_step=True,
        )
        event_times = single_results.t[1:]
        in_interval = (event_times > t_initial) & (event_times <= t_final)
        valid = single_results.reactions >= 0
        return jnp.sum(jnp.where(valid & in_interval, increments, 0.0))

    mean_medium = jnp.mean(eqx.filter_vmap(_medium)(results))

    kde_kwargs = {
        'n_grid_points': n_grid_points,
        'min_max_vals': min_max_vals,
        'kde_type': kde_type,
        'bw_multiplier': bw_multiplier,
        'dirichlet_alpha': dirichlet_alpha,
        'dirichlet_kappa': dirichlet_kappa,
    }
    s_initial = system_entropy(results, (species, t_initial), **kde_kwargs)
    s_final = system_entropy(results, (species, t_final), **kde_kwargs)

    return mean_medium + (s_final - s_initial)
