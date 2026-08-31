"""Tests for stochastic-thermodynamics observables (`analysis.thermo`)."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import stochastix as stx
from stochastix.analysis import thermo


# --------------------------------------------------------------------------- #
# Network builders
# --------------------------------------------------------------------------- #
def _reversible_pair(kf=2.0, kr=1.0, transform=None):
    """Closed reversible system A <-> B."""
    if transform is None:
        fk, rk = kf, kr
        kwargs = {}
    else:
        fk, rk = jnp.log(kf), jnp.log(kr)
        kwargs = {'transform': transform}
    return stx.ReactionNetwork(
        [
            stx.Reaction('A -> B', stx.kinetics.MassAction(k=fk, **kwargs), name='fwd'),
            stx.Reaction('B -> A', stx.kinetics.MassAction(k=rk, **kwargs), name='rev'),
        ]
    )


def _driven_cycle(k_fwd=2.0, k_rev=1.0):
    """Closed 3-cycle A -> B -> C -> A with reverses; a NESS with net current."""
    return stx.ReactionNetwork(
        [
            stx.Reaction('A -> B', stx.kinetics.MassAction(k=k_fwd)),
            stx.Reaction('B -> A', stx.kinetics.MassAction(k=k_rev)),
            stx.Reaction('B -> C', stx.kinetics.MassAction(k=k_fwd)),
            stx.Reaction('C -> B', stx.kinetics.MassAction(k=k_rev)),
            stx.Reaction('C -> A', stx.kinetics.MassAction(k=k_fwd)),
            stx.Reaction('A -> C', stx.kinetics.MassAction(k=k_rev)),
        ]
    )


def _simulate(network, x0, key, T, solver=None, **kwargs):
    solver = solver or stx.DirectMethod()
    return stx.stochsimsolve(key, network, x0, T=T, solver=solver, **kwargs)


# --------------------------------------------------------------------------- #
# Reverse-reaction pairing (structural primitive)
# --------------------------------------------------------------------------- #
def test_reverse_reaction_index_pairs_forward_and_reverse():
    net = _driven_cycle()
    # species sorted -> A,B,C; reactions in construction order.
    assert list(net.reverse_reaction_index) == [1, 0, 3, 2, 5, 4]
    assert net.has_reverse_channels


def test_reverse_reaction_index_marks_irreversible():
    net = stx.ReactionNetwork(
        [
            stx.Reaction('A -> B', stx.kinetics.MassAction(k=1.0)),
            stx.Reaction('B -> A', stx.kinetics.MassAction(k=1.0)),
            stx.Reaction('0 -> A', stx.kinetics.MassAction(k=1.0), name='synth'),
        ]
    )
    assert list(net.reverse_reaction_index) == [1, 0, -1]
    assert not net.has_reverse_channels
    missing = net._reactions_missing_reverse()
    assert len(missing) == 1 and 'synth' in missing[0]


def test_self_reverse_reaction_has_no_partner():
    # A -> A has a signature that equals its own swap but must not pair with itself.
    net = stx.ReactionNetwork([stx.Reaction('A -> A', stx.kinetics.MassAction(k=1.0))])
    assert list(net.reverse_reaction_index) == [-1]


def test_to_latex_still_pairs_reversible_reactions():
    net = _driven_cycle()
    latex = net.to_latex()
    # Three reversible pairs -> three leftrightarrow lines, no rightarrow.
    assert latex.count(r'\leftrightarrow') == 3
    assert r'\rightarrow' not in latex

    net_irr = stx.ReactionNetwork(
        [
            stx.Reaction('A -> B', stx.kinetics.MassAction(k=1.0)),
            stx.Reaction('B -> A', stx.kinetics.MassAction(k=1.0)),
            stx.Reaction('0 -> A', stx.kinetics.MassAction(k=1.0)),
        ]
    )
    latex_irr = net_irr.to_latex()
    assert latex_irr.count(r'\leftrightarrow') == 1
    assert latex_irr.count(r'\rightarrow') == 1


# --------------------------------------------------------------------------- #
# Entropy production physics
# --------------------------------------------------------------------------- #
def test_detailed_balance_has_zero_entropy_production_rate():
    kf, kr = 2.0, 1.0
    net = _reversible_pair(kf, kr)
    n = 300.0
    a_eq = n * kr / (kf + kr)
    x0 = jnp.array([a_eq, n - a_eq])  # start at equilibrium
    keys = jax.random.split(jax.random.PRNGKey(0), 128)

    def epr(key):
        return thermo.entropy_production_rate(net, _simulate(net, x0, key, T=10.0))

    mean_epr = jnp.mean(eqx.filter_vmap(epr)(keys))
    assert jnp.abs(mean_epr) < 0.5  # ~0 up to sampling noise


def test_low_copy_driven_cycle_epr_matches_schnakenberg():
    # Regression: channel-aggregated current times a separately averaged
    # same-state affinity gives the wrong answer at low copy number (about
    # log(2) / 3 for this system). The edge-resolved jump estimator gives log(2).
    net = _driven_cycle()
    x0 = jnp.array([1.0, 0.0, 0.0])
    keys = jax.random.split(jax.random.PRNGKey(1), 128)

    def both(key):
        res = _simulate(net, x0, key, T=50.0, max_steps=500)
        return (
            thermo.entropy_production_rate(net, res),
            thermo.schnakenberg_epr(net, res),
        )

    medium, schnak = eqx.filter_vmap(both)(keys)
    mean_medium, mean_schnak = jnp.mean(medium), jnp.mean(schnak)

    assert jnp.allclose(medium, schnak, rtol=2e-5, atol=1e-6)
    assert jnp.abs(mean_medium - jnp.log(2.0)) < 0.08
    assert jnp.abs(mean_schnak - jnp.log(2.0)) < 0.08


def test_low_copy_edge_affinity_uses_post_jump_reverse_propensity():
    net = _driven_cycle()
    res = _simulate(
        net,
        jnp.array([1.0, 0.0, 0.0]),
        jax.random.PRNGKey(21),
        T=10.0,
        max_steps=100,
    )

    edge_affinity = thermo.affinities(net, res, reduce='none')
    rows = jnp.arange(res.reactions.shape[0])
    mask = res.reactions >= 0
    safe_reactions = jnp.maximum(res.reactions, 0)
    realized = edge_affinity[rows, safe_reactions]
    medium_increments = thermo.entropy_production(net, res, per_step=True)

    assert jnp.allclose(realized, medium_increments)
    assert jnp.allclose(jnp.abs(realized[mask]), jnp.log(2.0), rtol=1e-5, atol=1e-5)


def test_entropy_production_per_step_sums_to_total():
    net = _reversible_pair()
    x0 = jnp.array([40.0, 60.0])
    res = _simulate(net, x0, jax.random.PRNGKey(2), T=3.0).clean()
    per_step = thermo.entropy_production(net, res, per_step=True)
    total = thermo.entropy_production(net, res)
    assert per_step.shape == res.reactions.shape
    assert jnp.allclose(jnp.sum(per_step), total)


# --------------------------------------------------------------------------- #
# Currents and affinities
# --------------------------------------------------------------------------- #
def test_reaction_currents_are_antisymmetric_across_pairs():
    net = _driven_cycle()
    x0 = jnp.array([100.0, 100.0, 100.0])
    res = _simulate(net, x0, jax.random.PRNGKey(3), T=15.0)
    currents = thermo.reaction_currents(net, res)
    # Each pair (rho, rho_bar) carries equal-and-opposite net current.
    assert jnp.allclose(currents[0], -currents[1])
    assert jnp.allclose(currents[2], -currents[3])
    assert jnp.allclose(currents[4], -currents[5])


def test_rate_normalization_uses_full_observation_window():
    net = _reversible_pair(kf=2e-6, kr=1e-6)
    res = _simulate(
        net,
        jnp.array([1.0, 0.0]),
        jax.random.PRNGKey(22),
        T=2.0,
        max_steps=8,
        solver=stx.DirectMethod(),
    )

    assert jnp.all(res.reactions < 0)
    assert jnp.allclose(res.t[-1] - res.t[0], 2.0)
    assert jnp.allclose(thermo.reaction_currents(net, res), jnp.zeros(2))
    assert jnp.allclose(thermo.entropy_production_rate(net, res), 0.0)
    time_average_affinity = thermo.affinities(net, res, reduce='time_average')
    assert jnp.allclose(time_average_affinity[0], jnp.log(2.0))


def test_edge_affinities_reductions():
    net = _reversible_pair(2.0, 1.0)
    x0 = jnp.array([80.0, 20.0])
    res = _simulate(net, x0, jax.random.PRNGKey(4), T=4.0)

    none = thermo.affinities(net, res, reduce='none')
    mean = thermo.affinities(net, res, reduce='mean')
    tavg = thermo.affinities(net, res, reduce='time_average')
    assert none.shape == (res.reactions.shape[0], net.n_reactions)
    assert mean.shape == (net.n_reactions,)
    assert tavg.shape == (net.n_reactions,)
    assert jnp.all(jnp.isfinite(mean))
    assert jnp.all(jnp.isfinite(tavg))


def test_affinities_rejects_bad_reduce():
    net = _reversible_pair()
    res = _simulate(net, jnp.array([40.0, 60.0]), jax.random.PRNGKey(5), T=2.0)
    with pytest.raises(ValueError, match='reduce must be one of'):
        thermo.affinities(net, res, reduce='sum')


# --------------------------------------------------------------------------- #
# Differentiability, JIT, vmap
# --------------------------------------------------------------------------- #
def test_entropy_production_rate_is_differentiable_under_jit():
    net = _reversible_pair(2.0, 1.0, transform=jnp.exp)
    x0 = jnp.array([70.0, 30.0])

    @eqx.filter_jit
    def loss(network, key):
        res = _simulate(network, x0, key, T=5.0, solver=stx.DifferentiableDirect())
        return thermo.entropy_production_rate(network, res)

    value, grads = eqx.filter_value_and_grad(loss)(net, jax.random.PRNGKey(6))
    grad_k = grads.reactions[0].kinetics.k
    assert jnp.isfinite(value)
    assert jnp.isfinite(grad_k)
    assert jnp.abs(grad_k) > 1e-6  # gradient actually flows through the rate


# --------------------------------------------------------------------------- #
# Error handling and edge cases
# --------------------------------------------------------------------------- #
def test_missing_reverse_raises_by_default():
    net = stx.ReactionNetwork(
        [
            stx.Reaction('A -> B', stx.kinetics.MassAction(k=1.0), name='decay'),
        ]
    )
    res = _simulate(net, jnp.array([50.0, 0.0]), jax.random.PRNGKey(7), T=2.0)
    with pytest.raises(ValueError, match='reverse channel'):
        thermo.entropy_production(net, res)


def test_missing_reverse_gives_inf_when_allowed():
    net = stx.ReactionNetwork([stx.Reaction('A -> B', stx.kinetics.MassAction(k=1.0))])
    res = _simulate(net, jnp.array([50.0, 0.0]), jax.random.PRNGKey(8), T=2.0)
    ep = thermo.entropy_production(net, res, require_reversible=False)
    assert jnp.isinf(ep)


def test_zero_reverse_propensity_gives_inf():
    net = _reversible_pair(kf=1.0, kr=0.0)
    res = _simulate(
        net,
        jnp.array([1.0, 0.0]),
        jax.random.PRNGKey(1),
        T=10.0,
        max_steps=4,
        solver=stx.DirectMethod(),
    )

    assert res.reactions[0] == 0
    assert jnp.isinf(thermo.entropy_production(net, res))
    edge_affinity = thermo.affinities(net, res, reduce='none')
    assert jnp.isinf(edge_affinity[0, 0])


def test_missing_reactions_record_raises():
    net = _reversible_pair()
    res = _simulate(
        net,
        jnp.array([40.0, 60.0]),
        jax.random.PRNGKey(9),
        T=2.0,
        save_trajectory=False,
    )
    assert res.reactions is None
    with pytest.raises(ValueError, match='save_trajectory'):
        thermo.entropy_production(net, res)


# --------------------------------------------------------------------------- #
# Stored-propensity reuse
# --------------------------------------------------------------------------- #
def test_stored_propensities_match_recompute():
    net = _reversible_pair(2.0, 1.0)
    x0 = jnp.array([60.0, 40.0])
    res = _simulate(net, x0, jax.random.PRNGKey(10), T=4.0).clean()

    recomputed = thermo.entropy_production(net, res, use_stored_propensities=False)
    reused = thermo.entropy_production(net, res)  # stored is the default
    assert jnp.allclose(recomputed, reused, rtol=1e-5, atol=1e-5)


def test_stored_propensities_match_recompute_uncleaned():
    # Regression: stored propensities are zeroed on padded steps, which must not
    # corrupt the last valid jump's reverse term on an uncleaned trajectory.
    net = _reversible_pair(2.0, 1.0)
    x0 = jnp.array([60.0, 40.0])
    res = _simulate(net, x0, jax.random.PRNGKey(20), T=4.0)  # NOT cleaned -> padded
    assert bool(jnp.any(res.reactions < 0))  # padding present

    recomputed = thermo.entropy_production(net, res, use_stored_propensities=False)
    reused = thermo.entropy_production(net, res)  # stored is the default
    assert jnp.allclose(recomputed, reused, rtol=1e-5, atol=1e-5)


def test_stored_propensities_requires_saved_propensities():
    net = _reversible_pair()
    res = _simulate(
        net,
        jnp.array([40.0, 60.0]),
        jax.random.PRNGKey(11),
        T=2.0,
        save_propensities=False,
    )
    assert res.propensities is None
    with pytest.raises(ValueError, match='save_propensities'):
        thermo.entropy_production(net, res, use_stored_propensities=True)


def test_log_prob_stored_propensities_match_recompute():
    net = _reversible_pair(2.0, 1.0)
    res = _simulate(net, jnp.array([60.0, 40.0]), jax.random.PRNGKey(12), T=3.0).clean()
    recomputed = net.log_prob(res)
    reused = net.log_prob(res, use_stored_propensities=True)
    assert jnp.allclose(recomputed, reused, rtol=1e-5, atol=1e-5)


# --------------------------------------------------------------------------- #
# Total entropy production and TUR bounds
# --------------------------------------------------------------------------- #
def test_total_entropy_production_uses_requested_interval():
    net = _reversible_pair(3.0, 1.0)
    x0 = jnp.array([80.0, 20.0])  # start away from equilibrium
    keys = jax.random.split(jax.random.PRNGKey(13), 64)
    results = eqx.filter_vmap(lambda k: _simulate(net, x0, k, T=6.0, max_steps=2000))(
        keys
    )

    t_initial, t_final = 0.5, 5.5
    kwargs = {'n_grid_points': 40, 'min_max_vals': (0.0, 100.0)}
    total = thermo.total_entropy_production(
        net, results, 'A', t_initial, t_final, **kwargs
    )

    def interval_medium(single_results):
        increments = thermo.entropy_production(net, single_results, per_step=True)
        event_times = single_results.t[1:]
        mask = (
            (single_results.reactions >= 0)
            & (event_times > t_initial)
            & (event_times <= t_final)
        )
        return jnp.sum(jnp.where(mask, increments, 0.0))

    mean_medium = jnp.mean(eqx.filter_vmap(interval_medium)(results))
    delta_system = thermo.system_entropy(
        results, ('A', t_final), **kwargs
    ) - thermo.system_entropy(results, ('A', t_initial), **kwargs)

    assert jnp.isfinite(total)
    assert jnp.allclose(total, mean_medium + delta_system)


def test_tur_bound_is_below_realized_entropy_production():
    net = _driven_cycle()
    x0 = jnp.array([100.0, 100.0, 100.0])
    keys = jax.random.split(jax.random.PRNGKey(14), 128)

    def per_traj(key):
        res = _simulate(net, x0, key, T=10.0)
        # integrated current through the first pair over the window
        current = thermo.reaction_currents(net, res, normalize_by_time=False)[0]
        ep = thermo.entropy_production(net, res)
        return current, ep

    currents, eps = eqx.filter_vmap(per_traj)(keys)
    bound = thermo.tur_bound(jnp.mean(currents), jnp.var(currents))
    q = thermo.thermodynamic_efficiency(
        jnp.mean(currents), jnp.var(currents), jnp.mean(eps)
    )
    assert bound <= jnp.mean(eps) + 1e-6
    assert q <= 1.0 + 1e-6
