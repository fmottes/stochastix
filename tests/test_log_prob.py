from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import stochastix as stx
from stochastix.utils.optimization import reinforce_loss


def birth_network(rate):
    return stx.ReactionNetwork(
        [stx.Reaction('0 -> A', stx.kinetics.MassAction(k=rate))]
    )


@pytest.mark.parametrize('n_events', [0, 2])
@pytest.mark.parametrize('as_pytree', [False, True])
def test_poisson_likelihood_and_score_include_readout(n_events, as_pytree):
    rate = jnp.array(1.5)
    network = birth_network(rate)
    # Nonzero start time and repeated terminal times exercise padding as well.
    times = [2.0, 5.0, 5.0] if n_events == 0 else [2.0, 2.5, 3.0, 5.0, 5.0]
    states = jnp.minimum(jnp.arange(len(times)), n_events).astype(float)
    results = stx.SimulationResults(
        x={'A': states} if as_pytree else states[:, None],
        t=jnp.array(times),
        reactions=jnp.array([-1, -1] if n_events == 0 else [0, 0, -1, -1]),
        species=network.species,
    )

    @eqx.filter_jit
    @eqx.filter_value_and_grad
    def likelihood(model):
        return model.log_prob(results).sum()

    expected = n_events * jnp.log(rate) - rate * 3.0
    expected_score = n_events / rate - 3.0
    value, grads = likelihood(network)
    assert value == pytest.approx(float(expected))
    assert grads.reactions[0].kinetics.k == pytest.approx(float(expected_score))
    assert network.log_prob(results)[-1] == 0.0
    assert network.log_prob(results.clean()).sum() == pytest.approx(float(expected))


def test_terminal_survival_uses_post_event_state_and_all_channels():
    network = stx.ReactionNetwork(
        [
            stx.Reaction('0 -> A', stx.kinetics.MassAction(k=0.5)),
            stx.Reaction('A -> 0', stx.kinetics.MassAction(k=2.0)),
        ]
    )
    results = stx.SimulationResults(
        x=jnp.array([[2.0], [1.0], [1.0], [1.0]]),
        t=jnp.array([0.0, 0.25, 1.0, 1.0]),
        reactions=jnp.array([1, -1, -1]),
    )
    expected = jnp.array([jnp.log(4.0) - 4.5 * 0.25, -2.5 * 0.75, 0.0])
    assert jnp.allclose(network.log_prob(results), expected)


@pytest.mark.parametrize(
    'solver',
    [stx.DirectMethod(), stx.FirstReactionMethod(), stx.DifferentiableDirect()],
)
@pytest.mark.parametrize('save_propensities', [False, True])
def test_simulated_poisson_likelihood_jit_vmap_and_clean(solver, save_propensities):
    rate, t0, end = 2.0, 1.0, 3.0
    network = birth_network(jnp.array(rate))
    simulate = partial(
        stx.stochsimsolve,
        t0=t0,
        T=end,
        max_steps=32,
        solver=solver,
        save_propensities=save_propensities,
    )
    results = eqx.filter_vmap(simulate, in_axes=(0, None, None))(
        jax.random.split(jax.random.key(24), 4), network, jnp.array([0.0])
    )
    terms = eqx.filter_jit(eqx.filter_vmap(network.log_prob))(results)
    counts = (results.reactions >= 0).sum(axis=1)
    expected = counts * jnp.log(rate) - rate * (end - t0)
    assert not jnp.any(results.time_overflow)
    assert jnp.allclose(results.t[:, -1], end)
    assert jnp.allclose(terms.sum(axis=1), expected)
    for i in range(4):
        assert network.log_prob(results[i].clean()).sum() == pytest.approx(
            float(expected[i]), rel=1e-5
        )


@pytest.mark.parametrize('wrap_model', [False, True])
def test_reinforce_readout_reward_weights_terminal_score(wrap_model):
    network = birth_network(jnp.array(2.0))
    model = (
        stx.StochasticModel(network, stx.DirectMethod(), T=3.0)
        if wrap_model
        else network
    )
    results = stx.SimulationResults(
        x=jnp.array([[0.0], [1.0], [1.0], [1.0]]),
        t=jnp.array([0.0, 0.5, 3.0, 3.0]),
        reactions=jnp.array([0, -1, -1]),
    )
    # The score for this path is N/k - T, and the loss is -advantage * log P.
    value, grads = eqx.filter_jit(eqx.filter_value_and_grad(reinforce_loss()))(
        model, results, jnp.array(4.0), baseline=1.0
    )
    network_grads = grads.network if wrap_model else grads
    assert value == pytest.approx(float(-3.0 * (jnp.log(2.0) - 6.0)))
    assert network_grads.reactions[0].kinetics.k == pytest.approx(-3.0 * (0.5 - 3.0))


def test_overflow_likelihood_stops_at_last_recorded_event():
    network = birth_network(jnp.array(2.0))
    results = stx.stochsimsolve(
        jax.random.key(3),
        network,
        jnp.array([0.0]),
        T=100.0,
        max_steps=1,
        solver=stx.DirectMethod(),
    )
    assert results.time_overflow
    expected = jnp.log(2.0) - 2.0 * results.t[-1]
    assert network.log_prob(results).sum() == pytest.approx(float(expected))


def test_absorbing_state_has_zero_terminal_survival_cost_and_finite_score():
    network = stx.ReactionNetwork(
        [stx.Reaction('A -> 0', stx.kinetics.MassAction(k=jnp.array(2.0)))]
    )
    results = stx.SimulationResults(
        x=jnp.array([[1.0], [0.0], [0.0]]),
        t=jnp.array([0.0, 0.5, 3.0]),
        reactions=jnp.array([0, -1]),
    )
    value, grads = eqx.filter_value_and_grad(lambda m: m.log_prob(results).sum())(
        network
    )
    assert value == pytest.approx(float(jnp.log(2.0) - 1.0))
    assert grads.reactions[0].kinetics.k == pytest.approx(0.0)


@pytest.mark.parametrize('reactions', [None, jnp.zeros((1, 1), dtype=int)])
def test_log_prob_requires_exact_full_trajectory(reactions):
    results = stx.SimulationResults(
        x=jnp.array([[0.0], [0.0]]),
        t=jnp.array([0.0, 1.0]),
        reactions=reactions,
    )
    with pytest.raises(ValueError, match='full trajectory with exact reaction indices'):
        birth_network(1.0).log_prob(results)
