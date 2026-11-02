import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import stochastix as stx
from stochastix._state_utils import pytree_to_state
from stochastix.utils.optimization import reinforce_loss


@pytest.mark.parametrize(
    'solver_class', [stx.DifferentiableDirect, stx.DifferentiableFirstReaction]
)
@pytest.mark.parametrize('reaction_grad', ['stgs', 'reinmax'])
@pytest.mark.parametrize('wrap_model', [False, True])
def test_reinforce_matches_fixed_trajectory_score(
    solver_class, reaction_grad, wrap_model
):
    solver = solver_class(reaction_grad=reaction_grad)

    def loss(log_rates):
        network = stx.ReactionNetwork(
            [
                stx.Reaction(
                    '0 -> A', stx.kinetics.MassAction(log_rates[0], transform=jnp.exp)
                ),
                stx.Reaction(
                    'A -> 0', stx.kinetics.MassAction(log_rates[1], transform=jnp.exp)
                ),
            ]
        )
        model = stx.StochasticModel(network, solver, T=2.0) if wrap_model else network
        results = stx.stochsimsolve(
            jax.random.key(99),
            network,
            {'A': 2.0} if wrap_model else jnp.array([2.0]),
            T=2.0,
            max_steps=32,
            solver=solver,
        )
        states = pytree_to_state(results.x, results.species)
        reward = 3.0 + states[-1, 0]
        baseline = log_rates.sum()
        value = reinforce_loss()(model, results, reward, baseline)
        return value, (results, reward - baseline, network.log_prob(results).sum())

    log_rates = jnp.log(jnp.array([2.0, 0.7]))
    (value, (results, advantage, log_p)), gradient = eqx.filter_jit(
        eqx.filter_value_and_grad(loss, has_aux=True)
    )(log_rates)
    assert not results.time_overflow
    states = pytree_to_state(results.x, results.species)
    exposure = jnp.sum(states[:-1, 0] * jnp.diff(results.t))
    counts = jnp.array([(results.reactions == i).sum() for i in range(2)])
    # Exact log-rate score for this fixed birth-death trajectory.
    score = counts - jnp.exp(log_rates) * jnp.array([2.0, exposure])
    np.testing.assert_allclose(value, -advantage * log_p, rtol=1e-6)
    np.testing.assert_allclose(gradient, -advantage * score, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize('stop_returns_grad', [False, True])
def test_reinforce_return_gradient_option_is_preserved(stop_returns_grad):
    results = stx.SimulationResults(
        x=jnp.array([[0.0], [1.0], [1.0]]),
        t=jnp.array([0.0, 0.5, 2.0]),
        reactions=jnp.array([0, -1]),
        species=('A',),
    )

    def loss(log_rate):
        network = stx.ReactionNetwork(
            [
                stx.Reaction(
                    '0 -> A', stx.kinetics.MassAction(log_rate, transform=jnp.exp)
                )
            ]
        )
        return reinforce_loss(stop_returns_grad)(
            network, results, log_rate, baseline=log_rate + 0.5
        )

    log_rate = jnp.log(2.0)
    value, gradient = eqx.filter_jit(eqx.filter_value_and_grad(loss))(log_rate)
    log_p = log_rate - 2.0 * jnp.exp(log_rate)
    expected = 0.5 * (1.0 - 2.0 * jnp.exp(log_rate))
    if not stop_returns_grad:
        expected -= log_p
    np.testing.assert_allclose(value, 0.5 * log_p, rtol=1e-6)
    np.testing.assert_allclose(gradient, expected, rtol=1e-6)
