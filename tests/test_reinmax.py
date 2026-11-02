import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from stochastix import stochsimsolve
from stochastix.kinetics import MassAction
from stochastix.reaction import Reaction, ReactionNetwork
from stochastix.solvers import (
    DifferentiableDirect,
    DifferentiableFirstReaction,
    FirstReactionMethod,
)
from stochastix.solvers._differentiable import _reinmax


@pytest.fixture
def network():
    return ReactionNetwork(
        [Reaction(f'0 -> {s}', MassAction(k=1.0)) for s in ('A', 'B', 'C')]
    )


def _step(network, solver, a, key):
    return solver.step(
        network, jnp.array(0.0), jnp.zeros(network.n_species), a, None, key=key
    )[0]


@pytest.mark.parametrize('temperature', [1.0, 2.0, 5.0])
@pytest.mark.parametrize(
    'solver_class', [DifferentiableDirect, DifferentiableFirstReaction]
)
def test_reinmax_step_gradient_matches_formula(network, temperature, solver_class):
    solver = solver_class(reaction_grad='reinmax', logits_scale=temperature)
    logits = jnp.log(jnp.array([0.2, 1.3, 2.5]))
    key = jax.random.PRNGKey(42)
    costs = jnp.array([-0.5, 1.3, 2.0])
    step = _step(network, solver, jnp.exp(logits), key)
    sample = jax.nn.one_hot(step.reaction_idx, 3)
    p = jax.nn.softmax(logits)
    q = (sample + jax.nn.softmax(logits / temperature)) / 2
    covariance = lambda p: jnp.diag(p) - jnp.outer(p, p)
    expected = (2 * covariance(q) - covariance(p) / 2) @ costs

    def loss(logits):
        return _step(network, solver, jnp.exp(logits), key).x_new @ costs

    actual = eqx.filter_jit(eqx.filter_grad(loss))(logits)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-7)
    direction = jnp.array([0.3, -0.8, 1.1])
    _, tangent = eqx.filter_jvp(loss, (logits,), (direction,))
    np.testing.assert_allclose(tangent, expected @ direction, rtol=2e-6, atol=1e-7)
    # Choice probabilities are invariant to uniform log-rate shifts.
    _, scale_tangent = eqx.filter_jvp(loss, (logits,), (jnp.ones_like(logits),))
    np.testing.assert_allclose(scale_tangent, 0, atol=5e-7)

    def waiting_time(logits):
        return _step(network, solver, jnp.exp(logits), key).dt

    np.testing.assert_allclose(
        eqx.filter_grad(waiting_time)(logits), -step.dt * p, rtol=2e-6
    )


def test_reinmax_expected_gradient_is_exact_for_quadratic_loss():
    logits = jnp.array([-0.7, 0.3, 1.2])
    costs = jnp.array([-2.0, 0.5, 3.0])
    target = 0.8

    def sampled_loss(logits, sample):
        return (_reinmax(logits, sample, 1.0) @ costs - target) ** 2

    gradients = eqx.filter_vmap(eqx.filter_grad(sampled_loss), in_axes=(None, 0))(
        logits, jnp.eye(3)
    )
    actual = jax.nn.softmax(logits) @ gradients
    exact_loss = lambda logits: jax.nn.softmax(logits) @ (costs - target) ** 2
    np.testing.assert_allclose(
        actual, eqx.filter_grad(exact_loss)(logits), rtol=2e-6, atol=1e-7
    )


@pytest.mark.parametrize('temperature', [1.0, 2.0])
@pytest.mark.parametrize('a', [[0.0, 0.3, 2.0], [0.0, 1e-30, 0.0]])
@pytest.mark.parametrize(
    'solver_class', [DifferentiableDirect, DifferentiableFirstReaction]
)
def test_reinmax_disabled_reactions_have_finite_zero_gradients(
    network, temperature, a, solver_class
):
    solver = solver_class(reaction_grad='reinmax', logits_scale=temperature)
    a = jnp.array(a)
    keys = jax.random.split(jax.random.PRNGKey(2026), 64)

    def state_and_gradient(key):
        def loss(a):
            return _step(network, solver, a, key).x_new @ jnp.array([1.0, 2.0, 4.0])

        step = _step(network, solver, a, key)
        return step.reaction_idx, eqx.filter_grad(loss)(a)

    selected, gradients = eqx.filter_jit(eqx.filter_vmap(state_and_gradient))(keys)
    assert jnp.all(a[selected] > 0)
    assert jnp.all(jnp.isfinite(gradients))
    np.testing.assert_array_equal(gradients[:, a == 0], 0.0)


@pytest.mark.parametrize('temperature', [1.0, 2.0])
@pytest.mark.parametrize(
    'solver_class', [DifferentiableDirect, DifferentiableFirstReaction]
)
def test_reinmax_trajectory_matches_stgs_and_supports_time_kernel(
    temperature, solver_class
):
    def simulate(log_rates, solver):
        network = ReactionNetwork(
            [
                Reaction('0 -> A', MassAction(k=log_rates[0], transform=jnp.exp)),
                Reaction('A -> 0', MassAction(k=log_rates[1], transform=jnp.exp)),
            ]
        )
        return stochsimsolve(
            jax.random.PRNGKey(123),
            network,
            jnp.array([0.0]),
            T=4.0,
            max_steps=64,
            solver=solver,
        )

    log_rates = jnp.log(jnp.array([3.0, 0.8]))
    solver = solver_class(reaction_grad='reinmax', logits_scale=temperature)
    expected = simulate(log_rates, solver_class(logits_scale=temperature))
    actual = simulate(log_rates, solver)
    for field in ('x', 't', 'reactions', 'propensities', 'time_overflow'):
        np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))
    assert not actual.time_overflow
    np.testing.assert_array_equal(actual.x, jnp.round(actual.x))

    def loss(log_rates):
        observed = simulate(log_rates, solver).interpolate(
            jnp.array([0.5, 1.0, 2.0]), time_kernel_bw='adaptive'
        )
        return jnp.square(observed.x - 4).mean()

    # Exercise the surrogate gradient inside scan, jit and ensemble vmap.
    rates_batch = jnp.stack([log_rates, log_rates + 0.1])
    losses, gradients = eqx.filter_jit(
        eqx.filter_vmap(eqx.filter_value_and_grad(loss))
    )(rates_batch)
    assert jnp.all(jnp.isfinite(losses))
    assert jnp.all(jnp.isfinite(gradients))
    assert jnp.all(jnp.linalg.norm(gradients, axis=-1) > 0)
    _, tangent = eqx.filter_jvp(loss, (log_rates,), (jnp.ones_like(log_rates),))
    np.testing.assert_allclose(tangent, gradients[0].sum(), rtol=2e-5, atol=1e-6)


@pytest.mark.parametrize(
    'kwargs, message',
    [
        ({'reaction_grad': 'unknown'}, 'reaction_grad'),
        ({'reaction_grad': 'reinmax', 'exact_fwd': False}, 'exact_fwd=True'),
        *[
            ({'reaction_grad': 'reinmax', 'logits_scale': t}, 'logits_scale >= 1')
            for t in (-1.0, 0.0, 0.5, float('inf'), float('nan'))
        ],
    ],
)
@pytest.mark.parametrize(
    'solver_class', [DifferentiableDirect, DifferentiableFirstReaction]
)
def test_invalid_reinmax_options(kwargs, message, solver_class):
    with pytest.raises(ValueError, match=message):
        solver_class(**kwargs)


@pytest.mark.parametrize(
    'solver_class', [DifferentiableDirect, DifferentiableFirstReaction]
)
def test_stgs_remains_default_and_supports_relaxed_forward(solver_class):
    assert solver_class().reaction_grad == 'stgs'
    assert not solver_class(logits_scale=0.3, exact_fwd=False).is_exact_solver


def test_frm_reinmax_preserves_exact_exponential_race(network):
    a = jnp.array([0.0, 0.3, 2.0])
    keys = jax.random.split(jax.random.PRNGKey(2026), 64)
    simulate = eqx.filter_jit(eqx.filter_vmap(_step, in_axes=(None, None, None, 0)))
    expected = simulate(network, FirstReactionMethod(), a, keys)
    actual = simulate(
        network, DifferentiableFirstReaction(reaction_grad='reinmax'), a, keys
    )
    for field in ('x_new', 'dt', 'reaction_idx', 'propensities'):
        np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))


def test_frm_reinmax_joint_reaction_and_time_gradient(network):
    """The combined channels must reproduce an exactly solvable joint moment."""
    solver = DifferentiableFirstReaction(reaction_grad='reinmax')
    logits = jnp.log(jnp.array([0.2, 1.3, 2.5]))
    costs = jnp.array([-0.5, 1.3, 2.0])

    def loss(logits, key):
        step = _step(network, solver, jnp.exp(logits), key)
        return (step.x_new @ costs) * step.dt

    keys = jax.random.split(jax.random.PRNGKey(17), 16384)
    gradients = eqx.filter_jit(
        eqx.filter_vmap(eqx.filter_grad(loss), in_axes=(None, 0))
    )(logits, keys)
    # E[cost(reaction) * dt] = sum(a * costs) / sum(a)**2.
    exact_loss = lambda logits: (
        jnp.sum(jnp.exp(logits) * costs) / jnp.sum(jnp.exp(logits)) ** 2
    )
    expected = eqx.filter_grad(exact_loss)(logits)
    mean = np.asarray(gradients).mean(axis=0)
    se = np.asarray(gradients).std(axis=0, ddof=1) / np.sqrt(len(keys))
    assert np.all(np.abs(mean - np.asarray(expected)) < 5 * se + 1e-6)


def test_frm_reinmax_absorbing_states_in_batched_time_kernel_loss():
    solver = DifferentiableFirstReaction(reaction_grad='reinmax')
    keys = jax.random.split(jax.random.PRNGKey(19), 3)
    initial_states = jnp.array([[0.0], [1.0], [2.0]])

    def loss(log_rate):
        network = ReactionNetwork(
            [Reaction('A -> 0', MassAction(k=log_rate, transform=jnp.exp))]
        )

        def simulate(key, x0):
            return stochsimsolve(key, network, x0, T=4.0, max_steps=16, solver=solver)

        results = eqx.filter_vmap(simulate)(keys, initial_states)
        return results.interpolate(
            jnp.array([0.5, 1.0]), time_kernel_bw='adaptive'
        ).x.mean()

    value, gradient = eqx.filter_jit(eqx.filter_value_and_grad(loss))(jnp.log(0.7))
    assert jnp.isfinite(value)
    assert jnp.isfinite(gradient)
