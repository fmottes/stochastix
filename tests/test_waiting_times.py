import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from stochastix import stochsimsolve
from stochastix.kinetics import MassAction
from stochastix.reaction import Reaction, ReactionNetwork
from stochastix.solvers import (
    DGA,
    DifferentiableDirect,
    DifferentiableFirstReaction,
    DirectMethod,
    FirstReactionMethod,
)

try:
    from jax import enable_x64
except ImportError:
    from jax.experimental import enable_x64

_SOLVERS = [
    DirectMethod(),
    DGA(),
    DifferentiableDirect(),
    DifferentiableDirect(reaction_grad='reinmax'),
    FirstReactionMethod(),
    DifferentiableFirstReaction(),
    DifferentiableFirstReaction(reaction_grad='reinmax'),
]


def _is_first_reaction(solver):
    return isinstance(solver, (FirstReactionMethod, DifferentiableFirstReaction))


def _step(solver, propensities, key):
    network = ReactionNetwork(
        [Reaction('0 -> A', MassAction(k=1.0)) for _ in range(propensities.size)]
    )
    return solver.step(
        network, jnp.array(0.0), jnp.array([0.0]), propensities, None, key=key
    )[0]


@pytest.mark.parametrize('solver', _SOLVERS)
def test_zero_uniform_waiting_time_and_masked_gradient_are_finite(solver):
    # These Threefry keys produce a genuine zero, without mocking the sampler.
    with enable_x64(False):
        key = jax.random.key(
            780233 if _is_first_reaction(solver) else 3353848,
            impl='threefry2x32',
        )
        time_key = key if _is_first_reaction(solver) else jax.random.split(key)[0]
        assert jax.random.uniform(time_key) == 0

        def waiting_time(rate):
            return _step(solver, rate[None], key).dt

        rate = jnp.array(2.0)
        dt, gradient = eqx.filter_jit(eqx.filter_value_and_grad(waiting_time))(rate)
        expected = -jnp.log(jnp.finfo(jnp.float32).tiny) / rate
        np.testing.assert_allclose(dt, expected)
        np.testing.assert_allclose(gradient, -expected / rate)

        def rejected_event(rate):
            dt = waiting_time(rate)
            return jnp.where(dt > 1.0, 1.0, dt)

        value, gradient = eqx.filter_jit(eqx.filter_value_and_grad(rejected_event))(
            rate
        )
        assert value == 1.0
        assert gradient == 0.0


@pytest.mark.parametrize('solver', _SOLVERS)
@pytest.mark.parametrize('use_x64', [False, True])
def test_nonzero_waiting_time_draws_are_unchanged(solver, use_x64):
    with enable_x64(use_x64):
        keys = jax.random.split(jax.random.key(42), 256)
        rate = jnp.array(2.0)

        def compare(key):
            if _is_first_reaction(solver):
                u = jax.random.uniform(key, shape=(1,))[0]
            else:
                u = jax.random.uniform(jax.random.split(key)[0])
            return u, -jnp.log(u) / rate, _step(solver, rate[None], key).dt

        draws, before, after = eqx.filter_jit(eqx.filter_vmap(compare))(keys)
        assert jnp.all(draws > 0)
        np.testing.assert_array_equal(after, before)


@pytest.mark.parametrize('solver', [DirectMethod(), DGA(), DifferentiableDirect()])
def test_zero_total_propensity_has_zero_waiting_time_gradient(solver):
    def waiting_time(rate):
        return _step(solver, rate[None], jax.random.key(0)).dt

    dt, gradient = eqx.filter_jit(eqx.filter_value_and_grad(waiting_time))(
        jnp.array(0.0)
    )
    assert jnp.isposinf(dt)
    assert gradient == 0.0


@pytest.mark.parametrize(
    'solver',
    [
        FirstReactionMethod(),
        DifferentiableFirstReaction(),
    ],
)
def test_disabled_candidate_has_zero_waiting_time_gradient(solver):
    def waiting_time(a):
        return _step(solver, a, jax.random.key(0)).dt

    dt, gradient = eqx.filter_jit(eqx.filter_value_and_grad(waiting_time))(
        jnp.array([0.0, 2.0])
    )
    assert jnp.isfinite(dt)
    assert jnp.all(jnp.isfinite(gradient))
    assert gradient[0] == 0.0


@pytest.mark.parametrize('solver', _SOLVERS)
def test_zero_uniform_overshoot_has_finite_adaptive_time_kernel_gradient(solver):
    with enable_x64(False):
        key = jax.random.key(
            12582090 if _is_first_reaction(solver) else 1407809,
            impl='threefry2x32',
        )
        key_loop = jax.random.split(key)[1]
        key_step = jax.random.split(key_loop)[0]
        key_solver = jax.random.split(key_step)[0]
        time_key = (
            key_solver
            if _is_first_reaction(solver)
            else jax.random.split(key_solver)[0]
        )
        assert jax.random.uniform(time_key) == 0

        def loss(rate, bandwidth):
            network = ReactionNetwork([Reaction('0 -> A', MassAction(k=rate))])
            results = stochsimsolve(
                key, network, jnp.array([0.0]), T=1.0, max_steps=4, solver=solver
            )
            return results.interpolate(
                jnp.array([0.25, 0.75]), time_kernel_bw=bandwidth
            ).x.sum()

        rate = jnp.array(2.0)
        value, gradient = eqx.filter_jit(eqx.filter_value_and_grad(loss))(
            rate, 'adaptive'
        )
        assert value == loss(rate, 0) == 0.0
        assert gradient == 0.0
