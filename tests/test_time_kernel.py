import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import stochastix as stx


def _trajectory(propensities=True):
    return stx.SimulationResults(
        x=jnp.array([[2.0], [5.0], [4.0], [4.0], [4.0]]),
        t=jnp.array([0.0, 1.0, 2.0, 3.0, 3.0]),
        propensities=jnp.array([[2.0, 2.0], [0.5, 1.5], [0.0, 0.0], [0.0, 0.0]])
        if propensities
        else None,
        reactions=jnp.array([0, 1, -1, -1]),
        species=('A',),
    )


@pytest.mark.parametrize('bandwidth', [0, 0.3, 'adaptive', jnp.array(0.0)])
def test_kernel_forward_values_are_exactly_forward_fill(bandwidth):
    results = _trajectory()
    times = jnp.array([-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0])
    expected = results.interpolate(times)
    actual = eqx.filter_jit(lambda r: r.interpolate(times, time_kernel_bw=bandwidth))(
        results
    )
    np.testing.assert_array_equal(actual.x, expected.x)
    np.testing.assert_array_equal(actual.t, times)
    assert actual.propensities is None
    assert actual.reactions is None
    assert actual.species == expected.species


@pytest.mark.parametrize('bandwidth', [0.3, 'adaptive'])
def test_kernel_event_time_derivative_and_forward_mode(bandwidth):
    results = _trajectory()
    observations = jnp.array([0.8, 1.8])

    def loss(t):
        r = eqx.tree_at(lambda r: r.t, results, t)
        return r.interpolate(observations, time_kernel_bw=bandwidth).x.sum()

    actual = eqx.filter_jit(eqx.filter_grad(loss))(results.t)
    # These two event widths use different pre-event rates (4 and 2).
    widths = jnp.array([0.25, 0.5]) if bandwidth == 'adaptive' else bandwidth
    s = jax.nn.sigmoid((observations[:, None] - results.t[None, 1:3]) / widths)
    expected = -(s * (1 - s) / widths).sum(axis=0) * jnp.array([3.0, -1.0])
    np.testing.assert_allclose(actual[1:3], expected, rtol=1e-6)
    np.testing.assert_array_equal(actual[jnp.array([0, 3, 4])], 0.0)

    direction = jnp.array([0.0, -1.0, -2.0, 0.0, 0.0])
    _, tangent = eqx.filter_jvp(loss, (results.t,), (direction,))
    np.testing.assert_allclose(tangent, actual @ direction, rtol=1e-6)


@pytest.mark.parametrize('bandwidth', [0.3, 'adaptive'])
def test_kernel_preserves_existing_state_gradients(bandwidth):
    results = _trajectory()
    times = jnp.array([0.5, 1.5, 2.5])

    def loss(x, bw):
        r = eqx.tree_at(lambda r: r.x, results, x)
        return jnp.square(r.interpolate(times, time_kernel_bw=bw).x).sum()

    expected = eqx.filter_grad(loss)(results.x, 0)
    actual = eqx.filter_grad(loss)(results.x, bandwidth)
    np.testing.assert_array_equal(actual, expected)


def test_kernel_bandwidths_are_detached():
    results = _trajectory()
    times = jnp.array([0.8, 1.8])

    def adaptive_loss(a):
        r = eqx.tree_at(lambda r: r.propensities, results, a)
        return r.interpolate(times, time_kernel_bw='adaptive').x.sum()

    np.testing.assert_array_equal(
        eqx.filter_grad(adaptive_loss)(results.propensities), 0.0
    )
    fixed_loss = lambda h: results.interpolate(times, time_kernel_bw=h).x.sum()
    assert eqx.filter_grad(fixed_loss)(jnp.array(0.3)) == 0.0


def test_adaptive_kernel_handles_large_finite_propensities():
    results = stx.SimulationResults(
        x=jnp.array([[0.0], [1.0]]),
        t=jnp.array([0.0, 1.0]),
        propensities=jnp.array([[1e38]]),
        reactions=jnp.array([0]),
    )

    def loss(t):
        r = eqx.tree_at(lambda r: r.t, results, t)
        return r.interpolate(jnp.array([1.0]), time_kernel_bw='adaptive').x.sum()

    # At the event time the sigmoid derivative is 1/4. Forming h=1/a0
    # explicitly can underflow to zero and discard this valid contribution.
    grad = eqx.filter_jit(eqx.filter_grad(loss))(results.t)
    np.testing.assert_allclose(grad[1], -2.5e37, rtol=1e-6)


def test_kernel_masks_absorbing_terminal_and_nonfinite_padding():
    results = stx.SimulationResults(
        x=jnp.array([[0.0], [1.0], [1.0], [1.0], [1.0]]),
        t=jnp.array([0.0, 1.0, 2.0, 2.0, jnp.inf]),
        propensities=jnp.array([[2.0], [0.0], [jnp.nan], [jnp.inf]]),
        reactions=jnp.array([0, -1, -1, -1]),
    )

    def loss(r):
        return r.interpolate(jnp.array([0.8]), time_kernel_bw='adaptive').x.sum()

    value, grads = eqx.filter_jit(eqx.filter_value_and_grad(loss))(results)
    assert value == 0.0
    assert jnp.all(jnp.isfinite(grads.t))
    assert grads.t[1] < 0
    np.testing.assert_array_equal(grads.t[jnp.array([0, 2, 3, 4])], 0.0)
    np.testing.assert_array_equal(grads.propensities, 0.0)


class _State(eqx.Module):
    A: jax.Array
    metadata: str = eqx.field(static=True, default='preserve me')


@pytest.mark.parametrize('state_type', ['array', 'dict', 'module'])
def test_kernel_supports_batched_pytrees(state_type):
    results = _trajectory()
    if state_type == 'dict':
        results = eqx.tree_at(lambda r: r.x, results, {'A': results.x[:, 0]})
    elif state_type == 'module':
        results = eqx.tree_at(lambda r: r.x, results, _State(results.x[:, 0]))
    batched = eqx.filter_vmap(lambda _: results)(jnp.arange(2))
    times = jnp.array([0.8, 1.8])
    expected = batched.interpolate(times)
    actual = eqx.filter_jit(lambda r: r.interpolate(times, time_kernel_bw='adaptive'))(
        batched
    )
    for a, b in zip(jax.tree.leaves(actual.x), jax.tree.leaves(expected.x)):
        np.testing.assert_array_equal(a, b)
    if state_type == 'module':
        assert actual.x.metadata == 'preserve me'


def test_adaptive_kernel_requires_per_event_propensities():
    results = _trajectory(propensities=False)
    with pytest.raises(ValueError, match='requires saved per-step propensities'):
        results.interpolate(jnp.array([1.5]), time_kernel_bw='adaptive')
    # Fixed kernels need no saved propensities.
    np.testing.assert_array_equal(
        results.interpolate(jnp.array([1.5]), time_kernel_bw=0.3).x, [[5.0]]
    )


@pytest.mark.parametrize('bandwidth', [0.3, 'adaptive'])
@pytest.mark.parametrize('save_propensities', [False, True])
def test_kernel_rejects_initial_final_only_results(bandwidth, save_propensities):
    final_only = stx.SimulationResults(
        x=jnp.array([[0.0], [1.0]]),
        t=jnp.array([0.0, 2.0]),
        propensities=jnp.ones((2, 1)) if save_propensities else None,
    )
    with pytest.raises(ValueError, match='full saved trajectory'):
        final_only.interpolate(jnp.array([1.0]), time_kernel_bw=bandwidth)
    np.testing.assert_array_equal(final_only.interpolate(jnp.array([1.0])).x, [[0.0]])


def test_adaptive_kernel_rejects_misaligned_propensities():
    results = _trajectory()
    results = eqx.tree_at(lambda r: r.propensities, results, jnp.ones((5, 2)))
    with pytest.raises(ValueError, match='one propensity vector per event interval'):
        results.interpolate(jnp.array([1.0]), time_kernel_bw='adaptive')


@pytest.mark.parametrize('bandwidth', [-0.1, float('inf'), float('nan'), 'unknown'])
def test_invalid_kernel_bandwidth(bandwidth):
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match='time_kernel_bw'):
        _trajectory().interpolate(jnp.array([1.5]), time_kernel_bw=bandwidth)


def test_kernel_reopens_rate_scaling_direction_through_stochsimsolve():
    times = jnp.array([1.0, 2.0])

    def simulate(log_rate):
        network = stx.ReactionNetwork(
            [
                stx.Reaction(
                    '0 -> A', stx.kinetics.MassAction(log_rate, transform=jnp.exp)
                )
            ]
        )
        return stx.stochsimsolve(
            jax.random.key(0), network, jnp.array([0.0]), T=6.0, max_steps=32
        )

    def loss(log_rate, bandwidth):
        results = simulate(log_rate)
        return results.interpolate(times, time_kernel_bw=bandwidth).x.sum()

    log_rate = jnp.array(0.0)
    assert eqx.filter_grad(loss)(log_rate, 0) == 0.0
    value, grad = eqx.filter_jit(eqx.filter_value_and_grad(loss))(log_rate, 'adaptive')
    assert grad > 0
    assert value == loss(log_rate, 0)
    _, tangent = eqx.filter_jvp(
        lambda p: loss(p, 'adaptive'), (log_rate,), (jnp.array(1.0),)
    )
    np.testing.assert_allclose(tangent, grad, rtol=1e-5)

    results = simulate(log_rate)
    np.testing.assert_array_equal(
        results.interpolate(times, time_kernel_bw='adaptive').x,
        results.clean().interpolate(times, time_kernel_bw='adaptive').x,
    )
