import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import stochastix as stx
from stochastix import SimulationResults
from stochastix.analysis.hist import state_kde


def _make_batched_results_for_time_tests() -> SimulationResults:
    x = jnp.array(
        [
            [[0.0], [1.0], [2.0], [3.0]],
            [[10.0], [11.0], [12.0], [13.0]],
        ]
    )
    t = jnp.array([[0.0, 5.0, 10.0, 20.0], [0.0, 5.0, 10.0, 20.0]])
    return SimulationResults(
        x=x,
        t=t,
        propensities=None,
        reactions=None,
        time_overflow=None,
        species=('A',),
    )


@pytest.mark.parametrize('bandwidth', [0, jnp.array(0.0)])
def test_state_kde_t_none_uses_final_recorded_state(bandwidth):
    results = _make_batched_results_for_time_tests()
    grid_none, values_none = state_kde(
        results,
        'A',
        n_grid_points=8,
        min_max_vals=(0.0, 13.0),
        t=None,
        time_kernel_bw=bandwidth,
    )
    grid_final_time, values_final_time = state_kde(
        results,
        'A',
        n_grid_points=8,
        min_max_vals=(0.0, 13.0),
        t=20.0,
        time_kernel_bw=0,
    )
    assert jnp.allclose(grid_none, grid_final_time)
    assert jnp.allclose(values_none, values_final_time)


@pytest.mark.parametrize('bandwidth', [None, 0.3, 'adaptive'])
@pytest.mark.parametrize('jit', [False, True])
def test_state_kde_time_kernel_requires_explicit_time(bandwidth, jit):
    results = _make_batched_results_for_time_tests()

    def evaluate(results):
        kwargs = {} if bandwidth is None else {'time_kernel_bw': bandwidth}
        return state_kde(
            results,
            'A',
            n_grid_points=8,
            min_max_vals=(0.0, 13.0),
            **kwargs,
        )

    if jit:
        evaluate = eqx.filter_jit(evaluate)
    with pytest.raises(ValueError, match='explicit observation time t is required'):
        evaluate(results)


def test_state_kde_dynamic_zero_bandwidth_needs_no_history():
    results = _make_batched_results_for_time_tests()

    @eqx.filter_jit
    def evaluate(bandwidth):
        return state_kde(
            results,
            'A',
            n_grid_points=8,
            min_max_vals=(0.0, 13.0),
            time_kernel_bw=bandwidth,
        )[1]

    expected = state_kde(
        results, 'A', n_grid_points=8, min_max_vals=(0.0, 13.0), time_kernel_bw=0
    )[1]
    np.testing.assert_array_equal(evaluate(jnp.array(0.0)), expected)
    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError),
        match='explicit observation time t is required',
    ):
        evaluate(jnp.array(0.3)).block_until_ready()


def test_state_kde_fixed_time_preserves_absorbing_process_time_gradient():
    keys = jax.random.split(jax.random.key(91), 32)

    def loss(log_rate, bandwidth):
        network = stx.ReactionNetwork(
            [
                stx.Reaction(
                    'A -> 0', stx.kinetics.MassAction(log_rate, transform=jnp.exp)
                )
            ]
        )

        def simulate(key):
            return stx.stochsimsolve(
                key,
                network,
                jnp.array([1.0]),
                T=2.0,
                max_steps=8,
                solver=stx.DifferentiableDirect(reaction_grad='reinmax'),
            )

        results = eqx.filter_vmap(simulate)(keys)
        _, values = state_kde(
            results,
            'A',
            n_grid_points=2,
            min_max_vals=(0.0, 1.0),
            t=2.0,
            time_kernel_bw=bandwidth,
            kde_type='gaussian',
            bw_multiplier=1.5,
        )
        return values[1, 0], (results.t[:, -1] < 2.0).sum()

    evaluate = eqx.filter_jit(eqx.filter_value_and_grad(loss, has_aux=True))
    log_rate = jnp.log(0.7)
    (value, absorbed), gradient = evaluate(log_rate, 'adaptive')
    (default_value, _), default_gradient = evaluate(log_rate, 0)
    assert 0 < absorbed < len(keys)
    np.testing.assert_array_equal(value, default_value)
    assert default_gradient == 0
    assert jnp.isfinite(gradient) and gradient < 0


@pytest.mark.parametrize('t_scalar', [2, np.int32(2), jnp.int32(2)])
def test_state_kde_integral_timestamps_warn_and_match_float_time(t_scalar):
    results = _make_batched_results_for_time_tests()
    with pytest.warns(FutureWarning):
        grid_int, values_int = state_kde(
            results,
            'A',
            n_grid_points=8,
            min_max_vals=(0.0, 13.0),
            t=t_scalar,
            time_kernel_bw=0,
        )
    grid_float, values_float = state_kde(
        results, 'A', n_grid_points=8, min_max_vals=(0.0, 13.0), t=2.0, time_kernel_bw=0
    )
    assert jnp.allclose(grid_int, grid_float)
    assert jnp.allclose(values_int, values_float)


@pytest.mark.parametrize('bad_t', [jnp.array([1.0]), [1.0], (1.0,)])
def test_state_kde_rejects_non_scalar_timestamps(bad_t):
    results = _make_batched_results_for_time_tests()
    with pytest.raises(ValueError, match='scalar numeric time value'):
        state_kde(results, 'A', t=bad_t)


def test_state_kde_filter_jit_accepts_jax_scalar_time():
    results = _make_batched_results_for_time_tests()

    @eqx.filter_jit
    def _kde(t):
        return state_kde(
            results,
            'A',
            n_grid_points=8,
            min_max_vals=(0.0, 13.0),
            t=t,
            time_kernel_bw=0,
        )

    t = jnp.asarray(10.0, dtype=jnp.float32)
    grid_eager, values_eager = state_kde(
        results, 'A', n_grid_points=8, min_max_vals=(0.0, 13.0), t=t, time_kernel_bw=0
    )
    grid_jit, values_jit = _kde(t)
    assert jnp.allclose(grid_eager, grid_jit)
    assert jnp.allclose(values_eager, values_jit)
