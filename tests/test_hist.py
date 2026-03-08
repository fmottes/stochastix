import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

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


def test_state_kde_t_none_uses_final_recorded_state():
    results = _make_batched_results_for_time_tests()
    grid_none, values_none = state_kde(
        results, 'A', n_grid_points=8, min_max_vals=(0.0, 13.0), t=None
    )
    grid_final_time, values_final_time = state_kde(
        results, 'A', n_grid_points=8, min_max_vals=(0.0, 13.0), t=20.0
    )
    assert jnp.allclose(grid_none, grid_final_time)
    assert jnp.allclose(values_none, values_final_time)


@pytest.mark.parametrize('t_scalar', [2, np.int32(2), jnp.int32(2)])
def test_state_kde_integral_timestamps_warn_and_match_float_time(t_scalar):
    results = _make_batched_results_for_time_tests()
    with pytest.warns(FutureWarning):
        grid_int, values_int = state_kde(
            results, 'A', n_grid_points=8, min_max_vals=(0.0, 13.0), t=t_scalar
        )
    grid_float, values_float = state_kde(
        results, 'A', n_grid_points=8, min_max_vals=(0.0, 13.0), t=2.0
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
        return state_kde(results, 'A', n_grid_points=8, min_max_vals=(0.0, 13.0), t=t)

    t = jnp.asarray(10.0, dtype=jnp.float32)
    grid_eager, values_eager = state_kde(
        results, 'A', n_grid_points=8, min_max_vals=(0.0, 13.0), t=t
    )
    grid_jit, values_jit = _kde(t)
    assert jnp.allclose(grid_eager, grid_jit)
    assert jnp.allclose(values_eager, values_jit)
