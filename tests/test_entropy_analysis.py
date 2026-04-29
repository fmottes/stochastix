import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from stochastix import SimulationResults
from stochastix.analysis import entropy, state_entropy
from stochastix.analysis.kde_1d import kde_triangular


def _make_integer_series(n: int = 4000) -> jnp.ndarray:
    idx = jnp.arange(n)
    x = (idx % 11).astype(jnp.float32)
    noise = ((idx * 7) % 3 == 0).astype(jnp.float32)
    return jnp.mod(x + noise, 11)


def test_entropy_without_smoothing_is_finite_and_matches_manual():
    x = _make_integer_series()

    h_est = entropy(
        x,
        n_grid_points=11,
        min_max_vals=(0.0, 10.0),
        kde_type='triangular',
        bw_multiplier=1.0,
        dirichlet_alpha=0.0,
        dirichlet_kappa=0.0,
    )

    _, counts_x = kde_triangular(
        x,
        n_grid_points=11,
        min_max_vals=(0.0, 10.0),
        density=False,
        bw_multiplier=1.0,
        dirichlet_alpha=0.0,
        dirichlet_kappa=0.0,
    )
    q_x = counts_x / jnp.sum(counts_x)
    support = q_x > 0
    h_manual = -jnp.sum(jnp.where(support, q_x * jnp.log(q_x), 0.0)) / jnp.log(2.0)

    assert jnp.isfinite(h_est)
    assert jnp.isclose(h_est, h_manual, atol=1e-5)


def test_entropy_smoothed_matches_manual_formula():
    x = _make_integer_series()
    alpha = 0.1

    h_est = entropy(
        x,
        n_grid_points=11,
        min_max_vals=(0.0, 10.0),
        kde_type='triangular',
        bw_multiplier=1.0,
        dirichlet_alpha=alpha,
        dirichlet_kappa=None,
    )

    _, counts_x = kde_triangular(
        x,
        n_grid_points=11,
        min_max_vals=(0.0, 10.0),
        density=False,
        bw_multiplier=1.0,
        dirichlet_alpha=0.0,
        dirichlet_kappa=0.0,
    )
    n_bins = counts_x.size
    q_x = (counts_x + alpha) / (jnp.sum(counts_x) + alpha * n_bins)
    h_manual = -jnp.sum(q_x * jnp.log(q_x)) / jnp.log(2.0)

    assert jnp.isclose(h_est, h_manual, atol=1e-5)


def test_entropy_default_smoothing_matches_alpha_first_convention():
    x = _make_integer_series()
    n_grid_points = 11
    alpha = 0.1

    h_default = entropy(
        x,
        n_grid_points=n_grid_points,
        min_max_vals=(0.0, 10.0),
        kde_type='triangular',
        bw_multiplier=1.0,
    )

    _, counts_x = kde_triangular(
        x,
        n_grid_points=n_grid_points,
        min_max_vals=(0.0, 10.0),
        density=False,
        bw_multiplier=1.0,
        dirichlet_alpha=0.0,
        dirichlet_kappa=0.0,
    )
    n_bins = counts_x.size
    q_x = (counts_x + alpha) / (jnp.sum(counts_x) + alpha * n_bins)
    h_manual = -jnp.sum(q_x * jnp.log(q_x)) / jnp.log(2.0)

    assert jnp.isclose(h_default, h_manual, atol=1e-5)


@pytest.mark.parametrize('base', [1.0, 0.0, -2.0, jnp.inf, jnp.nan])
def test_entropy_rejects_invalid_base(base):
    x = jnp.array([0.0, 1.0, 2.0, 3.0], dtype=jnp.float32)
    with pytest.raises(
        ValueError, match='base must be finite, positive, and not equal to 1'
    ):
        entropy(x, base=base)


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


@pytest.mark.parametrize('t_scalar', [2, np.int32(2), jnp.int32(2)])
def test_state_entropy_integral_timestamps_warn_and_match_float_time(t_scalar):
    results = _make_batched_results_for_time_tests()
    kwargs = dict(
        n_grid_points=4,
        min_max_vals=(0.0, 13.0),
        dirichlet_alpha=0.1,
        dirichlet_kappa=None,
    )
    with pytest.warns(FutureWarning):
        h_int = state_entropy(results, ('A', t_scalar), **kwargs)
    h_float = state_entropy(results, ('A', 2.0), **kwargs)
    assert jnp.isclose(h_int, h_float)


def test_state_entropy_filter_jit_accepts_jax_scalar_time():
    results = _make_batched_results_for_time_tests()

    @eqx.filter_jit
    def _h(t):
        return state_entropy(
            results,
            ('A', t),
            n_grid_points=4,
            min_max_vals=(0.0, 13.0),
            dirichlet_alpha=0.1,
            dirichlet_kappa=None,
        )

    t = jnp.asarray(2.0, dtype=jnp.float32)
    eager = state_entropy(
        results,
        ('A', t),
        n_grid_points=4,
        min_max_vals=(0.0, 13.0),
        dirichlet_alpha=0.1,
        dirichlet_kappa=None,
    )
    jitted = _h(t)
    assert jnp.isclose(eager, jitted)
