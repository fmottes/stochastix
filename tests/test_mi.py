import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from stochastix import SimulationResults
from stochastix.analysis import mutual_information, state_mutual_info
from stochastix.analysis.kde_2d import kde_triangular_2d


def _make_integer_series(n: int = 4000) -> tuple[jnp.ndarray, jnp.ndarray]:
    idx = jnp.arange(n)
    x = (idx % 11).astype(jnp.float32)
    noise = ((idx * 7) % 3 == 0).astype(jnp.float32)
    y = jnp.mod(x + noise, 11)
    return x, y


def test_mutual_information_without_smoothing_is_finite_and_matches_manual():
    x, y = _make_integer_series()

    mi_est = mutual_information(
        x,
        y,
        n_grid_points1=11,
        n_grid_points2=11,
        min_max_vals1=(0.0, 10.0),
        min_max_vals2=(0.0, 10.0),
        kde_type='triangular',
        bw_multiplier=1.0,
        dirichlet_alpha=None,
        dirichlet_kappa=None,
    )
    _, _, counts_xy = kde_triangular_2d(
        x,
        y,
        n_grid_points1=11,
        n_grid_points2=11,
        min_max_vals1=(0.0, 10.0),
        min_max_vals2=(0.0, 10.0),
        density=False,
        bw_multiplier=1.0,
        dirichlet_alpha=0.0,
        dirichlet_kappa=0.0,
    )
    n_eff = jnp.sum(counts_xy)
    q_xy = counts_xy / n_eff
    q_x = jnp.sum(q_xy, axis=1)
    q_y = jnp.sum(q_xy, axis=0)
    support = q_xy > 0
    mi_manual = jnp.sum(
        jnp.where(
            support,
            q_xy * (jnp.log(q_xy) - (jnp.log(q_x)[:, None] + jnp.log(q_y)[None, :])),
            0.0,
        )
    ) / jnp.log(2.0)

    assert jnp.isfinite(mi_est)
    assert jnp.isclose(mi_est, mi_manual, atol=1e-5)


def test_mutual_information_coarse_grid_uses_cell_mass():
    x, y = _make_integer_series()
    alpha = 0.1

    mi_est = mutual_information(
        x,
        y,
        n_grid_points1=6,
        n_grid_points2=6,
        min_max_vals1=(0.0, 10.0),
        min_max_vals2=(0.0, 10.0),
        kde_type='triangular',
        bw_multiplier=1.0,
        dirichlet_alpha=alpha,
        dirichlet_kappa=None,
    )
    _, _, counts_xy = kde_triangular_2d(
        x,
        y,
        n_grid_points1=6,
        n_grid_points2=6,
        min_max_vals1=(0.0, 10.0),
        min_max_vals2=(0.0, 10.0),
        density=False,
        bw_multiplier=1.0,
        dirichlet_alpha=0.0,
        dirichlet_kappa=0.0,
    )

    n_bins = counts_xy.size
    n_eff = jnp.sum(counts_xy)
    denom = n_eff + alpha * n_bins
    q_xy = (counts_xy + alpha) / denom
    q_x = jnp.sum(q_xy, axis=1)
    q_y = jnp.sum(q_xy, axis=0)
    mi_manual = jnp.sum(
        q_xy * (jnp.log(q_xy) - (jnp.log(q_x)[:, None] + jnp.log(q_y)[None, :]))
    ) / jnp.log(2.0)

    assert jnp.isclose(mi_est, mi_manual, atol=1e-5)


def test_mutual_information_default_smoothing_matches_alpha_path():
    x, y = _make_integer_series()

    mi_default = mutual_information(
        x,
        y,
        n_grid_points1=11,
        n_grid_points2=11,
        min_max_vals1=(0.0, 10.0),
        min_max_vals2=(0.0, 10.0),
        kde_type='triangular',
        bw_multiplier=1.0,
    )
    mi_alpha = mutual_information(
        x,
        y,
        n_grid_points1=11,
        n_grid_points2=11,
        min_max_vals1=(0.0, 10.0),
        min_max_vals2=(0.0, 10.0),
        kde_type='triangular',
        bw_multiplier=1.0,
        dirichlet_alpha=0.1,
        dirichlet_kappa=None,
    )
    assert jnp.isclose(mi_default, mi_alpha, atol=1e-8)


@pytest.mark.parametrize('base', [1.0, 0.0, -2.0, jnp.inf, jnp.nan])
def test_mutual_information_rejects_invalid_base(base):
    x = jnp.array([0.0, 1.0, 2.0, 3.0], dtype=jnp.float32)
    y = jnp.array([0.0, 1.0, 2.0, 3.0], dtype=jnp.float32)

    with pytest.raises(
        ValueError, match='base must be finite, positive, and not equal to 1'
    ):
        mutual_information(x, y, base=base)


def _make_batched_results_for_time_tests() -> SimulationResults:
    x = jnp.array(
        [
            [[0.0], [1.0], [2.0], [3.0]],
            [[10.0], [11.0], [12.0], [13.0]],
        ]
    )
    # Keep times batched to match interpolate's current batched convention.
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
def test_state_mutual_info_integral_timestamps_warn_and_match_float_time(t_scalar):
    results = _make_batched_results_for_time_tests()
    kwargs = dict(
        n_grid_points1=4,
        n_grid_points2=4,
        min_max_vals1=(0.0, 13.0),
        min_max_vals2=(0.0, 13.0),
        dirichlet_alpha=0.1,
        dirichlet_kappa=None,
    )
    with pytest.warns(FutureWarning):
        mi_int = state_mutual_info(
            results, [('A', t_scalar), ('A', t_scalar)], **kwargs
        )
    mi_float = state_mutual_info(results, [('A', 2.0), ('A', 2.0)], **kwargs)
    assert jnp.isclose(mi_int, mi_float)


def test_state_mutual_info_uses_time_not_index_for_integral_timestamps():
    results = _make_batched_results_for_time_tests()
    kwargs = dict(
        n_grid_points1=4,
        n_grid_points2=4,
        min_max_vals1=(0.0, 13.0),
        min_max_vals2=(0.0, 13.0),
        dirichlet_alpha=0.1,
        dirichlet_kappa=None,
    )
    mi_index_like = state_mutual_info(results, [('A', 2.0), ('A', 2.0)], **kwargs)
    mi_real_time = state_mutual_info(results, [('A', 10.0), ('A', 10.0)], **kwargs)
    assert not jnp.isclose(mi_index_like, mi_real_time)
    with pytest.warns(FutureWarning):
        mi_integral = state_mutual_info(
            results, [('A', np.int32(2)), ('A', np.int32(2))], **kwargs
        )
    assert jnp.isclose(mi_integral, mi_index_like)


@pytest.mark.parametrize('bad_t', [jnp.array([1.0]), [1.0], (1.0,)])
def test_state_mutual_info_rejects_non_scalar_timestamps(bad_t):
    results = _make_batched_results_for_time_tests()
    with pytest.raises(ValueError, match='scalar numeric time value'):
        state_mutual_info(results, [('A', bad_t), ('A', 2.0)])


def test_state_mutual_info_filter_jit_accepts_jax_scalar_times():
    results = _make_batched_results_for_time_tests()

    @eqx.filter_jit
    def _mi(t1, t2):
        return state_mutual_info(
            results,
            [('A', t1), ('A', t2)],
            n_grid_points1=4,
            n_grid_points2=4,
            min_max_vals1=(0.0, 13.0),
            min_max_vals2=(0.0, 13.0),
            dirichlet_alpha=0.1,
            dirichlet_kappa=None,
        )

    t1 = jnp.asarray(2.0, dtype=jnp.float32)
    t2 = jnp.asarray(10.0, dtype=jnp.float32)
    eager = state_mutual_info(
        results,
        [('A', t1), ('A', t2)],
        n_grid_points1=4,
        n_grid_points2=4,
        min_max_vals1=(0.0, 13.0),
        min_max_vals2=(0.0, 13.0),
        dirichlet_alpha=0.1,
        dirichlet_kappa=None,
    )
    jitted = _mi(t1, t2)
    assert jnp.isclose(eager, jitted)
