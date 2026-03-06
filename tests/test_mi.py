import jax.numpy as jnp
import pytest

from stochastix.analysis import mutual_information
from stochastix.analysis.kde_2d import kde_triangular_2d


def _make_integer_series(n: int = 4000) -> tuple[jnp.ndarray, jnp.ndarray]:
    idx = jnp.arange(n)
    x = (idx % 11).astype(jnp.float32)
    noise = ((idx * 7) % 3 == 0).astype(jnp.float32)
    y = jnp.mod(x + noise, 11)
    return x, y


def test_mutual_information_without_smoothing_returns_nan():
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
    assert jnp.isnan(mi_est)


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


@pytest.mark.parametrize('base', [1.0, 0.0, -2.0, jnp.inf, jnp.nan])
def test_mutual_information_rejects_invalid_base(base):
    x = jnp.array([0.0, 1.0, 2.0, 3.0], dtype=jnp.float32)
    y = jnp.array([0.0, 1.0, 2.0, 3.0], dtype=jnp.float32)

    with pytest.raises(
        ValueError, match='base must be finite, positive, and not equal to 1'
    ):
        mutual_information(x, y, base=base)
