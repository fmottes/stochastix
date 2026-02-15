import jax.numpy as jnp
import pytest

from stochastix.analysis import mutual_information
from stochastix.analysis.kde_1d import kde_triangular
from stochastix.analysis.kde_2d import kde_triangular_2d


def _empirical_discrete_mi_bits(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    x_int = [int(v) for v in jnp.asarray(x).tolist()]
    y_int = [int(v) for v in jnp.asarray(y).tolist()]

    x_vals = sorted(set(x_int))
    y_vals = sorted(set(y_int))
    x_map = {v: i for i, v in enumerate(x_vals)}
    y_map = {v: i for i, v in enumerate(y_vals)}

    counts = jnp.zeros((len(x_vals), len(y_vals)), dtype=jnp.float32)
    for xv, yv in zip(x_int, y_int):
        counts = counts.at[x_map[xv], y_map[yv]].add(1.0)

    q_xy = counts / jnp.sum(counts)
    q_x = jnp.sum(q_xy, axis=1)
    q_y = jnp.sum(q_xy, axis=0)

    tiny = jnp.finfo(q_xy.dtype).tiny
    log_ratio = jnp.log2(jnp.maximum(q_xy, tiny)) - (
        jnp.log2(jnp.maximum(q_x, tiny))[:, None]
        + jnp.log2(jnp.maximum(q_y, tiny))[None, :]
    )
    return jnp.sum(q_xy * log_ratio)


def _make_integer_series(n: int = 4000) -> tuple[jnp.ndarray, jnp.ndarray]:
    idx = jnp.arange(n)
    x = (idx % 11).astype(jnp.float32)
    noise = ((idx * 7) % 3 == 0).astype(jnp.float32)
    y = jnp.mod(x + noise, 11)
    return x, y


def test_mutual_information_matches_discrete_at_unit_grid():
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
    )
    mi_exact = _empirical_discrete_mi_bits(x, y)

    assert jnp.isclose(mi_est, mi_exact, atol=1e-5)


def test_mutual_information_coarse_grid_uses_cell_mass():
    x, y = _make_integer_series()

    mi_est = mutual_information(
        x,
        y,
        n_grid_points1=6,
        n_grid_points2=6,
        min_max_vals1=(0.0, 10.0),
        min_max_vals2=(0.0, 10.0),
        kde_type='triangular',
        bw_multiplier=1.0,
        dirichlet_alpha=None,
    )

    grid1, p_x = kde_triangular(
        x,
        n_grid_points=6,
        min_max_vals=(0.0, 10.0),
        density=True,
        bw_multiplier=1.0,
        dirichlet_alpha=None,
    )
    grid2, p_y = kde_triangular(
        y,
        n_grid_points=6,
        min_max_vals=(0.0, 10.0),
        density=True,
        bw_multiplier=1.0,
        dirichlet_alpha=None,
    )
    _, _, p_xy = kde_triangular_2d(
        x,
        y,
        n_grid_points1=6,
        n_grid_points2=6,
        min_max_vals1=(0.0, 10.0),
        min_max_vals2=(0.0, 10.0),
        density=True,
        bw_multiplier=1.0,
        dirichlet_alpha=None,
    )

    dx = grid1[1] - grid1[0]
    dy = grid2[1] - grid2[0]

    q_x = p_x * dx
    q_y = p_y * dy
    q_xy = p_xy * (dx * dy)

    tiny = jnp.finfo(q_xy.dtype).tiny
    log_ratio = jnp.log2(jnp.maximum(q_xy, tiny)) - (
        jnp.log2(jnp.maximum(q_x, tiny))[:, None]
        + jnp.log2(jnp.maximum(q_y, tiny))[None, :]
    )
    mi_manual = jnp.sum(q_xy * log_ratio)

    assert jnp.isclose(mi_est, mi_manual, atol=1e-5)


@pytest.mark.parametrize('base', [1.0, 0.0, -2.0, jnp.inf, jnp.nan])
def test_mutual_information_rejects_invalid_base(base):
    x = jnp.array([0.0, 1.0, 2.0, 3.0], dtype=jnp.float32)
    y = jnp.array([0.0, 1.0, 2.0, 3.0], dtype=jnp.float32)

    with pytest.raises(
        ValueError, match='base must be finite, positive, and not equal to 1'
    ):
        mutual_information(x, y, base=base)
