import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from stochastix.analysis import entropy, kde, kde_2d, mutual_information


@pytest.mark.parametrize('measure', ['entropy', 'mi'])
@pytest.mark.parametrize('kernel', ['wendland_c2', 'triangular'])
@pytest.mark.parametrize('alpha', [0.0, None])
def test_unsmoothed_information_gradients_with_empty_bins(measure, kernel, alpha):
    x = jnp.array([0.2, 0.3, 1.2, 2.1])
    y = jnp.array([0.3, 1.4, 0.9, 2.2])
    shared = {'bw_multiplier': 1.5, 'dirichlet_alpha': alpha}
    if measure == 'entropy':
        grid = {'n_grid_points': 7, 'min_max_vals': (0.0, 6.0)}
        loss = lambda x: entropy(x, kde_type=kernel, **grid, **shared)
        _, counts = kde(x, kernel=kernel, density=False, **grid, **shared)
        p = np.asarray(counts / counts.sum())
        assert np.any(p == 0)
        expected = -(p[p > 0] * np.log2(p[p > 0])).sum()
    else:
        grid = {
            'n_grid_points1': 7,
            'n_grid_points2': 7,
            'min_max_vals1': (0.0, 6.0),
            'min_max_vals2': (0.0, 6.0),
        }
        loss = lambda x: mutual_information(x, y, kde_type=kernel, **grid, **shared)
        _, _, counts = kde_2d(x, y, kernel=kernel, density=False, **grid, **shared)
        p = np.asarray(counts / counts.sum())
        independent = p.sum(axis=1)[:, None] * p.sum(axis=0)[None, :]
        support = p > 0
        assert np.any(p.sum(axis=0) == 0)
        assert np.any(p.sum(axis=1) == 0)
        expected = (p[support] * np.log2(p[support] / independent[support])).sum()

    value, gradient = eqx.filter_jit(eqx.filter_value_and_grad(loss))(x)
    np.testing.assert_allclose(value, expected, rtol=2e-6, atol=2e-7)
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.linalg.norm(gradient) > 0

    direction = jnp.array([0.3, -0.7, 0.8, -0.2])
    epsilon = 1e-3
    fd = (loss(x + epsilon * direction) - loss(x - epsilon * direction)) / (2 * epsilon)
    _, tangent = eqx.filter_jvp(loss, (x,), (direction,))
    np.testing.assert_allclose(gradient @ direction, fd, rtol=3e-3, atol=3e-4)
    np.testing.assert_allclose(tangent, gradient @ direction, rtol=2e-5, atol=1e-7)
