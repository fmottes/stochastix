import importlib

import jax.numpy as jnp


def test_kde_reuses_compiled_helper_for_repeated_calls():
    kde_1d_module = importlib.import_module('stochastix.analysis.kde_1d')
    kde_1d_module._probs._cached._clear_cache()

    x = jnp.array([0.0, 1.0, 2.0, 3.0], dtype=jnp.float32)
    _, values_first = kde_1d_module.kde(
        x,
        n_grid_points=4,
        min_max_vals=(0.0, 3.0),
        kernel='gaussian',
    )
    values_first.block_until_ready()
    cache_size_after_first = kde_1d_module._probs._cached._cache_size()

    _, values_second = kde_1d_module.kde(
        x,
        n_grid_points=4,
        min_max_vals=(0.0, 3.0),
        kernel='gaussian',
    )
    values_second.block_until_ready()

    assert cache_size_after_first == 1
    assert kde_1d_module._probs._cached._cache_size() == cache_size_after_first


def test_kde_2d_reuses_compiled_helper_for_repeated_calls():
    kde_2d_module = importlib.import_module('stochastix.analysis.kde_2d')
    kde_2d_module._probs._cached._clear_cache()

    x1 = jnp.array([0.0, 1.0, 2.0, 3.0], dtype=jnp.float32)
    x2 = jnp.array([3.0, 2.0, 1.0, 0.0], dtype=jnp.float32)
    _, _, values_first = kde_2d_module.kde_2d(
        x1,
        x2,
        n_grid_points1=4,
        n_grid_points2=4,
        min_max_vals1=(0.0, 3.0),
        min_max_vals2=(0.0, 3.0),
        kernel='gaussian',
    )
    values_first.block_until_ready()
    cache_size_after_first = kde_2d_module._probs._cached._cache_size()

    _, _, values_second = kde_2d_module.kde_2d(
        x1,
        x2,
        n_grid_points1=4,
        n_grid_points2=4,
        min_max_vals1=(0.0, 3.0),
        min_max_vals2=(0.0, 3.0),
        kernel='gaussian',
    )
    values_second.block_until_ready()

    assert cache_size_after_first == 1
    assert kde_2d_module._probs._cached._cache_size() == cache_size_after_first
