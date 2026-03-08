import importlib

import jax.numpy as jnp
import pytest

kde_2d_module = importlib.import_module('stochastix.analysis.kde_2d')


def _probs_reference_old(
    x1_data: jnp.ndarray,
    x2_data: jnp.ndarray,
    grid1_data: jnp.ndarray,
    grid2_data: jnp.ndarray,
    w_data: jnp.ndarray | None,
    grid_step1_data: jnp.ndarray,
    grid_step2_data: jnp.ndarray,
    bw_data: jnp.ndarray,
    alpha_eff_data: jnp.ndarray,
    *,
    density: bool,
    kernel_fn,
) -> jnp.ndarray:
    x1_b = x1_data[:, None]
    grid1_b = grid1_data[None, :]
    x2_b = x2_data[:, None]
    grid2_b = grid2_data[None, :]

    scale1 = grid_step1_data * bw_data
    dist1 = jnp.abs(x1_b - grid1_b) / scale1
    kernel_vals1 = kernel_fn(dist1, grid1_data.dtype)

    scale2 = grid_step2_data * bw_data
    dist2 = jnp.abs(x2_b - grid2_b) / scale2
    kernel_vals2 = kernel_fn(dist2, grid2_data.dtype)

    row_sum1 = jnp.sum(kernel_vals1, axis=1, keepdims=True)
    row_sum1 = jnp.where(
        row_sum1 > 0,
        row_sum1,
        jnp.asarray(1.0, dtype=kernel_vals1.dtype),
    )
    kernel_vals1 = kernel_vals1 / row_sum1

    row_sum2 = jnp.sum(kernel_vals2, axis=1, keepdims=True)
    row_sum2 = jnp.where(
        row_sum2 > 0,
        row_sum2,
        jnp.asarray(1.0, dtype=kernel_vals2.dtype),
    )
    kernel_vals2 = kernel_vals2 / row_sum2

    joint_kernel = kernel_vals1[:, :, None] * kernel_vals2[:, None, :]
    if w_data is None:
        counts = jnp.sum(joint_kernel, axis=0)
    else:
        counts = jnp.sum(joint_kernel * w_data[:, None, None], axis=0)

    if not density:
        return counts

    n_eff = jnp.sum(counts)
    alpha = jnp.asarray(alpha_eff_data, dtype=counts.dtype)
    n_bins = jnp.asarray(counts.size, dtype=counts.dtype)
    denom = n_eff + alpha * n_bins
    denom = jnp.where(denom > 0, denom, jnp.asarray(1.0, dtype=counts.dtype))
    pmf_smoothed = (counts + alpha) / denom
    return pmf_smoothed / (grid_step1_data * grid_step2_data)


@pytest.mark.parametrize('kernel', ['triangular', 'gaussian', 'wendland_c2'])
@pytest.mark.parametrize('density', [False, True])
@pytest.mark.parametrize('weighted', [False, True])
def test_kde_2d_probs_matches_old_broadcast_reference(
    kernel: str, density: bool, weighted: bool
):
    x1 = jnp.array([0.0, 1.0, 2.0, 3.0, 5.0], dtype=jnp.float32)
    x2 = jnp.array([3.0, 2.5, 1.0, 0.5, -1.0], dtype=jnp.float32)
    w = jnp.array([1.0, 0.2, 3.0, 0.5, 2.0], dtype=jnp.float32) if weighted else None
    grid1 = jnp.linspace(-1.0, 6.0, 17, dtype=jnp.float32)
    grid2 = jnp.linspace(-2.0, 4.0, 13, dtype=jnp.float32)
    grid_step1 = jnp.maximum(grid1[1] - grid1[0], jnp.asarray(1e-6, dtype=grid1.dtype))
    grid_step2 = jnp.maximum(grid2[1] - grid2[0], jnp.asarray(1e-6, dtype=grid2.dtype))
    bw = jnp.asarray(1.0, dtype=jnp.float32)
    alpha = jnp.asarray(0.1 / float(grid1.size * grid2.size), dtype=jnp.float32)
    kernel_fn = kde_2d_module._resolve_kernel_function(kernel)

    got = kde_2d_module._probs(
        x1_data=x1,
        x2_data=x2,
        grid1_data=grid1,
        grid2_data=grid2,
        w_data=w,
        grid_step1_data=grid_step1,
        grid_step2_data=grid_step2,
        bw_data=bw,
        alpha_eff_data=alpha,
        density=density,
        kernel_fn=kernel_fn,
    )
    expected = _probs_reference_old(
        x1_data=x1,
        x2_data=x2,
        grid1_data=grid1,
        grid2_data=grid2,
        w_data=w,
        grid_step1_data=grid_step1,
        grid_step2_data=grid_step2,
        bw_data=bw,
        alpha_eff_data=alpha,
        density=density,
        kernel_fn=kernel_fn,
    )

    assert jnp.allclose(got, expected, rtol=1e-5, atol=1e-6)
