"""Kernel density estimation functions for 2D data."""

from __future__ import annotations

import typing

import equinox as eqx
import jax.numpy as jnp

KDE2DKernel = typing.Literal['triangular', 'exponential', 'gaussian', 'wendland_c2']
_KERNEL_NAMES = ('triangular', 'exponential', 'gaussian', 'wendland_c2')


def _triangular_kernel(dist: jnp.ndarray, dtype: typing.Any) -> jnp.ndarray:
    return jnp.maximum(jnp.asarray(0.0, dtype), jnp.asarray(1.0, dtype) - dist)


def _exponential_kernel(dist: jnp.ndarray, _dtype: typing.Any) -> jnp.ndarray:
    return jnp.exp(-dist)


def _gaussian_kernel(dist: jnp.ndarray, _dtype: typing.Any) -> jnp.ndarray:
    return jnp.exp(-0.5 * (dist * dist))


def _wendland_c2_kernel(dist: jnp.ndarray, dtype: typing.Any) -> jnp.ndarray:
    r = jnp.clip(dist, jnp.asarray(0.0, dtype), jnp.asarray(1.0, dtype))
    return jnp.where(
        dist < jnp.asarray(1.0, dtype),
        (jnp.asarray(1.0, dtype) - r) ** 4
        * (jnp.asarray(4.0, dtype) * r + jnp.asarray(1.0, dtype)),
        jnp.asarray(0.0, dtype),
    )


def _resolve_dirichlet_alpha(
    n_grid_points1: int,
    n_grid_points2: int,
    dirichlet_alpha: float | None,
    dirichlet_kappa: float | None,
) -> float:
    n_total_bins = n_grid_points1 * n_grid_points2
    if dirichlet_kappa is not None:
        alpha_eff = float(dirichlet_kappa) / float(n_total_bins)
    elif dirichlet_alpha is not None:
        alpha_eff = float(dirichlet_alpha)
    else:
        alpha_eff = 0.0
    return max(alpha_eff, 0.0)


def _resolve_kernel_function(
    kernel: KDE2DKernel,
) -> typing.Callable[[jnp.ndarray, typing.Any], jnp.ndarray]:
    kernel_functions: dict[
        str, typing.Callable[[jnp.ndarray, typing.Any], jnp.ndarray]
    ] = {
        'triangular': _triangular_kernel,
        'exponential': _exponential_kernel,
        'gaussian': _gaussian_kernel,
        'wendland_c2': _wendland_c2_kernel,
    }
    return kernel_functions[kernel]


def kde_2d(
    x1: jnp.ndarray,
    x2: jnp.ndarray,
    n_grid_points1: int | None = None,
    n_grid_points2: int | None = None,
    min_max_vals1: tuple[float, float] | None = None,
    min_max_vals2: tuple[float, float] | None = None,
    density: bool = True,
    weights: jnp.ndarray | None = None,
    bw_multiplier: float = 1.0,
    *,
    kernel: KDE2DKernel = 'wendland_c2',
    dirichlet_alpha: float | None = 0.1,
    dirichlet_kappa: float | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute a 2D JAX-compatible KDE with selectable kernel.

    This computes a JAX-compatible 2D KDE using a product of two 1D kernels
    centered at each sample, one per dimension. Supported kernels are
    ``'triangular'``, ``'exponential'``, ``'gaussian'``, and ``'wendland_c2'``.
    Each sample's kernel is renormalized over the evaluation grid to avoid
    boundary mass loss on finite support.

    Note: Dirichlet smoothing
        When ``density=True``, applies add-α smoothing to the multinomial pmf
        implied by the soft counts before converting to a pdf:
        ``p_hat = (counts + α) / (N + α*K)``.
        When ``density=False``, returns raw soft counts (no smoothing).

    Note: JIT-compatibility
        For JIT-compatibility, provide concrete binning parameters. If
        `n_grid_points1`, `n_grid_points2`, `min_max_vals1`, or `min_max_vals2`
        are ``None``, bin parameters are derived from data outside of JIT.

    Args:
        x1: 1D array of samples for the first dimension. If not 1D, it is
            flattened.
        x2: 1D array of samples for the second dimension. If not 1D, it is
            flattened. Must have the same length as ``x1``.
        n_grid_points1: Number of grid points for the first dimension. If
            ``None``, inferred from the integer span
            ``[floor(min(x1)), ceil(max(x1))]``.
        n_grid_points2: Number of grid points for the second dimension. If
            ``None``, inferred from the integer span
            ``[floor(min(x2)), ceil(max(x2))]``.
        min_max_vals1: Tuple ``(min_val, max_val)`` defining the range for the
            first dimension. If ``None``, determined from data.
        min_max_vals2: Tuple ``(min_val, max_val)`` defining the range for the
            second dimension. If ``None``, determined from data.
        density: If ``True``, returns a probability density function whose
            Riemann sum over the grid integrates to 1 (via normalization by
            ``sum * grid_step1 * grid_step2``). If ``False``, returns
            unnormalized counts/weights per grid point.
        weights: Optional nonnegative weights per sample (same length as ``x1``
            and ``x2``). When provided, kernel contributions are multiplied by
            these weights.
        bw_multiplier: Kernel scale multiplier relative to the grid step in
            each dimension.
        kernel: Kernel family. Must be one of ``'triangular'``,
            ``'exponential'``, ``'gaussian'``, or ``'wendland_c2'``.
            Default is ``'wendland_c2'``.
        dirichlet_alpha: Per-bin pseudo-count for Dirichlet smoothing. Default
            is ``0.1``. Note: ``dirichlet_kappa`` takes priority over this
            parameter if provided.
        dirichlet_kappa: Total pseudo-count for Dirichlet smoothing. If
            provided, takes priority over ``dirichlet_alpha`` and
            ``alpha = kappa / K`` where K is the total number of grid points
            (K1 * K2). If ``None``, uses ``dirichlet_alpha`` instead.

    Returns:
        A tuple ``(grid1, grid2, values)`` where:

            - ``grid1``: 1D array of evaluation points (bin centers) for the
              first dimension, shape ``(n_grid_points1,)``.
            - ``grid2``: 1D array of evaluation points (bin centers) for the
              second dimension, shape ``(n_grid_points2,)``.
            - ``values``: 2D array of KDE values at the grid points, shape
              ``(n_grid_points1, n_grid_points2)``. If ``density=True``, these
              approximate a PDF.

    Raises:
        ValueError: If ``x1`` and ``x2`` do not have the same length.
        ValueError: If ``weights`` length does not match ``x1``/``x2`` length.
        ValueError: If ``kernel`` is not a supported kernel name.
    """
    x1 = jnp.asarray(x1).reshape(-1)
    x2 = jnp.asarray(x2).reshape(-1)
    if x1.shape[0] != x2.shape[0]:
        raise ValueError('x1 and x2 must have the same length')

    w = None if weights is None else jnp.asarray(weights).reshape(-1)
    if w is not None and w.shape[0] != x1.shape[0]:
        raise ValueError('weights must have the same length as x1 and x2')
    if kernel not in _KERNEL_NAMES:
        raise ValueError(f'kernel must be one of {_KERNEL_NAMES}, got {kernel!r}')

    if min_max_vals1 is None:
        min_val1 = jnp.floor(jnp.min(x1))
        max_val1 = jnp.ceil(jnp.max(x1))
    else:
        min_val1, max_val1 = min_max_vals1

    if n_grid_points1 is None:
        n_grid_points1 = int(max_val1 - min_val1) + 1 if max_val1 >= min_val1 else 1

    grid1 = jnp.linspace(min_val1, max_val1, int(n_grid_points1))

    if min_max_vals2 is None:
        min_val2 = jnp.floor(jnp.min(x2))
        max_val2 = jnp.ceil(jnp.max(x2))
    else:
        min_val2, max_val2 = min_max_vals2

    if n_grid_points2 is None:
        n_grid_points2 = int(max_val2 - min_val2) + 1 if max_val2 >= min_val2 else 1

    grid2 = jnp.linspace(min_val2, max_val2, int(n_grid_points2))

    if int(n_grid_points1) > 1:
        grid_step1 = grid1[1] - grid1[0]
    else:
        grid_step1 = jnp.asarray(1.0, dtype=grid1.dtype)
    grid_step1 = jnp.maximum(grid_step1, jnp.asarray(1e-6, dtype=grid1.dtype))

    if int(n_grid_points2) > 1:
        grid_step2 = grid2[1] - grid2[0]
    else:
        grid_step2 = jnp.asarray(1.0, dtype=grid2.dtype)
    grid_step2 = jnp.maximum(grid_step2, jnp.asarray(1e-6, dtype=grid2.dtype))

    bw = jnp.maximum(
        jnp.asarray(bw_multiplier, dtype=grid1.dtype),
        jnp.asarray(1e-6, dtype=grid1.dtype),
    )
    alpha_eff = _resolve_dirichlet_alpha(
        n_grid_points1=int(n_grid_points1),
        n_grid_points2=int(n_grid_points2),
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_kappa=dirichlet_kappa,
    )
    kernel_fn = _resolve_kernel_function(kernel)

    @eqx.filter_jit
    def _probs(
        x1_data: jnp.ndarray,
        x2_data: jnp.ndarray,
        grid1_data: jnp.ndarray,
        grid2_data: jnp.ndarray,
        w_data: jnp.ndarray | None,
        grid_step1_data: jnp.ndarray,
        grid_step2_data: jnp.ndarray,
        bw_data: jnp.ndarray,
        alpha_eff_data: jnp.ndarray,
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

        kernel_vals1_b = jnp.expand_dims(kernel_vals1, 2)
        kernel_vals2_b = jnp.expand_dims(kernel_vals2, 1)
        joint_kernel = kernel_vals1_b * kernel_vals2_b

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

    values = _probs(
        x1_data=x1,
        x2_data=x2,
        grid1_data=grid1,
        grid2_data=grid2,
        w_data=w,
        grid_step1_data=grid_step1,
        grid_step2_data=grid_step2,
        bw_data=bw,
        alpha_eff_data=jnp.asarray(alpha_eff, dtype=grid1.dtype),
    )
    return grid1, grid2, values


def kde_triangular_2d(
    x1: jnp.ndarray,
    x2: jnp.ndarray,
    n_grid_points1: int | None = None,
    n_grid_points2: int | None = None,
    min_max_vals1: tuple[float, float] | None = None,
    min_max_vals2: tuple[float, float] | None = None,
    density: bool = True,
    weights: jnp.ndarray | None = None,
    bw_multiplier: float = 1.0,
    *,
    dirichlet_alpha: float | None = 0.1,
    dirichlet_kappa: float | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Backward-compatible triangular specialization of ``kde_2d``."""
    return kde_2d(
        x1=x1,
        x2=x2,
        n_grid_points1=n_grid_points1,
        n_grid_points2=n_grid_points2,
        min_max_vals1=min_max_vals1,
        min_max_vals2=min_max_vals2,
        density=density,
        weights=weights,
        bw_multiplier=bw_multiplier,
        kernel='triangular',
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_kappa=dirichlet_kappa,
    )


def kde_exponential_2d(
    x1: jnp.ndarray,
    x2: jnp.ndarray,
    n_grid_points1: int | None = None,
    n_grid_points2: int | None = None,
    min_max_vals1: tuple[float, float] | None = None,
    min_max_vals2: tuple[float, float] | None = None,
    density: bool = True,
    weights: jnp.ndarray | None = None,
    bw_multiplier: float = 1.0,
    *,
    dirichlet_alpha: float | None = 0.1,
    dirichlet_kappa: float | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Backward-compatible exponential specialization of ``kde_2d``."""
    return kde_2d(
        x1=x1,
        x2=x2,
        n_grid_points1=n_grid_points1,
        n_grid_points2=n_grid_points2,
        min_max_vals1=min_max_vals1,
        min_max_vals2=min_max_vals2,
        density=density,
        weights=weights,
        bw_multiplier=bw_multiplier,
        kernel='exponential',
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_kappa=dirichlet_kappa,
    )


def kde_gaussian_2d(
    x1: jnp.ndarray,
    x2: jnp.ndarray,
    n_grid_points1: int | None = None,
    n_grid_points2: int | None = None,
    min_max_vals1: tuple[float, float] | None = None,
    min_max_vals2: tuple[float, float] | None = None,
    density: bool = True,
    weights: jnp.ndarray | None = None,
    bw_multiplier: float = 1.0,
    *,
    dirichlet_alpha: float | None = 0.1,
    dirichlet_kappa: float | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Backward-compatible Gaussian specialization of ``kde_2d``."""
    return kde_2d(
        x1=x1,
        x2=x2,
        n_grid_points1=n_grid_points1,
        n_grid_points2=n_grid_points2,
        min_max_vals1=min_max_vals1,
        min_max_vals2=min_max_vals2,
        density=density,
        weights=weights,
        bw_multiplier=bw_multiplier,
        kernel='gaussian',
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_kappa=dirichlet_kappa,
    )


def kde_wendland_c2_2d(
    x1: jnp.ndarray,
    x2: jnp.ndarray,
    n_grid_points1: int | None = None,
    n_grid_points2: int | None = None,
    min_max_vals1: tuple[float, float] | None = None,
    min_max_vals2: tuple[float, float] | None = None,
    density: bool = True,
    weights: jnp.ndarray | None = None,
    bw_multiplier: float = 1.0,
    *,
    dirichlet_alpha: float | None = 0.1,
    dirichlet_kappa: float | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Backward-compatible Wendland C2 specialization of ``kde_2d``."""
    return kde_2d(
        x1=x1,
        x2=x2,
        n_grid_points1=n_grid_points1,
        n_grid_points2=n_grid_points2,
        min_max_vals1=min_max_vals1,
        min_max_vals2=min_max_vals2,
        density=density,
        weights=weights,
        bw_multiplier=bw_multiplier,
        kernel='wendland_c2',
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_kappa=dirichlet_kappa,
    )
