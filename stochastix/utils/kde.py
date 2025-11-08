"""Kernel density estimation functions."""

from __future__ import annotations

import typing

import equinox as eqx
import jax.numpy as jnp

if typing.TYPE_CHECKING:
    pass


def kde_triangular(
    x: jnp.ndarray,
    n_grid_points: int | None = None,
    min_max_vals: tuple[float, float] | None = None,
    density: bool = True,
    weights: jnp.ndarray | None = None,
    bw_multiplier: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Kernel density estimation with a triangular kernel.

    This computes a JAX-compatible KDE using a triangular kernel centered at
    each sample. The kernel support is ±(`bw_multiplier` * grid step). Each
    sample's kernel is renormalized over the evaluation grid to avoid boundary mass
    loss on finite support.

    Note:
        For JIT-compatibility, provide concrete binning parameters. If
        `n_grid_points` or `min_max_vals` are ``None``, bin parameters are derived
        from data outside of JIT.

    Args:
        x: 1D array of samples. If not 1D, it will be flattened.
        n_grid_points: Number of grid points. If ``None``, inferred from the
            integer span ``[floor(min(x)), ceil(max(x))]``.
        min_max_vals: Tuple ``(min_val, max_val)`` defining the bin range. If
            ``None``, determined from data.
        density: If ``True``, returns a probability density function whose
            Riemann sum over the grid integrates to 1 (via normalization by
            ``sum * grid_step``). If ``False``, returns unnormalized
            counts/weights per grid point.
        weights: Optional nonnegative weights per sample (same length as `x`).
            When provided, kernel contributions are multiplied by these weights.
        bw_multiplier: Kernel half-width as a multiple of the bin width.

    Returns:
        A tuple ``(grid, values)`` where:
            - ``grid``: 1D array of evaluation points (bin centers), shape
              ``(n_grid_points,)``.
            - ``values``: 1D array of KDE values at the grid points, shape
              ``(n_grid_points,)``. If ``density=True``, these approximate a PDF.
    """
    x = jnp.asarray(x).reshape(-1)
    w = None if weights is None else jnp.asarray(weights).reshape(-1)
    if w is not None and w.shape[0] != x.shape[0]:
        raise ValueError('weights must have the same length as x')

    # Grid parameters
    if min_max_vals is None:
        min_val = jnp.floor(jnp.min(x))
        max_val = jnp.ceil(jnp.max(x))
    else:
        min_val, max_val = min_max_vals

    if n_grid_points is None:
        n_grid_points = int(max_val - min_val) + 1 if max_val >= min_val else 1

    grid = jnp.linspace(min_val, max_val, int(n_grid_points))

    # Compute grid step outside jit using static n_grid_points
    if int(n_grid_points) > 1:
        grid_step = grid[1] - grid[0]
    else:
        grid_step = jnp.asarray(1.0, dtype=grid.dtype)
    grid_step = jnp.maximum(grid_step, jnp.asarray(1e-6, dtype=grid.dtype))
    bw = jnp.maximum(
        jnp.asarray(bw_multiplier, dtype=grid.dtype),
        jnp.asarray(1e-6, dtype=grid.dtype),
    )

    @eqx.filter_jit
    def _probs(x, grid, w, grid_step, bw):
        x_b = x[:, None]  # (N, 1)
        grid_b = grid[None, :]  # (1, G)

        scale = grid_step * bw
        dist = jnp.abs(x_b - grid_b) / scale
        kernel_vals = jnp.maximum(
            jnp.asarray(0.0, grid.dtype), jnp.asarray(1.0, grid.dtype) - dist
        )

        # Per-sample renormalization (prevents boundary mass loss)
        row_sum = jnp.sum(kernel_vals, axis=1, keepdims=True)
        row_sum = jnp.where(
            row_sum > 0,
            row_sum,
            jnp.asarray(1.0, dtype=kernel_vals.dtype),
        )
        kernel_vals = kernel_vals / row_sum

        if w is None:
            counts = jnp.sum(kernel_vals, axis=0)
        else:
            counts = jnp.sum(kernel_vals * w[:, None], axis=0)

        if density:
            denom = jnp.sum(counts)
            denom = jnp.where(denom > 0, denom, jnp.asarray(1.0, dtype=counts.dtype))
            counts = counts / (denom * grid_step)

        return counts

    probabilities = _probs(x, grid, w, grid_step, bw)
    return grid, probabilities


def kde_exponential(
    x: jnp.ndarray,
    n_grid_points: int | None = None,
    min_max_vals: tuple[float, float] | None = None,
    density: bool = True,
    weights: jnp.ndarray | None = None,
    bw_multiplier: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Kernel density estimation with an exponential (Laplace) kernel.

    This computes a JAX-compatible KDE using an exponential kernel centered at
    each sample. The kernel is ``k(d) = exp(-|d| / scale)`` where
    ``scale = bw_multiplier * grid_step``. Each sample's kernel is renormalized
    over the evaluation grid to avoid boundary mass loss on finite support.

    Note:
        For JIT-compatibility, provide concrete binning parameters. If
        `n_grid_points` or `min_max_vals` are ``None``, bin parameters are derived
        from data outside of JIT.

    Args:
        x: 1D array of samples. If not 1D, it will be flattened.
        n_grid_points: Number of grid points. If ``None``, inferred from the
            integer span ``[floor(min(x)), ceil(max(x))]``.
        min_max_vals: Tuple ``(min_val, max_val)`` defining the bin range. If
            ``None``, determined from data.
        density: If ``True``, returns a probability density function whose
            Riemann sum over the grid integrates to 1 (via normalization by
            ``sum * grid_step``). If ``False``, returns unnormalized
            counts/weights per grid point.
        weights: Optional nonnegative weights per sample (same length as `x`).
            When provided, kernel contributions are multiplied by these weights.
        bw_multiplier: Positive decay scale as a multiple of the bin width.

    Returns:
        A tuple ``(grid, values)`` where:
            - ``grid``: 1D array of evaluation points (bin centers), shape
              ``(n_grid_points,)``.
            - ``values``: 1D array of KDE values at the grid points, shape
              ``(n_grid_points,)``. If ``density=True``, these approximate a PDF.
    """
    x = jnp.asarray(x).reshape(-1)
    w = None if weights is None else jnp.asarray(weights).reshape(-1)
    if w is not None and w.shape[0] != x.shape[0]:
        raise ValueError('weights must have the same length as x')

    # Grid parameters
    if min_max_vals is None:
        min_val = jnp.floor(jnp.min(x))
        max_val = jnp.ceil(jnp.max(x))
    else:
        min_val, max_val = min_max_vals

    if n_grid_points is None:
        n_grid_points = int(max_val - min_val) + 1 if max_val >= min_val else 1

    grid = jnp.linspace(min_val, max_val, int(n_grid_points))

    # Compute grid step outside jit using static n_grid_points
    if int(n_grid_points) > 1:
        grid_step = grid[1] - grid[0]
    else:
        grid_step = jnp.asarray(1.0, dtype=grid.dtype)
    grid_step = jnp.maximum(grid_step, jnp.asarray(1e-6, dtype=grid.dtype))
    bw = jnp.maximum(
        jnp.asarray(bw_multiplier, dtype=grid.dtype),
        jnp.asarray(1e-6, dtype=grid.dtype),
    )

    @eqx.filter_jit
    def _probs(x, grid, w, grid_step, bw):
        x_b = x[:, None]  # (N, 1)
        grid_b = grid[None, :]  # (1, G)

        scale = grid_step * bw
        dist = jnp.abs(x_b - grid_b) / scale
        kernel_vals = jnp.exp(-dist)

        # Per-sample renormalization (prevents boundary mass loss)
        row_sum = jnp.sum(kernel_vals, axis=1, keepdims=True)
        row_sum = jnp.where(
            row_sum > 0,
            row_sum,
            jnp.asarray(1.0, dtype=kernel_vals.dtype),
        )
        kernel_vals = kernel_vals / row_sum

        if w is None:
            counts = jnp.sum(kernel_vals, axis=0)
        else:
            counts = jnp.sum(kernel_vals * w[:, None], axis=0)

        if density:
            denom = jnp.sum(counts)
            denom = jnp.where(denom > 0, denom, jnp.asarray(1.0, dtype=counts.dtype))
            counts = counts / (denom * grid_step)

        return counts

    probabilities = _probs(x, grid, w, grid_step, bw)
    return grid, probabilities


def kde_gaussian(
    x: jnp.ndarray,
    n_grid_points: int | None = None,
    min_max_vals: tuple[float, float] | None = None,
    density: bool = True,
    weights: jnp.ndarray | None = None,
    bw_multiplier: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Kernel density estimation with a Gaussian kernel.

    This computes a JAX-compatible KDE using a Gaussian kernel centered at
    each sample. The kernel is ``k(d) = exp(-0.5 * (d / scale)^2)`` where
    ``scale = bw_multiplier * grid_step``. Each sample's kernel is renormalized
    over the evaluation grid to avoid boundary mass loss on finite support.

    Note:
        For JIT-compatibility, provide concrete binning parameters. If
        `n_grid_points` or `min_max_vals` are ``None``, bin parameters are derived
        from data outside of JIT.

    Args:
        x: 1D array of samples. If not 1D, it will be flattened.
        n_grid_points: Number of grid points. If ``None``, inferred from the
            integer span ``[floor(min(x)), ceil(max(x))]``.
        min_max_vals: Tuple ``(min_val, max_val)`` defining the bin range. If
            ``None``, determined from data.
        density: If ``True``, returns a probability density function whose
            Riemann sum over the grid integrates to 1 (via normalization by
            ``sum * grid_step``). If ``False``, returns unnormalized
            counts/weights per grid point.
        weights: Optional nonnegative weights per sample (same length as `x`).
            When provided, kernel contributions are multiplied by these weights.
        bw_multiplier: Positive decay scale as a multiple of the bin width.

    Returns:
        A tuple ``(grid, values)`` where:
            - ``grid``: 1D array of evaluation points (bin centers), shape
              ``(n_grid_points,)``.
            - ``values``: 1D array of KDE values at the grid points, shape
              ``(n_grid_points,)``. If ``density=True``, these approximate a PDF.
    """
    x = jnp.asarray(x).reshape(-1)
    w = None if weights is None else jnp.asarray(weights).reshape(-1)
    if w is not None and w.shape[0] != x.shape[0]:
        raise ValueError('weights must have the same length as x')

    # Grid parameters
    if min_max_vals is None:
        min_val = jnp.floor(jnp.min(x))
        max_val = jnp.ceil(jnp.max(x))
    else:
        min_val, max_val = min_max_vals

    if n_grid_points is None:
        n_grid_points = int(max_val - min_val) + 1 if max_val >= min_val else 1

    grid = jnp.linspace(min_val, max_val, int(n_grid_points))

    # Compute grid step outside jit using static n_grid_points
    if int(n_grid_points) > 1:
        grid_step = grid[1] - grid[0]
    else:
        grid_step = jnp.asarray(1.0, dtype=grid.dtype)
    grid_step = jnp.maximum(grid_step, jnp.asarray(1e-6, dtype=grid.dtype))
    bw = jnp.maximum(
        jnp.asarray(bw_multiplier, dtype=grid.dtype),
        jnp.asarray(1e-6, dtype=grid.dtype),
    )

    @eqx.filter_jit
    def _probs(x, grid, w, grid_step, bw):
        x_b = x[:, None]  # (N, 1)
        grid_b = grid[None, :]  # (1, G)

        scale = grid_step * bw
        z = (x_b - grid_b) / scale
        kernel_vals = jnp.exp(-0.5 * (z * z))

        # Per-sample renormalization (prevents boundary mass loss)
        row_sum = jnp.sum(kernel_vals, axis=1, keepdims=True)
        row_sum = jnp.where(
            row_sum > 0,
            row_sum,
            jnp.asarray(1.0, dtype=kernel_vals.dtype),
        )
        kernel_vals = kernel_vals / row_sum

        if w is None:
            counts = jnp.sum(kernel_vals, axis=0)
        else:
            counts = jnp.sum(kernel_vals * w[:, None], axis=0)

        if density:
            denom = jnp.sum(counts)
            denom = jnp.where(denom > 0, denom, jnp.asarray(1.0, dtype=counts.dtype))
            counts = counts / (denom * grid_step)

        return counts

    probabilities = _probs(x, grid, w, grid_step, bw)
    return grid, probabilities
