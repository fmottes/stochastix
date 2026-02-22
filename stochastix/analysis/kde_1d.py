"""Kernel density estimation functions."""

from __future__ import annotations

import typing

import equinox as eqx
import jax.numpy as jnp

KDEKernel = typing.Literal['triangular', 'exponential', 'gaussian', 'wendland_c2']
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
    n_grid_points: int,
    dirichlet_alpha: float | None,
    dirichlet_kappa: float | None,
) -> float:
    if dirichlet_kappa is not None:
        alpha_eff = float(dirichlet_kappa) / float(n_grid_points)
    elif dirichlet_alpha is not None:
        alpha_eff = float(dirichlet_alpha)
    else:
        alpha_eff = 0.0
    return max(alpha_eff, 0.0)


def _resolve_kernel_function(
    kernel: KDEKernel,
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


def kde(
    x: jnp.ndarray,
    n_grid_points: int | None = None,
    min_max_vals: tuple[float, float] | None = None,
    density: bool = True,
    weights: jnp.ndarray | None = None,
    bw_multiplier: float = 1.0,
    *,
    kernel: KDEKernel = 'wendland_c2',
    dirichlet_alpha: float | None = 0.1,
    dirichlet_kappa: float | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute a 1D JAX-compatible KDE with selectable kernel.

    Each sample contributes a kernel centered at its value. Per-sample kernels
    are renormalized over the evaluation grid to avoid boundary mass loss on
    finite supports.

    Args:
        x: 1D array of samples. If not 1D, it is flattened.
        n_grid_points: Number of grid points. If ``None``, inferred from the
            integer span ``[floor(min(x)), ceil(max(x))]``.
        min_max_vals: Tuple ``(min_val, max_val)`` defining the grid range. If
            ``None``, inferred from data.
        density: If ``True``, return a probability density function. If
            ``False``, return unnormalized soft counts.
        weights: Optional nonnegative weights per sample (same length as ``x``).
        bw_multiplier: Kernel scale multiplier relative to the grid step.
        kernel: Kernel family. Must be one of ``'triangular'``,
            ``'exponential'``, ``'gaussian'``, or ``'wendland_c2'``.
        dirichlet_alpha: Per-bin pseudo-count for Dirichlet smoothing when
            ``density=True``. Ignored when ``density=False``.
        dirichlet_kappa: Total pseudo-count for Dirichlet smoothing. When set,
            overrides ``dirichlet_alpha`` with ``alpha = kappa / K``.

    Returns:
        Tuple ``(grid, values)`` where ``grid`` has shape ``(n_grid_points,)`` and
        ``values`` has shape ``(n_grid_points,)``.

    Raises:
        ValueError: If ``weights`` length does not match ``x`` length.
        ValueError: If ``kernel`` is not a supported kernel name.
    """
    x = jnp.asarray(x).reshape(-1)
    w = None if weights is None else jnp.asarray(weights).reshape(-1)
    if w is not None and w.shape[0] != x.shape[0]:
        raise ValueError('weights must have the same length as x')
    if kernel not in _KERNEL_NAMES:
        raise ValueError(f'kernel must be one of {_KERNEL_NAMES}, got {kernel!r}')

    if min_max_vals is None:
        min_val = jnp.floor(jnp.min(x))
        max_val = jnp.ceil(jnp.max(x))
    else:
        min_val, max_val = min_max_vals

    if n_grid_points is None:
        n_grid_points = int(max_val - min_val) + 1 if max_val >= min_val else 1

    grid = jnp.linspace(min_val, max_val, int(n_grid_points))
    if int(n_grid_points) > 1:
        grid_step = grid[1] - grid[0]
    else:
        grid_step = jnp.asarray(1.0, dtype=grid.dtype)
    grid_step = jnp.maximum(grid_step, jnp.asarray(1e-6, dtype=grid.dtype))
    bw = jnp.maximum(
        jnp.asarray(bw_multiplier, dtype=grid.dtype),
        jnp.asarray(1e-6, dtype=grid.dtype),
    )
    alpha_eff = _resolve_dirichlet_alpha(
        n_grid_points=int(n_grid_points),
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_kappa=dirichlet_kappa,
    )
    kernel_fn = _resolve_kernel_function(kernel)

    @eqx.filter_jit
    def _probs(
        x_data: jnp.ndarray,
        grid_data: jnp.ndarray,
        w_data: jnp.ndarray | None,
        grid_step_data: jnp.ndarray,
        bw_data: jnp.ndarray,
        alpha_eff_data: jnp.ndarray,
    ) -> jnp.ndarray:
        x_b = x_data[:, None]
        grid_b = grid_data[None, :]

        scale = grid_step_data * bw_data
        dist = jnp.abs(x_b - grid_b) / scale
        kernel_vals = kernel_fn(dist, grid_data.dtype)

        row_sum = jnp.sum(kernel_vals, axis=1, keepdims=True)
        row_sum = jnp.where(
            row_sum > 0,
            row_sum,
            jnp.asarray(1.0, dtype=kernel_vals.dtype),
        )
        kernel_vals = kernel_vals / row_sum

        if w_data is None:
            counts = jnp.sum(kernel_vals, axis=0)
        else:
            counts = jnp.sum(kernel_vals * w_data[:, None], axis=0)

        if not density:
            return counts

        n_eff = jnp.sum(counts)
        alpha = jnp.asarray(alpha_eff_data, dtype=counts.dtype)
        n_bins = jnp.asarray(counts.shape[-1], dtype=counts.dtype)
        denom = n_eff + alpha * n_bins
        denom = jnp.where(denom > 0, denom, jnp.asarray(1.0, dtype=counts.dtype))
        pmf_smoothed = (counts + alpha) / denom
        return pmf_smoothed / grid_step_data

    values = _probs(
        x_data=x,
        grid_data=grid,
        w_data=w,
        grid_step_data=grid_step,
        bw_data=bw,
        alpha_eff_data=jnp.asarray(alpha_eff, dtype=grid.dtype),
    )
    return grid, values


def kde_triangular(
    x: jnp.ndarray,
    n_grid_points: int | None = None,
    min_max_vals: tuple[float, float] | None = None,
    density: bool = True,
    weights: jnp.ndarray | None = None,
    bw_multiplier: float = 1.0,
    *,
    dirichlet_alpha: float | None = 0.1,
    dirichlet_kappa: float | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Backward-compatible triangular specialization of ``kde``."""
    return kde(
        x=x,
        n_grid_points=n_grid_points,
        min_max_vals=min_max_vals,
        density=density,
        weights=weights,
        bw_multiplier=bw_multiplier,
        kernel='triangular',
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_kappa=dirichlet_kappa,
    )


def kde_exponential(
    x: jnp.ndarray,
    n_grid_points: int | None = None,
    min_max_vals: tuple[float, float] | None = None,
    density: bool = True,
    weights: jnp.ndarray | None = None,
    bw_multiplier: float = 1.0,
    *,
    dirichlet_alpha: float | None = 0.1,
    dirichlet_kappa: float | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Backward-compatible exponential specialization of ``kde``."""
    return kde(
        x=x,
        n_grid_points=n_grid_points,
        min_max_vals=min_max_vals,
        density=density,
        weights=weights,
        bw_multiplier=bw_multiplier,
        kernel='exponential',
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_kappa=dirichlet_kappa,
    )


def kde_gaussian(
    x: jnp.ndarray,
    n_grid_points: int | None = None,
    min_max_vals: tuple[float, float] | None = None,
    density: bool = True,
    weights: jnp.ndarray | None = None,
    bw_multiplier: float = 1.0,
    *,
    dirichlet_alpha: float | None = 0.1,
    dirichlet_kappa: float | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Backward-compatible Gaussian specialization of ``kde``."""
    return kde(
        x=x,
        n_grid_points=n_grid_points,
        min_max_vals=min_max_vals,
        density=density,
        weights=weights,
        bw_multiplier=bw_multiplier,
        kernel='gaussian',
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_kappa=dirichlet_kappa,
    )


def kde_wendland_c2(
    x: jnp.ndarray,
    n_grid_points: int | None = None,
    min_max_vals: tuple[float, float] | None = None,
    density: bool = True,
    weights: jnp.ndarray | None = None,
    bw_multiplier: float = 1.0,
    *,
    dirichlet_alpha: float | None = 0.1,
    dirichlet_kappa: float | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Backward-compatible Wendland C2 specialization of ``kde``.

    Bandwidths below ``1.0`` approach a discrete pmf and usually lose useful
    gradients. Around ``1.5`` is a practical differentiation heuristic.
    """
    return kde(
        x=x,
        n_grid_points=n_grid_points,
        min_max_vals=min_max_vals,
        density=density,
        weights=weights,
        bw_multiplier=bw_multiplier,
        kernel='wendland_c2',
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_kappa=dirichlet_kappa,
    )
