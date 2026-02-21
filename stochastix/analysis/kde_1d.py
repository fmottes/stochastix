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
    *,
    dirichlet_alpha: float | None = 0.1,
    dirichlet_kappa: float | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Kernel density estimation with a triangular kernel.

    This computes a JAX-compatible KDE using a triangular kernel centered at
    each sample. The kernel support is ±(`bw_multiplier` * grid step). Each
    sample's kernel is renormalized over the evaluation grid to avoid boundary mass
    loss on finite support.

    Note: Dirichlet smoothing
        When ``density=True``, applies add-α smoothing to the multinomial pmf
        implied by the soft counts before converting to a pdf:
        ``p_hat = (counts + α) / (N + α*K)``.
        When ``density=False``, returns raw soft counts (no smoothing).

    Note: JIT-compatibility
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
        dirichlet_alpha: Per-bin pseudo-count for Dirichlet smoothing. Default is
            ``0.1``. Note: ``dirichlet_kappa`` takes priority over this parameter if
            provided.
        dirichlet_kappa: Total pseudo-count for Dirichlet smoothing. If provided,
            takes priority over ``dirichlet_alpha`` and ``alpha = kappa / K`` where K
            is the number of grid points. If ``None``, uses ``dirichlet_alpha``
            instead.

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

    # Resolve effective Dirichlet alpha (per bin)
    # If kappa is provided, alpha = kappa / K; else use dirichlet_alpha or 0
    K = int(n_grid_points)
    if dirichlet_kappa is not None:
        alpha_eff = float(dirichlet_kappa) / float(K)
    elif dirichlet_alpha is not None:
        alpha_eff = float(dirichlet_alpha)
    else:
        alpha_eff = 0.0
    alpha_eff = max(alpha_eff, 0.0)  # guard

    @eqx.filter_jit
    def _probs(x, grid, w, grid_step, bw, alpha_eff):
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

        # Soft counts (weighted if provided)
        if w is None:
            counts = jnp.sum(kernel_vals, axis=0)  # shape (G,)
        else:
            counts = jnp.sum(kernel_vals * w[:, None], axis=0)

        if not density:
            # Return raw soft counts; no Dirichlet smoothing here by design.
            return counts

        # Convert to a *pmf* with optional Dirichlet smoothing:
        # p_hat = (counts + alpha) / (N + alpha*K)
        N = jnp.sum(counts)
        # Avoid 0/0 when N=0 and alpha=0: fall back to uniform
        alpha = jnp.asarray(alpha_eff, dtype=counts.dtype)
        Kf = jnp.asarray(counts.shape[-1], dtype=counts.dtype)

        denom = N + alpha * Kf
        denom = jnp.where(denom > 0, denom, jnp.asarray(1.0, dtype=counts.dtype))

        pmf_smoothed = (counts + alpha) / denom

        # Convert pmf to pdf on the grid
        pdf = pmf_smoothed / grid_step
        return pdf

    values = _probs(x, grid, w, grid_step, bw, jnp.asarray(alpha_eff, dtype=grid.dtype))
    return grid, values


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
    """Kernel density estimation with an exponential (Laplace) kernel.

    This computes a JAX-compatible KDE using an exponential kernel centered at
    each sample. The kernel is ``k(d) = exp(-|d| / scale)`` where
    ``scale = bw_multiplier * grid_step``. Each sample's kernel is renormalized
    over the evaluation grid to avoid boundary mass loss on finite support.

    Note: Dirichlet smoothing
        When ``density=True``, applies add-α smoothing to the multinomial pmf
        implied by the soft counts before converting to a pdf:
        ``p_hat = (counts + α) / (N + α*K)``.
        When ``density=False``, returns raw soft counts (no smoothing).

    Note: JIT-compatibility
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
        dirichlet_alpha: Per-bin pseudo-count for Dirichlet smoothing. Default is
            ``0.1``. Note: ``dirichlet_kappa`` takes priority over this parameter if
            provided.
        dirichlet_kappa: Total pseudo-count for Dirichlet smoothing. If provided,
            takes priority over ``dirichlet_alpha`` and ``alpha = kappa / K`` where K
            is the number of grid points. If ``None``, uses ``dirichlet_alpha``
            instead.

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

    # Resolve effective Dirichlet alpha (per bin)
    # If kappa is provided, alpha = kappa / K; else use dirichlet_alpha or 0
    K = int(n_grid_points)
    if dirichlet_kappa is not None:
        alpha_eff = float(dirichlet_kappa) / float(K)
    elif dirichlet_alpha is not None:
        alpha_eff = float(dirichlet_alpha)
    else:
        alpha_eff = 0.0
    alpha_eff = max(alpha_eff, 0.0)  # guard

    @eqx.filter_jit
    def _probs(x, grid, w, grid_step, bw, alpha_eff):
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

        # Soft counts (weighted if provided)
        if w is None:
            counts = jnp.sum(kernel_vals, axis=0)  # shape (G,)
        else:
            counts = jnp.sum(kernel_vals * w[:, None], axis=0)

        if not density:
            # Return raw soft counts; no Dirichlet smoothing here by design.
            return counts

        # Convert to a *pmf* with optional Dirichlet smoothing:
        # p_hat = (counts + alpha) / (N + alpha*K)
        N = jnp.sum(counts)
        # Avoid 0/0 when N=0 and alpha=0: fall back to uniform
        alpha = jnp.asarray(alpha_eff, dtype=counts.dtype)
        Kf = jnp.asarray(counts.shape[-1], dtype=counts.dtype)

        denom = N + alpha * Kf
        denom = jnp.where(denom > 0, denom, jnp.asarray(1.0, dtype=counts.dtype))

        pmf_smoothed = (counts + alpha) / denom

        # Convert pmf to pdf on the grid
        pdf = pmf_smoothed / grid_step
        return pdf

    values = _probs(x, grid, w, grid_step, bw, jnp.asarray(alpha_eff, dtype=grid.dtype))
    return grid, values


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
    """Kernel density estimation with a Gaussian kernel.

    This computes a JAX-compatible KDE using a Gaussian kernel centered at
    each sample. The kernel is ``k(d) = exp(-0.5 * (d / scale)^2)`` where
    ``scale = bw_multiplier * grid_step``. Each sample's kernel is renormalized
    over the evaluation grid to avoid boundary mass loss on finite support.

    Note: Dirichlet smoothing
        When ``density=True``, applies add-α smoothing to the multinomial pmf
        implied by the soft counts before converting to a pdf:
        ``p_hat = (counts + α) / (N + α*K)``.
        When ``density=False``, returns raw soft counts (no smoothing).

    Note: JIT-compatibility
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
        dirichlet_alpha: Per-bin pseudo-count for Dirichlet smoothing. Default is
            ``0.1``. Note: ``dirichlet_kappa`` takes priority over this parameter if
            provided.
        dirichlet_kappa: Total pseudo-count for Dirichlet smoothing. If provided,
            takes priority over ``dirichlet_alpha`` and ``alpha = kappa / K`` where K
            is the number of grid points. If ``None``, uses ``dirichlet_alpha``
            instead.

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

    # Resolve effective Dirichlet alpha (per bin)
    # If kappa is provided, alpha = kappa / K; else use dirichlet_alpha or 0
    K = int(n_grid_points)
    if dirichlet_kappa is not None:
        alpha_eff = float(dirichlet_kappa) / float(K)
    elif dirichlet_alpha is not None:
        alpha_eff = float(dirichlet_alpha)
    else:
        alpha_eff = 0.0
    alpha_eff = max(alpha_eff, 0.0)  # guard

    @eqx.filter_jit
    def _probs(x, grid, w, grid_step, bw, alpha_eff):
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

        # Soft counts (weighted if provided)
        if w is None:
            counts = jnp.sum(kernel_vals, axis=0)  # shape (G,)
        else:
            counts = jnp.sum(kernel_vals * w[:, None], axis=0)

        if not density:
            # Return raw soft counts; no Dirichlet smoothing here by design.
            return counts

        # Convert to a *pmf* with optional Dirichlet smoothing:
        # p_hat = (counts + alpha) / (N + alpha*K)
        N = jnp.sum(counts)
        # Avoid 0/0 when N=0 and alpha=0: fall back to uniform
        alpha = jnp.asarray(alpha_eff, dtype=counts.dtype)
        Kf = jnp.asarray(counts.shape[-1], dtype=counts.dtype)

        denom = N + alpha * Kf
        denom = jnp.where(denom > 0, denom, jnp.asarray(1.0, dtype=counts.dtype))

        pmf_smoothed = (counts + alpha) / denom

        # Convert pmf to pdf on the grid
        pdf = pmf_smoothed / grid_step
        return pdf

    values = _probs(x, grid, w, grid_step, bw, jnp.asarray(alpha_eff, dtype=grid.dtype))
    return grid, values


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
    """Kernel density estimation with a Wendland C2 kernel."""
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

    # Resolve effective Dirichlet alpha (per bin)
    # If kappa is provided, alpha = kappa / K; else use dirichlet_alpha or 0
    K = int(n_grid_points)
    if dirichlet_kappa is not None:
        alpha_eff = float(dirichlet_kappa) / float(K)
    elif dirichlet_alpha is not None:
        alpha_eff = float(dirichlet_alpha)
    else:
        alpha_eff = 0.0
    alpha_eff = max(alpha_eff, 0.0)  # guard

    @eqx.filter_jit
    def _probs(x, grid, w, grid_step, bw, alpha_eff):
        x_b = x[:, None]  # (N, 1)
        grid_b = grid[None, :]  # (1, G)

        scale = grid_step * bw
        dist = jnp.abs(x_b - grid_b) / scale
        # Wendland C2 kernel: (1 - r)^4 * (4r + 1), compactly supported on [0, 1]
        r = jnp.clip(dist, jnp.asarray(0.0, grid.dtype), jnp.asarray(1.0, grid.dtype))
        kernel_vals = jnp.where(
            dist < jnp.asarray(1.0, grid.dtype),
            (jnp.asarray(1.0, grid.dtype) - r) ** 4
            * (jnp.asarray(4.0, grid.dtype) * r + jnp.asarray(1.0, grid.dtype)),
            jnp.asarray(0.0, grid.dtype),
        )

        # Per-sample renormalization (prevents boundary mass loss)
        row_sum = jnp.sum(kernel_vals, axis=1, keepdims=True)
        row_sum = jnp.where(
            row_sum > 0,
            row_sum,
            jnp.asarray(1.0, dtype=kernel_vals.dtype),
        )
        kernel_vals = kernel_vals / row_sum

        # Soft counts (weighted if provided)
        if w is None:
            counts = jnp.sum(kernel_vals, axis=0)  # shape (G,)
        else:
            counts = jnp.sum(kernel_vals * w[:, None], axis=0)

        if not density:
            # Return raw soft counts; no Dirichlet smoothing here by design.
            return counts

        # Convert to a *pmf* with optional Dirichlet smoothing:
        # p_hat = (counts + alpha) / (N + alpha*K)
        N = jnp.sum(counts)
        # Avoid 0/0 when N=0 and alpha=0: fall back to uniform
        alpha = jnp.asarray(alpha_eff, dtype=counts.dtype)
        Kf = jnp.asarray(counts.shape[-1], dtype=counts.dtype)

        denom = N + alpha * Kf
        denom = jnp.where(denom > 0, denom, jnp.asarray(1.0, dtype=counts.dtype))

        pmf_smoothed = (counts + alpha) / denom

        # Convert pmf to pdf on the grid
        pdf = pmf_smoothed / grid_step
        return pdf

    values = _probs(x, grid, w, grid_step, bw, jnp.asarray(alpha_eff, dtype=grid.dtype))
    return grid, values
