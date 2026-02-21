"""Kernel density estimation functions for 2D data."""

from __future__ import annotations

import typing

import equinox as eqx
import jax.numpy as jnp

if typing.TYPE_CHECKING:
    pass


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
    """Kernel density estimation with a 2D triangular kernel.

    This computes a JAX-compatible 2D KDE using a product of two 1D triangular
    kernels centered at each sample. The kernel support is ±(`bw_multiplier` *
    grid step) in each dimension. Each sample's kernel is renormalized over the
    evaluation grid to avoid boundary mass loss on finite support.

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
        x1: 1D array of samples for the first dimension. If not 1D, it will be
            flattened.
        x2: 1D array of samples for the second dimension. If not 1D, it will be
            flattened. Must have the same length as ``x1``.
        n_grid_points1: Number of grid points for the first dimension. If
            ``None``, inferred from the integer span
            ``[floor(min(x1)), ceil(max(x1))]``.
        n_grid_points2: Number of grid points for the second dimension. If
            ``None``, inferred from the integer span
            ``[floor(min(x2)), ceil(max(x2))]``.
        min_max_vals1: Tuple ``(min_val, max_val)`` defining the bin range for
            the first dimension. If ``None``, determined from data.
        min_max_vals2: Tuple ``(min_val, max_val)`` defining the bin range for
            the second dimension. If ``None``, determined from data.
        density: If ``True``, returns a probability density function whose
            Riemann sum over the grid integrates to 1 (via normalization by
            ``sum * grid_step1 * grid_step2``). If ``False``, returns
            unnormalized counts/weights per grid point.
        weights: Optional nonnegative weights per sample (same length as `x1`
            and `x2`). When provided, kernel contributions are multiplied by
            these weights.
        bw_multiplier: Kernel half-width as a multiple of the bin width in
            each dimension.
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
    """
    x1 = jnp.asarray(x1).reshape(-1)
    x2 = jnp.asarray(x2).reshape(-1)
    if x1.shape[0] != x2.shape[0]:
        raise ValueError('x1 and x2 must have the same length')

    w = None if weights is None else jnp.asarray(weights).reshape(-1)
    if w is not None and w.shape[0] != x1.shape[0]:
        raise ValueError('weights must have the same length as x1 and x2')

    # Grid parameters for first dimension
    if min_max_vals1 is None:
        min_val1 = jnp.floor(jnp.min(x1))
        max_val1 = jnp.ceil(jnp.max(x1))
    else:
        min_val1, max_val1 = min_max_vals1

    if n_grid_points1 is None:
        n_grid_points1 = int(max_val1 - min_val1) + 1 if max_val1 >= min_val1 else 1

    grid1 = jnp.linspace(min_val1, max_val1, int(n_grid_points1))

    # Grid parameters for second dimension
    if min_max_vals2 is None:
        min_val2 = jnp.floor(jnp.min(x2))
        max_val2 = jnp.ceil(jnp.max(x2))
    else:
        min_val2, max_val2 = min_max_vals2

    if n_grid_points2 is None:
        n_grid_points2 = int(max_val2 - min_val2) + 1 if max_val2 >= min_val2 else 1

    grid2 = jnp.linspace(min_val2, max_val2, int(n_grid_points2))

    # Compute grid steps outside jit using static n_grid_points
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

    # Resolve effective Dirichlet alpha (per bin)
    # If kappa is provided, alpha = kappa / K; else use dirichlet_alpha or 0
    K1 = int(n_grid_points1)
    K2 = int(n_grid_points2)
    K_total = K1 * K2
    if dirichlet_kappa is not None:
        alpha_eff = float(dirichlet_kappa) / float(K_total)
    elif dirichlet_alpha is not None:
        alpha_eff = float(dirichlet_alpha)
    else:
        alpha_eff = 0.0
    alpha_eff = max(alpha_eff, 0.0)  # guard

    @eqx.filter_jit
    def _probs(x1, x2, grid1, grid2, w, grid_step1, grid_step2, bw, alpha_eff):
        x1_b = x1[:, None]  # (N, 1)
        grid1_b = grid1[None, :]  # (1, G1)
        x2_b = x2[:, None]  # (N, 1)
        grid2_b = grid2[None, :]  # (1, G2)

        # Compute 1D triangular kernels
        scale1 = grid_step1 * bw
        dist1 = jnp.abs(x1_b - grid1_b) / scale1
        kernel_vals1 = jnp.maximum(
            jnp.asarray(0.0, grid1.dtype), jnp.asarray(1.0, grid1.dtype) - dist1
        )

        scale2 = grid_step2 * bw
        dist2 = jnp.abs(x2_b - grid2_b) / scale2
        kernel_vals2 = jnp.maximum(
            jnp.asarray(0.0, grid2.dtype), jnp.asarray(1.0, grid2.dtype) - dist2
        )

        # Per-sample renormalization (prevents boundary mass loss)
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

        # Combine kernels: product gives contribution of each point to each 2D bin
        kernel_vals1_b = jnp.expand_dims(kernel_vals1, 2)  # (N, G1, 1)
        kernel_vals2_b = jnp.expand_dims(kernel_vals2, 1)  # (N, 1, G2)
        joint_kernel = kernel_vals1_b * kernel_vals2_b  # (N, G1, G2)

        # Soft counts (weighted if provided)
        if w is None:
            counts = jnp.sum(joint_kernel, axis=0)  # shape (G1, G2)
        else:
            counts = jnp.sum(joint_kernel * w[:, None, None], axis=0)

        if not density:
            # Return raw soft counts; no Dirichlet smoothing here by design.
            return counts

        # Convert to a *pmf* with optional Dirichlet smoothing:
        # p_hat = (counts + alpha) / (N + alpha*K)
        N = jnp.sum(counts)
        # Avoid 0/0 when N=0 and alpha=0: fall back to uniform
        alpha = jnp.asarray(alpha_eff, dtype=counts.dtype)
        Kf = jnp.asarray(counts.size, dtype=counts.dtype)

        denom = N + alpha * Kf
        denom = jnp.where(denom > 0, denom, jnp.asarray(1.0, dtype=counts.dtype))

        pmf_smoothed = (counts + alpha) / denom

        # Convert pmf to pdf on the grid
        pdf = pmf_smoothed / (grid_step1 * grid_step2)
        return pdf

    values = _probs(
        x1,
        x2,
        grid1,
        grid2,
        w,
        grid_step1,
        grid_step2,
        bw,
        jnp.asarray(alpha_eff, dtype=grid1.dtype),
    )
    return grid1, grid2, values


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
    """Kernel density estimation with a 2D exponential (Laplace) kernel.

    This computes a JAX-compatible 2D KDE using a product of two 1D exponential
    kernels centered at each sample. The kernel is ``k(d) = exp(-|d| / scale)``
    where ``scale = bw_multiplier * grid_step`` in each dimension. Each sample's
    kernel is renormalized over the evaluation grid to avoid boundary mass loss on
    finite support.

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
        x1: 1D array of samples for the first dimension. If not 1D, it will be
            flattened.
        x2: 1D array of samples for the second dimension. If not 1D, it will be
            flattened. Must have the same length as ``x1``.
        n_grid_points1: Number of grid points for the first dimension. If
            ``None``, inferred from the integer span
            ``[floor(min(x1)), ceil(max(x1))]``.
        n_grid_points2: Number of grid points for the second dimension. If
            ``None``, inferred from the integer span
            ``[floor(min(x2)), ceil(max(x2))]``.
        min_max_vals1: Tuple ``(min_val, max_val)`` defining the bin range for
            the first dimension. If ``None``, determined from data.
        min_max_vals2: Tuple ``(min_val, max_val)`` defining the bin range for
            the second dimension. If ``None``, determined from data.
        density: If ``True``, returns a probability density function whose
            Riemann sum over the grid integrates to 1 (via normalization by
            ``sum * grid_step1 * grid_step2``). If ``False``, returns
            unnormalized counts/weights per grid point.
        weights: Optional nonnegative weights per sample (same length as `x1`
            and `x2`). When provided, kernel contributions are multiplied by
            these weights.
        bw_multiplier: Positive decay scale as a multiple of the bin width in
            each dimension.
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
    """
    x1 = jnp.asarray(x1).reshape(-1)
    x2 = jnp.asarray(x2).reshape(-1)
    if x1.shape[0] != x2.shape[0]:
        raise ValueError('x1 and x2 must have the same length')

    w = None if weights is None else jnp.asarray(weights).reshape(-1)
    if w is not None and w.shape[0] != x1.shape[0]:
        raise ValueError('weights must have the same length as x1 and x2')

    # Grid parameters for first dimension
    if min_max_vals1 is None:
        min_val1 = jnp.floor(jnp.min(x1))
        max_val1 = jnp.ceil(jnp.max(x1))
    else:
        min_val1, max_val1 = min_max_vals1

    if n_grid_points1 is None:
        n_grid_points1 = int(max_val1 - min_val1) + 1 if max_val1 >= min_val1 else 1

    grid1 = jnp.linspace(min_val1, max_val1, int(n_grid_points1))

    # Grid parameters for second dimension
    if min_max_vals2 is None:
        min_val2 = jnp.floor(jnp.min(x2))
        max_val2 = jnp.ceil(jnp.max(x2))
    else:
        min_val2, max_val2 = min_max_vals2

    if n_grid_points2 is None:
        n_grid_points2 = int(max_val2 - min_val2) + 1 if max_val2 >= min_val2 else 1

    grid2 = jnp.linspace(min_val2, max_val2, int(n_grid_points2))

    # Compute grid steps outside jit using static n_grid_points
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

    # Resolve effective Dirichlet alpha (per bin)
    # If kappa is provided, alpha = kappa / K; else use dirichlet_alpha or 0
    K1 = int(n_grid_points1)
    K2 = int(n_grid_points2)
    K_total = K1 * K2
    if dirichlet_kappa is not None:
        alpha_eff = float(dirichlet_kappa) / float(K_total)
    elif dirichlet_alpha is not None:
        alpha_eff = float(dirichlet_alpha)
    else:
        alpha_eff = 0.0
    alpha_eff = max(alpha_eff, 0.0)  # guard

    @eqx.filter_jit
    def _probs(x1, x2, grid1, grid2, w, grid_step1, grid_step2, bw, alpha_eff):
        x1_b = x1[:, None]  # (N, 1)
        grid1_b = grid1[None, :]  # (1, G1)
        x2_b = x2[:, None]  # (N, 1)
        grid2_b = grid2[None, :]  # (1, G2)

        # Compute 1D exponential kernels
        scale1 = grid_step1 * bw
        dist1 = jnp.abs(x1_b - grid1_b) / scale1
        kernel_vals1 = jnp.exp(-dist1)

        scale2 = grid_step2 * bw
        dist2 = jnp.abs(x2_b - grid2_b) / scale2
        kernel_vals2 = jnp.exp(-dist2)

        # Per-sample renormalization (prevents boundary mass loss)
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

        # Combine kernels: product gives contribution of each point to each 2D bin
        kernel_vals1_b = jnp.expand_dims(kernel_vals1, 2)  # (N, G1, 1)
        kernel_vals2_b = jnp.expand_dims(kernel_vals2, 1)  # (N, 1, G2)
        joint_kernel = kernel_vals1_b * kernel_vals2_b  # (N, G1, G2)

        # Soft counts (weighted if provided)
        if w is None:
            counts = jnp.sum(joint_kernel, axis=0)  # shape (G1, G2)
        else:
            counts = jnp.sum(joint_kernel * w[:, None, None], axis=0)

        if not density:
            # Return raw soft counts; no Dirichlet smoothing here by design.
            return counts

        # Convert to a *pmf* with optional Dirichlet smoothing:
        # p_hat = (counts + alpha) / (N + alpha*K)
        N = jnp.sum(counts)
        # Avoid 0/0 when N=0 and alpha=0: fall back to uniform
        alpha = jnp.asarray(alpha_eff, dtype=counts.dtype)
        Kf = jnp.asarray(counts.size, dtype=counts.dtype)

        denom = N + alpha * Kf
        denom = jnp.where(denom > 0, denom, jnp.asarray(1.0, dtype=counts.dtype))

        pmf_smoothed = (counts + alpha) / denom

        # Convert pmf to pdf on the grid
        pdf = pmf_smoothed / (grid_step1 * grid_step2)
        return pdf

    values = _probs(
        x1,
        x2,
        grid1,
        grid2,
        w,
        grid_step1,
        grid_step2,
        bw,
        jnp.asarray(alpha_eff, dtype=grid1.dtype),
    )
    return grid1, grid2, values


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
    """Kernel density estimation with a 2D Gaussian kernel.

    This computes a JAX-compatible 2D KDE using a product of two 1D Gaussian
    kernels centered at each sample. The kernel is ``k(d) = exp(-0.5 * (d /
    scale)^2)`` where ``scale = bw_multiplier * grid_step`` in each dimension.
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
        x1: 1D array of samples for the first dimension. If not 1D, it will be
            flattened.
        x2: 1D array of samples for the second dimension. If not 1D, it will be
            flattened. Must have the same length as ``x1``.
        n_grid_points1: Number of grid points for the first dimension. If
            ``None``, inferred from the integer span
            ``[floor(min(x1)), ceil(max(x1))]``.
        n_grid_points2: Number of grid points for the second dimension. If
            ``None``, inferred from the integer span
            ``[floor(min(x2)), ceil(max(x2))]``.
        min_max_vals1: Tuple ``(min_val, max_val)`` defining the bin range for
            the first dimension. If ``None``, determined from data.
        min_max_vals2: Tuple ``(min_val, max_val)`` defining the bin range for
            the second dimension. If ``None``, determined from data.
        density: If ``True``, returns a probability density function whose
            Riemann sum over the grid integrates to 1 (via normalization by
            ``sum * grid_step1 * grid_step2``). If ``False``, returns
            unnormalized counts/weights per grid point.
        weights: Optional nonnegative weights per sample (same length as `x1`
            and `x2`). When provided, kernel contributions are multiplied by
            these weights.
        bw_multiplier: Positive decay scale as a multiple of the bin width in
            each dimension.
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
    """
    x1 = jnp.asarray(x1).reshape(-1)
    x2 = jnp.asarray(x2).reshape(-1)
    if x1.shape[0] != x2.shape[0]:
        raise ValueError('x1 and x2 must have the same length')

    w = None if weights is None else jnp.asarray(weights).reshape(-1)
    if w is not None and w.shape[0] != x1.shape[0]:
        raise ValueError('weights must have the same length as x1 and x2')

    # Grid parameters for first dimension
    if min_max_vals1 is None:
        min_val1 = jnp.floor(jnp.min(x1))
        max_val1 = jnp.ceil(jnp.max(x1))
    else:
        min_val1, max_val1 = min_max_vals1

    if n_grid_points1 is None:
        n_grid_points1 = int(max_val1 - min_val1) + 1 if max_val1 >= min_val1 else 1

    grid1 = jnp.linspace(min_val1, max_val1, int(n_grid_points1))

    # Grid parameters for second dimension
    if min_max_vals2 is None:
        min_val2 = jnp.floor(jnp.min(x2))
        max_val2 = jnp.ceil(jnp.max(x2))
    else:
        min_val2, max_val2 = min_max_vals2

    if n_grid_points2 is None:
        n_grid_points2 = int(max_val2 - min_val2) + 1 if max_val2 >= min_val2 else 1

    grid2 = jnp.linspace(min_val2, max_val2, int(n_grid_points2))

    # Compute grid steps outside jit using static n_grid_points
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

    # Resolve effective Dirichlet alpha (per bin)
    # If kappa is provided, alpha = kappa / K; else use dirichlet_alpha or 0
    K1 = int(n_grid_points1)
    K2 = int(n_grid_points2)
    K_total = K1 * K2
    if dirichlet_kappa is not None:
        alpha_eff = float(dirichlet_kappa) / float(K_total)
    elif dirichlet_alpha is not None:
        alpha_eff = float(dirichlet_alpha)
    else:
        alpha_eff = 0.0
    alpha_eff = max(alpha_eff, 0.0)  # guard

    @eqx.filter_jit
    def _probs(x1, x2, grid1, grid2, w, grid_step1, grid_step2, bw, alpha_eff):
        x1_b = x1[:, None]  # (N, 1)
        grid1_b = grid1[None, :]  # (1, G1)
        x2_b = x2[:, None]  # (N, 1)
        grid2_b = grid2[None, :]  # (1, G2)

        # Compute 1D Gaussian kernels
        scale1 = grid_step1 * bw
        z1 = (x1_b - grid1_b) / scale1
        kernel_vals1 = jnp.exp(-0.5 * (z1 * z1))

        scale2 = grid_step2 * bw
        z2 = (x2_b - grid2_b) / scale2
        kernel_vals2 = jnp.exp(-0.5 * (z2 * z2))

        # Per-sample renormalization (prevents boundary mass loss)
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

        # Combine kernels: product gives contribution of each point to each 2D bin
        kernel_vals1_b = jnp.expand_dims(kernel_vals1, 2)  # (N, G1, 1)
        kernel_vals2_b = jnp.expand_dims(kernel_vals2, 1)  # (N, 1, G2)
        joint_kernel = kernel_vals1_b * kernel_vals2_b  # (N, G1, G2)

        # Soft counts (weighted if provided)
        if w is None:
            counts = jnp.sum(joint_kernel, axis=0)  # shape (G1, G2)
        else:
            counts = jnp.sum(joint_kernel * w[:, None, None], axis=0)

        if not density:
            # Return raw soft counts; no Dirichlet smoothing here by design.
            return counts

        # Convert to a *pmf* with optional Dirichlet smoothing:
        # p_hat = (counts + alpha) / (N + alpha*K)
        N = jnp.sum(counts)
        # Avoid 0/0 when N=0 and alpha=0: fall back to uniform
        alpha = jnp.asarray(alpha_eff, dtype=counts.dtype)
        Kf = jnp.asarray(counts.size, dtype=counts.dtype)

        denom = N + alpha * Kf
        denom = jnp.where(denom > 0, denom, jnp.asarray(1.0, dtype=counts.dtype))

        pmf_smoothed = (counts + alpha) / denom

        # Convert pmf to pdf on the grid
        pdf = pmf_smoothed / (grid_step1 * grid_step2)
        return pdf

    values = _probs(
        x1,
        x2,
        grid1,
        grid2,
        w,
        grid_step1,
        grid_step2,
        bw,
        jnp.asarray(alpha_eff, dtype=grid1.dtype),
    )
    return grid1, grid2, values


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
    """Kernel density estimation with a 2D Wendland C2 kernel.

    This computes a JAX-compatible 2D KDE using a product of two 1D Wendland C2
    kernels centered at each sample. The kernel is ``k(r) = (1 - r)^4 * (4r + 1)``
    for r in [0, 1], zero elsewhere, where ``r = |d| / scale`` and
    ``scale = bw_multiplier * grid_step`` in each dimension. The kernel is
    compactly supported. Each sample's kernel is renormalized over the evaluation
    grid to avoid boundary mass loss on finite support.

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
        x1: 1D array of samples for the first dimension. If not 1D, it will be
            flattened.
        x2: 1D array of samples for the second dimension. If not 1D, it will be
            flattened. Must have the same length as ``x1``.
        n_grid_points1: Number of grid points for the first dimension. If
            ``None``, inferred from the integer span
            ``[floor(min(x1)), ceil(max(x1))]``.
        n_grid_points2: Number of grid points for the second dimension. If
            ``None``, inferred from the integer span
            ``[floor(min(x2)), ceil(max(x2))]``.
        min_max_vals1: Tuple ``(min_val, max_val)`` defining the bin range for
            the first dimension. If ``None``, determined from data.
        min_max_vals2: Tuple ``(min_val, max_val)`` defining the bin range for
            the second dimension. If ``None``, determined from data.
        density: If ``True``, returns a probability density function whose
            Riemann sum over the grid integrates to 1 (via normalization by
            ``sum * grid_step1 * grid_step2``). If ``False``, returns
            unnormalized counts/weights per grid point.
        weights: Optional nonnegative weights per sample (same length as `x1`
            and `x2`). When provided, kernel contributions are multiplied by
            these weights.
        bw_multiplier: Kernel half-width as a multiple of the bin width in
            each dimension.
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
    """
    x1 = jnp.asarray(x1).reshape(-1)
    x2 = jnp.asarray(x2).reshape(-1)
    if x1.shape[0] != x2.shape[0]:
        raise ValueError('x1 and x2 must have the same length')

    w = None if weights is None else jnp.asarray(weights).reshape(-1)
    if w is not None and w.shape[0] != x1.shape[0]:
        raise ValueError('weights must have the same length as x1 and x2')

    # Grid parameters for first dimension
    if min_max_vals1 is None:
        min_val1 = jnp.floor(jnp.min(x1))
        max_val1 = jnp.ceil(jnp.max(x1))
    else:
        min_val1, max_val1 = min_max_vals1

    if n_grid_points1 is None:
        n_grid_points1 = int(max_val1 - min_val1) + 1 if max_val1 >= min_val1 else 1

    grid1 = jnp.linspace(min_val1, max_val1, int(n_grid_points1))

    # Grid parameters for second dimension
    if min_max_vals2 is None:
        min_val2 = jnp.floor(jnp.min(x2))
        max_val2 = jnp.ceil(jnp.max(x2))
    else:
        min_val2, max_val2 = min_max_vals2

    if n_grid_points2 is None:
        n_grid_points2 = int(max_val2 - min_val2) + 1 if max_val2 >= min_val2 else 1

    grid2 = jnp.linspace(min_val2, max_val2, int(n_grid_points2))

    # Compute grid steps outside jit using static n_grid_points
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

    # Resolve effective Dirichlet alpha (per bin)
    # If kappa is provided, alpha = kappa / K; else use dirichlet_alpha or 0
    K1 = int(n_grid_points1)
    K2 = int(n_grid_points2)
    K_total = K1 * K2
    if dirichlet_kappa is not None:
        alpha_eff = float(dirichlet_kappa) / float(K_total)
    elif dirichlet_alpha is not None:
        alpha_eff = float(dirichlet_alpha)
    else:
        alpha_eff = 0.0
    alpha_eff = max(alpha_eff, 0.0)  # guard

    @eqx.filter_jit
    def _probs(x1, x2, grid1, grid2, w, grid_step1, grid_step2, bw, alpha_eff):
        x1_b = x1[:, None]  # (N, 1)
        grid1_b = grid1[None, :]  # (1, G1)
        x2_b = x2[:, None]  # (N, 1)
        grid2_b = grid2[None, :]  # (1, G2)

        # Compute 1D Wendland C2 kernels
        scale1 = grid_step1 * bw
        dist1 = jnp.abs(x1_b - grid1_b) / scale1
        r1 = jnp.clip(
            dist1,
            jnp.asarray(0.0, grid1.dtype),
            jnp.asarray(1.0, grid1.dtype),
        )
        kernel_vals1 = jnp.where(
            dist1 < jnp.asarray(1.0, grid1.dtype),
            (jnp.asarray(1.0, grid1.dtype) - r1) ** 4
            * (jnp.asarray(4.0, grid1.dtype) * r1 + jnp.asarray(1.0, grid1.dtype)),
            jnp.asarray(0.0, grid1.dtype),
        )

        scale2 = grid_step2 * bw
        dist2 = jnp.abs(x2_b - grid2_b) / scale2
        r2 = jnp.clip(
            dist2,
            jnp.asarray(0.0, grid2.dtype),
            jnp.asarray(1.0, grid2.dtype),
        )
        kernel_vals2 = jnp.where(
            dist2 < jnp.asarray(1.0, grid2.dtype),
            (jnp.asarray(1.0, grid2.dtype) - r2) ** 4
            * (jnp.asarray(4.0, grid2.dtype) * r2 + jnp.asarray(1.0, grid2.dtype)),
            jnp.asarray(0.0, grid2.dtype),
        )

        # Per-sample renormalization (prevents boundary mass loss)
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

        # Combine kernels: product gives contribution of each point to each 2D bin
        kernel_vals1_b = jnp.expand_dims(kernel_vals1, 2)  # (N, G1, 1)
        kernel_vals2_b = jnp.expand_dims(kernel_vals2, 1)  # (N, 1, G2)
        joint_kernel = kernel_vals1_b * kernel_vals2_b  # (N, G1, G2)

        # Soft counts (weighted if provided)
        if w is None:
            counts = jnp.sum(joint_kernel, axis=0)  # shape (G1, G2)
        else:
            counts = jnp.sum(joint_kernel * w[:, None, None], axis=0)

        if not density:
            # Return raw soft counts; no Dirichlet smoothing here by design.
            return counts

        # Convert to a *pmf* with optional Dirichlet smoothing:
        # p_hat = (counts + alpha) / (N + alpha*K)
        N = jnp.sum(counts)
        # Avoid 0/0 when N=0 and alpha=0: fall back to uniform
        alpha = jnp.asarray(alpha_eff, dtype=counts.dtype)
        Kf = jnp.asarray(counts.size, dtype=counts.dtype)

        denom = N + alpha * Kf
        denom = jnp.where(denom > 0, denom, jnp.asarray(1.0, dtype=counts.dtype))

        pmf_smoothed = (counts + alpha) / denom

        # Convert pmf to pdf on the grid
        pdf = pmf_smoothed / (grid_step1 * grid_step2)
        return pdf

    values = _probs(
        x1,
        x2,
        grid1,
        grid2,
        w,
        grid_step1,
        grid_step2,
        bw,
        jnp.asarray(alpha_eff, dtype=grid1.dtype),
    )
    return grid1, grid2, values
