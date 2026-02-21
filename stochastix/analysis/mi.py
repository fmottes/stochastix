"""Mutual information functions."""

from __future__ import annotations

import math
import typing

import jax.numpy as jnp

from .._state_utils import pytree_to_state
from .kde_1d import kde_exponential, kde_gaussian, kde_triangular, kde_wendland_c2
from .kde_2d import (
    kde_exponential_2d,
    kde_gaussian_2d,
    kde_triangular_2d,
    kde_wendland_c2_2d,
)

if typing.TYPE_CHECKING:
    from .._simulation_results import SimulationResults


def mutual_information(
    x1: jnp.ndarray,
    x2: jnp.ndarray,
    n_grid_points1: int | None = None,
    n_grid_points2: int | None = None,
    min_max_vals1: tuple[float, float] | None = None,
    min_max_vals2: tuple[float, float] | None = None,
    base: float = 2.0,
    *,
    kde_type: str = 'wendland_c2',
    bw_multiplier: float = 1.0,
    dirichlet_alpha: float | None = 0.1,
    dirichlet_kappa: float | None = None,
) -> jnp.ndarray:
    """Compute the mutual information between two arrays.

    This function uses KDE functions to compute the mutual information between
    two arrays `x1` and `x2`. The mutual information is a measure of the mutual
    dependence between the two variables.

    Note: JIT-compatibility
        For JIT-compatibility, provide concrete values for all grid parameters
        (`n_grid_points1`, `n_grid_points2`, `min_max_vals1`, `min_max_vals2`).
        If left as ``None``, grid parameters are determined from the data (not
        JIT-able).

    Args:
        x1: 1D array of the first input data.
        x2: 1D array of the second input data. Must have the same length as
            ``x1``.
        n_grid_points1: Number of grid points for the first dimension. If
            ``None``, determined automatically.
        n_grid_points2: Number of grid points for the second dimension. If
            ``None``, determined automatically.
        min_max_vals1: Tuple ``(min_val, max_val)`` for the first dimension's
            grid range. If ``None``, determined automatically.
        min_max_vals2: Tuple ``(min_val, max_val)`` for the second dimension's
            grid range. If ``None``, determined automatically.
        base: The logarithmic base to use for the entropy calculation. Default
            is ``2.0`` (bits).
        kde_type: Type of kernel to use. One of ``'triangular'``, ``'exponential'``,
            ``'gaussian'``, or ``'wendland_c2'``. Default is ``'wendland_c2'``.
        bw_multiplier: Kernel bandwidth multiplier. Controls the width of the
            kernel relative to the grid step size. Default is ``1.0``.
        dirichlet_alpha: Per-bin pseudo-count for Dirichlet smoothing. Default
            is ``0.1``. Note: ``dirichlet_kappa`` takes priority over this
            parameter if provided.
        dirichlet_kappa: Total pseudo-count for Dirichlet smoothing. If
            provided, takes priority over ``dirichlet_alpha``. If ``None``, uses
            ``dirichlet_alpha`` instead.

    Returns:
        The mutual information between `x1` and `x2` in the specified base.
    """
    base = float(base)
    if not math.isfinite(base) or base <= 0.0 or base == 1.0:
        raise ValueError(
            f'base must be finite, positive, and not equal to 1, got {base}'
        )

    # Select the appropriate KDE functions (happens at trace time, JIT-compatible)
    kde_1d_functions = {
        'triangular': kde_triangular,
        'exponential': kde_exponential,
        'gaussian': kde_gaussian,
        'wendland_c2': kde_wendland_c2,
    }
    kde_2d_functions = {
        'triangular': kde_triangular_2d,
        'exponential': kde_exponential_2d,
        'gaussian': kde_gaussian_2d,
        'wendland_c2': kde_wendland_c2_2d,
    }
    if kde_type not in kde_1d_functions:
        raise ValueError(
            f'kde_type must be one of {list(kde_1d_functions.keys())}, got {kde_type}'
        )
    kde_1d_func = kde_1d_functions[kde_type]
    kde_2d_func = kde_2d_functions[kde_type]

    # p(x1)
    grid1, p_x1 = kde_1d_func(
        x1,
        n_grid_points=n_grid_points1,
        min_max_vals=min_max_vals1,
        density=True,
        bw_multiplier=bw_multiplier,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_kappa=dirichlet_kappa,
    )
    # p(x2)
    grid2, p_x2 = kde_1d_func(
        x2,
        n_grid_points=n_grid_points2,
        min_max_vals=min_max_vals2,
        density=True,
        bw_multiplier=bw_multiplier,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_kappa=dirichlet_kappa,
    )
    # p(x1, x2)
    _, _, p_x1_x2 = kde_2d_func(
        x1,
        x2,
        n_grid_points1=n_grid_points1,
        n_grid_points2=n_grid_points2,
        min_max_vals1=min_max_vals1,
        min_max_vals2=min_max_vals2,
        density=True,
        bw_multiplier=bw_multiplier,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_kappa=dirichlet_kappa,
    )

    # Convert densities to per-cell probabilities (masses).
    # This keeps MI correct for non-unit grid spacing.
    if grid1.shape[0] > 1:
        dx1 = grid1[1] - grid1[0]
    else:
        dx1 = jnp.asarray(1.0, dtype=grid1.dtype)
    dx1 = jnp.maximum(dx1, jnp.asarray(1e-6, dtype=grid1.dtype))

    if grid2.shape[0] > 1:
        dx2 = grid2[1] - grid2[0]
    else:
        dx2 = jnp.asarray(1.0, dtype=grid2.dtype)
    dx2 = jnp.maximum(dx2, jnp.asarray(1e-6, dtype=grid2.dtype))

    q_x1 = p_x1 * dx1
    q_x2 = p_x2 * dx2
    q_x1_x2 = p_x1_x2 * (dx1 * dx2)

    # More numerically stable computation using direct log-ratio formula:
    # I(X;Y) = sum_{x,y} q(x,y) * log(q(x,y) / (q(x) * q(y)))
    # Computed in log-space to avoid underflow and cancellation errors.

    tiny = jnp.finfo(q_x1_x2.dtype).tiny

    # Compute log probabilities in log-space
    log_q_x1_x2 = jnp.log2(jnp.maximum(q_x1_x2, tiny))
    log_q_x1 = jnp.log2(jnp.maximum(q_x1, tiny))
    log_q_x2 = jnp.log2(jnp.maximum(q_x2, tiny))

    # Create outer product for log(q(x) * q(y)) = log q(x) + log q(y)
    log_q_x1_2d = log_q_x1[:, None]  # shape: (n_grid_points1, 1)
    log_q_x2_2d = log_q_x2[None, :]  # shape: (1, n_grid_points2)
    log_q_x1_x2_indep = (
        log_q_x1_2d + log_q_x2_2d
    )  # shape: (n_grid_points1, n_grid_points2)

    # I(X;Y) = sum q(x,y) * (log q(x,y) - log(q(x) * q(y)))
    log_ratio = log_q_x1_x2 - log_q_x1_x2_indep

    # Sum over all (x,y) pairs. Terms where q(x,y) = 0 contribute 0, so safe to sum all
    mi = jnp.sum(q_x1_x2 * log_ratio)

    # Convert to desired base if needed
    if base != 2.0:
        mi = mi / jnp.log2(base)

    # Ensure non-negativity (should be naturally non-negative, but clamp for numerical safety)
    return mi  # jnp.maximum(mi, 0.0)


def state_mutual_info(
    results: SimulationResults,
    species_at_t: typing.Iterable[tuple[str, int | float]],
    n_grid_points1: int | None = None,
    n_grid_points2: int | None = None,
    min_max_vals1: tuple[float, float] | None = None,
    min_max_vals2: tuple[float, float] | None = None,
    base: float = 2.0,
    *,
    kde_type: str = 'wendland_c2',
    bw_multiplier: float = 1.0,
    dirichlet_alpha: float | None = 0.1,
    dirichlet_kappa: float | None = None,
) -> jnp.ndarray:
    """Compute the mutual information between two species at specific time points.

    This function calculates the mutual information between the distributions of
    two species at two potentially different time points, `t1` and `t2`, from
    batched simulation results. It uses KDE functions to ensure the entire
    computation is end-to-end differentiable, which is useful for gradient-based
    optimization of simulation parameters.

    Note: JIT-compatibility
        For JIT-compatibility, provide concrete values for all grid parameters
        (`n_grid_points1`, `n_grid_points2`, `min_max_vals1`, `min_max_vals2`).
        If left as ``None``, grid parameters are determined from the data (not
        JIT-able).

    Args:
        results: The `SimulationResults` from a `stochsimsolve` simulation.
            This should contain a batch of simulation trajectories (e.g., from
            vmapping over `stochsimsolve`).
        species_at_t: An iterable containing two tuples, where each tuple consists
            of a species name and a time point, e.g., `[('S1', t1), ('S2', t2)]`.
            The time point can be an integer index or a float time value.
        n_grid_points1: Number of grid points for the first species. If ``None``,
            determined automatically (not JIT-compatible).
        n_grid_points2: Number of grid points for the second species. If ``None``,
            determined automatically (not JIT-compatible).
        min_max_vals1: Tuple ``(min_val, max_val)`` for the first species' grid
            range. If ``None``, determined automatically (not JIT-compatible).
        min_max_vals2: Tuple ``(min_val, max_val)`` for the second species' grid
            range. If ``None``, determined automatically (not JIT-compatible).
        base: The logarithmic base for the entropy calculation. Default is ``2.0``
            (bits).
        kde_type: Type of kernel to use. One of ``'triangular'``, ``'exponential'``,
            ``'gaussian'``, or ``'wendland_c2'``. Default is ``'wendland_c2'``.
        bw_multiplier: Kernel bandwidth multiplier. Controls the width of the
            kernel relative to the grid step size. Default is ``1.0``.
        dirichlet_alpha: Per-bin pseudo-count for Dirichlet smoothing. Default is
            ``0.1``. Note: ``dirichlet_kappa`` takes priority over this parameter
            if provided.
        dirichlet_kappa: Total pseudo-count for Dirichlet smoothing. If provided,
            takes priority over ``dirichlet_alpha``. If ``None``, uses
            ``dirichlet_alpha`` instead.

    Returns:
        The mutual information between the distributions of the two specified
        species at their respective time points.
    """
    # All Python operations below happen at trace time (JIT-compatible)
    species_at_t_list = list(species_at_t)
    if len(species_at_t_list) != 2:
        raise ValueError(
            'species_at_t must be an iterable of two (species, time) tuples.'
        )

    (s1_name, t1), (s2_name, t2) = species_at_t_list

    s1_idx = results.species.index(s1_name)
    s2_idx = results.species.index(s2_name)

    # Extract data for the first species at time t1
    if isinstance(t1, int):
        x1 = pytree_to_state(results.x, results.species)[:, t1, s1_idx]
    else:
        x1 = pytree_to_state(results.interpolate(t1).x, results.species)[:, s1_idx]

    # Extract data for the second species at time t2
    if isinstance(t2, int):
        x2 = pytree_to_state(results.x, results.species)[:, t2, s2_idx]
    else:
        # Re-interpolate even if t1==t2 for simplicity and to handle
        # the case where results.interpolate is not memoized.
        x2 = pytree_to_state(results.interpolate(t2).x, results.species)[:, s2_idx]

    return mutual_information(
        x1,
        x2,
        n_grid_points1=n_grid_points1,
        n_grid_points2=n_grid_points2,
        min_max_vals1=min_max_vals1,
        min_max_vals2=min_max_vals2,
        base=base,
        kde_type=kde_type,
        bw_multiplier=bw_multiplier,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_kappa=dirichlet_kappa,
    )
