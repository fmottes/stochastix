"""Mutual information functions."""

from __future__ import annotations

import math
import typing

import jax.numpy as jnp

from .._state_utils import pytree_to_state
from ._utils import normalize_time_scalar
from .kde_2d import kde_2d

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
    dirichlet_alpha: float | None = 0.0,
    dirichlet_kappa: float | None = 0.1,
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
            is ``0.0``. Note: ``dirichlet_kappa`` takes priority over this
            parameter if provided.
        dirichlet_kappa: Total pseudo-count for Dirichlet smoothing. If
            provided, takes priority over ``dirichlet_alpha``. If ``None``, uses
            ``dirichlet_alpha`` instead. Default is ``0.1``.

    Returns:
        The mutual information between `x1` and `x2` in the specified base.
    """
    base = float(base)
    if not math.isfinite(base) or base <= 0.0 or base == 1.0:
        raise ValueError(
            f'base must be finite, positive, and not equal to 1, got {base}'
        )

    valid_kde_types = ('triangular', 'exponential', 'gaussian', 'wendland_c2')
    if kde_type not in valid_kde_types:
        raise ValueError(
            f'kde_type must be one of {list(valid_kde_types)}, got {kde_type}'
        )

    # Raw joint soft-counts only; force KDE smoothing off defensively.
    _, _, counts_xy = kde_2d(
        x1,
        x2,
        n_grid_points1=n_grid_points1,
        n_grid_points2=n_grid_points2,
        min_max_vals1=min_max_vals1,
        min_max_vals2=min_max_vals2,
        density=False,
        bw_multiplier=bw_multiplier,
        kernel=kde_type,
        dirichlet_alpha=0.0,
        dirichlet_kappa=0.0,
    )
    dtype = counts_xy.dtype
    n_eff = jnp.sum(counts_xy)

    def _alpha_eff(n_bins: int) -> jnp.ndarray:
        # Joint-only Dirichlet smoothing parameter (per-bin).
        if dirichlet_kappa is not None:
            alpha = float(dirichlet_kappa) / float(n_bins)
        elif dirichlet_alpha is not None:
            alpha = float(dirichlet_alpha)
        else:
            alpha = 0.0
        return jnp.asarray(max(alpha, 0.0), dtype=dtype)

    # Joint pmf with K=K1*K2-bin smoothing (or none if alpha==0).
    n_bins = counts_xy.size
    alpha = _alpha_eff(n_bins)
    denom = n_eff + alpha * jnp.asarray(n_bins, dtype=dtype)
    q_xy = (counts_xy + alpha) / denom

    # Marginals derived from the same joint pmf -> true KL-based MI.
    q_x = jnp.sum(q_xy, axis=1)
    q_y = jnp.sum(q_xy, axis=0)

    # True mutual information:
    # I = sum_{x,y} q(x,y) * log(q(x,y) / (q(x) q(y)))
    # No masking/flooring: if smoothing is off and zeros occur, log(0) will break.
    log_q_xy = jnp.log(q_xy)
    log_q_x = jnp.log(q_x)
    log_q_y = jnp.log(q_y)
    log_indep = log_q_x[:, None] + log_q_y[None, :]
    mi_nat = jnp.sum(q_xy * (log_q_xy - log_indep))

    # Convert from nats to requested base.
    mi = mi_nat / jnp.log(jnp.asarray(base, dtype=dtype))
    return mi


def state_mutual_info(
    results: SimulationResults,
    species_at_t: typing.Iterable[tuple[str, typing.Any]],
    n_grid_points1: int | None = None,
    n_grid_points2: int | None = None,
    min_max_vals1: tuple[float, float] | None = None,
    min_max_vals2: tuple[float, float] | None = None,
    base: float = 2.0,
    *,
    kde_type: str = 'wendland_c2',
    bw_multiplier: float = 1.0,
    dirichlet_alpha: float | None = 0.0,
    dirichlet_kappa: float | None = 0.1,
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
            of a species name and a scalar physical time point, e.g.,
            `[('S1', t1), ('S2', t2)]`.
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
            ``0.0``. Note: ``dirichlet_kappa`` takes priority over this parameter
            if provided.
        dirichlet_kappa: Total pseudo-count for Dirichlet smoothing. If provided,
            takes priority over ``dirichlet_alpha``. If ``None``, uses
            ``dirichlet_alpha`` instead. Default is ``0.1``.

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

    t1 = normalize_time_scalar(t1, arg_name='species_at_t[0][1]')
    t2 = normalize_time_scalar(t2, arg_name='species_at_t[1][1]')

    x1 = pytree_to_state(results.interpolate(t1).x, results.species)[:, s1_idx]
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
