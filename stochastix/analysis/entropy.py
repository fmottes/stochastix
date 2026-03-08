"""Entropy functions."""

from __future__ import annotations

import math
import typing

import jax.numpy as jnp

from .._state_utils import pytree_to_state
from ._utils import normalize_time_scalar
from .kde_1d import kde

if typing.TYPE_CHECKING:
    from .._simulation_results import SimulationResults


def entropy(
    x: jnp.ndarray,
    n_grid_points: int | None = None,
    min_max_vals: tuple[float, float] | None = None,
    base: float = 2.0,
    *,
    kde_type: str = 'wendland_c2',
    bw_multiplier: float = 1.0,
    dirichlet_alpha: float | None = 0.0,
    dirichlet_kappa: float | None = 0.1,
) -> jnp.ndarray:
    """Compute robust entropy of a 1D sample distribution.

    This function computes entropy on grid cell masses induced by a 1D KDE soft
    histogram. It uses raw soft counts (`density=False`) and then applies
    optional Dirichlet smoothing to the implied pmf before entropy evaluation.

    Note: JIT-compatibility
        For JIT-compatibility, provide concrete values for all grid parameters
        (`n_grid_points`, `min_max_vals`). If left as ``None``, grid parameters
        are determined from the data (not JIT-able).

    Note: Taking entropy gradients
        For gradient-based optimization, using Dirichlet smoothing
        (``dirichlet_alpha > 0`` or ``dirichlet_kappa > 0``) is generally
        preferred and is the safest choice for stable autodiff. With no
        smoothing (both set to ``0``), entropy can remain finite, but gradients
        are piecewise and may be numerically fragile near zero-mass bins.

    Args:
        x: 1D sample array.
        n_grid_points: Number of grid points. If ``None``, inferred from data.
        min_max_vals: Tuple ``(min_val, max_val)`` for grid range. If ``None``,
            inferred from data.
        base: Logarithmic base for entropy. Default is ``2.0`` (bits).
        kde_type: Type of kernel to use. One of ``'triangular'``,
            ``'exponential'``, ``'gaussian'``, or ``'wendland_c2'``. Default is
            ``'wendland_c2'``.
        bw_multiplier: Kernel bandwidth multiplier. Default is ``1.0``.
        dirichlet_alpha: Per-bin pseudo-count for Dirichlet smoothing. Default
            is ``0.0``. Note: ``dirichlet_kappa`` takes priority over this
            parameter if provided.
        dirichlet_kappa: Total pseudo-count for Dirichlet smoothing. If
            provided, takes priority over ``dirichlet_alpha``. If ``None``, uses
            ``dirichlet_alpha`` instead. Default is ``0.1``.

    Returns:
        Entropy of the smoothed grid cell-mass distribution in the requested
        logarithmic base.
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

    # Raw soft-counts only; force KDE smoothing off defensively.
    _, counts_x = kde(
        x,
        n_grid_points=n_grid_points,
        min_max_vals=min_max_vals,
        density=False,
        bw_multiplier=bw_multiplier,
        kernel=kde_type,
        dirichlet_alpha=0.0,
        dirichlet_kappa=0.0,
    )
    dtype = counts_x.dtype
    n_eff = jnp.sum(counts_x)

    def _alpha_eff(n_bins: int) -> jnp.ndarray:
        if dirichlet_kappa is not None:
            alpha = float(dirichlet_kappa) / float(n_bins)
        elif dirichlet_alpha is not None:
            alpha = float(dirichlet_alpha)
        else:
            alpha = 0.0
        return jnp.asarray(max(alpha, 0.0), dtype=dtype)

    n_bins = counts_x.size
    alpha = _alpha_eff(n_bins)
    denom = n_eff + alpha * jnp.asarray(n_bins, dtype=dtype)
    q_x = (counts_x + alpha) / denom

    log_q_x = jnp.log(q_x)
    support = q_x > 0
    h_nat = -jnp.sum(jnp.where(support, q_x * log_q_x, 0.0))

    h = h_nat / jnp.log(jnp.asarray(base, dtype=dtype))
    return h


def state_entropy(
    results: SimulationResults,
    species_at_t: tuple[str, typing.Any],
    n_grid_points: int | None = None,
    min_max_vals: tuple[float, float] | None = None,
    base: float = 2.0,
    *,
    kde_type: str = 'wendland_c2',
    bw_multiplier: float = 1.0,
    dirichlet_alpha: float | None = 0.0,
    dirichlet_kappa: float | None = 0.1,
) -> jnp.ndarray:
    """Compute entropy of one species at a specific time point.

    This function computes entropy of a species distribution at a selected
    physical time point from batched simulation results.

    Note: JIT-compatibility
        For JIT-compatibility, provide concrete values for all grid parameters
        (`n_grid_points`, `min_max_vals`). If left as ``None``, grid parameters
        are determined from the data (not JIT-able).

    Note: Taking entropy gradients
        For gradient-based optimization, using Dirichlet smoothing
        (``dirichlet_alpha > 0`` or ``dirichlet_kappa > 0``) is generally
        preferred and is the safest choice for stable autodiff. With no
        smoothing (both set to ``0``), entropy can remain finite, but gradients
        are piecewise and may be numerically fragile near zero-mass bins.

    Args:
        results: Batched `SimulationResults` from simulation.
        species_at_t: Tuple ``(species_name, t)`` where ``t`` is a scalar
            physical time.
        n_grid_points: Number of grid points. If ``None``, inferred from data.
        min_max_vals: Tuple ``(min_val, max_val)`` for grid range. If ``None``,
            inferred from data.
        base: Logarithmic base for entropy. Default is ``2.0`` (bits).
        kde_type: Type of kernel to use. One of ``'triangular'``,
            ``'exponential'``, ``'gaussian'``, or ``'wendland_c2'``. Default is
            ``'wendland_c2'``.
        bw_multiplier: Kernel bandwidth multiplier. Default is ``1.0``.
        dirichlet_alpha: Per-bin pseudo-count for Dirichlet smoothing.
        dirichlet_kappa: Total pseudo-count for Dirichlet smoothing.

    Returns:
        Entropy of the selected species at the requested time point.
    """
    if len(species_at_t) != 2:
        raise ValueError('species_at_t must be a tuple of (species_name, scalar_time).')
    species_name, t = species_at_t
    species_idx = results.species.index(species_name)
    t = normalize_time_scalar(t, arg_name='species_at_t[1]')

    x = pytree_to_state(results.interpolate(t).x, results.species)[:, species_idx]
    return entropy(
        x,
        n_grid_points=n_grid_points,
        min_max_vals=min_max_vals,
        base=base,
        kde_type=kde_type,
        bw_multiplier=bw_multiplier,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_kappa=dirichlet_kappa,
    )
