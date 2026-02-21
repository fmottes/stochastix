"""KDE and histogram functions for simulation results."""

from __future__ import annotations

import typing

import jax
import jax.numpy as jnp

from .._state_utils import pytree_to_state
from .kde_1d import kde_exponential, kde_gaussian, kde_triangular, kde_wendland_c2

if typing.TYPE_CHECKING:
    from .._simulation_results import SimulationResults


def state_kde(
    results: SimulationResults,
    species: str | tuple[str, ...],
    n_grid_points: int | None = None,
    min_max_vals: tuple[float, float] | None = None,
    density: bool = True,
    t: int | float = -1,
    *,
    kde_type: str = 'wendland_c2',
    bw_multiplier: float = 1.0,
    dirichlet_alpha: float | None = 0.1,
    dirichlet_kappa: float | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute a kernel density estimate (KDE) of the state distribution.

    This function is a convenience wrapper around KDE functions that specifically
    operates on the state of batched `SimulationResults`. It supports multiple
    kernel types for density estimation.

    Args:
        results: The `SimulationResults` from a `stochsimsolve` simulation.
            This should contain a batch of simulation trajectories (e.g. from
            vmapping over `stochsimsolve`).
        species: The species for which to compute the KDE. Can be a single
            species name or a tuple of names.
        n_grid_points: Number of grid points to use. If ``None``, inferred from
            the integer span ``[floor(min(x)), ceil(max(x))]``.
        min_max_vals: Tuple ``(min_val, max_val)`` defining the grid range. If
            ``None``, determined from data.
        density: If ``True``, returns a probability density function whose
            Riemann sum over the grid integrates to 1. If ``False``, returns
            unnormalized counts/weights per grid point.
        t: The time point (float) or time index (int) at which to compute the
            KDE. If ``-1`` (default), uses the final time point.
        kde_type: Type of kernel to use. One of ``'triangular'``, ``'exponential'``,
            ``'gaussian'``, or ``'wendland_c2'``. Default is ``'wendland_c2'``.
        bw_multiplier: Kernel bandwidth multiplier. Controls the width of the
            kernel relative to the grid step size. Default is ``1.0``.
        dirichlet_alpha: Per-bin pseudo-count for Dirichlet smoothing. Default is
            ``0.1``. Note: ``dirichlet_kappa`` takes priority over this parameter
            if provided.
        dirichlet_kappa: Total pseudo-count for Dirichlet smoothing. If provided,
            takes priority over ``dirichlet_alpha`` and ``alpha = kappa / K`` where
            K is the number of grid points. If ``None``, uses ``dirichlet_alpha``
            instead.

    Returns:
        A tuple ``(grid, values)`` where:

            - ``grid``: 1D array of evaluation points (grid centers), shape
              ``(n_grid_points,)``.
            - ``values``: 2D array where ``values[:, i]`` is the KDE values for the
              i-th species at the specified time point, shape
              ``(n_grid_points, n_species)``. If ``density=True``, these approximate
              a PDF.
    """
    if isinstance(species, str):
        species = (species,)

    species_indices = jnp.array([results.species.index(s) for s in species])

    if isinstance(t, int):
        t_idx = t
        x = pytree_to_state(results.x, results.species)[:, t_idx, species_indices]
    else:
        results = results.interpolate(t)
        x = pytree_to_state(results.x, results.species)[:, species_indices]

    # Select the appropriate KDE function
    kde_functions = {
        'triangular': kde_triangular,
        'exponential': kde_exponential,
        'gaussian': kde_gaussian,
        'wendland_c2': kde_wendland_c2,
    }
    if kde_type not in kde_functions:
        raise ValueError(
            f'kde_type must be one of {list(kde_functions.keys())}, got {kde_type}'
        )
    kde_func = kde_functions[kde_type]

    # Determine grid parameters based on all species data
    if min_max_vals is None:
        min_val = jnp.floor(jnp.min(x))
        max_val = jnp.ceil(jnp.max(x))
    else:
        min_val, max_val = min_max_vals

    if n_grid_points is None:
        n_grid_points = int(max_val - min_val) + 1 if max_val >= min_val else 1

    # Compute grid once (all species will use the same grid)
    grid = jnp.linspace(float(min_val), float(max_val), int(n_grid_points))

    # Use consistent grid parameters for all species
    grid_params = {
        'n_grid_points': n_grid_points,
        'min_max_vals': (float(min_val), float(max_val)),
        'density': density,
        'bw_multiplier': bw_multiplier,
        'dirichlet_alpha': dirichlet_alpha,
        'dirichlet_kappa': dirichlet_kappa,
    }

    def _get_kde(data):
        return kde_func(data, **grid_params)[1]

    # Vectorize over the species
    values = jax.vmap(_get_kde, in_axes=1, out_axes=1)(x)

    return grid, values
