"""Data analysis functions for simulation results."""

from __future__ import annotations

import typing

import equinox as eqx
import jax
import jax.numpy as jnp

from .._state_utils import pytree_to_state

if typing.TYPE_CHECKING:
    from .._simulation_results import SimulationResults


@eqx.filter_jit
def _autocorr_1d(signal):
    """JIT-compiled 1D autocorrelation."""
    n = signal.shape[0]
    signal_norm = signal - jnp.mean(signal)
    corr = jnp.correlate(signal_norm, signal_norm, mode='full')
    corr_positive_lags = corr[n - 1 :]
    norm = jnp.sum(signal_norm**2)
    norm = jnp.where(norm == 0, 1.0, norm)
    return corr_positive_lags / norm


def autocorrelation(
    results: SimulationResults,
    n_points: int = 1000,
    species: str | tuple[str, ...] = '*',
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the autocorrelation of species trajectories.

    This function first interpolates the simulation data onto a regular time
    grid. Then, it calculates the normalized autocorrelation for each specified
    species' trajectory. The core computation is JIT-compiled for performance.

    Args:
        results: The `SimulationResults` object from a `stochsimsolve` simulation.
        n_points: The number of points for the interpolation grid.
        species: The species for which to compute the autocorrelation. Can be
            "*" for all species, or a string or tuple of strings for specific species.

    Returns:
        Tuple `(lags, autocorrs)` where:
            - `lags`: The time lags for the autocorrelation, in the same units as
              the simulation time.
            - `autocorrs`: A 2D array where `autocorrs[:, i]` is the
              autocorrelation of the i-th species.
    """
    t_start = results.t[0]
    t_end = results.t[-1]
    dt = (t_end - t_start) / (n_points - 1)
    t_interp = jnp.linspace(t_start, t_end, n_points)

    # Interpolate the state trajectory to the new time grid.
    x = pytree_to_state(results.interpolate(t_interp).x, results.species)

    if species != '*':
        if isinstance(species, str):
            species = (species,)
        species_indices = [results.species.index(s) for s in species]
        x = x[:, jnp.array(species_indices)]

    autocorrs = jax.vmap(_autocorr_1d, in_axes=1, out_axes=1)(x)
    lags = jnp.arange(autocorrs.shape[0]) * dt
    return lags, autocorrs


@eqx.filter_jit
def _cross_corr_1d(x1, x2):
    """JIT-compiled 1D cross-correlation."""
    n_timesteps = x1.shape[0]
    x1_norm = x1 - jnp.mean(x1)
    x2_norm = x2 - jnp.mean(x2)

    cross_corr_raw = jnp.correlate(x1_norm, x2_norm, mode='full')

    # More robust normalization
    norm1_sq = jnp.sum(x1_norm**2)
    norm2_sq = jnp.sum(x2_norm**2)
    norm = jnp.sqrt(norm1_sq * norm2_sq)
    norm = jnp.where(norm == 0, 1.0, norm)

    return cross_corr_raw / norm, n_timesteps


def cross_correlation(
    results: SimulationResults,
    species1: str,
    species2: str,
    n_points: int = 1000,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the cross-correlation between two species trajectories.

    This function interpolates the simulation data onto a regular time grid and
    then computes the normalized cross-correlation between two specified species.

    Args:
        results: The `SimulationResults` object from a `stochsimsolve` simulation.
        species1: The name of the first species.
        species2: The name of the second species.
        n_points: The number of points for the interpolation grid.

    Returns:
        A tuple `(lags, cross_corr)` where:
            - `lags`: The time lags for the cross-correlation.
            - `cross_corr`: A 1D array of the cross-correlation values.
    """
    t_start = results.t[0]
    t_end = results.t[-1]
    dt = (t_end - t_start) / (n_points - 1)
    t_interp = jnp.linspace(t_start, t_end, n_points)

    # Interpolate the state trajectory to the new time grid.
    x = pytree_to_state(results.interpolate(t_interp).x, results.species)

    idx1 = results.species.index(species1)
    idx2 = results.species.index(species2)
    x1 = x[:, idx1]
    x2 = x[:, idx2]

    cross_corr, n_timesteps = _cross_corr_1d(x1, x2)
    lags = (jnp.arange(cross_corr.shape[0]) - (n_timesteps - 1)) * dt
    return lags, cross_corr
