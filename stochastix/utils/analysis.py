"""Data analysis functions for simulation results."""

from __future__ import annotations

import typing

import equinox as eqx
import jax
import jax.numpy as jnp

from .._state_utils import pytree_to_state
from ._utils import entropy

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


def differentiable_histogram(
    x: jnp.ndarray,
    n_bins: int | None = None,
    min_max_vals: tuple[float, float] | None = None,
    density: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Differentiable histogram with a triangular kernel.

    This function uses a soft-binning approach with a triangular kernel to
    create a differentiable histogram. This is particularly useful for
    gradient-based optimization of parameters that affect the distribution of
    the input data.

    For integer inputs, this function behaves like a standard categorical
    histogram, but remains differentiable with respect to its inputs.

    Note: JIT-compatibility.
        To make this function JIT-compatible, you must provide concrete values for
        all binning parameters (`n_bins`, `min_max_vals`).
        If left as `None`, bin parameters are determined from the data, which is
        not a JIT-able operation.

    Args:
        x: 1D array of the input data to compute the histogram of. If not 1D, will be flattened.
        n_bins: The number of bins to use. If None, determined automatically from data range.
        min_max_vals: Tuple of (min_val, max_val) for bin range. If None, determined
            automatically from data.
        density: Whether to return the density or the counts.

    Returns:
        A tuple `(bins, probabilities)` where:
            - `bins`: The center of the histogram bins.
            - `probabilities`: A 1D array of the probability mass of the data.
    """
    x = x.flatten()

    # Determine bin parameters
    if min_max_vals is None:
        # For discrete counts, bins should be integers.
        # This must be done outside JIT to avoid errors with traced values.
        min_val = jnp.floor(jnp.min(x))
        max_val = jnp.ceil(jnp.max(x))
    else:
        min_val, max_val = min_max_vals

    if n_bins is None:
        n_bins = int(max_val - min_val) + 1 if max_val >= min_val else 1

    bins = jnp.linspace(min_val, max_val, n_bins)

    @eqx.filter_jit
    def _probs(x, bins):
        bin_width = jnp.where(bins.shape[0] > 1, bins[1] - bins[0], 1.0)
        bin_width = jnp.maximum(bin_width, 1e-6)

        # Soft binning using a triangular kernel
        # Broadcast data and bins to compute pairwise distances
        x_b = jnp.expand_dims(x, -1)
        bins_b = jnp.expand_dims(bins, 0)

        # Compute normalized distance to each bin center.
        dist = jnp.abs(x_b - bins_b) / bin_width

        # Triangular kernel: max(0, 1 - dist)
        kernel_vals = jnp.maximum(0, 1 - dist)

        # Sum contributions for each bin and normalize
        probability = jnp.sum(kernel_vals, axis=0)

        if density:
            probability = probability / jnp.sum(probability)

        return probability

    probabilities = _probs(x, bins)

    return bins, probabilities


def differentiable_histogram2d(
    x1: jnp.ndarray,
    x2: jnp.ndarray,
    n_bins1: int | None = None,
    n_bins2: int | None = None,
    min_max_vals1: tuple[float, float] | None = None,
    min_max_vals2: tuple[float, float] | None = None,
    density: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Differentiable 2D histogram with a triangular kernel.

    This function uses a soft-binning approach with a product of two 1D
    triangular kernels to create a differentiable 2D histogram. This is useful
    for gradient-based optimization of parameters that affect the joint
    distribution of the input data.

    Note: JIT-compatibility.
        To make this function JIT-compatible, you must provide concrete values for
        all binning parameters (`n_bins1`, `n_bins2`, `min_max_vals1`, `min_max_vals2`).
        If left as `None`, bin parameters are determined from the data, which is
        not a JIT-able operation.

    Args:
        x1: 1D array of the first input data. If not 1D, will be flattened.
        x2: 1D array of the second input data. If not 1D, will be flattened.
            Must have the same length as x1.
        n_bins1: Number of bins for the first dimension. If None, determined automatically.
        n_bins2: Number of bins for the second dimension. If None, determined automatically.
        min_max_vals1: Tuple of (min, max) for the first dimension's bin range.
            If None, determined automatically.
        min_max_vals2: Tuple of (min, max) for the second dimension's bin range.
            If None, determined automatically.
        density: Whether to return the density or the counts.

    Returns:
        A tuple `(bins1, bins2, probabilities)` where:
            - `bins1`: The center of the histogram bins for the first dimension.
            - `bins2`: The center of the histogram bins for the second dimension.
            - `probabilities`: A 2D array of the probability mass of the data.
    """
    x1 = x1.flatten()
    x2 = x2.flatten()

    if x1.shape[0] != x2.shape[0]:
        raise ValueError('x1 and x2 must have the same length.')

    # Determine bin parameters
    if min_max_vals1 is None:
        min_val1, max_val1 = jnp.floor(jnp.min(x1)), jnp.ceil(jnp.max(x1))
    else:
        min_val1, max_val1 = min_max_vals1

    if min_max_vals2 is None:
        min_val2, max_val2 = jnp.floor(jnp.min(x2)), jnp.ceil(jnp.max(x2))
    else:
        min_val2, max_val2 = min_max_vals2

    if n_bins1 is None:
        n_bins1 = int(max_val1 - min_val1) + 1 if max_val1 >= min_val1 else 1
    if n_bins2 is None:
        n_bins2 = int(max_val2 - min_val2) + 1 if max_val2 >= min_val2 else 1

    bins1 = jnp.linspace(min_val1, max_val1, n_bins1)
    bins2 = jnp.linspace(min_val2, max_val2, n_bins2)

    @eqx.filter_jit
    def _probs(x1, x2, bins1, bins2):
        # Calculate 1D kernel values for each dimension
        bin_width1 = jnp.where(bins1.shape[0] > 1, bins1[1] - bins1[0], 1.0)
        bin_width1 = jnp.maximum(bin_width1, 1e-6)
        dist1 = (
            jnp.abs(jnp.expand_dims(x1, -1) - jnp.expand_dims(bins1, 0)) / bin_width1
        )
        kernel_vals1 = jnp.maximum(0, 1 - dist1)

        bin_width2 = jnp.where(bins2.shape[0] > 1, bins2[1] - bins2[0], 1.0)
        bin_width2 = jnp.maximum(bin_width2, 1e-6)
        dist2 = (
            jnp.abs(jnp.expand_dims(x2, -1) - jnp.expand_dims(bins2, 0)) / bin_width2
        )
        kernel_vals2 = jnp.maximum(0, 1 - dist2)

        # Combine kernels and sum contributions
        kernel_vals1_b = jnp.expand_dims(kernel_vals1, 2)  # (n_points, n_bins1, 1)
        kernel_vals2_b = jnp.expand_dims(kernel_vals2, 1)  # (n_points, 1, n_bins2)

        # Product gives contribution of each point to each 2D bin
        joint_kernel = kernel_vals1_b * kernel_vals2_b  # (n_points, n_bins1, n_bins2)

        # Sum over all points
        probability = jnp.sum(joint_kernel, axis=0)

        if density:
            probability = probability / jnp.sum(probability)

        return probability

    probabilities = _probs(x1, x2, bins1, bins2)

    return bins1, bins2, probabilities


def mutual_information(
    x1: jnp.ndarray,
    x2: jnp.ndarray,
    n_bins1: int | None = None,
    n_bins2: int | None = None,
    min_max_vals1: tuple[float, float] | None = None,
    min_max_vals2: tuple[float, float] | None = None,
    base: float = 2.0,
) -> jnp.ndarray:
    """Compute the mutual information between two arrays.

    This function uses the differentiable histogram functions to compute the
    mutual information between two arrays `x1` and `x2`. The mutual
    information is a measure of the mutual dependence between the two variables.

    Note: JIT-compatibility.
        To make this function JIT-compatible, you must provide concrete values for
        all binning parameters (`n_bins1`, `n_bins2`, `min_max_vals1`, `min_max_vals2`).
        If left as `None`, bin parameters are determined from the data, which is
        not a JIT-able operation.

    Args:
        x1: 1D array of the first input data.
        x2: 1D array of the second input data.
        n_bins1: Number of bins for the first dimension. If None, determined automatically.
        n_bins2: Number of bins for the second dimension. If None, determined automatically.
        min_max_vals1: Tuple of (min, max) for the first dimension's bin range.
            If None, determined automatically.
        min_max_vals2: Tuple of (min, max) for the second dimension's bin range.
            If None, determined automatically.
        base: The logarithmic base to use for the entropy calculation.

    Returns:
        The mutual information between `x1` and `x2`.
    """
    # p(x1)
    _, p_x1 = differentiable_histogram(
        x1, n_bins=n_bins1, min_max_vals=min_max_vals1, density=True
    )
    # p(x2)
    _, p_x2 = differentiable_histogram(
        x2, n_bins=n_bins2, min_max_vals=min_max_vals2, density=True
    )
    # p(x1, x2)
    _, _, p_x1_x2 = differentiable_histogram2d(
        x1,
        x2,
        n_bins1=n_bins1,
        n_bins2=n_bins2,
        min_max_vals1=min_max_vals1,
        min_max_vals2=min_max_vals2,
        density=True,
    )

    # H(X) = - sum(p(x) * log(p(x)))
    h_x1 = entropy(p_x1, base=base)
    # H(Y) = - sum(p(y) * log(p(y)))
    h_x2 = entropy(p_x2, base=base)
    # H(X, Y) = - sum(p(x, y) * log(p(x, y)))
    h_x1_x2 = entropy(p_x1_x2.flatten(), base=base)

    # I(X;Y) = H(X) + H(Y) - H(X,Y)
    mi = h_x1 + h_x2 - h_x1_x2
    return mi


def differentiable_state_histogram(
    results: SimulationResults,
    species: str | tuple[str, ...],
    n_bins: int | None = None,
    min_max_vals: tuple[float, float] | None = None,
    density: bool = True,
    t: int | float = -1,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute a differentiable histogram of the final state distribution.

    This function is a convenience wrapper around `differentiable_histogram`
    that specifically operates on the final state of batched `SimulationResults`.

    Args:
        results: The `SimulationResults` from a `stochsimsolve` simulation.
            This should contain a batch of simulation trajectories (e.g. from
            vmapping over `stochsimsolve`).
        species: The species for which to compute the histogram. Can be a single
            species name or a tuple of names.
        n_bins: Number of bins to use. If None, determined automatically from data range.
        min_max_vals: Tuple of (min_val, max_val) for bin range. If None, determined
            automatically from data (not JIT-able).
        density: Return normalized (True) or unnormalized probabilities (False).
        t: The time point or time index at which to compute the histogram. If None, the final time point is used.

    Returns:
        A tuple `(bins, probabilities)` where:
            - `bins`: The center of the histogram bins.
            - `probabilities`: A 2D array where `probabilities[:, i]` is the
              probability mass of the i-th species at the final time point.
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

    def _get_histogram(data):
        return differentiable_histogram(data, n_bins, min_max_vals, density)[1]

    # Vectorize over the species
    probabilities = jax.vmap(_get_histogram, in_axes=1, out_axes=1)(x)

    # Determine bin parameters for the output
    if min_max_vals is None:
        min_val = jnp.floor(jnp.min(x))
        max_val = jnp.ceil(jnp.max(x))
    else:
        min_val, max_val = min_max_vals

    if n_bins is None:
        n_bins = int(max_val - min_val) + 1 if max_val >= min_val else 1

    bins = jnp.linspace(min_val, max_val, n_bins)
    return bins, probabilities


def differentiable_state_mi(
    results: SimulationResults,
    species_at_t: typing.Iterable[tuple[str, int | float]],
    n_bins1: int | None = None,
    n_bins2: int | None = None,
    min_max_vals1: tuple[float, float] | None = None,
    min_max_vals2: tuple[float, float] | None = None,
    base: float = 2.0,
) -> jnp.ndarray:
    """Compute the mutual information between two species at specific time points.

    This function calculates the mutual information between the distributions of
    two species at two potentially different time points, `t1` and `t2`, from
    batched simulation results. It leverages differentiable histograms to ensure
    the entire computation is end-to-end differentiable, which is useful for
    gradient-based optimization of simulation parameters.

    Note: JIT-compatibility.
        To make this function JIT-compatible, you must provide concrete values for
        all binning parameters (`n_bins1`, `n_bins2`, `min_max_vals1`, `min_max_vals2`).
        If left as `None`, bin parameters are determined from the data, which is
        not a JIT-able operation.

    Args:
        results: The `SimulationResults` from a `stochsimsolve` simulation.
            This should contain a batch of simulation trajectories (e.g., from
            vmapping over `stochsimsolve`).
        species_at_t: An iterable containing two tuples, where each tuple consists
            of a species name and a time point, e.g., `[('S1', t1), ('S2', t2)]`.
            The time point can be an integer index or a float time value.
        n_bins1: Number of bins for the first species' histogram. If None,
            determined automatically (not JIT-compatible).
        n_bins2: Number of bins for the second species' histogram. If None,
            determined automatically (not JIT-compatible).
        min_max_vals1: Tuple of (min, max) for the first species' bin range. If
            None, determined automatically (not JIT-compatible).
        min_max_vals2: Tuple of (min, max) for the second species' bin range. If
            None, determined automatically (not JIT-compatible).
        base: The logarithmic base for the entropy calculation.

    Returns:
        The mutual information between the distributions of the two specified
        species at their respective time points.
    """
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
        n_bins1=n_bins1,
        n_bins2=n_bins2,
        min_max_vals1=min_max_vals1,
        min_max_vals2=min_max_vals2,
        base=base,
    )
