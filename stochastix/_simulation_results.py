"""Data structures for storing and managing simulation results."""

from __future__ import annotations

import typing
import warnings

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu


class SimulationResults(eqx.Module):
    """Container for simulation results.

    This class is used to store the results of a stochastic simulation. Includes
    utilities for indexing, cleaning, and interpolating the results.

    Attributes:
        x: State trajectory over time, shape (n_timepoints, n_species).
        t: Time points corresponding to state changes.
        propensities: Reaction propensities at each time step.
        reactions: Index of reactions that occurred at each step.
        time_overflow: True if simulation stopped due to reaching max_steps before time T.
        species: Names of the species in the simulation.

    Methods:
        clean: Remove padding steps from the simulation results (returns a new object).
        interpolate: Interpolate the simulation results to new time points (returns a new object).

    Example: Indexing into batched simulation results
        It is pretty common to generate a batch of simulation results, e.g. by doing:

        ```python
        keys = jax.random.split(key, num_simulations)
        results = equinox.filter_vmap(stochsimsolve, in_axes=(0, None, None, None))(keys, network, x0, T)
        ```

        In this case, `results` is a `SimulationResults` object where each field but `species` has an extra first dimension of size `num_simulations`. For convenience, you can directly index into the batch of results (this is *not* standard JAX behavior):

        ```python
        >>> results[0]  # First simulation result
        # SimulationResults(x[0, ...], t[0, ...], propensities[0, ...], reactions[0, ...], time_overflow=[0, ...], species=species)
        >>> results[1:]  # All but the first simulation result
        # SimulationResults(x[1:, ...], t[1:, ...], propensities[1:, ...], reactions[1:, ...], time_overflow=[1, ...], species=species)
        ```

    Example: Removing padding from simulation results
        If simulation time did not overflow, i.e. `max_steps` was not reached,
        `SimulationResults` will contain unused steps at the end of the simulation,
        with `reactions` set to `-1`. You can remove these steps using the `clean` method:

        ```python
        cleaned_results = results.clean()
        ```

    Example: Interpolating simulation results
        You can interpolate the simulation results to new time points using the `interpolate` method. This is especially useful for plotting or calculating statistics over the simulation results that require a regular time grid.

        ```python
        t_interp = jnp.linspace(0, T, 100)
        interpolated_results = results.interpolate(t_interp)
        ```

    """

    x: typing.Any
    t: jnp.ndarray
    propensities: jnp.ndarray
    reactions: jnp.ndarray
    time_overflow: bool
    species: tuple[str, ...]

    def __init__(
        self,
        x: typing.Any,
        t: jnp.ndarray,
        propensities: jnp.ndarray | None = None,
        reactions: jnp.ndarray | None = None,
        time_overflow: bool | None = None,
        species: tuple[str, ...] | None = None,
    ):
        self.x = x
        self.t = t
        self.propensities = propensities
        self.reactions = reactions
        self.time_overflow = time_overflow
        self.species = species

    def __getitem__(self, idx: int | slice | jnp.ndarray) -> SimulationResults:
        """Allows indexing into a batch of simulation results.

        This method enables accessing individual or slices of simulations from a
        batched `SimulationResults` object. For non-batched results, it behaves
        as if indexing a batch of size 1, following the standard JAX convention.

        Args:
            idx: An integer index, slice, or other valid JAX array index.

        Returns:
            SimulationResults: A new container with the subset of results
                corresponding to the given index.
        """
        is_pytree = not isinstance(self.x, jnp.ndarray)
        if is_pytree:
            # Assumes at least one leaf exists
            first_leaf = jtu.tree_leaves(self.x)[0]
            is_batched = first_leaf.ndim > 1
        else:
            is_batched = self.x.ndim > 2

        def _apply_idx(arr, is_scalar=False):
            if arr is None:
                return None

            arr_is_pytree = not isinstance(arr, jnp.ndarray)

            promoted_arr = arr
            if not is_batched:
                if arr_is_pytree:
                    promoted_arr = jtu.tree_map(lambda leaf: leaf[None, ...], arr)
                else:
                    promoted_arr = arr[None] if is_scalar else arr[None, ...]

            if arr_is_pytree:  # is batched, index into pytree
                return jtu.tree_map(lambda leaf: leaf[idx], promoted_arr)

            # else input is batched array, just index into array
            return promoted_arr[idx]

        # time_overflow is a scalar in the non-batched case.
        return SimulationResults(
            x=_apply_idx(self.x),
            t=_apply_idx(self.t),
            propensities=_apply_idx(self.propensities),
            reactions=_apply_idx(self.reactions),
            time_overflow=_apply_idx(self.time_overflow, is_scalar=True),
            species=self.species,
        )

    def clean(self) -> SimulationResults:
        """Cleans simulation results by removing padded, unused steps.

        Note:
            Does not work with batched simulations.

        Stochastic simulations pre-allocate arrays of a fixed size for JIT
        compilation. This function removes the trailing steps that were allocated
        but not used because the simulation terminated early (e.g., reached the
        final time or ran out of reactants).

        It identifies unused steps by checking for negative reaction indices, which
        are used as padding values by the solvers. It correctly handles results
        from both exact solvers (where `reactions` is 1D) and approximate solvers
        (where `reactions` is 2D).

        Returns:
            SimulationResults: A new container with only the valid steps of the simulation trajectory.

        Raises:
            NotImplementedError: If called on batched simulation results.
        """
        if self.reactions is None:
            warnings.warn(
                'This object does not contain reaction information, returning original object.'
            )
            return self

        is_pytree = not isinstance(self.x, jnp.ndarray)
        if is_pytree:
            # Assumes at least one leaf exists
            first_leaf = jtu.tree_leaves(self.x)[0]
            is_batched = first_leaf.ndim > 1
        else:
            is_batched = self.x.ndim > 2

        if is_batched:
            raise NotImplementedError(
                'The `clean` method does not currently support batched simulation results '
                'because trajectories may have different lengths. To clean a specific '
                'trajectory from a batch, first select it using indexing. For example: '
                '`cleaned_result = batched_results[0].clean()`'
            )
        # Handle both exact solvers (1D reactions) and approximate solvers (2D reactions)
        if self.reactions.ndim == 1:
            # Exact solver: reactions is 1D array of reaction indices
            mask = self.reactions >= 0
        else:
            # Approximate solver (e.g., TauLeapingSolver): reactions is 2D array of reaction counts
            # Keep steps where at least one reaction occurred (any count >= 0)
            mask = jnp.any(self.reactions >= 0, axis=-1)

        if is_pytree:
            clean_leaf = lambda leaf: jnp.hstack([leaf[0], leaf[1:][mask]])
            x_cleaned = jtu.tree_map(clean_leaf, self.x)
        else:
            x_cleaned = jnp.vstack([self.x[0], self.x[1:][mask]])

        t = jnp.hstack([self.t[0], self.t[1:][mask]])
        a = self.propensities[mask]
        r = self.reactions[mask]

        return SimulationResults(x_cleaned, t, a, r, self.time_overflow, self.species)

    def interpolate(self, t_interp: jnp.ndarray) -> SimulationResults:
        """Interpolates stochastic simulation results to new time points.

        This function performs a piecewise-constant (or "forward-fill") interpolation
        of the simulation's state trajectory `x`. The state `x[i]` is considered
        constant over the time interval `[t[i], t[i+1])`.

        Args:
            t_interp: A sorted array of time points at which to interpolate
                the state. Values outside the simulation time range [self.t[0], self.t[-1]]
                will be automatically clamped to the nearest boundary.

        Returns:
            SimulationResults: A new container with the interpolated state trajectory
                at the specified time points. The `reactions` and `propensities` fields
                are set to None, as they are not meaningful at arbitrary time points.
                `time_overflow` is also set to None, since interpolations do not exceed
                the simulation time range.

        Note:
            This method is JIT-compatible. Time points outside the simulation range
            are automatically clamped to `[self.t[0], self.t[-1]]` to prevent
            out-of-bounds errors. Input validation should be performed before calling
            this method if strict bounds checking is required.
        """

        def _interpolate(results, t_interp):
            # Find the index of the last event for each interpolation time point (forward-fill).
            # Clamp t_interp to valid range to ensure indices are within bounds
            t_clamped = jnp.clip(t_interp, results.t[0], results.t[-1])
            indices = jnp.searchsorted(results.t, t_clamped, side='right') - 1

            is_pytree_local = not isinstance(results.x, jnp.ndarray)
            if is_pytree_local:
                x_interp = jtu.tree_map(lambda leaf: leaf[indices], results.x)
            else:
                x_interp = results.x[indices]

            return SimulationResults(
                x=x_interp,
                t=t_interp,
                propensities=None,
                reactions=None,
                time_overflow=None,
                species=results.species,
            )

        is_pytree = not isinstance(self.x, jnp.ndarray)
        if is_pytree:
            first_leaf = jtu.tree_leaves(self.x)[0]
            is_batched = first_leaf.ndim > 1
        else:
            is_batched = self.x.ndim > 2

        if is_batched:
            # vmap over batch dimension of self, t_interp is broadcast
            vmap_interp = eqx.filter_vmap(
                _interpolate,
                in_axes=(eqx.if_array(0), None),
                out_axes=eqx.if_array(0),
            )
            new_results = vmap_interp(self, t_interp)
        else:
            new_results = _interpolate(self, t_interp)

        return new_results
