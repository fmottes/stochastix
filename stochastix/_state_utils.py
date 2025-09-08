import typing
from typing import Any

import jax.numpy as jnp
import jax.tree_util as jtu
from equinox import is_array


def add_to_state(
    x: jnp.ndarray,
    delta_x: float | int | jnp.floating | jnp.integer | jnp.ndarray,
    species: None | tuple[str, ...] = None,
) -> jnp.ndarray | Any:
    """Additively applies state updates to a flat state vector or PyTree.

    Updates can be a single number, a PyTree, or a flat array. If a single number, it is added to all species. If a PyTree, must have the same structure as `x`. If a flat array, must have the same length as `x` (if `x` is an array) or the number of species (if `x` is a PyTree).

    Args:
        x: The state vector or PyTree.
        delta_x: The state updates. Must have the same structure as `x` or be a single number.
        species: The species names. If `x` is a PyTree, this must be provided. If `x` is an array, this is ignored.

    Returns:
        The updated state vector or PyTree.
    """
    if is_array(x):
        # fast updates if state is an array
        return x + delta_x
    else:
        # general (slower) updates if state is a PyTree
        if species is None:
            raise ValueError('species must be provided if x is a PyTree')

        # if delta_x is not a single number, convert it to a PyTree like x
        if not isinstance(delta_x, int | float | jnp.floating | jnp.integer):
            delta_x = state_to_pytree(x, species, delta_x)

        _apply = lambda x, dx: x + dx if is_array(x) else x
        return jtu.tree_map(_apply, x, delta_x)


def pytree_to_state(tree: typing.Any, species: tuple[str, ...]) -> jnp.ndarray:
    """Converts a PyTree or other initial state formats to a flat JAX array.

    This function processes an initial state `tree` which can be a dictionary, an
    object with attributes (like a named tuple or an Equinox module), or an
    array-like object, and converts it into a flat JAX array of species counts.
    The order of species in the output array is determined by `species`.

    Args:
        tree: The initial state. Can be a PyTree (dictionary, custom object) with
            leaves named after species, or an array-like object.
        species: The species names, in the order they should appear in the output array.
        dtype: The data type for the output array.

    Returns:
        A 1D JAX array representing the state vector, ordered according to `species`.
    """
    dtype = jnp.result_type(float)

    # Fast-path: array-like provided in species order
    if isinstance(tree, (jnp.ndarray | list | tuple)) and not hasattr(tree, '_fields'):
        arr = jnp.asarray(tree, dtype=dtype)
        if arr.ndim == 0 and len(species) == 1:
            return arr.reshape(1)
        if arr.ndim == 1 and arr.shape[0] != len(species):
            raise ValueError(
                'If tree is an iterable, its length must match the number of species. '
                f'Got tree with shape {arr.shape} for {len(species)} species.'
            )
        if arr.ndim > 1 and arr.shape[-1] != len(species):
            raise ValueError(
                'If tree is an iterable, its last dimension must match the number of species. '
                f'Got tree with shape {arr.shape} for {len(species)} species.'
            )
        return arr

    # Generic PyTree path: collect leaves whose path names match species
    species_set = set(species)
    collected: dict[str, jnp.ndarray] = {}

    def _collect(path, leaf):
        # Determine the name of this leaf from the last path key
        name = None
        if path:
            key = path[-1]
            if isinstance(key, jtu.GetAttrKey):
                name = key.name
            elif isinstance(key, jtu.DictKey):
                name = key.key
        if name in species_set:
            if name in collected:
                raise ValueError(
                    f'Duplicate leaf for species "{name}" found in the provided PyTree.'
                )
            collected[name] = jnp.asarray(leaf, dtype=dtype)
        return None

    # Traverse the tree to populate `collected`. The return value is ignored.
    try:
        jtu.tree_map_with_path(_collect, tree)
    except Exception:
        # Fall back to attribute/dict handling if traversal fails for exotic objects
        collected = {}

    if collected:
        missing = [s for s in species if s not in collected]
        if missing:
            raise ValueError(
                f'Species {", ".join(missing)} not found in the provided PyTree.'
            )
        # Ensure scalars; squeeze 0-d arrays to scalars and stack
        values = [jnp.asarray(collected[s]).squeeze() for s in species]
        return jnp.stack(values, axis=-1)

    # Fallbacks for simple dict or attribute-based objects
    if isinstance(tree, dict):
        missing_species = [s for s in species if s not in tree]
        if missing_species:
            raise ValueError(f'Species {", ".join(missing_species)} not found in tree')
        return jnp.asarray([tree[s] for s in species], dtype=dtype)

    try:
        is_attr_based = bool(species) and all(hasattr(tree, s) for s in species)
    except Exception:
        is_attr_based = False

    if is_attr_based:
        return jnp.asarray([getattr(tree, s) for s in species], dtype=dtype)

    # Last resort: attempt array conversion (will error if shape is wrong)
    arr = jnp.asarray(tree, dtype=dtype)
    if arr.ndim == 0 and len(species) == 1:
        return arr.reshape(1)
    if arr.shape != (..., len(species)):
        raise ValueError(
            'If tree is an iterable, its last dimension must match the number of species. '
            f'Got tree with shape {arr.shape} for {len(species)} species.'
        )
    return arr


def state_to_pytree(
    template: typing.Any,
    species: tuple[str, ...],
    x_trajectory: jnp.ndarray,
) -> typing.Any:
    """Converts a flat state trajectory array back into a PyTree.

    This function reconstructs a PyTree with the same structure as the original
    initial state `template`. For each leaf in `template` that corresponds to a species,
    it replaces the initial value with the full time-series trajectory from
    `x_trajectory`. Leaves that do not correspond to species are preserved.

    If `template` was originally an array-like object, the trajectory is returned as is.

    Args:
        template: The original initial state PyTree, used as a template for the output.
        species: The species names to substitute in the output.
        x_trajectory: A 2D array of shape `(n_timepoints, n_species)` representing
            the state trajectory.

    Returns:
        A PyTree of the same structure as `template`, but with species leaves replaced by their trajectories, or the original `x_trajectory` if `template` was array-like.
    """
    # Fast-path: array-like input was provided initially
    if isinstance(template, (jnp.ndarray | list | tuple | int | float)):
        if not hasattr(template, '_fields'):
            return x_trajectory

    species_to_idx = {s: i for i, s in enumerate(species)}

    def _replace(path, leaf):
        name = None
        if path:
            key = path[-1]
            if isinstance(key, jtu.GetAttrKey):
                name = key.name
            elif isinstance(key, jtu.DictKey):
                name = key.key
        if name in species_to_idx:
            return x_trajectory[..., species_to_idx[name]]
        return leaf

    try:
        return jtu.tree_map_with_path(_replace, template)
    except Exception:
        # If mapping fails (non-standard object), return trajectory as-is
        return x_trajectory
