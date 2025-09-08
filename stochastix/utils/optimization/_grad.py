"""Gradient estimation utilities (finite differences, SPSA)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from equinox import is_array


def gradfd(
    func: Callable[..., jnp.ndarray], epsilon: float = 1e-5
) -> Callable[..., Any]:
    """Compute the gradient of a function using central first order finite differences.

    NOTE: only supports differentiation with respect to the first argument of `func`.

    Args:
        func: The function to differentiate. Should take a PyTree of arrays
            as input and return a scalar.
        epsilon: The step size to use for finite differences.

    Returns:
        A function that takes the same arguments as `func` and returns the gradient of `func` at that point as a PyTree with the same structure as the input.
    """

    def _central_difference_element(treedef, model_leaves, j, x, i, *args):
        def eval_f(x_mod):
            model = jax.tree.unflatten(
                treedef, model_leaves[:j] + [x_mod] + model_leaves[j + 1 :]
            )
            return func(model, *args)

        multi_dim_index = jnp.unravel_index(i, x.shape)
        x_plus = x.at[multi_dim_index].add(epsilon)
        x_minus = x.at[multi_dim_index].add(-epsilon)
        return (eval_f(x_plus) - eval_f(x_minus)) / (2 * epsilon)

    def _grad_fn(model, *args):
        model_leaves, treedef = jax.tree.flatten(model)

        def process_arg(leaf, j):
            if is_array(leaf) and jnp.issubdtype(leaf.dtype, jnp.inexact):
                grad_arg = jax.vmap(
                    lambda i: _central_difference_element(
                        treedef, model_leaves, j, leaf, i, *args
                    )
                )(jnp.arange(leaf.size))

                grad_arg = grad_arg.reshape(leaf.shape)
            else:
                grad_arg = None

            return grad_arg

        flat_grad = [process_arg(leaf, j) for j, leaf in enumerate(model_leaves)]
        return jax.tree.unflatten(treedef, flat_grad)

    return _grad_fn


def gradspsa(
    func: Callable[..., jnp.ndarray],
    epsilon: float = 1e-3,
    num_samples: int = 20,
    *,
    split_first_arg_key: bool = True,
) -> Callable[..., Any]:
    """Compute the gradient of a function using SPSA (Simultaneous Perturbation Stochastic Approximation).

    NOTE: only supports differentiation with respect to the first argument of `func`.

    Args:
        func: The function to differentiate. Should take a PyTree of arrays
            as input and return a scalar.
        epsilon: The perturbation size to use for SPSA.
        num_samples: Number of SPSA samples to average over.
        split_first_arg_key: If True, assumes the first positional argument passed
            to the returned gradient function is a PRNGKey and splits it into
            `num_samples` independent keys, using one per SPSA sample. The same
            per-sample key is used for both + and - perturbations (common random
            numbers) to reduce variance. If False, the exact same arguments are
            used for all SPSA samples.

    Returns:
        A function that takes the same arguments as `func` (plus a PRNGKey) and returns the gradient of `func` at that point as a PyTree with the same structure as the input.
    """

    def _grad_fn(model, *args, key: jnp.ndarray):
        model_leaves, treedef = jax.tree.flatten(model)

        def get_grad_sample(subkey, *sample_args):
            # Generate delta with the same structure as the model
            tree_key, _ = jax.random.split(subkey)
            delta = jax.tree.unflatten(
                treedef,
                [
                    (
                        jax.random.rademacher(k, shape=leaf.shape, dtype=leaf.dtype)
                        if is_array(leaf) and jnp.issubdtype(leaf.dtype, jnp.inexact)
                        else None
                    )
                    for leaf, k in zip(
                        model_leaves, jax.random.split(tree_key, len(model_leaves))
                    )
                ],
            )

            # Perturb model simultaneously in all directions
            model_plus = jax.tree.map(
                lambda m, d: m + epsilon * d if d is not None else m, model, delta
            )
            model_minus = jax.tree.map(
                lambda m, d: m - epsilon * d if d is not None else m, model, delta
            )

            # Evaluate function at perturbed points
            y_plus = func(model_plus, *sample_args)
            y_minus = func(model_minus, *sample_args)

            # Compute gradient estimate
            diff = y_plus - y_minus
            grad_est = jax.tree.map(
                lambda d: diff / (2 * epsilon * d) if d is not None else None, delta
            )
            return grad_est

        # Average gradients over a number of samples
        keys = jax.random.split(key, num_samples)

        if split_first_arg_key and len(args) >= 1 and is_array(args[0]):
            # Split the first arg (assumed PRNGKey) across samples for lower-variance SPSA.
            sim_keys = jax.random.split(args[0], num_samples)
            rest = args[1:]

            def vmapped_one(k_delta, k_sim):
                return get_grad_sample(k_delta, k_sim, *rest)

            grad_samples = jax.vmap(vmapped_one)(keys, sim_keys)
        else:
            # Reuse identical arguments for all samples
            grad_samples = jax.vmap(lambda k_delta: get_grad_sample(k_delta, *args))(
                keys
            )
        # Reduce over the sample axis for array leaves; keep None leaves as None
        grad = jax.tree.map(
            lambda x: None if x is None else jnp.mean(x, axis=0), grad_samples
        )

        return grad

    return _grad_fn
