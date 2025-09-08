"""General utilities for optimization workflows."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def discounted_returns(rewards: jnp.ndarray, GAMMA: float = 0.9) -> jnp.ndarray:
    """Calculate the discounted returns for a sequence of rewards.

    Args:
        rewards: A 1D array of rewards.
        GAMMA: The discount factor.

    Returns:
        A 1D array of discounted returns, where each element represents the
        cumulative discounted return from that point forward.
    """

    def discounting_add(
        carry: jnp.ndarray, reward: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        return GAMMA * carry + reward, GAMMA * carry + reward

    # Reverse the rewards array
    reversed_rewards = jnp.flip(rewards, axis=0)

    # Initialize the carry to be the last element
    _, discounted_returns_reversed = jax.lax.scan(
        discounting_add, 0.0, reversed_rewards
    )
    discounted_returns = jnp.flip(discounted_returns_reversed, axis=0)

    return discounted_returns


def dataloader(
    key: jnp.ndarray,
    arrays: list[jnp.ndarray],
    batch_size: int,
):
    """Create a generator function that yields batches of data.

    This function creates an infinite generator that repeatedly yields batches
    of data from the provided arrays.

    Args:
        key: A JAX random key for shuffling the data.
        arrays: A list of arrays, each with the same size in the first dimension.
        batch_size: The size of each batch.

    Yields:
        A tuple of batched arrays.

    Raises:
        ValueError: If the arrays do not all have the same size in the first
            dimension.
    """
    dataset_size = arrays[0].shape[0]

    if not all(array.shape[0] == dataset_size for array in arrays):
        sizes = [array.shape[0] for array in arrays]
        raise ValueError(
            f'All arrays must have the same first dimension size. Got sizes: {sizes}, expected all to be {dataset_size}'
        )

    indices = jnp.arange(dataset_size)

    while True:
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, indices)
        start = 0
        end = batch_size
        while end <= dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size
