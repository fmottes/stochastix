"""Neural network-based kinetics for data-driven rate modeling."""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from ..utils.nn import MultiLayerPerceptron
from ._base import AbstractKinetics


class MLP(AbstractKinetics):
    """Multi-Layer Perceptron kinetics for data-driven rate modeling.

    This class allows for flexible, data-driven modeling of reaction rates where
    the functional form is unknown and can be learned from data. The MLP takes
    species concentrations/counts as input and outputs a reaction rate.

    Note: `input_species="*"` triggers deferred initialization.
        Supports deferred initialization when `input_species` is `"*"`. This creates
        the concrete MLP architecture upon initialization of the ReactionNetwork it is a part of.
        This allows the MLP to be created when the full list of species in the network is known.

    Attributes:
        mlp: The Multi-Layer Perceptron model for rate computation.
        _input_species_idx: Tuple of species indices used as MLP inputs.
        _hidden_sizes: Tuple of integers specifying the number of neurons in
            each hidden layer.
        _activation: The activation function for the hidden layers.
        _final_activation: The activation function for the output layer. This should
            ensure the rate is always non-negative (e.g., `jax.nn.softplus`).
        _weight_init: The weight initialization function for the MLP.
        _bias_init: The bias initialization function for the MLP.
        _mlp_init_key: A `jax.random.PRNGKey` for the random initialization of MLP weights.

    Methods:
        propensity_fn: Calculates the propensity using the MLP.
        ode_rate_fn: Calculates the ODE rate using the MLP.
    """

    _input_species_idx: tuple[int, ...] | None
    mlp: MultiLayerPerceptron | None

    # Hyperparameters for deferred MLP creation
    _hidden_sizes: tuple[int, ...] = eqx.field(static=True)
    _activation: Callable = eqx.field(static=True)
    _final_activation: Callable = eqx.field(static=True)
    _weight_init: Callable | None = eqx.field(static=True)
    _bias_init: Callable | None = eqx.field(static=True)
    _mlp_init_key: jnp.ndarray | None

    def __init__(
        self,
        input_species: str | tuple[str, ...],
        hidden_sizes: tuple[int, ...],
        activation: Callable = jax.nn.relu,
        final_activation: Callable = jax.nn.softplus,
        weight_init: Callable | None = None,
        bias_init: Callable | None = None,
        *,
        key: jnp.ndarray,
    ):
        """Initializes the MLP kinetics.

        Args:
            input_species: Species names to use as MLP inputs. Can be a single
                string, a tuple of strings, or `"*"` to use all network species (triggers deferred initialization).
            hidden_sizes: The number of neurons in each hidden layer.
            activation: Activation function for hidden layers.
            final_activation: Activation for the output layer.
            weight_init: Weight initializer.
            bias_init: Bias initializer.
            key: JAX random key for weight initialization.
        """
        if isinstance(input_species, str):
            input_species = (input_species,)

        self._input_species_idx = None

        if input_species == ('*',):
            self._requires_species = ()
            self.mlp = None
            self._mlp_init_key = key
        else:
            self._requires_species = input_species
            self.mlp = MultiLayerPerceptron(
                in_size=len(input_species),
                out_size='scalar',
                hidden_sizes=hidden_sizes,
                activation=activation,
                final_activation=final_activation,
                weight_init=weight_init,
                bias_init=bias_init,
                key=key,
            )
            self._mlp_init_key = None

        # Store hyperparameters for deferred creation
        self._hidden_sizes = hidden_sizes
        self._activation = activation
        self._final_activation = final_activation
        self._weight_init = weight_init
        self._bias_init = bias_init

    def _bind_to_network(self, species_map: dict[str, int]):
        """Binds the kinetics to a network.

        Creates the MLP if deferred and sets the input species indices.

        Raises:
            ValueError: If the MLP was initialized with deferred creation (`"*"`) but no JAX random key was provided.
        """
        if self.mlp is None:  # Deferred MLP creation
            if self._mlp_init_key is None:
                raise ValueError(
                    'A JAX random key must be provided to initialize the MLP.'
                )
            # Re-initialize the class with the full list of species.
            all_species = tuple(sorted(species_map.keys()))
            return MLP(
                input_species=all_species,
                hidden_sizes=self._hidden_sizes,
                activation=self._activation,
                final_activation=self._final_activation,
                weight_init=self._weight_init,
                bias_init=self._bias_init,
                key=self._mlp_init_key,
            )

        else:
            # Use specified input species
            indices = tuple(species_map[s] for s in self._requires_species)
            return eqx.tree_at(
                lambda x: x._input_species_idx,
                self,
                indices,
                is_leaf=lambda x: x is None,
            )

    def propensity_fn(
        self,
        x: jnp.ndarray,
        reactants: jnp.ndarray,
        t: float | None = None,
        volume: float = 1.0,
    ):
        """Calculates the propensity using the MLP.

        The MLP takes species concentrations as input and outputs a rate in
        concentration/time. This is then multiplied by the volume to get the
        final propensity in molecules/time.

        Args:
            x: The current state vector (species counts).
            reactants: The reactant stoichiometry vector (unused).
            t: The current time (unused).
            volume: The system volume.

        Returns:
            The stochastic propensity of the reaction.

        Raises:
            RuntimeError: If the MLP has not been initialized by binding it to a network.
        """
        if self.mlp is None:
            raise RuntimeError('MLP is not initialized. Call `_bind_to_network` first.')

        if self._input_species_idx is None:
            # Case ('*',) - use all species
            inputs = x / volume
        else:
            inputs = x[jnp.array(self._input_species_idx)] / volume

        rate_in_conc = self.mlp(inputs)

        return rate_in_conc * volume

    def ode_rate_fn(
        self,
        x: jnp.ndarray,
        reactants: jnp.ndarray,
        t: float | None = None,
        volume: float = 1.0,
    ):
        """Calculates the ODE rate using the MLP.

        The MLP computes a rate in concentration/time, which is then divided by
        the volume to obtain the final rate in molecules/time.

        Args:
            x: The current state vector (species counts).
            reactants: The reactant stoichiometry vector (unused).
            t: The current time (unused).
            volume: The system volume.

        Returns:
            The deterministic rate in molecules per unit time.
        """
        return self.propensity_fn(x, reactants, t, volume) / volume
