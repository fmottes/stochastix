"""Basic neural network layers and architectures for kinetic modeling."""

from collections.abc import Callable
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as rng


class Linear(eqx.Module):
    """Linear layer with custom initializers.

    Attributes:
        weights: Weight matrix for the linear transformation.
        bias: Bias vector for the linear transformation.
        in_features: Number of input features or 'scalar'.
        out_features: Number of output features or 'scalar'.
        use_bias: Boolean flag indicating whether bias is used.
    """

    weights: jnp.ndarray
    bias: jnp.ndarray | None
    in_features: int | Literal['scalar'] = eqx.field(static=True)
    out_features: int | Literal['scalar'] = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)

    def __init__(
        self,
        in_features: int | Literal['scalar'],
        out_features: int | Literal['scalar'],
        use_bias: bool = True,
        weight_init: Callable | None = None,
        bias_init: Callable | None = None,
        dtype=None,
        *,
        key: jnp.ndarray,
    ):
        """Initialize the Linear layer.

        Args:
            in_features: The number of input features. Can be an integer or "scalar".
            out_features: The number of output features. Can be an integer or "scalar".
            use_bias: Whether to include a bias term.
            weight_init: A function to initialize the weights.
            bias_init: A function to initialize the bias.
            dtype: The data type of the weights and bias.
            key: A JAX random key for initialization.
        """
        wkey, bkey = jax.random.split(key)
        in_size = 1 if in_features == 'scalar' else in_features
        out_size = 1 if out_features == 'scalar' else out_features
        if dtype is None:
            dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32

        if weight_init is None:
            weight_init = jax.nn.initializers.glorot_uniform()
        if bias_init is None:
            bias_init = jax.nn.initializers.zeros

        self.weights = weight_init(wkey, (out_size, in_size), dtype)
        if use_bias:
            self.bias = bias_init(bkey, (out_size,), dtype)
        else:
            self.bias = None

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias

    def __call__(
        self,
        x: jnp.ndarray,
        *,
        key: jnp.ndarray | None = None,
    ):
        """Apply the linear transformation.

        Args:
            x: The input array.
            key: A JAX random key (not used).

        Returns:
            The transformed array.
        """
        if self.in_features == 'scalar':
            if jnp.shape(x) != ():
                raise ValueError(f'Expected input shape () but got {x.shape}')
            x = jnp.broadcast_to(x, (1,))

        x = self.weights @ x

        if self.bias is not None:
            x = x + self.bias

        if self.out_features == 'scalar':
            x = jnp.squeeze(x, axis=-1)

        return x


class MultiLayerPerceptron(eqx.Module, strict=True):
    """Multi-Layer Perceptron (MLP) for feed-forward neural networks.

    This is a simple MLP with a configurable number of hidden layers and neurons,
    custom activation functions, and initializers.

    Attributes:
        layers: Tuple of Linear layers.
        activation: The activation function for the hidden layers.
        final_activation: The activation function for the output layer.
        in_size: The input size of the MLP.
        out_size: The output size of the MLP.
        hidden_sizes: Tuple of integers specifying the size of each hidden layer.
        use_bias: Whether to use a bias in the hidden layers.
        use_final_bias: Whether to use a bias in the final layer.
    """

    layers: tuple[eqx.nn.Linear, ...]
    activation: Callable
    final_activation: Callable
    in_size: int | Literal['scalar'] = eqx.field(static=True)
    out_size: int | Literal['scalar'] = eqx.field(static=True)
    hidden_sizes: tuple[int, ...] = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)
    use_final_bias: bool = eqx.field(static=True)

    def __init__(
        self,
        in_size: int | Literal['scalar'],
        out_size: int | Literal['scalar'],
        hidden_sizes: tuple[int, ...] = (),
        activation: Callable = jax.nn.relu,
        final_activation: Callable = jax.nn.softplus,
        use_bias: bool = True,
        use_final_bias: bool = True,
        weight_init: Callable | None = None,
        bias_init: Callable | None = None,
        dtype=None,
        *,
        key: jnp.ndarray,
    ):
        """Initialize the MultiLayerPerceptron.

        Args:
            in_size: The input size. The input to the module should be a vector of
                shape (in_features,). It also supports the string "scalar" as a
                special value, in which case the input to the module should be of shape ().
            out_size: The output size. The output from the module will be a vector
                of shape (out_features,). It also supports the string "scalar"
                as a special value, in which case the output from the module will have shape ().
            hidden_sizes: The size of each hidden layer.
            activation: The activation function after each hidden layer.
            final_activation: The activation function after the output layer.
            use_bias: Whether to add on a bias to internal layers.
            use_final_bias: Whether to add on a bias to the final layer.
            weight_init: The initializer for the weights.
            bias_init: The initializer for the biases.
            dtype: The dtype to use for all the weights and biases in this MLP.
                Defaults to either `jax.numpy.float32` or `jax.numpy.float64`
                depending on whether JAX is in 64-bit mode.
            key: A `jax.random.PRNGKey` used to provide randomness for parameter
                initialisation. (Keyword only argument.)
        """
        if dtype is None:
            dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
        all_sizes = [in_size, *hidden_sizes, out_size]
        keys = rng.split(key, len(all_sizes) - 1)
        layers = []
        for i, (in_dim, out_dim) in enumerate(zip(all_sizes[:-1], all_sizes[1:])):
            is_final_layer = i == len(all_sizes) - 2
            bias = use_final_bias if is_final_layer else use_bias
            layers.append(
                Linear(
                    in_dim,
                    out_dim,
                    use_bias=bias,
                    weight_init=weight_init,
                    bias_init=bias_init,
                    dtype=dtype,
                    key=keys[i],
                )
            )

        self.layers = tuple(layers)
        self.activation = activation
        self.final_activation = final_activation
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_sizes = tuple(hidden_sizes)
        self.use_bias = use_bias
        self.use_final_bias = use_final_bias

    def __call__(self, x: jnp.ndarray, *, key: jnp.ndarray | None = None):
        """Execute the forward pass of the MLP.

        Args:
            x: A JAX array with shape (in_size,). (Or shape () if
                in_size="scalar".)
            key: Ignored; provided for compatibility with the rest of the Equinox API.

        Returns:
            A JAX array with shape (out_size,). (Or shape () if
            out_size="scalar".)
        """
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.final_activation(self.layers[-1](x))
        return x
