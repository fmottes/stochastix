---
hide:
  - toc
---


`stochastix` is a JAX-based library for stochastic simulation of chemical reaction networks.
It provides a simple and flexible API for defining models, simulating them using the Gillespie algorithm (and its variants), and optimizing their parameters with modern gradient-based methods.

## Key features

*   **JAX-powered**: Built on JAX and [Equinox](https://github.com/patrick-kidger/equinox). Models are PyTrees, enabling `jit`, `vmap`, `grad`, and GPU/TPU acceleration.
*   **Flexible model definition**: Compose networks from `Reaction` and `ReactionNetwork` with built-in kinetics (MassAction, Hill, Michaelis–Menten) and NN-based kinetics.
*   **Exact, approximate, and differentiable SSA**: `DirectMethod`, `FirstReactionMethod`, `TauLeaping`, plus differentiable variants (`DifferentiableDirect`, `DifferentiableFirstReaction`, `DGA`).
*   **Deterministic and CLE support**: ODE (`vector_field`, `diffrax_ode_term`) and CLE/SDE (`drift_fn`, `noise_coupling`, `diffrax_sde_term`) compatible with Diffrax.
*   **Controllers**: Time-based interventions (e.g., `Timer`) to manipulate species during simulations.
*   **Likelihood and RL utilities**: `ReactionNetwork.log_prob` for exact trajectories and helpers for REINFORCE-style training.
*   **Analysis and visualization**: Differentiable autocorrelation, cross-correlation, differentiable histograms/MI, and plotting utilities.

## Installation

**GPU support**

`stochastix` (as all other JAX-based libraries) relies on JAX for hardware acceleration support. To run on GPU or other accelerators, you need to install the appropriate JAX version. If JAX is not already present, the standard `stochastix` installation will automatically install the CPU version.


Please refer to the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) for the latest guidelines.


- **pip**

To install the package and core dependencies:

```bash
pip install stochastix
```

or directly from the repository:

```bash
pip install git+https://github.com/fmottes/stochastix.git
```

**Note:** in order to run the example Jupyter notebooks, you need to install the optional dependencies:

```bash
pip install stochastix[notebooks]
```

- **uv**

You can add the package to your project dependencies with:

```bash
uv add stochastix
```

For all other `uv` installation options, see the [uv docs](https://docs.astral.sh/uv/).



## Quick Example

A basic simulation of a chain reaction with Gillespie's direct method:

```python
import jax
import jax.numpy as jnp

import stochastix as stx
from stochastix.kinetics import MassAction

# simple reaction chain with mass action rates
network = stx.ReactionNetwork([
    stx.Reaction("0 -> X", MassAction(k=0.01)),
    stx.Reaction("X -> Y", MassAction(k=0.002))
])


x0 = jnp.array([0,0]) #initial conditions [X,Y]
sim_key = jax.random.PRNGKey(0) #key for jax random number generator

#solve with direct method from t0=0s to t1=100s
sim_results = stx.stochsimsolve(sim_key, network, x0, T=100.0)
```

## Diving deeper

- [Basic Usage](basic-usage.md): quick examples of basic library functionalities.
- [User Guide](user-guide/key-concepts.md): more detailed examples and explanations of the library features.
- [Example notebooks](https://github.com/fmottes/stochastix/tree/main/example-notebooks): Miscellaneous examples in Jupyter notebook format.


## API Reference

The full API reference is available [here](api/index.md).


## Citation

Coming soon...

```
```
