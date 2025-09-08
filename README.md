<div align="center">

# stochastix
*Differentiable, hardware accelerated (stochastic) kinetic models in JAX.*

</div>



The main purpose of this library is to provide a high-level, fairly flexible implementation of stochastic simulations of kinetic models (also commonly referred to as *Gillespie simulations* from the author of the most used solvers).

The design is heavily inspired by the [Diffrax](https://github.com/patrick-kidger/diffrax) library for solution of differential equations and heavily relies on the infrastructure provided by [Equinox](https://github.com/patrick-kidger/equinox). All of this is built on top of [JAX](https://github.com/jax-ml/jax), if you are new to this world you might want to take a quick look at some basic [JAX Tutorials](https://docs.jax.dev/en/latest/tutorials.html).

Basic usage of the main `stochastix` API for running forward simulations should require only a minimal knowledge of JAX.

**Main features:**

- Automatic GPU acceleration, JIT compilation, and vectorization via JAX
- Exact and pathwise differentiable stochastic solvers
- Seamless integration with the JAX ecosystem (Equinox, Diffrax, Optax, etc.)
- Automatic conversion between CME (jump process), CLE (stochastic differential equation), and ODE (deterministic differential equation) models for Diffrax integration
- Easy parameter selection for gradient computation via Equinox
- Support for learnable neural network propensities


## Installation

### GPU support

`stochastix` (as all other JAX-based libraries) relies on JAX for hardware acceleration support. To run on GPU or other accelerators, you need to install the appropriate JAX version. If JAX is not already present, the standard `stochastix` installation will automatically install the CPU version.


Please refer to the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) for the latest guidelines.


### pip

To install the package and core dependencies:

```bash
pip install stochastix
```

or directly from the repository:

```bash
pip install git+https://github.com/fmottes/stochastix.git
```

**Note:** in order to run the Jupyter notebooks, you need to install the optional dependencies:

```bash
pip install stochastix[notebooks]
```

### uv

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

To dive deeper take a look at the [stochastix documentation](https://fmottes.github.io/stochastix/).

## Citation

See [`CITATION.cff`](CITATION.cff). For BibTeX format, click on the “Cite this repository” button in the top right corner of the repository page.

