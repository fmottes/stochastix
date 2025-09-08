# Running simulations

This guide shows how to run simulations after you have defined reactions and networks. It builds on the concepts in [Key Concepts](key-concepts.md) and [Defining Models](defining-models.md).

## Quick example: stochastic simulation with an exact solver

```python
import jax
import jax.numpy as jnp
import stochastix as stx
from stochastix.reaction import Reaction, ReactionNetwork
from stochastix.kinetics import MassAction

# Define a simple network
reactions = [
    Reaction("0 -> A", MassAction(0.5), name="A_prod"),
    Reaction("A -> 0", MassAction(0.1), name="A_deg"),
]
network = ReactionNetwork(reactions)

# Initial state: array (species order is network.species)
key = jax.random.PRNGKey(0)
x0 = jnp.array([0.0])  # or: {"A": 0.0}

results = stx.stochsimsolve(
    key, network, x0,
    T=100.0,
    solver=stx.DirectMethod(),
    max_steps=int(1e5),
)

# Remove padded steps for plotting/analysis
results = results.clean()
```

Notes:
- `x0` can be a JAX array (ordered like `network.species`), a dict of `{species_name: value}`, or a PyTree/object with dict keys or attributes matching species names.
- `T` is in seconds by default. `max_steps` is a compile-time bound to preallocate arrays.
- `results.time_overflow` is `True` if `max_steps` was exhausted before reaching `T`.

## Choosing a solver

- Exact (event-by-event): `DirectMethod`, `FirstReactionMethod`.
- Approximate (leaps in time): `TauLeaping(epsilon=0.03)`.
- Differentiable variants (optional): `DifferentiableDirect`, `DifferentiableFirstReaction`, `DGA`.

```python
from stochastix import TauLeaping
results_tau = stx.stochsimsolve(key, network, x0, T=100.0, solver=TauLeaping(epsilon=0.03))
```

## Controllers

```python
import jax.numpy as jnp
from stochastix.controllers import Timer

# At t=50, set A to 100 molecules
controller = Timer(
    controlled_species="A",
    time_triggers=jnp.array([50.0]),
    species_at_triggers=jnp.array([[100.0]]),
)

results_ctrl = stx.stochsimsolve(key, network, x0, T=100.0, solver=stx.DirectMethod(), controller=controller)
```

The controller sets `reaction_idx = -2` on trigger steps to distinguish them from padded steps (`-1`).

## Batching simulations

```python
import equinox as eqx

key, *subkeys = jax.random.split(key, 33)
subkeys = jnp.array(subkeys) # shape (32,)

vmapped_sim = eqx.filter_vmap(stx.stochsimsolve, in_axes=(0, None, None, None, None))

batched_results = vmapped_sim(subkeys, network, x0, 0.0, 100.0)

# Index into the batch and clean padded steps
first = batched_results[0].clean()
```

You can also vmap over parameterized networks or initial states; match `in_axes` accordingly.

## Working with SimulationResults

- `clean()`: remove padded (unused) steps for exact-solver trajectories when `save_trajectory=True`. Not supported in `jit` and not supported on batched results (index first, e.g. `results[i].clean()`).
- `interpolate(t_interp)`: piecewise-constant interpolation onto a regular grid.
- Batched results support indexing: `res_i = results[i]`.

```python
t_grid = jnp.linspace(0.0, 100.0, 201)
interp = results.interpolate(t_grid)

# Access fields
x = interp.x  # (len(t_grid), n_species)
t = interp.t
```

## Plotting trajectories

```python
fig, ax = stx.plot_abundance_dynamic(results, species="A")
plt.show()
```

For batched results, multiple trajectories are plotted with the same color. Use `species=['A', 'B']` to overlay.

## Systems (high-level wrappers)

```python
# Stochastic wrapper (SSA)
ssa_model = stx.StochasticModel(network=network, solver=stx.DirectMethod(), T=100.0, max_steps=int(1e5))
ssa_results = ssa_model(key, x0)

# Deterministic ODE wrapper (diffrax)
ode_model = stx.MeanFieldModel(network=network, T=100.0, saveat_steps=201)
ode_results = ode_model(None, x0=jnp.array([0.0]))
```

## Trajectory log-likelihood

For trajectories from exact solvers, you can compute per-step log-likelihood terms:

```python
log_terms = network.log_prob(results)
total_logp = log_terms.sum()
```

Pair with utilities in `stochastix.utils.optimization` (e.g., `reinforce_loss`, `discounted_returns`) for policy-gradient training. See [Optimization](../api/utils/optimization.md) for more details.
