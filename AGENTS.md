# AGENTS.md

This file provides guidance to AI coding agents when working with code in this repository.

## Commands

```bash
# Install dev dependencies
uv sync --group dev

# Run all tests
uv run pytest tests/ -v

# Run a single test file / single test
uv run pytest tests/test_mi.py -v
uv run pytest tests/test_mi.py::test_function_name -v

# Lint and format (ruff auto-fixes on pre-commit)
ruff check .
ruff format .

# Build and serve docs locally
uv sync --group docs
uv run mkdocs serve
```

Pre-commit hooks: ruff lint+format (single quotes enforced), trailing whitespace, and pip-compile of `requirements.txt` / `requirements-all.txt`.

## Architecture

Stochastix is a JAX library for differentiable stochastic simulation of chemical reaction networks. Everything is built on **Equinox** — all core classes are `eqx.Module` subclasses (immutable JAX PyTrees). Always prefer Equinox transformations (`eqx.filter_jit`, `eqx.filter_vmap`, `eqx.filter_value_and_grad`, `eqx.tree_at`, `eqx.filter`, `eqx.apply_updates`) over raw JAX equivalents.

### Core design: composable PyTree models

- **`ReactionNetwork`** (`reaction/_network.py`): Compiles a list of `Reaction` objects into stoichiometric matrices. Provides multiple executable representations of the same chemical system:
  - `propensity_fn()` — stochastic propensities for SSA solvers
  - `vector_field()` / `ode_rhs()` — deterministic ODE (diffrax-compatible `f(t, x, args)`)
  - `drift_fn()` / `noise_coupling()` — Chemical Langevin Equation SDE terms
  - `diffrax_ode_term()` / `diffrax_sde_term()` — ready-made diffrax terms
  - `log_prob()` — trajectory log-likelihood for inference
  - `diffusion_coeff_matrix()` — Fokker-Planck diffusion matrix

- **`StochasticModel` / `MeanFieldModel`** (`_systems.py`): Convenience wrappers that bundle a `ReactionNetwork` + solver into a single callable `eqx.Module`. Their `__call__` runs a full simulation. Useful as self-contained differentiable models — the entire model (network params, kinetics, solver) lives in one PyTree.

### Simulation execution (`_stochsimsolve.py`)

The primary API is `stochsimsolve` and `faststochsimsolve`, both `@eqx.filter_jit`. Ensemble simulations vmap over these directly:

```python
results = eqx.filter_vmap(stx.stochsimsolve, in_axes=(0, None, None))(keys, network, x0, T=T)
```

- **`stochsimsolve`**: Uses `jax.lax.scan` — stores full trajectory, supports reverse-mode autodiff. Default solver is `DifferentiableDirect`.
- **`faststochsimsolve`**: Uses `jax.lax.while_loop` — early stopping, forward-mode only, no trajectory.

Both use `jax.lax.cond` for branching (not Python `if`) in hot paths to remain JIT-traceable.

### Solver hierarchy (`solvers/`)

All inherit from `AbstractStochasticSolver` (`eqx.Module`). Key flags: `is_exact_solver`, `is_pathwise_differentiable`.

- **Exact** (`_exact.py`): `DirectMethod`, `FirstReactionMethod`
- **Approximate** (`_approximate.py`): `TauLeaping`
- **Differentiable** (`_differentiable.py`): `DifferentiableDirect`, `DifferentiableFirstReaction`, `DGA`

### Kinetics (`kinetics/`)

All inherit from `AbstractKinetics` (`eqx.Module`). Each implements `propensity_fn()` (stochastic) and `ode_rate_fn()` (deterministic). `_bind_to_network()` resolves species names to indices when attached to a `ReactionNetwork`.

Types: `Constant`, `MassAction`, `MichaelisMenten`, Hill family (`HillActivator`, `HillRepressor`, `HillAA`, `HillAR`, `HillRR`), `MLP`.

When all reactions use `MassAction`, `ReactionNetwork.propensity_fn` takes a fast vectorized path via `eqx.filter_vmap`.

### Analysis (`analysis/`)

Differentiable post-simulation analysis (all JAX-traceable for gradient-based optimization):
- KDE histograms: `kde_1d.py`, `kde_2d.py` (multiple kernel types)
- `entropy.py`: entropy estimation from KDE
- `mi.py`: mutual information from joint/marginal KDEs
- `corr.py`: autocorrelation, cross-correlation

### Controllers (`controllers/`)

Controllers are optional runtime hooks for stochastic simulations. `AbstractController` defines `init()` and `step()`, and `stochsimsolve` will call them during execution to let a controller inspect and modify per-step behavior (for example, event timing or state interventions) while remaining JIT-compatible.

### State handling (`_state_utils.py`)

`pytree_to_state()` / `state_to_pytree()` convert between user-friendly formats (dicts, named tuples, Equinox modules) and flat JAX arrays used internally. Initial conditions can be `jnp.array`, `dict`, named tuples, or custom pytrees — the format is preserved through simulation results.

### Simulation results (`_simulation_results.py`)

`SimulationResults` is the main return container for simulation APIs. It preserves pytree state formats when possible and provides convenience helpers that agents should know about: direct indexing into batched results, `clean()` for removing padded trailing steps, and `interpolate()` for forward-fill interpolation onto a new time grid.

## Usage patterns

### Parameter tracing and static/dynamic split

Kinetics parameters that are `jax.Array` become dynamic PyTree leaves — they are traced by JAX and support `jit`/`grad`. Plain Python `float`/`int` values are treated as static by Equinox and baked into the compiled graph. Older code in the repository may still use deprecated `jnp.ndarray` annotations, but new or touched code should prefer `jax.Array`. To make a parameter trainable, pass it as a JAX array:

```python
# k is trainable (traced by JAX)
MassAction(k=jnp.array(-4.6), transform=jnp.exp)

# k is static (compiled as a constant, no gradient)
MassAction(k=0.01)
```

### Transform pattern for constrained parameters

Each kinetics class stores raw parameters and a `transform` callable (`eqx.field(static=True)`). The transform is applied inside `propensity_fn`/`ode_rate_fn` to map unconstrained values to valid rates. Standard pattern is log-parameterization for positive-constrained parameters:

```python
stx.Reaction('0 -> mRNA', stx.kinetics.MassAction(k=jnp.log(0.01), transform=jnp.exp))
```

To recover the actual rate constant from a trained model, apply the transform:

```python
kinetics = model.network.production.kinetics  # access by reaction name
actual_k = kinetics.transform(kinetics.k)
```

Hill kinetics have per-parameter transforms (`transform_v`, `transform_K`, `transform_n`, `transform_v0`).

### Ensemble simulation

Vmap over `stochsimsolve` directly with split keys:

```python
keys = jax.random.split(key, n_simulations)
results = eqx.filter_vmap(stx.stochsimsolve, in_axes=(0, None, None))(
    keys, network, x0, T=3600.0, solver=stx.DifferentiableDirect()
)
```

### Gradient-based training

Always use Equinox utilities for the optimization loop:

```python
opt = optax.adam(1e-2)
opt_state = opt.init(eqx.filter(model, eqx.is_array))

# Vmap over seeds, grad over the model
loss_and_grads = eqx.filter_vmap(
    eqx.filter_value_and_grad(loss_fn), in_axes=(None, None, 0)
)

for step in range(n_steps):
    keys = jax.random.split(key, batch_size)
    losses, grads = loss_and_grads(model, x0, keys)
    avg_grads = jax.tree.map(lambda g: g.mean(axis=0), grads)
    updates, opt_state = opt.update(avg_grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
```

### Functional parameter updates

Since all models are immutable Equinox modules, use `eqx.tree_at` for modifications:

```python
new_model = eqx.tree_at(lambda m: m.network.reactions[0].kinetics.k, model, new_value)
```

### ODE/SDE from the same network

A `ReactionNetwork` can drive stochastic (SSA), deterministic (ODE), or Langevin (SDE) simulations:

```python
# ODE via MeanFieldModel utility
mf_model = stx.MeanFieldModel(network, T=3600.0, saveat_steps=100)
result = mf_model(None, x0)

# SDE via diffrax directly
terms = network.diffrax_sde_term(T, key=subkey)
sol = diffrax.diffeqsolve(terms, diffrax.Euler(), t0=0.0, t1=T, dt0=0.1, y0=x0)
```

## Code conventions

- **Array type hints**: use `jax.Array` — never `np.ndarray`, `jnp.ndarray`, or `from jax import Array`
- **Docstrings**: Google Python Style Guide. Type hints in function signatures only, not in docstring arg descriptions
- **Ruff ignores**: `E501` (line length), `E731` (lambda assignment). Pydocstyle (`D` rules) enforced on library code but not on tests or notebooks
- **Quote style**: single quotes
