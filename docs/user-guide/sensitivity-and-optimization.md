# Sensitivity analysis and optimization

## Setup

Set up dependencies, a simple model, and the objective function.

### Model definition

```python
import jax
import jax.numpy as jnp
import equinox as eqx
import stochastix as stx
from stochastix import Reaction, ReactionNetwork
from stochastix.kinetics import MassAction

# Simple birth-death system
# Make the production rate differentiable (jnp.array) and the degradation static (float).
net = ReactionNetwork([
    Reaction("0 -> A", MassAction(jnp.array(5.0)), name="prod"),  # will receive gradients
    Reaction("A -> 0", MassAction(0.1), name="deg"),               # kept static
])

key = jax.random.PRNGKey(0)
x0 = jnp.array([0.0])  # species order matches net.species
T = 100.0
```

### Objective: match a target final-time abundance of A

Avoid using `clean()` inside losses so code remains JIT-compatible. The last state is already padded with the last valid value, so it safely represents the final abundance.

```python
def final_A_sqerr(results: stx.SimulationResults, TARGET_A: float = 20.0) -> jnp.ndarray:
    return (results.x[-1, 0] - TARGET_A) ** 2
```

## Sensitivity analysis

This guide shows how to compute parameter sensitivities of objectives using two types of estimators:

- Score-based (likelihood-ratio / REINFORCE): unbiased, works with exact SSA solvers, higher variance.
- Pathwise (reparameterization): low-variance, requires differentiable solvers.

Both approaches can be used with `ReactionNetwork` directly or the high-level `StochasticModel` wrappers (see [Running simulations](running-simulations.md)).


### Score-based estimator (likelihood ratio / REINFORCE)

Use exact solvers (e.g., `DirectMethod`) and differentiate the log-likelihood of the path w.r.t. parameters. Use `eqx.filter_grad` so only array leaves (e.g., `prod.k`) receive gradients.

```python
from stochastix.utils.optimization import reinforce_loss

solver = stx.DirectMethod()

def simulate(key, net, x0):
    return stx.stochsimsolve(key, net, x0, T=T, solver=solver, max_steps=int(1e5))

def score_loss(net, key):
    res = simulate(key, net, x0)
    # Reward is negative squared error to encourage matching TARGET_A
    r = -final_A_sqerr(res)
    # Broadcast final reward over valid steps; padded steps contribute 0 via log_prob masking
    returns = jnp.where(res.reactions >= 0, r, 0.0)
    # Maximize returns -> minimize negative REINFORCE loss
    return -reinforce_loss()(net, res, returns)

# Sensitivity wrt parameters marked as arrays (prod.k)
g_score = eqx.filter_grad(score_loss)(net, key)
```
Tip: you can avoid defining `simulate` by using `stx.StochasticModel` or your custom callable `eqx.Module`:

```python
 model = stx.StochasticModel(network=net, solver=solver, T=T, max_steps=int(1e5))
 res = model(key, x0)
```

Basic variance reduction tricks (that keep the estimator unbiased):
- Provide a baseline to `reinforce_loss`.
- Average gradients over multiple keys with `vmap`.

```python
def batched_score_loss(net, key):
    keys = jax.random.split(key, 32)

    # Vectorize over keys while broadcasting the model
    vmapped_score = eqx.filter_vmap(score_loss, in_axes=(None, 0))
    return jnp.mean(vmapped_score(net, keys))

g_score_batched = eqx.filter_grad(batched_score_loss)(net, key)
```

### Pathwise estimator (reparameterization)

Use differentiable SSA solvers to push gradients through the sampled path. Keep the forward pass exact with straight-through gradients (`exact_fwd=True`) or relax it.

Available solvers:
- `stx.DifferentiableDirect(logits_scale=1.0, exact_fwd=True)`
- `stx.DifferentiableFirstReaction(logits_scale=1.0, exact_fwd=True)`
- `stx.DGA(a=..., sigma=...)` (does not admit exact forward pass; less stable)

```python
solver_pw = stx.DifferentiableDirect(logits_scale=1.0, exact_fwd=True)

def pathwise_loss(net, key):
    res = stx.stochsimsolve(key, net, x0, T=T, solver=solver_pw, max_steps=int(1e5))
    return final_A_sqerr(res)  # minimize squared error to TARGET_A

grad_pw = eqx.filter_grad(pathwise_loss)(net, key)
```

Tips:
- Smaller `logits_scale` tightens the relaxation but can increase gradient variance; tune per model.
- `exact_fwd=True` (default) keeps the trajectory exact and uses straight-through gradients.
- Consider averaging over multiple keys with `vmap`.


## Gradient-based optimization

```python
import optax

opt = optax.adam(1e-2)
opt_state = opt.init(net)

@eqx.filter_jit
def step(net, opt_state, key):
    loss, grads = eqx.filter_value_and_grad(pathwise_loss)(net, key)
    updates, opt_state = opt.update(grads, opt_state, params=net)
    net = eqx.apply_updates(net, updates)
    return net, opt_state, loss

for i in range(100):
    key, sub = jax.random.split(key)
    net, opt_state, loss = step(net, opt_state, sub)
```

To switch to score-based training, replace `pathwise_loss` with `score_loss` (or its batched variant).

Batching to match a target average final abundance across multiple trajectories:

```python
def batched_pathwise_loss(net, key):
    keys = jax.random.split(key, 32)
    def one(k):
        res = stx.stochsimsolve(k, net, x0, T=T, solver=solver_pw, max_steps=int(1e5))
        return final_A_sqerr(res)
    return jnp.mean(jax.vmap(one)(keys))

@eqx.filter_jit
def step_batched(net, opt_state, key):
    loss, grads = eqx.filter_value_and_grad(batched_pathwise_loss)(net, key)
    updates, opt_state = opt.update(grads, opt_state, params=net)
    net = eqx.apply_updates(net, updates)
    return net, opt_state, loss
```


## Notes and estimator choice

For more thorough details and practical examples, see [Example Notebooks](../example-notebooks/README.md).

- Do not call `clean()` inside losses; keep functions JIT-compatible. The last state is padded with the last valid state.
- With `eqx.filter_grad`, only array leaves receive gradients; in the example, only `prod.k` is differentiated, while `deg.k` (float) stays static. You can also define custom filters to use with `eqx.filter_grad` that change this default behavior.
- Prefer **pathwise** for smooth objectives and when differentiable solvers are acceptable (lower variance, faster convergence).
- Use **score-based** for unbiased gradients under exact SSA or when pathwise relaxations are unsuitable.

### TEST: FD and SPSA estimators

For testing and sanity checks, two basic implementations of zeroth-order estimators are available under `stochastix.utils.optimization.grad`.

```python
from stochastix.utils.optimization.grad import gradfd, gradspsa

# Central difference on parameter PyTrees (w.r.t. first arg)
fd = gradfd(pathwise_loss, epsilon=1e-2)
g_fd = fd(net, key)

# SPSA with common random numbers (per-sample key split)
spsa = optgrad.gradspsa(pathwise_loss, epsilon=1e-1, num_samples=64, split_first_arg_key=True)
g_spsa = spsa(net, key)  # uses internal delta keys; splits 'key' per sample for simulation
```

Notes:
- Both estimators differentiate w.r.t. the first argument of the target function.
- SPSA uses Rademacher perturbations and averages unbiased per-sample estimates. With `split_first_arg_key=True`, each SPSA sample gets its own simulation key but shares the same key between +/− evaluations (common random numbers), typically reducing variance.
