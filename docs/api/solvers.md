# Solvers

## Classic Solvers

---
::: stochastix.DirectMethod
    options:
      members: false
      inherited_members: false

---
::: stochastix.FirstReactionMethod
    options:
      members: false
      inherited_members: false

---

---

## Pathwise Differentiable Solvers

Both differentiable exact solvers support ReinMax reaction gradients:

```python
solver = stochastix.DifferentiableDirect(reaction_grad='reinmax', logits_scale=1.0)
# Or keep First Reaction's candidate-time sampling:
solver = stochastix.DifferentiableFirstReaction(
    reaction_grad='reinmax', logits_scale=1.0
)
```

ReinMax requires `exact_fwd=True` (the default) and a finite `logits_scale >= 1`.
The temperature changes the surrogate reaction-choice gradient; sampled reactions
and waiting times remain exact. In ReinMax mode, both solvers differentiate the
waiting time through total propensity: `d(dt) = -dt * d(log(sum(a)))`. First
Reaction keeps its sampled minimum forward, but uses this clock gradient instead
of its existing soft-argmin waiting-time gradient.

The default `reaction_grad='stgs'` preserves Direct's Gumbel-softmax relaxation
and First Reaction's existing soft-argmin relaxation of candidate firing times.
For First Reaction, `logits_scale` is a time scale in the existing mode and a
dimensionless temperature in ReinMax mode. Both modes use biased gradients.

---
::: stochastix.DifferentiableDirect
    options:
      members: false
      inherited_members: false

---
::: stochastix.DifferentiableFirstReaction
    options:
      members: false
      inherited_members: false

---
::: stochastix.DGA
    options:
      members: false
      inherited_members: false

---

---

## Approximate Solvers

---
::: stochastix.TauLeaping
    options:
      members: false
      inherited_members: false

---

---

## Abstract Base Class

---
::: stochastix.solvers.AbstractStochasticSolver
