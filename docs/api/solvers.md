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

`DifferentiableDirect` uses STGS reaction gradients by default. To use ReinMax:

```python
solver = stochastix.DifferentiableDirect(reaction_grad='reinmax', logits_scale=1.0)
```

ReinMax requires `exact_fwd=True` (the default) and a finite `logits_scale >= 1`.
The temperature changes the surrogate reaction-choice gradient; sampled reactions
and waiting times remain exact. Waiting-time gradients are unchanged. Both STGS
and ReinMax are biased gradient estimators.

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
