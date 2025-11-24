---
hide:
  - toc
---

This function performs stochastic simulation using `jax.lax.while_loop`, which allows early termination when the simulation finishes before reaching `max_steps`. This makes it considerably faster than `stochsimsolve` for simulations that finish early. For the same random key, `faststochsimsolve` produces the exact same results as `stochsimsolve` (same final state, same reaction sequence), but comes with some trade-offs:

- It **does not support backward differentiation** (reverse-mode autodiff), only forward differentiation.
- It **does not support saving the full trajectory** (only initial and final states are saved).

---
::: stochastix.faststochsimsolve
