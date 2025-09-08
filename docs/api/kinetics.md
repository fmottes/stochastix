# Kinetic Laws

Kinetic laws for chemical reactions.


---

::: stochastix.kinetics.Constant
    options:
      members: false
      inherited_members: false

---
::: stochastix.kinetics.MassAction
    options:
      members: false
      inherited_members: false

---
::: stochastix.kinetics.MichaelisMenten
    options:
      members: false
      inherited_members: false


---

## 1D Hill Functions

---
::: stochastix.kinetics.HillActivator
    options:
      members: false
      inherited_members: false

---
::: stochastix.kinetics.HillRepressor
    options:
      members: false
      inherited_members: false

---
::: stochastix.kinetics.HillSingleRegulator
    options:
      members: false
      inherited_members: false


---

## 2D Hill Functions

---
::: stochastix.kinetics.HillAA
    options:
      members: false
      inherited_members: false

---
::: stochastix.kinetics.HillAR
    options:
      members: false
      inherited_members: false

---
::: stochastix.kinetics.HillRR
    options:
      members: false
      inherited_members: false

---

## Neural Kinetic Laws

---
::: stochastix.kinetics.MLP
    options:
      members: false
      inherited_members: false


---

## Abstract Base Class

---
::: stochastix.kinetics.AbstractKinetics
    options:
      members:
        - propensity_fn
        - ode_rate_fn
        - _bind_to_network
