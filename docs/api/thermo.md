# Stochastic Thermodynamics

The `analysis.thermo` module turns a stochastic trajectory (together with the
`ReactionNetwork` that generated it) into differentiable thermodynamic
observables suitable for gradient-based optimization: entropy production and its
rate, per-reaction affinities and currents, total entropy production, and
Thermodynamic Uncertainty Relation diagnostics.

These observables require the trajectory to record which reaction fired at each
step, so run `stochsimsolve` with `save_trajectory=True` (the default) using an
exact solver such as [`DirectMethod`][stochastix.DirectMethod] or
[`DifferentiableDirect`][stochastix.DifferentiableDirect]. A structural reverse
channel is necessary, but not sufficient, for finite entropy production: its
propensity must also be positive on every traversed reverse edge. The network
exposes `ReactionNetwork.reverse_reaction_index` and
`ReactionNetwork.has_reverse_channels` to inspect the structural requirement.

## Entropy Production

To differentiate entropy production through reaction selection, save the
reaction weights from a solver with an exact forward pass.

```python
import equinox as eqx
import stochastix as stx
from stochastix.analysis import thermo


@eqx.filter_value_and_grad
def loss(network, key, x0):
    trajectory = stx.stochsimsolve(
        key,
        network,
        x0,
        T=10.0,
        solver=stx.DifferentiableDirect(reaction_grad='reinmax'),
        save_reaction_weights=True,
    )
    return thermo.entropy_production_rate(network, trajectory)
```

The solver records one weight per reaction per interval. The forward value
uses the fired reaction. The backward pass weights the complete log propensity
ratio for each candidate, evaluating its reverse rate at its own successor
state. These derivatives are a surrogate for multi-event ensemble gradients.

---
::: stochastix.analysis.thermo.entropy_production

---
::: stochastix.analysis.thermo.entropy_production_rate

---
::: stochastix.analysis.thermo.system_entropy

---
::: stochastix.analysis.thermo.total_entropy_production

## Currents and Affinities

---
::: stochastix.analysis.thermo.affinities

---
::: stochastix.analysis.thermo.reaction_currents

---
::: stochastix.analysis.thermo.schnakenberg_epr

## Thermodynamic Bounds

---
::: stochastix.analysis.thermo.tur_bound

---
::: stochastix.analysis.thermo.thermodynamic_efficiency
