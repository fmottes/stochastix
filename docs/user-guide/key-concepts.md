# Key Concepts in `stochastix`

This document outlines the core components for defining and simulating biochemical reaction networks in `stochastix`.

## Object Hierarchy

Here is a schematic representation of the object hierarchy in `stochastix`, from model definition to simulation results.

```
reaction_string ─┐
                 ├─> Reaction ──> ReactionNetwork  ─┐
Kinetics ────────┘                                  │
                                                    ├─> stochsimsolve ──> SimulationResults
                                         Solver  ───┤
                                                    │
                                     Controller  ───┘
```

`StochasticModel` and `MeanFieldModel` are convenience wrappers that combine a `ReactionNetwork` with, respectively, a stochastic solver and a diffrax ODE solver.

---

## Model Components

### Kinetics
The `Kinetics` classes define the mathematical form of the reaction rates.

-   **Parameter Definition:** They hold the parameters for the rate laws (e.g., rate constants).
-   **Rate Functions:** They provide two key methods: `propensity_fn` for stochastic simulations (calculating reaction propensities) and `ode_rate_fn` for deterministic simulations (calculating reaction rates for ODEs).
-   **Custom Laws:** You can implement your own custom kinetic laws by subclassing `kinetics.AbstractKinetics`.

### Reaction
A `Reaction` object represents a single reaction channel, uniting a reaction string, a kinetic law, and an optional name.

-   **Reaction String:** Defines the reaction's stoichiometry (e.g., `"A + B -> C"`).
-   **Kinetics:** A `Kinetics` object that defines the reaction rate.

```python
from stochastix.reaction import Reaction
from stochastix.kinetics import MassAction

# Defines the reaction A + B -> C with a mass-action rate constant of 0.1
reaction = Reaction("A + B -> C", MassAction(k=0.1))
```

### ReactionNetwork
The `ReactionNetwork` is the central object for representing a biochemical system.

-   **Reaction Handling:** It takes a list of `Reaction` objects and automatically discovers all chemical species, building the stoichiometric matrix.
-   **System Representation:** It can be converted into other representations, like systems of ODEs or SDEs.
-   **Functional Manipulation:** `ReactionNetwork` objects can be manipulated functionally (e.g., slicing, addition).

```python
from stochastix.reaction import Reaction, ReactionNetwork
from stochastix.kinetics import MassAction

reactions = [
    Reaction("S + E -> SE", MassAction(k=0.1)),
    Reaction("SE -> S + E", MassAction(k=0.05)),
    Reaction("SE -> P + E", MassAction(k=0.2))
]
network = ReactionNetwork(reactions)
```

---

## Simulation Components

### stochsimsolve
`stochsimsolve` is the main function for running stochastic simulations. It orchestrates the interplay between the `ReactionNetwork`, the `Solver`, and the (optional) `Controller`, returning a `SimulationResults` object.

```python
from stochastix import stochsimsolve

results = stochsimsolve(key, network, x0, T=T)
```

### Solver
The `Solver` implements the core algorithm for advancing the simulation in time.

-   **Exact Solvers:** Gillespie methods such as `DirectMethod` and `FirstReactionMethod` simulate every reaction event.
-   **Approximate Solvers:** `TauLeaping` takes discrete time steps and can fire multiple reactions per step.
-   **Differentiable Variants (optional):** `DifferentiableDirect`, `DifferentiableFirstReaction`, and `DGA` enable pathwise gradients.

### Controller
A `Controller` allows for implementing time-dependent perturbations or events during the simulation, such as adding a species or changing a rate at a specific time.

### Systems
`StochasticModel` wraps a `ReactionNetwork` with a stochastic `Solver` for repeated simulations; `MeanFieldModel` wraps it with a diffrax ODE solver for deterministic dynamics.
