# Defining Models

## Kinetics

The `Kinetics` classes define the mathematical form of the reaction rates. They are not supposed to be instantiated directly, but rather used to define the kinetic laws of `Reaction` objects.

- **Parameter Definition:** They hold the parameters for the rate laws (e.g., rate constants).
- **Rate Functions:** They provide two key methods: `propensity_fn` for stochastic simulations (calculating reaction propensities) and `ode_rate_fn` for deterministic simulations (calculating reaction rates for ODEs).
- **Custom Laws:** You can implement your own custom kinetic laws by subclassing `stx.kinetics.AbstractKinetics` and implementing the `propensity_fn` and `ode_rate_fn` methods. `stochastix` supports several built-in kinetic laws (see [Kinetics API docs](../api/kinetics.md)).

## Reactions

A `Reaction` object is a container that represents a single reaction channel, uniting a reaction string, a kinetic law, and an optional name.

- **Reaction String:** Defines the stoichiometry of the reaction (e.g., `"A + B -> C"`). The parser is flexible and handles various formats, including stoichiometric coefficients (`"2 A -> A"`) and null products (`"A -> 0"`).
- **Kinetics:** A `Kinetics` object that defines the reaction rate.
- **Name:** An optional string to identify the reaction.

Here is an example of how to define a reaction:

```python
from stochastix import Reaction
from stochastix.kinetics import MassAction

# Defines the reaction A + B -> C with a mass-action rate constant of 0.1
reaction = Reaction("A + B -> C", MassAction(k=0.1))
```

**Notes:**
- Reaction strings support coefficients and null sides: e.g., `"2A + B -> 0"`.
- Bidirectional arrows `<->` are not supported; add forward and reverse as separate reactions.
- Species names may include underscores; numeric coefficients are parsed as floats.

## Reaction Networks

The `ReactionNetwork` class is the central object for representing a biochemical system. It acts as an interface between a set of reactions and the simulation machinery.

**Reaction Handling:** It takes a list of `Reaction` objects and automatically discovers all chemical species, building the stoichiometric matrix. Species are stored alphabetically in the `.species` attribute.

```python
from stochastix import Reaction, ReactionNetwork
from stochastix.kinetics import MassAction

reactions = [
    Reaction("S + E -> SE", MassAction(k=0.1), name="binding"),
    Reaction("SE -> S + E", MassAction(k=0.05), name="unbinding"),
    Reaction("SE -> P + E", MassAction(k=0.2), name="conversion")
]

network = ReactionNetwork(reactions)
```

**Textual and LaTeX Representations:** It provides text and LaTeX representations for easy visualization and analysis of the model.

```python
# Pretty-print a human-readable summary
print(network)
```

```text
R0 (binding):    S + E -> SE    |  MassAction
R1 (unbinding):  SE -> S + E    |  MassAction
R2 (conversion): SE -> P + E    |  MassAction
```

```python
# Generate a LaTeX representation (optionally include kinetics types)
latex_str = network.to_latex()
```

**Functional Manipulation:** `ReactionNetwork` objects can be manipulated functionally, for example by slicing or by adding reactions or other networks together.

```python
# add new reaction to the network
network = network + Reaction("P -> 0", MassAction(k=0.3), name="degradation")

# slice the network
network = network[1:2]  # remove the first and last reactions

# add two networks together
network1 = ReactionNetwork([Reaction("S -> 0", MassAction(k=0.1)), Reaction("P -> 0", MassAction(k=0.2))])
network = network + network1

# access the `degradation` reaction
degradation_reaction = network.degradation
```

**Dynamics Representation:** It provides utilities to simulate the network dynamics in different representations, such as an ordinary differential equation (ODE) or stochastic differential equation (SDE) using the Chemical Langevin Equation (CLE). These representations are fully compatible with `diffrax`.

```python
# simulate the network dynamics as an ODE
vector_field = network.vector_field(t, x)

# simulate the network dynamics as an SDE
drift = network.drift_fn(t, x)
diffusion = network.diffusion_fn(t, x)
```

Using `diffrax` for ODE/SDE integration:

```python
# ODE solve
term = network.diffrax_ode_term()
sol_ode = diffrax.diffeqsolve(
    term,
    ...
)

# SDE (CLE) solve
sde_term = network.diffrax_sde_term(t1=100.0, key=key)
sol_sde = diffrax.diffeqsolve(
    sde_term,
    ...
)
```

**Log-Likelihood Definition:** It defines the log-likelihood of a system's state or trajectory, for inference and optimization tasks.

```python
results = stochsimsolve(key, network, x0, T=100.0)

# Per-step log-probability terms (masked for padded steps)
log_terms = network.log_prob(results)
total_logp = log_terms.sum()
```

`log_prob` is defined for trajectories from exact solvers (e.g., `DirectMethod`, `FirstReactionMethod`). It recomputes propensities internally and masks padded steps, you can sum the terms to obtain a scalar log-likelihood.


**Other features:**
- Named access to reactions: if a reaction has `name="binding"`, access it as `network.binding`.
- Structural matrices are stored as species×reactions lists: `reactant_matrix`, `product_matrix`, and `stoichiometry_matrix`.
- Units and volume: rate parameters for concentration-based kinetics (e.g., Hill, Michaelis–Menten) are in concentration/time; conversion to molecules/time uses `network.volume` internally for both stochastic propensities and ODE rates.



## Systems

`StochasticModel` and `MeanFieldModel` are convenience wrappers that combine a `ReactionNetwork` with a solver. They are useful when you repeatedly simulate the same network (for example, for inference or optimization).

```python
from stochastix import StochasticModel, MeanFieldModel, DirectMethod

# Stochastic SSA wrapper
ssa_model = StochasticModel(network=network, solver=DirectMethod(), T=100.0, max_steps=int(1e5))

# Deterministic ODE wrapper (diffrax)
ode_model = MeanFieldModel(network=network, T=100.0, saveat_steps=101)
```

For more complex use cases, use the `stochsimsolve` function directly or define a custom system class. Making your system an `equinox.Module` makes it easy to bundle all reaction network parameters into a single object, which is especially useful for optimizations.

```python
import equinox as eqx
import stochastix as stx

class MySystem(eqx.Module):
    network: stx.ReactionNetwork
    # ... other parameters

    def __init__(self, network, ...):
        self.network = network
        # ...

    def __call__(self, x0, t, ...):
        # ... custom simulation logic
        return results
```
