# Systems

For convenience `stochastix` defines systems as a bundle of a reaction network and a solution method (simulation routine + solver). They are based on the assumption that most modelling aim to represent the behavior of a specific system (a `ReactionNetwork`) over a specified time period (`T`).

In practice, they are high-level convenience wrappers for running simulations with a simple functional interface. `stochastix` provides two simple templates for this:

- `StochasticModel`: for stochastic simulations using `stochastix.stochsimsolve`
- `MeanFieldModel`: for deterministic ODE simulations using `diffrax.diffeqsolve`

A whole simulation can be now carried out with a simple function call:

```python
model = stochastix.StochasticModel(network, solver, T=10.0)
result = model(x0)
```



Both implementations have the same backbone:

```python
class System(equinox.Module):
    network: ReactionNetwork
    solver: Solver
    ...

    def __init__(self, network: ReactionNetwork, solver: Solver):
        ...

    def __call__(self, x0: Array, T: float, **kwargs) -> Array:
        ...

```

For more complex use cases, you can fall back to the standard solution API or easily create your own custom system by appropriately subclassing `equinox.Module`.


::: stochastix.StochasticModel
    options:
      members:
        - __call__


::: stochastix.MeanFieldModel
    options:
      members:
        - __call__