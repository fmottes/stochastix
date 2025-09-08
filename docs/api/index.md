---
title: API Reference
hide:
  - toc
---


This is the API reference for `stochastix`. For usage examples and guides see [Home](../index.md).

## Core

**Models**

*   [`stochastix.Reaction`][reaction] : A reaction.
*   [`stochastix.kinetics`][kinetics] : Kinetics of a reaction.
*   [`stochastix.ReactionNetwork`][network] : A network of reactions.
*   [`stochastix.generators`][generators] : Generators of ReactionNetwork models for some common kinetic models.


**Simulation**

*   [`stochastix.stochsimsolve`][stochsimsolve] : Main simulation entry point. Solve a stochastic initial value problem with a given solver.
*   [`stochastix.SimulationResults`][simulation-results] : Results of a stochastic simulation with utilities (interpolation and cleaning results).
*   [`stochastix.solvers`][solvers] : Solvers for stochastic simulations.
*   [`stochastix.controllers`][controllers] : Controllers for stochastic simulations.

**Systems**

*   [`stochastix.System`][systems] : Convenience wrappers. Bundle a ReactionNetwork with a forward solver.


## Utilities

*   [`stochastix.utils.optimization`][optimization] : Optimization utilities.
*   [`stochastix.utils.visualization`][visualization] : Visualization utilities.
*   [`stochastix.utils.analysis`][analysis] : Analysis utilities (e.g. correlation functions and histograms).
*   [`stochastix.utils.nn`][neural-networks] : Neural network utilities for [neural kinetics](kinetics.md#neural-kinetic-laws).
*   [`stochastix.utils.misc`][misc] : Miscellaneous utilities.



[stochsimsolve]: stochsimsolve.md
[simulation-results]: simulation-results.md
[systems]: systems.md
[reaction]: reaction.md
[kinetics]: kinetics.md
[network]: network.md
[generators]: generators.md
[solvers]: solvers.md
[controllers]: controllers.md
[analysis]: utils/analysis.md
[visualization]: utils/visualization.md
[neural-networks]: utils/neural-networks.md
[optimization]: utils/optimization.md
[misc]: utils/misc.md
