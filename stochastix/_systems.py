"""High-level simulation models for stochastic and deterministic systems."""

from __future__ import annotations

import typing
from typing import Any

import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp

from ._simulation_results import SimulationResults
from ._stochsimsolve import stochsimsolve
from .reaction import ReactionNetwork

if typing.TYPE_CHECKING:
    from .solvers import AbstractStochasticSolver


class StochasticModel(eqx.Module):
    """A stochastic model combining a reaction network and a solver.

    This class serves as a high-level convenience wrapper for running simulations.
    It composes a `ReactionNetwork` with a `solver` to define a complete
    executable simulation system.

    Attributes:
        network: The reaction network defining the system's structure.
        solver: The solver for the simulation.
        T: The final time of the simulation.
        max_steps: The maximum number of simulation steps.
    """

    network: ReactionNetwork
    solver: AbstractStochasticSolver
    T: float | jnp.ndarray | None = None
    max_steps: int = int(1e4)

    def __init__(
        self,
        network: ReactionNetwork,
        solver: AbstractStochasticSolver,
        T: float | jnp.ndarray | None = None,
        max_steps: int = int(1e4),
    ):
        """Initialize the stochastic model.

        Args:
            network: The reaction network defining the system's structure.
            solver: The solver for the simulation.
            T: The final time of the simulation. If not provided, it must be
                specified during the call.
            max_steps: The maximum number of simulation steps.
        """
        self.network = network
        self.solver = solver
        self.T = T
        self.max_steps = max_steps

    @eqx.filter_jit()
    def __call__(
        self,
        key: jnp.ndarray,
        x0: Any,
        T: float | jnp.ndarray | None = None,
        **kwargs: Any,
    ) -> SimulationResults:
        """Execute the simulation.

        This method internally calls the `stochsimsolve` function to run the
        simulation and returns the simulation results.

        Args:
            key: The JAX random key.
            x0: The initial state of the system.
            T: The final time of the simulation. If provided, it overrides the `T`
                attribute of the model. If not provided, the model's `T` is used.
            **kwargs: Additional keyword arguments to pass to `stochsimsolve`.

        Returns:
            solution: The results of the simulation.
        """
        if T is None:
            if self.T is None:
                T = jnp.nan
            T = self.T

        solution = stochsimsolve(
            key,
            self.network,
            x0,
            T=T,
            solver=self.solver,
            max_steps=self.max_steps,
            **kwargs,
        )

        return solution


class MeanFieldModel(StochasticModel):
    """A mean field model combining a reaction network and a diffrax solver.

    This class serves as a high-level convenience wrapper for running deterministic
    simulations. It composes a `ReactionNetwork` with a `diffrax` solver to define
    a complete executable simulation system for ordinary differential equations (ODEs).

    Note:
        This class accommodates only basic use cases and is not very flexible.
        If you need more control, use `diffrax` explicitly or create your own
        callable model.

    Attributes:
        network: The reaction network defining the system's structure.
        solver: The `diffrax` solver for the ODE simulation.
        T: The final time of the simulation.
        saveat: A `diffrax.SaveAt` object specifying when to save the solution.
        max_steps: The maximum number of steps for the solver.
    """

    network: ReactionNetwork
    solver: dfx.AbstractSolver
    T: float | jnp.ndarray | None = None
    saveat: dfx.SaveAt = dfx.SaveAt(t1=True)
    max_steps: int = int(1e4)

    def __init__(
        self,
        network: ReactionNetwork,
        solver: dfx.AbstractSolver = dfx.Dopri5(),
        T: float | jnp.ndarray | None = None,
        saveat_steps: int | list[float] | jnp.ndarray = -1,
        max_steps: int = int(1e4),
    ):
        """Initialize the mean field model.

        Args:
            network: The reaction network defining the system's structure.
            solver: The `diffrax` solver for the ODE simulation. Defaults to
                `diffrax.Dopri5()`.
            T: The final time of the simulation. If not provided, it must be
                specified during the call.
            saveat_steps: The time points at which to save the solution. Defaults
                to -1 (save only at the final time).
            max_steps: The maximum number of steps to take.
        """
        self.network = network
        self.solver = solver
        self.T = T
        self.max_steps = int(max_steps)

        if isinstance(saveat_steps, int) and saveat_steps == -1:
            self.saveat = -1
        elif isinstance(saveat_steps, int):
            self.saveat = jnp.linspace(0.0, T, int(saveat_steps)).tolist()
        else:
            self.saveat = saveat_steps

    @eqx.filter_jit()
    def __call__(
        self,
        key: jnp.ndarray | None,
        x0: jnp.ndarray,
        T: float | jnp.ndarray | None = None,
        max_steps: int | None = None,
        saveat_steps: int | list[float] | jnp.ndarray = None,
        **kwargs: Any,
    ) -> SimulationResults:
        """Execute the ODE simulation using diffrax.

        Args:
            key: For compatibility with stochastic models. This is ignored and can
                be safely set to `None`.
            x0: The initial state of the system (species concentrations/counts).
            T: The final time of the simulation. Overrides the model's `T` attribute.
            max_steps: Maximum number of steps to take. Overrides the model's
                `max_steps` attribute.
            saveat_steps: The time points to save the solution at. Overrides the
                model's `saveat` attribute.
            **kwargs: Additional keyword arguments to pass to `diffrax.diffeqsolve`.

        Returns:
            solution: The results of the simulation, adapted for ODE output.
        """
        if T is None:
            if self.T is None:
                t1 = jnp.nan
            t1 = self.T
        else:
            t1 = T

        if max_steps is None:
            if self.max_steps is None:
                max_steps = jnp.nan
            max_steps = self.max_steps

        if saveat_steps is None:
            saveat_steps = self.saveat

        if isinstance(saveat_steps, int) and saveat_steps == -1:
            saveat = dfx.SaveAt(t1=True)
        elif isinstance(saveat_steps, int):
            saveat = dfx.SaveAt(ts=jnp.linspace(0.0, t1, int(saveat_steps)))
        else:
            saveat = dfx.SaveAt(ts=saveat_steps)

        t0 = 0.0  # Assuming simulation starts at t=0

        # Get the ODE term from the network
        term = dfx.ODETerm(self.network.vector_field)

        stepsize_controller = dfx.PIDController(rtol=1e-3, atol=1e-6)

        # Solve the ODE
        solution = dfx.diffeqsolve(
            term,
            self.solver,
            t0=t0,
            t1=t1,
            dt0=None,
            y0=x0,
            max_steps=max_steps,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            throw=False,
            **kwargs,
        )

        # Adapt diffrax solution to SimulationResults format for compatibility with StochasticModel
        solution = SimulationResults(
            t=solution.ts,
            x=solution.ys,
            propensities=None,
            reactions=None,
            time_overflow=jnp.bool(solution.stats['num_steps'] == max_steps),
            species=self.network.species,
        )

        return solution
