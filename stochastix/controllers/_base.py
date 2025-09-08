"""Abstract base class for simulation controllers."""

from __future__ import annotations

import abc
import typing

import equinox as eqx
import jax.numpy as jnp

if typing.TYPE_CHECKING:
    from ..reaction import ReactionNetwork
    from ..solvers import SimulationStep


class AbstractController(eqx.Module):
    """Base class for controllers.

    A controller is a module that can be used to modify the simulation behavior
    during execution.

    Methods:
        init: Initializes the controller's state.
        step: Performs a single control step.
    """

    def init(
        self,
        network: ReactionNetwork,
        t: jnp.floating,
        x: jnp.ndarray,
        a: jnp.ndarray,
        *,
        key: jnp.ndarray,
    ) -> typing.Any:
        """Initialize the controller's state.

        This method is called once at the beginning of the simulation to set up
        the initial state of the controller.

        Args:
            network: The reaction network being simulated.
            t: The initial time of the simulation.
            x: The initial state vector (species counts).
            a: The initial propensity vector.
            key: A JAX random key for any stochastic initialization.

        Returns:
            The initial state of the controller. Can be any PyTree.
        """
        return None

    @abc.abstractmethod
    def step(
        self,
        t: jnp.floating,
        step_result: SimulationStep,
        controller_state: typing.Any,
        key: jnp.ndarray,
    ) -> tuple[SimulationStep, typing.Any]:
        """Perform a single control step.

        This method is called at each step of the simulation, allowing the
        controller to inspect the simulation's progress and modify the state
        if necessary.

        Args:
            t: The current time of the simulation.
            step_result: The result from the solver's step.
            controller_state: The current state of the controller.
            key: A JAX random key for any stochastic operations.

        Returns:
            A tuple containing the (potentially modified) simulation step result
            and the new state of the controller.
        """
        raise NotImplementedError
