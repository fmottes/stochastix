"""Timer-based controller for species count manipulation at predefined time points."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from ..solvers import SimulationStep
from ._base import AbstractController


class Timer(AbstractController):
    """A controller that activates at specific time points.

    This controller allows for direct manipulation of species counts at predefined
    time points during a simulation. When a trigger time is reached, the controller
    updates the counts of the specified species to predefined values.

    Attributes:
        controlled_species: The species whose counts will be manipulated.
        time_triggers: The time points at which the controller activates.
        species_at_triggers: The new counts for the controlled species at each
            trigger time.

    Methods:
        init: Initialize the controller's state for the simulation.
        step: Execute a step of the Timer controller.
    """

    controlled_species: str | tuple[str, ...] = eqx.field(static=True)
    time_triggers: jnp.ndarray
    species_at_triggers: jnp.ndarray

    def __init__(
        self,
        controlled_species: str | list[str] | tuple[str, ...],
        time_triggers: jnp.ndarray,
        species_at_triggers: jnp.ndarray,
    ):
        """Initialize the Timer controller.

        Args:
            controlled_species: The name of the species or a list/tuple of
                species names to be controlled.
            time_triggers: A JAX array of time points at which the controller
                should activate.
            species_at_triggers: A JAX array where each row corresponds to a
                time trigger and contains the new counts for the controlled
                species. The order of species counts in each row must match
                the order of species names in `controlled_species`.

        Raises:
            ValueError: If `controlled_species` is not a string, list, or tuple.
            ValueError: If the number of time triggers does not match the number
                of species count updates.
            ValueError: If the number of controlled species does not match the
                number of species counts in each update.
        """
        if isinstance(controlled_species, str):
            _controlled_species = (controlled_species,)
        elif isinstance(controlled_species, list):
            _controlled_species = tuple(controlled_species)
        elif isinstance(controlled_species, tuple):
            _controlled_species = controlled_species
        else:
            raise ValueError(
                '`controlled_species` must be a string, a list or a tuple of strings of species names'
            )

        # check species_at_triggers and triggers have the same length
        if len(time_triggers) != len(species_at_triggers):
            raise ValueError('The number of triggers and states must be the same')

        if len(_controlled_species) != len(species_at_triggers[0]):
            raise ValueError(
                'The number of controlled species and provided updates must be the same'
            )

        self.controlled_species = _controlled_species
        self.time_triggers = tuple(jnp.array(time_triggers).tolist())
        self.species_at_triggers = tuple(
            [tuple(s) for s in jnp.array(species_at_triggers).tolist()]
        )

    def init(self, network, t, x, a, *, key=None):
        """Initialize the controller's state for the simulation.

        Args:
            network: The reaction network model.
            t: The initial time of the simulation.
            x: The initial species counts.
            a: The initial propensities.
            key: A JAX random key (not used by this controller).

        Returns:
            A tuple representing the initial state of the controller, containing:
                - The index of the next time trigger.
                - The time of the next trigger.
                - An array of indices for the controlled species.
        """
        trigger_idx = jnp.int16(0)
        next_trigger_time = self.time_triggers[0]
        species_idx = jnp.array(
            [network.species.index(species) for species in self.controlled_species]
        )

        return trigger_idx, next_trigger_time, species_idx

    def step(self, t, step_result, controller_state, key):
        """Execute a step of the Timer controller.

        This method checks if the simulation time has crossed a trigger point.
        If so, it updates the species counts to the predefined values for that
        trigger and advances to the next trigger time. Otherwise, it returns
        the simulation state unchanged.

        Args:
            t: The current simulation time.
            step_result: A `SimulationStep` object from the solver.
            controller_state: The current state of the controller.
            key: A JAX random key (not used by this controller).

        Returns:
            A tuple containing the (potentially modified) `SimulationStep` and
            the updated controller state.
        """

        def _triggered(step_result, controller_state):
            x = step_result.x_new
            a = step_result.propensities

            trigger_idx, next_trigger_time, species_idx = controller_state

            new_species = jnp.asarray(self.species_at_triggers)[trigger_idx]

            x_new = x.at[species_idx].set(new_species)
            dt = next_trigger_time - t
            a_new = jnp.zeros(a.shape, dtype=a.dtype)
            r_new = -2

            new_trigger_idx = trigger_idx + 1
            new_next_trigger_time = jax.lax.cond(
                new_trigger_idx < len(self.time_triggers),
                lambda: jnp.asarray(self.time_triggers)[new_trigger_idx],
                lambda: jnp.inf,
            )
            new_state = (new_trigger_idx, new_next_trigger_time, species_idx)

            new_step_result = SimulationStep(
                x_new=x_new,
                dt=dt,
                reaction_idx=r_new,
                propensities=a_new,
            )
            return new_step_result, new_state

        def _not_triggered(step_result, controller_state):
            return step_result, controller_state

        _, next_trigger_time, _ = controller_state
        triggered = t + step_result.dt >= next_trigger_time

        return jax.lax.cond(
            triggered,
            _triggered,
            _not_triggered,
            step_result,
            controller_state,
        )
