"""High-level interface for stochastic simulation algorithm execution."""

from __future__ import annotations

import typing

import equinox as eqx
import jax
import jax.numpy as jnp

from ._simulation_results import SimulationResults
from ._state_utils import pytree_to_state, state_to_pytree
from .solvers import DirectMethod, SimulationStep

if typing.TYPE_CHECKING:
    from .controllers import AbstractController
    from .reaction import ReactionNetwork
    from .solvers import AbstractStochasticSolver


@eqx.filter_jit
def stochsimsolve(
    key: jnp.ndarray,
    network: ReactionNetwork,
    x0: jnp.ndarray | dict[str, float] | typing.Any,
    t0: float = 0.0,
    T: float = 3600.0,
    max_steps: int = int(1e5),
    solver: AbstractStochasticSolver = DirectMethod(),
    controller: AbstractController | None = None,
) -> SimulationResults:
    """Run a stochastic simulation of a reaction network.

    This function performs stochastic simulation of reaction networks. The simulation
    continues until either the specified time limit is reached, no more reactions can
    occur, or the maximum number of updates is exceeded.

    Args:
        key: JAX random number generator key.
        network: The reaction network object containing reactions and their properties.
        x0: Initial state vector (counts). Can be an array, a dictionary mapping
            species names to their initial counts, a named tuple, or an Equinox module.
        t0: Initial time.
        T: Final simulation time in seconds.
        max_steps: Maximum number of simulation steps (needed for jit compilation).
        solver: The stochastic solver to use.
        controller: External state control during simulation.

    Returns:
        SimulationResults. An object with the following attributes:

            - `x`: State trajectory over time, shape (n_timepoints, n_species).
            - `t`: Time points corresponding to state changes.
            - `propensities`: Reaction propensities at each time step.
            - `reactions`: Index of reactions that occurred at each step.
            - `time_overflow`: True if simulation stopped due to reaching
              max_steps before time T.

    Note:
        This function is automatically JIT-compiled with `equinox.filter_jit`.

    Example: Basic usage
        See [Running Simulations](../user-guide/running-simulations.md) for more details.

        ```python
        network = stochastix.ReactionNetwork(...)
        x0 = jax.numpy.array(...)
        ssa_results = stochastix.stochsimsolve(key, network, x0, T=1000.0)
        ```
    """

    # Stepping function for jax.lax.scan
    def _step(carry, key):
        def _do_step(args):
            network, t, x, a, solver_state, key_solver = args
            # we need to pass the key twice, once for the step, once for the stop
            step_result, solver_state_new = solver.step(
                network, t, x, a, solver_state, key=key_solver
            )
            return step_result, solver_state_new

        def _stop(args):
            network, t, x, a, solver_state, key_solver = args
            dt = jnp.array(0.0)
            a_stop = jnp.zeros_like(a)
            if solver.is_exact_solver:
                r = jnp.array(-1).astype(jnp.result_type(int))
            else:
                r = -jnp.ones_like(a)

            step_result = SimulationStep(
                x_new=x,
                dt=dt,
                reaction_idx=r,
                propensities=a_stop,
            )

            return step_result, solver_state

        t, x, controller_state, solver_state = carry

        key_solver, key_controller = jax.random.split(key)

        # compute propensities
        a = solver.propensities(network, x, t)
        a0 = jnp.sum(a)

        do_step = jnp.logical_and(t < T, a0 > 0)

        # stop if time is up or if there are no more reactions
        step_args = (network, t, x, a, solver_state, key_solver)

        step_result, solver_state_new = jax.lax.cond(
            do_step, _do_step, _stop, step_args
        )

        if controller is not None:
            step_result, controller_state = controller.step(
                t, step_result, controller_state, key_controller
            )

        new_t = t + step_result.dt
        new_x = step_result.x_new

        new_carry = (new_t, new_x, controller_state, solver_state_new)
        return new_carry, step_result

    #########################################################
    # Solver and Controller Initialization
    #########################################################
    key_init, key_loop = jax.random.split(key)

    t0 = jnp.array(t0)

    x_init = pytree_to_state(x0, network.species)

    a_init = solver.propensities(network, x_init, t0)

    solver_state0 = solver.init(network, t0, x_init, a_init, key=key_init)

    controller_state = None
    if controller is not None:
        key_init, key_controller_init = jax.random.split(key_init)
        controller_state = controller.init(
            network, t0, x_init, a_init, key=key_controller_init
        )

    #########################################################
    # Simulation Loop
    #########################################################
    keys = jax.random.split(key_loop, max_steps)
    carry0 = (t0, x_init, controller_state, solver_state0)

    _, history = jax.lax.scan(_step, carry0, keys)

    x_h = history.x_new
    dt_h = history.dt
    a_h = history.propensities
    r_h = history.reaction_idx

    # Prepend initial state
    xs = jnp.vstack([x_init, x_h])
    ts = jnp.cumsum(jnp.hstack([jnp.array(t0), dt_h]))

    # Check if simulation stopped due to time limit or max steps
    time_overflow = jnp.logical_and(ts[-1] < T, a_h[-1].sum() > 0)

    xs = state_to_pytree(x0, network.species, xs)

    results = SimulationResults(
        x=xs,
        t=ts,
        propensities=a_h,
        reactions=r_h,
        time_overflow=time_overflow,
        species=network.species,
    )

    return results
