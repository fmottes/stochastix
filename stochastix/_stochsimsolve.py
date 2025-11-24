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


def _do_step_logic(
    network: ReactionNetwork,
    solver: AbstractStochasticSolver,
    controller: AbstractController | None,
    T: float,
    t: jnp.ndarray,
    x: jnp.ndarray,
    controller_state: typing.Any,
    solver_state: typing.Any,
    key_step: jnp.ndarray,
    reaction_null: jnp.ndarray,
):
    """Core stepping logic that performs one simulation step.

    This function is shared between stochsimsolve and faststochsimsolve.
    """
    key_solver, key_controller = jax.random.split(key_step)

    # compute propensities
    a = solver.propensities(network, x, t)
    a0 = jnp.sum(a)

    def _do_step(args):
        network, t, x, a, solver_state, key_solver = args
        step_result, solver_state_new = solver.step(
            network, t, x, a, solver_state, key=key_solver
        )
        return step_result, solver_state_new

    def _stop(args):
        network, t, x, a, solver_state, key_solver = args
        dt = jnp.array(0.0)
        a_stop = jnp.zeros_like(a)

        step_result = SimulationStep(
            x_new=x,
            dt=dt,
            reaction_idx=reaction_null,
            propensities=a_stop,
        )

        return step_result, solver_state

    do_step = jnp.logical_and(t < T, a0 > 0)

    # stop if time is up or if there are no more reactions
    step_args = (network, t, x, a, solver_state, key_solver)

    step_result, solver_state_new = jax.lax.cond(do_step, _do_step, _stop, step_args)

    if controller is not None:
        step_result, controller_state = controller.step(
            t, step_result, controller_state, key_controller
        )

    # If controller is triggered it will set a new dt
    # It the new dt keeps the sim within the sim T boundary then it will be accepted
    new_t_candidate = t + step_result.dt

    # Reject state update if event would exceed time T
    # Can be confusing when calculating final state statistics
    would_exceed_T = new_t_candidate > T
    new_t = jnp.where(would_exceed_T, T, new_t_candidate)
    new_dt = jnp.where(would_exceed_T, T - t, step_result.dt)  # for consitency
    new_x = jnp.where(would_exceed_T, x, step_result.x_new)
    new_r = jnp.where(would_exceed_T, reaction_null, step_result.reaction_idx)
    new_a = jnp.where(would_exceed_T, jnp.zeros_like(a), step_result.propensities)

    new_step_result = SimulationStep(
        x_new=new_x,
        dt=new_dt,
        reaction_idx=new_r,
        propensities=new_a,
    )

    return new_t, new_x, controller_state, solver_state_new, new_step_result


def _initialize_simulation(
    key: jnp.ndarray,
    network: ReactionNetwork,
    solver: AbstractStochasticSolver,
    controller: AbstractController | None,
    x0: jnp.ndarray | dict[str, float] | typing.Any,
    t0: float,
):
    """Initialize solver and controller states.

    Returns:
        tuple: (key_loop, t0, x_init, a_init, solver_state0, controller_state, reaction_null)
    """
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

    # Define reaction_null matching solver type and shapes
    if solver.is_exact_solver:
        reaction_null = jnp.array(-1).astype(jnp.result_type(int))
    else:
        reaction_null = jnp.zeros_like(a_init)

    return key_loop, t0, x_init, a_init, solver_state0, controller_state, reaction_null


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
    save_trajectory: bool = True,
    save_propensities: bool = True,
    checkpoint: bool = False,
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
        save_trajectory: If True, store full trajectory. If False, store only initial
            and final states (shape (2, ...)). Defaults to True.
        save_propensities: If True, store propensities at each step. If False, set
            propensities to None. Defaults to True.
        checkpoint: If True, use `jax.checkpoint` (gradient checkpointing) on the
            simulation step. This reduces memory usage during backpropagation by
            recomputing intermediate states, at the cost of increased computation time.
            This is particularly useful for:
            1. Very long simulations (large `max_steps`), where storing the
               gradient tape for every step would cause OOM.
            2. Simulations with complex kinetics (e.g., neural network propensities),
               where the computational graph per step is large.
            When `save_trajectory=True`, checkpointing is especially efficient as
            recomputation starts from the stored trajectory states. Defaults to False.

    Returns:
        SimulationResults. An object with the following attributes:

            - `x`: State trajectory over time. Shape (n_timepoints, n_species) if
              `save_trajectory=True`, or (2, n_species) if `save_trajectory=False`.
            - `t`: Time points corresponding to state changes. Shape (n_timepoints,)
              if `save_trajectory=True`, or (2,) if `save_trajectory=False`.
            - `propensities`: Reaction propensities at each time step, or None if
              `save_propensities=False`.
            - `reactions`: Index of reactions that occurred at each step.
            - `time_overflow`: True if simulation stopped due to reaching
              max_steps before time T.

    Note: Performance considerations.
        This function is automatically JIT-compiled with `equinox.filter_jit`.
        The `save_trajectory` and `save_propensities` parameters are static (compile-time
        constants) and changing them will trigger re-compilation.
        Both modes support backward differentiation through `jax.lax.scan`. For maximum
        performance without differentiation (early stopping), use `faststochsimsolve`.

    Example: Basic usage
        See [Running Simulations](../user-guide/running-simulations.md) for more details.

        ```python
        network = stochastix.ReactionNetwork(...)
        x0 = jax.numpy.array(...)
        ssa_results = stochastix.stochsimsolve(key, network, x0, T=1000.0)
        ```

    Example: Memory-efficient simulation
        ```python
        # Only save initial and final states, no propensities
        results = stochastix.stochsimsolve(
            key, network, x0, T=1000.0,
            save_trajectory=False, save_propensities=False
        )
        ```
    """
    #########################################################
    # Solver and Controller Initialization
    #########################################################
    key_loop, t0, x_init, a_init, solver_state0, controller_state, reaction_null = (
        _initialize_simulation(key, network, solver, controller, x0, t0)
    )

    #########################################################
    # Simulation Loop using scan (supports differentiation)
    #########################################################
    # Stepping function for jax.lax.scan
    # Use recursive key splitting for consistency
    def _step(carry, _):
        t, x, controller_state, solver_state, key_current = carry

        # Recursive split for consistent random sequence
        key_step, key_next = jax.random.split(key_current)

        new_t, new_x, new_controller_state, new_solver_state, new_step_result = (
            _do_step_logic(
                network,
                solver,
                controller,
                T,
                t,
                x,
                controller_state,
                solver_state,
                key_step,
                reaction_null,
            )
        )
        new_carry = (new_t, new_x, new_controller_state, new_solver_state, key_next)

        if not save_trajectory:
            # Return None to avoid allocating history arrays
            return new_carry, None

        if not save_propensities:
            # Create a step result without propensities to save memory
            new_step_result = SimulationStep(
                x_new=new_step_result.x_new,
                dt=new_step_result.dt,
                reaction_idx=new_step_result.reaction_idx,
                propensities=None,
            )

        return new_carry, new_step_result

    if checkpoint:
        _step = jax.checkpoint(_step)

    # Include key_loop in the carry and use length parameter
    carry0 = (t0, x_init, controller_state, solver_state0, key_loop)

    final_carry, history = jax.lax.scan(_step, carry0, None, length=max_steps)

    # Extract final state from carry
    t_final, x_final, controller_state_final, solver_state_final, _ = final_carry

    if save_trajectory:
        # Mode: Full trajectory
        x_h = history.x_new
        dt_h = history.dt
        a_h = history.propensities
        r_h = history.reaction_idx

        # Prepend initial state
        xs = jnp.vstack([x_init, x_h])
        ts = jnp.cumsum(jnp.hstack([jnp.array(t0), dt_h]))

        # Check if simulation stopped due to time limit or max steps
        if save_propensities:
            time_overflow = jnp.logical_and(ts[-1] < T, a_h[-1].sum() > 0)
        else:
            # Need to check final propensities for time_overflow
            final_a = solver.propensities(network, x_h[-1], ts[-1])
            time_overflow = jnp.logical_and(ts[-1] < T, final_a.sum() > 0)

        xs = state_to_pytree(x0, network.species, xs)

        results = SimulationResults(
            x=xs,
            t=ts,
            propensities=a_h,  # Already None when save_propensities=False
            reactions=r_h,
            time_overflow=time_overflow,
            species=network.species,
        )
    else:
        # Mode: Only initial and final states
        # Stack initial and final states
        xs = jnp.vstack([x_init, x_final])
        ts = jnp.array([t0, t_final])

        # Compute propensities if needed
        if save_propensities:
            a_init_array = a_init
            a_final = solver.propensities(network, x_final, t_final)
            a_stack = jnp.vstack([a_init_array, a_final])
        else:
            a_stack = None

        # Check if simulation stopped due to time limit or max steps
        a_final_check = solver.propensities(network, x_final, t_final)
        time_overflow = jnp.logical_and(t_final < T, a_final_check.sum() > 0)

        xs = state_to_pytree(x0, network.species, xs)

        results = SimulationResults(
            x=xs,
            t=ts,
            propensities=a_stack,
            reactions=None,
            time_overflow=time_overflow,
            species=network.species,
        )

    return results


@eqx.filter_jit
def faststochsimsolve(
    key: jnp.ndarray,
    network: ReactionNetwork,
    x0: jnp.ndarray | dict[str, float] | typing.Any,
    t0: float = 0.0,
    T: float = 3600.0,
    max_steps: int = int(1e5),
    solver: AbstractStochasticSolver = DirectMethod(),
    controller: AbstractController | None = None,
    save_propensities: bool = True,
) -> SimulationResults:
    """Fast stochastic simulation using while_loop (stops early, forward differentiation only).

    This function performs stochastic simulation using `jax.lax.while_loop`, which allows
    early termination when the simulation finishes before reaching max_steps. This makes
    it considerably faster than `stochsimsolve` for simulations that finish early, but it does NOT
    support backward differentiation (reverse-mode autodiff). It also does not support saving the trajectory.

    Args:
        key: JAX random number generator key.
        network: The reaction network object containing reactions and their properties.
        x0: Initial state vector (counts). Can be an array, a dictionary mapping
            species names to their initial counts, a named tuple, or an Equinox module.
        t0: Initial time.
        T: Final simulation time in seconds.
        max_steps: Maximum number of simulation steps (safety limit).
        solver: The stochastic solver to use.
        controller: External state control during simulation.
        save_propensities: If True, store initial and final propensities. If False, set
            propensities to None. Defaults to True.

    Returns:
        SimulationResults. An object with the following attributes:

            - `x`: Initial and final states only, shape (2, n_species).
            - `t`: Initial and final times, shape (2,).
            - `propensities`: Initial and final propensities (shape (2, n_reactions)) if
              `save_propensities=True`, or None otherwise.
            - `reactions`: None (not tracked in while_loop mode).
            - `time_overflow`: True if simulation stopped due to reaching max_steps before time T.

    Note: Forward differentiation only.
        This function is automatically JIT-compiled with `equinox.filter_jit`.
        Supports forward differentiation, DOES NOT support backward differentiation.
        Use `stochsimsolve` if you need backward differentiation.

    Note: No trajectory saving.
        This function does not support saving the whole trajectory. Use `stochsimsolve` if you need to save the trajectory.

    Example:
        ```python
        # Fast simulation without backward differentiation
        results = stochastix.faststochsimsolve(key, network, x0, T=1000.0)
        ```
    """
    #########################################################
    # Solver and Controller Initialization
    #########################################################
    key_loop, t0, x_init, a_init, solver_state0, controller_state, reaction_null = (
        _initialize_simulation(key, network, solver, controller, x0, t0)
    )

    #########################################################
    # Simulation Loop using while_loop (stops early)
    #########################################################
    def _cond(carry):
        t, x, controller_state, solver_state, step_count, key_current = carry
        a = solver.propensities(network, x, t)
        a0 = jnp.sum(a)
        return jnp.logical_and(jnp.logical_and(t < T, a0 > 0), step_count < max_steps)

    def _body(carry):
        t, x, controller_state, solver_state, step_count, key_current = carry
        key_step, key_next = jax.random.split(key_current)
        new_t, new_x, new_controller_state, new_solver_state, new_step_result = (
            _do_step_logic(
                network,
                solver,
                controller,
                T,
                t,
                x,
                controller_state,
                solver_state,
                key_step,
                reaction_null,
            )
        )
        return (
            new_t,
            new_x,
            new_controller_state,
            new_solver_state,
            step_count + 1,
            key_next,
        )

    carry0 = (t0, x_init, controller_state, solver_state0, 0, key_loop)
    final_carry = jax.lax.while_loop(_cond, _body, carry0)
    (
        t_final,
        x_final,
        controller_state_final,
        solver_state_final,
        step_count_final,
        _,
    ) = final_carry

    # Stack initial and final states
    xs = jnp.vstack([x_init, x_final])
    ts = jnp.array([t0, t_final])

    # Compute propensities if needed
    if save_propensities:
        a_init_array = a_init
        a_final = solver.propensities(network, x_final, t_final)
        a_stack = jnp.vstack([a_init_array, a_final])
    else:
        a_stack = None

    # Check if simulation stopped due to time limit or max steps
    a_final_check = solver.propensities(network, x_final, t_final)
    time_overflow = jnp.logical_and(
        t_final < T,
        jnp.logical_and(a_final_check.sum() > 0, step_count_final >= max_steps),
    )

    xs = state_to_pytree(x0, network.species, xs)

    results = SimulationResults(
        x=xs,
        t=ts,
        propensities=a_stack,
        reactions=None,
        time_overflow=time_overflow,
        species=network.species,
    )

    return results
