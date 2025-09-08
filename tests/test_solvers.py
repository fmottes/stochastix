import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from stochastix import stochsimsolve
from stochastix.kinetics import MassAction
from stochastix.reaction import Reaction, ReactionNetwork
from stochastix.solvers import (
    DGA,
    DifferentiableDirect,
    DifferentiableFirstReaction,
    DirectMethod,
    FirstReactionMethod,
    TauLeaping,
)


def test_direct_method_birth_process():
    """Tests the DirectMethod solver on a pure birth process.

    The number of molecules at time T should follow a Poisson distribution.
    This test is deterministic by using a fixed random seed.
    """
    # Define a pure birth process: 0 -> A
    k = 10.0
    r1 = Reaction('0 -> A', MassAction(k=k))
    network = ReactionNetwork([r1])

    # Simulation parameters
    x0 = jnp.array([0])
    T = 1000.0  # Use a single, long run for stability
    key = jax.random.PRNGKey(42)

    # Run one long simulation
    results = stochsimsolve(
        key, network, x0, T=T, solver=DirectMethod(), max_steps=int(2e5)
    )

    # The final count should be close to the expected mean k * T
    final_count = results.x[-1, 0]
    expected_mean = k * T

    # Check if the final count is within a reasonable tolerance of the mean.
    # For a single long run, the final count should be very close to the mean.
    assert jnp.isclose(final_count, expected_mean, rtol=0.1)


def test_first_reaction_method_birth_process():
    """Tests the FirstReactionMethod solver on a pure birth process.

    The number of molecules at time T should follow a Poisson distribution.
    This test is deterministic by using a fixed random seed.
    """
    # Define a pure birth process: 0 -> A
    k = 10.0
    r1 = Reaction('0 -> A', MassAction(k=k))
    network = ReactionNetwork([r1])

    # Simulation parameters
    x0 = jnp.array([0])
    T = 1000.0  # Use a single, long run for stability
    key = jax.random.PRNGKey(42)

    # Run one long simulation
    results = stochsimsolve(
        key, network, x0, T=T, solver=FirstReactionMethod(), max_steps=int(2e5)
    )

    # The final count should be close to the expected mean k * T
    final_count = results.x[-1, 0]
    expected_mean = k * T

    # Check if the final count is within a reasonable tolerance of the mean.
    # For a single long run, the final count should be very close to the mean.
    assert jnp.isclose(final_count, expected_mean, rtol=0.1)


def test_tau_leaping_mean_matches_theory():
    """Compare TauLeaping to analytic steady-state mean for birth–death.

    For birth (rate k_birth) and death (rate k_death * X), the steady-state mean is
    E[X] = k_birth / k_death. We average multiple TauLeaping runs and compare to this
    analytic mean. This avoids running an exact solver and keeps the test fast.
    """
    k_birth = 10.0
    k_death = 1.0
    expected_mean = k_birth / k_death

    r_birth = Reaction('0 -> A', MassAction(k=k_birth))
    r_death = Reaction('A -> 0', MassAction(k=k_death))
    network = ReactionNetwork([r_birth, r_death])

    x0 = jnp.array([0])
    T = 300.0  # Enough to reach near steady state, but faster
    key = jax.random.PRNGKey(123)

    num_runs = 16
    keys = jax.random.split(key, num_runs)

    solver_approx = TauLeaping(epsilon=0.01)

    def _run(k):
        return stochsimsolve(
            k, network, x0, T=T, solver=solver_approx, max_steps=int(1e5)
        )

    vm = eqx.filter_vmap(_run, in_axes=0)
    res_batched = vm(keys)
    final_counts = res_batched.x[:, -1, 0]
    mean_final = jnp.mean(final_counts)

    assert jnp.isclose(mean_final, expected_mean, rtol=0.3)


@pytest.mark.parametrize(
    'solver_class', [DifferentiableDirect, DGA, DifferentiableFirstReaction]
)
def test_differentiable_solvers_grad(solver_class):
    """Tests that differentiable solvers produce non-zero gradients with the correct sign."""

    # Define a birth-death process where we can control the rates
    def simulate_and_loss(k_birth, k_death, solver):
        r_birth = Reaction('0 -> A', MassAction(k=k_birth))
        r_death = Reaction('A -> 0', MassAction(k=k_death))
        network = ReactionNetwork([r_birth, r_death])
        x0 = jnp.array([0.0])
        T = 100.0
        key = jax.random.PRNGKey(42)

        # We need to jit the simulation for the gradients to flow
        @jax.jit
        def run_sim():
            results = stochsimsolve(
                key, network, x0, T=T, solver=solver, max_steps=int(1e4)
            )
            return results.x[-1, 0]

        final_count = run_sim()

        # Loss is the squared difference from a target
        target_count = 20.0
        return (final_count - target_count) ** 2

    # Initial parameters
    k_birth_initial = 10.0
    k_death_initial = 1.0
    solver = solver_class()

    # The steady state is k_birth / k_death = 10.
    # The target is 20.
    # To decrease the loss, we need to increase the final count.
    # Increasing k_birth increases the final count, so grad(loss) wrt k_birth should be negative.
    # Increasing k_death decreases the final count, so grad(loss) wrt k_death should be positive.

    # Gradient with respect to k_birth
    grad_k_birth = jax.grad(simulate_and_loss, argnums=0)(
        k_birth_initial, k_death_initial, solver
    )
    assert grad_k_birth < 0

    # Gradient with respect to k_death
    grad_k_death = jax.grad(simulate_and_loss, argnums=1)(
        k_birth_initial, k_death_initial, solver
    )
    assert grad_k_death > 0
