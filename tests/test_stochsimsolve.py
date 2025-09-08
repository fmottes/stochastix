import collections

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from stochastix import (
    faststochsimsolve,
    pytree_to_state,
    state_to_pytree,
    stochsimsolve,
)
from stochastix.kinetics import MassAction
from stochastix.reaction import Reaction, ReactionNetwork


@pytest.fixture
def simple_network():
    reactions = [
        Reaction('0 -> A', MassAction(k=1.0)),
        Reaction('A -> 0', MassAction(k=0.1)),
        Reaction('0 -> B', MassAction(k=1.0)),
        Reaction('B -> 0', MassAction(k=0.1)),
    ]
    return ReactionNetwork(reactions)


def test_array_input_smoke(simple_network):
    key = jax.random.PRNGKey(0)
    x0 = jnp.array([10, 20])
    res = stochsimsolve(key, simple_network, x0, T=0.5)
    assert isinstance(res.x, jnp.ndarray)
    assert res.x.shape[1] == 2
    assert jnp.allclose(res.x[0], jnp.array([10.0, 20.0]))


class SpeciesContainer(eqx.Module):
    A: jnp.ndarray
    B: jnp.ndarray

    def __init__(self, A, B):
        self.A = A
        self.B = B


class SpeciesContainerWithExtra(eqx.Module):
    A: jnp.ndarray
    B: jnp.ndarray
    meta: dict

    def __init__(self, A, B, meta):
        self.A = A
        self.B = B
        self.meta = meta


@pytest.mark.parametrize(
    'x0_input',
    [
        {'A': 10, 'B': 20},
        collections.namedtuple('Species', ['A', 'B'])(A=10, B=20),
        SpeciesContainer(A=10, B=20),
        SpeciesContainerWithExtra(A=10, B=20, meta={'info': 123}),
    ],
)
def test_pytree_roundtrip(simple_network, x0_input):
    key = jax.random.PRNGKey(1)
    res = stochsimsolve(key, simple_network, x0_input, T=0.5)

    # Check structure and content
    if isinstance(x0_input, dict):
        assert isinstance(res.x, dict)
        assert set(res.x.keys()) >= {'A', 'B'}
        assert res.x['A'].shape[0] == res.t.shape[0]
        assert res.x['B'].shape[0] == res.t.shape[0]
        assert float(res.x['A'][0]) == 10.0
        assert float(res.x['B'][0]) == 20.0
    else:
        assert hasattr(res.x, 'A') and hasattr(res.x, 'B')
        assert res.x.A.shape[0] == res.t.shape[0]
        assert res.x.B.shape[0] == res.t.shape[0]
        assert float(res.x.A[0]) == 10.0
        assert float(res.x.B[0]) == 20.0
        if hasattr(x0_input, 'meta'):
            assert hasattr(res.x, 'meta')
            assert res.x.meta == x0_input.meta


@pytest.mark.parametrize(
    'invalid_x0, match_str',
    [
        ({'A': 10}, 'Species B'),
        (jnp.array([1, 2, 3]), 'length must match'),
    ],
)
def test_invalid_inputs(simple_network, invalid_x0, match_str):
    key = jax.random.PRNGKey(2)
    with pytest.raises(ValueError, match=match_str):
        stochsimsolve(key, simple_network, invalid_x0, T=0.5)


def test_single_species_scalar():
    key = jax.random.PRNGKey(3)
    net = ReactionNetwork([Reaction('0 -> A', MassAction(k=1.0))])
    res = stochsimsolve(key, net, x0=10, T=0.5)
    print(res.x)
    assert isinstance(res.x, jnp.ndarray)
    assert res.x.shape[1] == 1
    assert float(res.x[0, 0]) == 10.0


def test_helpers_basic(simple_network):
    # Dict
    x0 = {'A': 3, 'B': 7}
    state = pytree_to_state(x0, simple_network.species)
    assert jnp.allclose(state, jnp.array([3.0, 7.0]))

    traj = jnp.stack([state, state + 1], axis=0)
    x_py = state_to_pytree(x0, simple_network.species, traj)
    assert isinstance(x_py, dict)
    assert jnp.allclose(x_py['A'], jnp.array([3.0, 4.0]))
    assert jnp.allclose(x_py['B'], jnp.array([7.0, 8.0]))

    # Namedtuple
    NT = collections.namedtuple('Species', ['A', 'B'])
    x0_nt = NT(A=5, B=9)
    st = pytree_to_state(x0_nt, simple_network.species)
    assert jnp.allclose(st, jnp.array([5.0, 9.0]))

    traj_nt = jnp.stack([st, st + 2], axis=0)
    x_nt = state_to_pytree(x0_nt, simple_network.species, traj_nt)
    assert hasattr(x_nt, 'A') and hasattr(x_nt, 'B')
    assert jnp.allclose(x_nt.A, jnp.array([5.0, 7.0]))
    assert jnp.allclose(x_nt.B, jnp.array([9.0, 11.0]))


def test_jit_with_pytree(simple_network):
    key = jax.random.PRNGKey(4)
    x0 = {'A': 2, 'B': 4}
    # The stochsimsolve function is already decorated with @eqx.filter_jit.
    # We test that it runs correctly by simply calling it.
    res = stochsimsolve(key, simple_network, x0, T=0.4)
    assert isinstance(res.x, dict)
    assert res.x['A'].shape[0] == res.t.shape[0]


def test_vmap_array_input(simple_network):
    key = jax.random.PRNGKey(5)
    keys = jax.random.split(key, 3)
    x0 = jnp.array([5.0, 6.0])

    fn = lambda k: stochsimsolve(k, simple_network, x0, T=0.3)
    vmapped = eqx.filter_vmap(fn, in_axes=0)
    batched = vmapped(keys)

    assert isinstance(batched.x, jnp.ndarray)
    assert batched.x.shape[0] == 3
    assert batched.x.shape[2] == 2


def test_vmap_pytree_input(simple_network):
    key = jax.random.PRNGKey(6)
    keys = jax.random.split(key, 2)
    x0 = {'A': 1, 'B': 2}

    fn = lambda k: stochsimsolve(k, simple_network, x0, T=0.2)
    vmapped = eqx.filter_vmap(fn, in_axes=0)
    batched = vmapped(keys)

    assert isinstance(batched.x, dict)
    assert batched.x['A'].shape[0] == 2
    assert batched.x['A'].shape[1] == batched.t.shape[1]


def test_save_trajectory_false(simple_network):
    """Test that save_trajectory=False returns only initial and final states."""
    key = jax.random.PRNGKey(7)
    x0 = jnp.array([10, 20])
    res = stochsimsolve(key, simple_network, x0, T=0.5, save_trajectory=False)

    assert isinstance(res.x, jnp.ndarray)
    assert res.x.shape == (2, 2)  # (initial, final) x (A, B)
    assert res.t.shape == (2,)
    assert jnp.allclose(res.x[0], jnp.array([10.0, 20.0]))  # Initial state
    assert res.t[0] == 0.0  # Initial time
    assert res.t[1] > res.t[0]  # Final time > initial time


def test_save_trajectory_false_pytree(simple_network):
    """Test save_trajectory=False with pytree input."""
    key = jax.random.PRNGKey(8)
    x0 = {'A': 10, 'B': 20}
    res = stochsimsolve(key, simple_network, x0, T=0.5, save_trajectory=False)

    assert isinstance(res.x, dict)
    assert res.x['A'].shape == (2,)
    assert res.x['B'].shape == (2,)
    assert res.t.shape == (2,)
    assert float(res.x['A'][0]) == 10.0
    assert float(res.x['B'][0]) == 20.0


def test_save_propensities_false(simple_network):
    """Test that save_propensities=False sets propensities to None."""
    key = jax.random.PRNGKey(9)
    x0 = jnp.array([10, 20])
    res = stochsimsolve(key, simple_network, x0, T=0.5, save_propensities=False)

    assert res.propensities is None
    assert res.x.shape[0] > 2  # Full trajectory still saved
    assert res.t.shape[0] > 2


def test_save_trajectory_and_propensities_false(simple_network):
    """Test both save_trajectory=False and save_propensities=False."""
    key = jax.random.PRNGKey(10)
    x0 = jnp.array([10, 20])
    res = stochsimsolve(
        key,
        simple_network,
        x0,
        T=0.5,
        save_trajectory=False,
        save_propensities=False,
    )

    assert res.x.shape == (2, 2)  # Only initial and final
    assert res.t.shape == (2,)
    assert res.propensities is None
    assert res.reactions is None  # Reactions also None in while_loop mode


def test_save_trajectory_consistency(simple_network):
    """Test that save_trajectory=True and False produce identical final states with same key."""
    key = jax.random.PRNGKey(11)
    x0 = jnp.array([10, 20])

    # Run with full trajectory
    res_full = stochsimsolve(key, simple_network, x0, T=0.5, save_trajectory=True)

    # Run with only initial/final (same key)
    res_final = stochsimsolve(key, simple_network, x0, T=0.5, save_trajectory=False)

    # Final states should be identical
    assert jnp.allclose(res_full.x[-1], res_final.x[-1])
    assert jnp.allclose(res_full.t[-1], res_final.t[-1])

    # Initial states should be identical
    assert jnp.allclose(res_full.x[0], res_final.x[0])
    assert jnp.allclose(res_full.t[0], res_final.t[0])

    # time_overflow should be the same
    assert res_full.time_overflow == res_final.time_overflow


def test_save_trajectory_consistency_pytree(simple_network):
    """Test consistency with pytree inputs."""
    key = jax.random.PRNGKey(12)
    x0 = {'A': 5, 'B': 15}

    # Run with full trajectory
    res_full = stochsimsolve(key, simple_network, x0, T=0.5, save_trajectory=True)

    # Run with only initial/final (same key)
    res_final = stochsimsolve(key, simple_network, x0, T=0.5, save_trajectory=False)

    # Final states should be identical
    if isinstance(res_full.x, dict):
        assert jnp.allclose(res_full.x['A'][-1], res_final.x['A'][-1])
        assert jnp.allclose(res_full.x['B'][-1], res_final.x['B'][-1])
    else:
        assert jnp.allclose(res_full.x[-1], res_final.x[-1])

    assert jnp.allclose(res_full.t[-1], res_final.t[-1])
    assert res_full.time_overflow == res_final.time_overflow


def test_differentiation_with_save_trajectory_false(simple_network):
    """Test that gradients work with save_trajectory=False using differentiable solver."""
    from stochastix.solvers import DifferentiableDirect

    solver = DifferentiableDirect(logits_scale=1.0, exact_fwd=True)

    def simulate_and_loss(k_birth, k_death):
        r_birth = Reaction('0 -> A', MassAction(k=k_birth))
        r_death = Reaction('A -> 0', MassAction(k=k_death))
        network = ReactionNetwork([r_birth, r_death])
        x0 = jnp.array([0.0])
        T = 10.0
        key = jax.random.PRNGKey(42)

        @jax.jit
        def run_sim():
            # Now save_trajectory=False supports differentiation (uses scan)
            results = stochsimsolve(
                key,
                network,
                x0,
                T=T,
                solver=solver,
                max_steps=int(1e4),
                save_trajectory=False,
            )
            return results.x[-1, 0]  # Final state

        final_count = run_sim()
        target_count = 20.0
        return (final_count - target_count) ** 2

    k_birth_initial = 10.0
    k_death_initial = 1.0

    # Test that gradients can be computed
    grad_k_birth = jax.grad(simulate_and_loss, argnums=0)(
        k_birth_initial, k_death_initial
    )
    grad_k_death = jax.grad(simulate_and_loss, argnums=1)(
        k_birth_initial, k_death_initial
    )

    # Gradients should be finite
    assert jnp.isfinite(grad_k_birth)
    assert jnp.isfinite(grad_k_death)
    # Expected signs: increasing birth increases final count (negative grad for loss)
    # increasing death decreases final count (positive grad for loss)
    assert grad_k_birth < 0
    assert grad_k_death > 0


def test_faststochsimsolve(simple_network):
    """Test that faststochsimsolve works correctly."""
    key = jax.random.PRNGKey(13)
    x0 = jnp.array([10, 20])
    res = faststochsimsolve(key, simple_network, x0, T=0.5)

    assert isinstance(res.x, jnp.ndarray)
    assert res.x.shape == (2, 2)  # (initial, final) x (A, B)
    assert res.t.shape == (2,)
    assert jnp.allclose(res.x[0], jnp.array([10.0, 20.0]))  # Initial state
    assert res.t[0] == 0.0  # Initial time
    assert res.t[1] > res.t[0]  # Final time > initial time
    assert res.reactions is None  # Reactions not tracked in while_loop mode


def test_faststochsimsolve_consistency(simple_network):
    """Test that faststochsimsolve produces same results as stochsimsolve with same key."""
    key = jax.random.PRNGKey(14)
    x0 = jnp.array([10, 20])

    # Run with faststochsimsolve (while_loop, stops early)
    res_fast = faststochsimsolve(key, simple_network, x0, T=0.5)

    # Run with stochsimsolve save_trajectory=False (scan, same key)
    res_scan = stochsimsolve(key, simple_network, x0, T=0.5, save_trajectory=False)

    # Final states should be identical (same random sequence)
    assert jnp.allclose(res_fast.x[-1], res_scan.x[-1])
    assert jnp.allclose(res_fast.t[-1], res_scan.t[-1])
    assert jnp.allclose(res_fast.x[0], res_scan.x[0])
    assert jnp.allclose(res_fast.t[0], res_scan.t[0])


def test_faststochsimsolve_no_differentiation(simple_network):
    """Test that faststochsimsolve does not support differentiation."""
    from stochastix.solvers import DifferentiableDirect

    solver = DifferentiableDirect(logits_scale=1.0, exact_fwd=True)

    def simulate_and_loss(k_birth, k_death):
        r_birth = Reaction('0 -> A', MassAction(k=k_birth))
        r_death = Reaction('A -> 0', MassAction(k=k_death))
        network = ReactionNetwork([r_birth, r_death])
        x0 = jnp.array([0.0])
        T = 10.0
        key = jax.random.PRNGKey(42)

        @jax.jit
        def run_sim():
            results = faststochsimsolve(
                key,
                network,
                x0,
                T=T,
                solver=solver,
                max_steps=int(1e4),
            )
            return results.x[-1, 0]  # Final state

        final_count = run_sim()
        target_count = 20.0
        return (final_count - target_count) ** 2

    k_birth_initial = 10.0
    k_death_initial = 1.0

    # Test that gradients cannot be computed (should raise error)
    with pytest.raises(ValueError, match='Reverse-mode differentiation'):
        jax.grad(simulate_and_loss, argnums=0)(k_birth_initial, k_death_initial)
