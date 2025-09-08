import collections

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from stochastix import (
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
    # We test that it runs correctly by simply calling it. Nesting JITs is
    # an anti-pattern and can cause low-level memory errors.
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
