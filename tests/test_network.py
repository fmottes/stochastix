import jax.numpy as jnp

from stochastix.kinetics import MassAction
from stochastix.reaction import Reaction, ReactionNetwork


def test_network_species_discovery():
    """Test that the ReactionNetwork correctly discovers all species."""
    r1 = Reaction('A -> B', MassAction(k=0.1))
    r2 = Reaction('B -> C', MassAction(k=0.2))
    r3 = Reaction('C -> D', MassAction(k=0.3))
    network = ReactionNetwork([r1, r2, r3])
    # Species should be sorted alphabetically
    assert network.species == ('A', 'B', 'C', 'D')


def test_network_stoichiometry_simple():
    """Test the stoichiometry matrix for a simple linear chain of reactions."""
    r1 = Reaction('A -> B', MassAction(k=0.1))
    r2 = Reaction('B -> C', MassAction(k=0.2))
    network = ReactionNetwork([r1, r2])
    # Species: A, B, C
    # Reactions: r1, r2
    # Expected S:
    #    r1  r2
    # A: -1   0
    # B:  1  -1
    # C:  0   1
    expected_S = jnp.array([[-1, 0], [1, -1], [0, 1]])
    assert jnp.array_equal(jnp.asarray(network.stoichiometry_matrix), expected_S)


def test_network_stoichiometry_complex():
    """Test the stoichiometry matrix for more complex reactions."""
    r1 = Reaction('2A + B -> C', MassAction(k=0.1))
    r2 = Reaction('C -> A + D', MassAction(k=0.2))
    network = ReactionNetwork([r1, r2])
    # Species: A, B, C, D
    # Reactions: r1, r2
    # Expected S:
    #    r1  r2
    # A: -2   1
    # B: -1   0
    # C:  1  -1
    # D:  0   1
    expected_S = jnp.array([[-2, 1], [-1, 0], [1, -1], [0, 1]])
    assert jnp.array_equal(jnp.asarray(network.stoichiometry_matrix), expected_S)


def test_network_stoichiometry_synthesis_degradation():
    """Test stoichiometry for synthesis (0 -> X) and degradation (X -> 0)."""
    r1 = Reaction('0 -> A', MassAction(k=10.0))
    r2 = Reaction('A -> 0', MassAction(k=0.1))
    r3 = Reaction('A -> B', MassAction(k=0.5))
    network = ReactionNetwork([r1, r2, r3])
    # Species: A, B
    # Reactions: r1, r2, r3
    # Expected S:
    #    r1  r2  r3
    # A:  1  -1  -1
    # B:  0   0   1
    expected_S = jnp.array([[1, -1, -1], [0, 0, 1]])
    assert jnp.array_equal(jnp.asarray(network.stoichiometry_matrix), expected_S)


def test_network_indexing_and_attributes():
    """Test indexing, slicing, and attribute access of reactions in the network."""
    r_birth = Reaction('0 -> A', MassAction(k=1.0), name='birth')
    r_death = Reaction('A -> 0', MassAction(k=0.1), name='death')
    r_dimer = Reaction('2A -> D', MassAction(k=0.01), name='dimerization')
    network = ReactionNetwork([r_birth, r_death, r_dimer])

    # Test __len__
    assert len(network) == 3

    # Test __getitem__ with integer
    assert network[0].name == r_birth.name
    assert network[1].name == r_death.name
    assert network[-1].name == r_dimer.name

    # Test __getattr__ for named reactions
    assert network.birth.name == r_birth.name
    assert network.death.name == r_death.name
    assert network.dimerization.name == r_dimer.name

    # Test __iter__
    reactions_list = list(iter(network))
    assert len(reactions_list) == 3
    assert reactions_list[0].name == r_birth.name
    assert reactions_list[1].name == r_death.name
    assert reactions_list[2].name == r_dimer.name

    # Test __getitem__ with slice
    sub_network = network[0:2]
    assert len(sub_network) == 2
    assert sub_network[0].name == r_birth.name
    assert sub_network[1].name == r_death.name
    assert sub_network.species == (
        'A',
    )  # Only species A is involved in birth and death reactions

    # Test __getitem__ with a list of names
    sub_network_named = network[['birth', 'dimerization']]
    assert len(sub_network_named) == 2
    assert sub_network_named[0].name == r_birth.name
    assert sub_network_named[1].name == r_dimer.name


def test_network_addition():
    """Test adding reactions to the network using the + operator."""
    r1 = Reaction('A -> B', MassAction(k=0.1), name='r1')
    r2 = Reaction('B -> C', MassAction(k=0.2), name='r2')
    network1 = ReactionNetwork([r1])

    # Test adding a single reaction
    network2 = network1 + r2
    assert len(network2) == 2
    assert network2[1].name == 'r2'

    # Test adding a list of reactions
    r3 = Reaction('C -> D', MassAction(k=0.3), name='r3')
    network3 = network2 + [r3]
    assert len(network3) == 3
    assert network3[2].name == 'r3'

    # Test adding another network
    r4 = Reaction('D -> A', MassAction(k=0.4), name='r4')
    network4 = ReactionNetwork([r4])
    network5 = network3 + network4
    assert len(network5) == 4
    assert network5[3].name == 'r4'
    assert network5.species == ('A', 'B', 'C', 'D')


def test_network_propensity_and_ode():
    """Test the propensity function and the ODE vector field of the network."""
    k_birth = 10.0
    k_death = 1.0
    r_birth = Reaction('0 -> A', MassAction(k=k_birth))
    r_death = Reaction('A -> 0', MassAction(k=k_death))
    network = ReactionNetwork([r_birth, r_death])

    x = jnp.array([5])  # 5 molecules of A

    # Test propensity_fn
    propensities = network.propensity_fn(x)
    expected_propensities = jnp.array([k_birth, k_death * x[0]])
    assert jnp.allclose(propensities, expected_propensities)

    # Test vector_field (ode_rhs)
    ode_rhs_fn = network.vector_field
    rates = ode_rhs_fn(t=0.0, x=x, args=None)
    expected_rates = jnp.array([k_birth - k_death * x[0]])
    assert jnp.allclose(rates, expected_rates)


def test_network_string_representations():
    """Test the __str__ and to_latex methods of the network."""
    r1 = Reaction('2A + B -> C', MassAction(k=0.1), name='r1')
    r2 = Reaction('C -> 0', MassAction(k=0.2), name='r2')
    network = ReactionNetwork([r1, r2])

    # Test __str__
    string_repr = str(network)
    assert 'R0 (r1):' in string_repr
    assert '2 A + B -> C' in string_repr
    assert 'MassAction' in string_repr
    assert 'R1 (r2):' in string_repr
    assert 'C -> 0' in string_repr

    # Test to_latex
    latex_repr = network.to_latex()
    assert '2\\,A + B' in latex_repr
    assert '\\rightarrow' in latex_repr
    assert 'C' in latex_repr
    assert '\\emptyset' in latex_repr
