import pytest

from stochastix import generators
from stochastix.reaction import ReactionNetwork

# A list of all generator functions and their expected properties
# (generator_function, expected_species_count, expected_reaction_count)
GENERATOR_TEST_CASES = [
    (generators.lotka_volterra_model, 2, 3),
    (generators.sirs_model, 3, 3),
    (generators.single_gene_expression_model, 2, 4),
    (generators.michaelis_menten_explicit_model, 4, 3),
    (generators.schlogl_model, 1, 4),
    (generators.hopfield_kinetic_proofreading_model, 5, 5),
    (generators.toggle_switch_model, 2, 4),
    (generators.repressilator_model, 3, 6),
]


@pytest.mark.parametrize('generator_func, n_species, n_reactions', GENERATOR_TEST_CASES)
def test_model_generator(generator_func, n_species, n_reactions):
    """Tests that a model generator correctly creates a ReactionNetwork.

    Args:
        generator_func (callable): The generator function to test.
        n_species (int): Expected number of species in the generated network.
        n_reactions (int): Expected number of reactions in the generated network.
    """
    # Call the generator with default parameters
    network = generator_func()

    # Check that the output is a ReactionNetwork instance
    assert isinstance(network, ReactionNetwork)

    # Check the number of species and reactions
    assert len(network.species) == n_species
    assert len(network.reactions) == n_reactions
