import jax.numpy as jnp
import pytest

from stochastix.controllers import Timer
from stochastix.solvers import SimulationStep


# Mock ReactionNetwork for testing
class MockReactionNetwork:
    def __init__(self, species):
        self._species = species

    @property
    def species(self):
        return self._species


def test_timer_init():
    # Test correct initialization
    controller = Timer(
        controlled_species=['S0'],
        time_triggers=[1.0],
        species_at_triggers=jnp.array([[10]]),
    )
    assert controller.controlled_species == ('S0',)
    assert controller.time_triggers == (1.0,)
    assert controller.species_at_triggers == ((10,),)

    # Test incorrect initialization (mismatched lengths)
    with pytest.raises(ValueError):
        Timer(
            controlled_species=['S0'],
            time_triggers=[1.0, 2.0],
            species_at_triggers=jnp.array([[10]]),
        )

    with pytest.raises(ValueError):
        Timer(
            controlled_species=['S0', 'S1'],
            time_triggers=[1.0],
            species_at_triggers=jnp.array([[10]]),
        )


def test_timer_single_trigger():
    controller = Timer(
        controlled_species=('S0',),
        time_triggers=[1.0],
        species_at_triggers=jnp.array([[10]]),
    )
    network = MockReactionNetwork(species=['S0', 'S1'])

    x = jnp.array([0, 5])
    t_init = 0.0
    a = jnp.array([1.0, 1.0])
    key = None
    state = controller.init(network, t_init, x, a, key=key)

    t = 0.9
    reaction_dt = 0.2
    r = 0

    step_result = SimulationStep(x, reaction_dt, r, a)
    new_step_result, new_state = controller.step(t, step_result, state, key)

    assert jnp.array_equal(new_step_result.x_new, jnp.array([10, 5]))
    assert new_step_result.dt == pytest.approx(1.0 - t)
    assert jnp.all(new_step_result.propensities == 0)
    assert new_step_result.reaction_idx == -2

    trigger_idx, next_trigger_time, species_idx = new_state
    assert trigger_idx == 1
    assert next_trigger_time == jnp.inf
    assert jnp.array_equal(species_idx, jnp.array([0]))


def test_timer_multiple_triggers():
    controller = Timer(
        controlled_species=('S0', 'S1'),
        time_triggers=[1.0, 2.0],
        species_at_triggers=jnp.array([[10, 10], [20, 20]]),
    )

    network = MockReactionNetwork(species=['S0', 'S1'])

    x = jnp.array([0, 0])
    t_init = 0.0
    a = jnp.array([1.0, 1.0])
    key = None
    state = controller.init(network, t_init, x, a, key=key)

    # First trigger
    t = 0.9
    reaction_dt = 0.2
    r = 0
    step_result = SimulationStep(x, reaction_dt, r, a)
    new_step_result, new_state = controller.step(t, step_result, state, key)

    assert jnp.array_equal(new_step_result.x_new, jnp.array([10, 10]))
    assert new_step_result.dt == pytest.approx(1.0 - t)

    trigger_idx, next_trigger_time, species_idx = new_state
    assert trigger_idx == 1
    assert next_trigger_time == 2.0

    # Second trigger
    state = new_state
    x = new_step_result.x_new
    t = 1.9
    reaction_dt = 0.2
    r = 0
    step_result = SimulationStep(x, reaction_dt, r, a)
    new_step_result, new_state = controller.step(t, step_result, state, key)
    assert jnp.array_equal(new_step_result.x_new, jnp.array([20, 20]))
    assert new_step_result.dt == pytest.approx(2.0 - t)

    trigger_idx, next_trigger_time, species_idx = new_state
    assert trigger_idx == 2
    assert next_trigger_time == jnp.inf


def test_timer_no_trigger():
    controller = Timer(
        controlled_species=['S0'],
        time_triggers=[1.0],
        species_at_triggers=jnp.array([[10]]),
    )
    network = MockReactionNetwork(species=['S0', 'S1'])

    x = jnp.array([0, 5])
    t_init = 0.0
    a = jnp.array([1.0, 1.0])
    key = None
    state = controller.init(network, t_init, x, a, key=key)

    t = 0.5
    reaction_dt = 0.2
    r = 0

    step_result = SimulationStep(x, reaction_dt, r, a)
    new_step_result, new_state = controller.step(t, step_result, state, key)

    assert jnp.array_equal(new_step_result.x_new, x)
    assert new_step_result.dt == reaction_dt
    assert jnp.array_equal(new_step_result.propensities, a)
    assert new_step_result.reaction_idx == r

    trigger_idx, next_trigger_time, species_idx = new_state
    assert trigger_idx == 0
    assert next_trigger_time == 1.0
