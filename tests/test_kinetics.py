import equinox as eqx
import jax
import jax.nn
import jax.numpy as jnp
import numpy as np
import pytest

from stochastix.kinetics import (
    MLP,
    Constant,
    HillAA,
    HillActivator,
    HillAR,
    HillRepressor,
    HillRR,
    HillSingleRegulator,
    MassAction,
    MichaelisMenten,
)


def test_constant_kinetics():
    """Test Constant kinetics."""
    k = 10.0
    kinetics = Constant(k)
    propensity = kinetics.propensity_fn(x=None, reactants=None)
    assert propensity == k

    ode_rate = kinetics.ode_rate_fn(x=None, reactants=None, volume=1.0)
    assert np.isclose(ode_rate, k)


def test_mass_action_kinetics_first_order():
    """Test MassAction kinetics for a first-order reaction."""
    k = 2.0
    kinetics = MassAction(k)
    # Reaction: A -> B, with count of A = 5
    x = jnp.array([5])
    reactants = jnp.array([1])
    propensity = kinetics.propensity_fn(x, reactants)
    # Expected: k * C(5, 1) = 2.0 * 5 = 10.0
    assert np.isclose(propensity, 10.0)


def test_mass_action_kinetics_second_order():
    """Test MassAction kinetics for a second-order reaction."""
    k = 0.1
    kinetics = MassAction(k)
    # Reaction: 2A -> B, with count of A = 10
    x = jnp.array([10])
    reactants = jnp.array([2])
    propensity = kinetics.propensity_fn(x, reactants)
    # Expected: k * C(10, 2) = 0.1 * (10 * 9 / 2) = 0.1 * 45 = 4.5
    assert np.isclose(propensity, 4.5)


def test_mass_action_kinetics_bimolecular():
    """Test MassAction kinetics for a bimolecular reaction."""
    k = 0.5
    kinetics = MassAction(k)
    # Reaction: A + B -> C, with counts A=4, B=5
    x = jnp.array([4, 5])
    reactants = jnp.array([1, 1])
    propensity = kinetics.propensity_fn(x, reactants)
    # Expected: k * C(4, 1) * C(5, 1) = 0.5 * 4 * 5 = 10.0
    assert np.isclose(propensity, 10.0)


def test_michaelis_menten_fixed_enzyme():
    """Test MichaelisMenten kinetics with a fixed enzyme count."""
    k_cat = 100.0
    k_m = 50.0
    enzyme_count = 10
    kinetics = MichaelisMenten(enzyme=enzyme_count, k_cat=k_cat, k_m=k_m)
    # Substrate S count = 200
    x = jnp.array([200.0])  # Substrate is at index 0
    reactants = jnp.array([1])  # S -> P
    propensity = kinetics.propensity_fn(x, reactants, volume=1.0)
    # v_max = k_cat * E = 100 * 10 = 1000
    # Expected: v_max * S / (k_m + S) = 1000 * 200 / (50 + 200) = 200000 / 250 = 800
    assert np.isclose(propensity, 800.0)


def test_michaelis_menten_dynamic_enzyme():
    """Test MichaelisMenten kinetics with a dynamic enzyme species."""
    k_cat = 100.0
    k_m = 50.0
    # Simulate the binding process
    species_map = {'S': 0, 'E': 1}
    kinetics = MichaelisMenten(enzyme='E', k_cat=k_cat, k_m=k_m)
    kinetics = kinetics._bind_to_network(species_map)

    # Substrate S=200, Enzyme E=10
    x = jnp.array([200.0, 10.0])
    reactants = jnp.array([1, 0])  # Reaction is S -> P
    propensity = kinetics.propensity_fn(x, reactants, volume=1.0)
    # v_max = k_cat * E = 100 * 10 = 1000
    # Expected: v_max * S / (k_m + S) = 1000 * 200 / (50 + 200) = 800
    assert np.isclose(propensity, 800.0)


def test_hill_activator():
    """Test HillActivator kinetics."""
    v = 100.0
    K = 50.0
    n = 2.0
    species_map = {'X': 0}
    kinetics = HillActivator(regulator='X', v=v, K=K, n=n)
    kinetics = kinetics._bind_to_network(species_map)
    # Regulator X count = 50
    x = jnp.array([50.0])
    propensity = kinetics.propensity_fn(x, reactants=None, volume=1.0)
    # Expected: v * X^n / (K^n + X^n)
    # = 100 * 50^2 / (50^2 + 50^2) = 100 * 0.5 = 50
    assert np.isclose(propensity, 50.0)

    # Test ODE rate
    ode_rate = kinetics.ode_rate_fn(x, reactants=None, volume=1.0)
    # Expected: v * (X/vol)^n / (K^n + (X/vol)^n) = 100 * 50^2 / (50^2 + 50^2) = 50
    assert np.isclose(ode_rate, 50.0)


def test_hill_activator_high_regulator():
    """Test HillActivator kinetics with high regulator concentration."""
    v = 100.0
    K = 50.0
    n = 2.0
    species_map = {'X': 0}
    kinetics = HillActivator(regulator='X', v=v, K=K, n=n)
    kinetics = kinetics._bind_to_network(species_map)
    # Regulator X count = 100
    x = jnp.array([100.0])
    propensity = kinetics.propensity_fn(x, reactants=None, volume=1.0)
    # Expected: v * X^n / (K^n + X^n) = 100 * 100^2 / (50^2 + 100^2) = 100 * 10000 / (2500 + 10000) = 100 * 0.8 = 80
    assert np.isclose(propensity, 80.0)

    # Test ODE rate
    ode_rate = kinetics.ode_rate_fn(x, reactants=None, volume=1.0)
    # Expected: v * (X/vol)^n / (K^n + (X/vol)^n) = 100 * 100^2 / (50^2 + 100^2) = 80
    assert np.isclose(ode_rate, 80.0)


def test_hill_repressor():
    """Test HillRepressor kinetics."""
    v = 100.0
    K = 50.0
    n = 2.0
    species_map = {'X': 0}
    kinetics = HillRepressor(regulator='X', v=v, K=K, n=n)
    kinetics = kinetics._bind_to_network(species_map)
    # Regulator X count = 50
    x = jnp.array([50.0])
    propensity = kinetics.propensity_fn(x, reactants=None, volume=1.0)
    # Expected: v / (1 + (X/K)^n) = 100 / (1 + (50/50)^2) = 100 / 2 = 50
    assert np.isclose(propensity, 50.0)

    # Test ODE rate
    ode_rate = kinetics.ode_rate_fn(x, reactants=None, volume=1.0)
    # Expected: v / (1 + (X/vol/K)^n) = 100 / (1 + (50/1/50)^2) = 50
    assert np.isclose(ode_rate, 50.0)


def test_hill_repressor_high_regulator():
    """Test HillRepressor kinetics with high regulator concentration."""
    v = 100.0
    K = 50.0
    n = 2.0
    species_map = {'X': 0}
    kinetics = HillRepressor(regulator='X', v=v, K=K, n=n)
    kinetics = kinetics._bind_to_network(species_map)
    # Regulator X count = 100
    x = jnp.array([100.0])
    propensity = kinetics.propensity_fn(x, reactants=None, volume=1.0)
    # Expected: v / (1 + (X/K)^n) = 100 / (1 + (100/50)^2) = 100 / 5 = 20
    assert np.isclose(propensity, 20.0)

    # Test ODE rate
    ode_rate = kinetics.ode_rate_fn(x, reactants=None, volume=1.0)
    # Expected: v / (1 + (X/vol/K)^n) = 100 / (1 + (100/1/50)^2) = 20
    assert np.isclose(ode_rate, 20.0)


def test_hill_activator_with_leakage():
    """Test HillActivator kinetics with leakage."""
    v = 100.0
    K = 50.0
    n = 2.0
    v0 = 10.0
    species_map = {'X': 0}
    kinetics = HillActivator(regulator='X', v=v, K=K, n=n, v0=v0)
    kinetics = kinetics._bind_to_network(species_map)
    # Regulator X count = 50
    x = jnp.array([50.0])
    propensity = kinetics.propensity_fn(x, reactants=None, volume=1.0)
    # Expected: v0 + v * X^n / (K^n + X^n) = 10 + 100 * 0.5 = 60
    assert np.isclose(propensity, 60.0)

    # Test ODE rate
    ode_rate = kinetics.ode_rate_fn(x, reactants=None, volume=1.0)
    # Expected: v0 + v * (X/vol)^n / (K^n + (X/vol)^n) = 10 + 50 = 60
    assert np.isclose(ode_rate, 60.0)


def test_hill_repressor_with_leakage():
    """Test HillRepressor kinetics with leakage."""
    v = 100.0
    K = 50.0
    n = 2.0
    v0 = 10.0
    species_map = {'X': 0}
    kinetics = HillRepressor(regulator='X', v=v, K=K, n=n, v0=v0)
    kinetics = kinetics._bind_to_network(species_map)
    # Regulator X count = 50
    x = jnp.array([50.0])
    propensity = kinetics.propensity_fn(x, reactants=None, volume=1.0)
    # Expected: v0 + v / (1 + (X/K)^n) = 10 + 100 / 2 = 60
    assert np.isclose(propensity, 60.0)

    # Test ODE rate
    ode_rate = kinetics.ode_rate_fn(x, reactants=None, volume=1.0)
    # Expected: v0 + v / (1 + (X/vol/K)^n) = 10 + 50 = 60
    assert np.isclose(ode_rate, 60.0)


def test_hill_single_regulator_activation():
    """Test HillSingleRegulator kinetics for activation."""
    v = 100.0
    K = 50.0
    n = 2.0
    species_map = {'X': 0}
    kinetics = HillSingleRegulator(regulator='X', v=v, K=K, n=n)
    kinetics = kinetics._bind_to_network(species_map)
    x = jnp.array([50.0])
    propensity = kinetics.propensity_fn(x, reactants=None, volume=1.0)
    assert np.isclose(propensity, 50.0)

    # Test ODE rate
    ode_rate = kinetics.ode_rate_fn(x, reactants=None, volume=1.0)
    assert np.isclose(ode_rate, 50.0)


def test_hill_single_regulator_inhibition():
    """Test HillSingleRegulator kinetics for inhibition."""
    v = 100.0
    K = 50.0
    n = -2.0
    species_map = {'X': 0}
    kinetics = HillSingleRegulator(regulator='X', v=v, K=K, n=n)
    kinetics = kinetics._bind_to_network(species_map)
    x = jnp.array([50.0])
    propensity = kinetics.propensity_fn(x, reactants=None, volume=1.0)
    assert np.isclose(propensity, 50.0)

    # Test ODE rate
    ode_rate = kinetics.ode_rate_fn(x, reactants=None, volume=1.0)
    assert np.isclose(ode_rate, 50.0)


def test_hill_single_regulator_with_leakage():
    """Test HillSingleRegulator kinetics with leakage."""
    v = 100.0
    K = 50.0
    n = 2.0
    v0 = 10.0
    species_map = {'X': 0}
    kinetics = HillSingleRegulator(regulator='X', v=v, K=K, n=n, v0=v0)
    kinetics = kinetics._bind_to_network(species_map)
    x = jnp.array([50.0])
    propensity = kinetics.propensity_fn(x, reactants=None, volume=1.0)
    assert np.isclose(propensity, 60.0)

    # Test ODE rate
    ode_rate = kinetics.ode_rate_fn(x, reactants=None, volume=1.0)
    assert np.isclose(ode_rate, 60.0)


@pytest.mark.filterwarnings('ignore:.* is negative')
@pytest.mark.parametrize(
    'kinetics_class, n_val, v, K, v0, transform, x_val, expected_prop',
    [
        # Rates > 1, x=50
        (HillActivator, 2.0, 100.0, 50.0, 10.0, None, 50, 60.0),
        (
            HillActivator,
            2.0,
            np.log(100.0),
            np.log(50.0),
            np.log(10.0),
            jnp.exp,
            50,
            60.0,
        ),
        (HillRepressor, 2.0, 100.0, 50.0, 10.0, None, 50, 60.0),
        (
            HillRepressor,
            2.0,
            np.log(100.0),
            np.log(50.0),
            np.log(10.0),
            jnp.exp,
            50,
            60.0,
        ),
        (HillSingleRegulator, 2.0, 100.0, 50.0, 10.0, None, 50, 60.0),  # Activation
        (
            HillSingleRegulator,
            2.0,
            np.log(100.0),
            np.log(50.0),
            np.log(10.0),
            jnp.exp,
            50,
            60.0,
        ),  # Activation
        (HillSingleRegulator, -2.0, 100.0, 50.0, 10.0, None, 50, 60.0),  # Repression
        (
            HillSingleRegulator,
            -2.0,
            np.log(100.0),
            np.log(50.0),
            np.log(10.0),
            jnp.exp,
            50,
            60.0,
        ),  # Repression
        # Rates < 1 (negative log-rates), x=1
        (HillActivator, 2.0, 0.8, 0.5, 0.1, None, 1, 0.74),
        (
            HillActivator,
            2.0,
            np.log(0.8),
            np.log(0.5),
            np.log(0.1),
            jnp.exp,
            1,
            0.74,
        ),
        (HillRepressor, 2.0, 0.8, 0.5, 0.1, None, 1, 0.26),
        (
            HillRepressor,
            2.0,
            np.log(0.8),
            np.log(0.5),
            np.log(0.1),
            jnp.exp,
            1,
            0.26,
        ),
        (HillSingleRegulator, 2.0, 0.8, 0.5, 0.1, None, 1, 0.74),  # Activation
        (
            HillSingleRegulator,
            2.0,
            np.log(0.8),
            np.log(0.5),
            np.log(0.1),
            jnp.exp,
            1,
            0.74,
        ),  # Activation
        (HillSingleRegulator, -2.0, 0.8, 0.5, 0.1, None, 1, 0.26),  # Repression
        (
            HillSingleRegulator,
            -2.0,
            np.log(0.8),
            np.log(0.5),
            np.log(0.1),
            jnp.exp,
            1,
            0.26,
        ),  # Repression
    ],
)
def test_hill_transforms(
    kinetics_class, n_val, v, K, v0, transform, x_val, expected_prop
):
    """Test parameter transforms in Hill kinetics."""
    species_map = {'X': 0}
    x = jnp.array([x_val])

    kinetics = kinetics_class(
        regulator='X',
        v=v,
        K=K,
        n=n_val,
        v0=v0,
        transform_v=transform,
        transform_K=transform,
        transform_n=None,
        transform_v0=transform,
    )
    kinetics = kinetics._bind_to_network(species_map)
    propensity = kinetics.propensity_fn(x, reactants=None, volume=1.0)

    assert np.isclose(propensity, expected_prop)


def test_mlp_kinetics_explicit_species():
    """Test MLP kinetics with explicit input species and custom initializers."""
    key = jax.random.PRNGKey(0)
    species_map = {'S1': 0, 'S2': 1, 'S3': 2}

    # Use initializers that make the output predictable
    weight_init = jax.nn.initializers.ones
    bias_init = jax.nn.initializers.zeros

    kinetics = MLP(
        input_species=('S1', 'S3'),
        hidden_sizes=(4,),
        key=key,
        weight_init=weight_init,
        bias_init=bias_init,
    )

    kinetics = kinetics._bind_to_network(species_map)

    assert kinetics._input_species_idx == (0, 2)
    assert kinetics.mlp is not None

    x = jnp.array([10.0, 5.0, 2.0])  # S1, S2, S3 counts

    # Manually calculate the expected output
    # inputs are S1 and S3 concentrations. volume=1.0 for simplicity.
    # inputs = [10.0, 2.0]
    # hidden layer: inputs (2) -> 4 neurons. weights are all 1.
    # layer1_out = relu(ones(4,2) @ [10, 2] + zeros(4)) = relu([12, 12, 12, 12]) = [12, 12, 12, 12]
    # output layer: 4 neurons -> 1 output. weights are all 1.
    # final_out = softplus(ones(1,4) @ [12,12,12,12] + zeros(1)) = softplus(48)
    # jax.nn.softplus(48.0) is approx 48.0

    propensity = kinetics.propensity_fn(x, reactants=None, volume=1.0)
    assert np.isclose(propensity, 48.0)

    # Test with volume
    propensity_vol = kinetics.propensity_fn(x, reactants=None, volume=2.0)
    # inputs = [5.0, 1.0]
    # layer1_out = relu([6,6,6,6]) = [6,6,6,6]
    # final_out = softplus(24) = 24
    # return rate_in_conc * volume = 24 * 2.0 = 48.0
    assert np.isclose(propensity_vol, 48.0)


def test_mlp_kinetics_all_species():
    """Test MLP kinetics using all species as input."""
    key = jax.random.PRNGKey(1)
    species_map = {'S1': 0, 'S2': 1, 'S3': 2}

    # Use initializers that make the output predictable
    weight_init = jax.nn.initializers.ones
    bias_init = jax.nn.initializers.zeros

    kinetics = MLP(
        input_species='*',
        hidden_sizes=(4,),
        key=key,
        weight_init=weight_init,
        bias_init=bias_init,
    )

    assert kinetics.mlp is None  # Deferred creation

    kinetics = kinetics._bind_to_network(species_map)

    assert kinetics.mlp is not None
    assert kinetics._requires_species == ('S1', 'S2', 'S3')  # Sorted
    assert kinetics._input_species_idx is None

    x = jnp.array([10.0, 5.0, 2.0])

    # Manually calculate the expected output
    # All species are used as input. inputs = [10.0, 5.0, 2.0]
    # hidden layer: inputs (3) -> 4 neurons. weights are all 1.
    # layer1_out = relu(ones(4,3) @ [10, 5, 2] + zeros(4)) = relu([17, 17, 17, 17]) = [17, 17, 17, 17]
    # output layer: 4 neurons -> 1 output. weights are all 1.
    # final_out = softplus(ones(1,4) @ [17,17,17,17] + zeros(1)) = softplus(68)
    # jax.nn.softplus(68.0) is approx 68.0

    propensity = kinetics.propensity_fn(x, reactants=None, volume=1.0)
    assert np.isclose(propensity, 68.0)


def test_mlp_ode_rate_fn():
    """Test the ODE rate function for MLP kinetics."""
    key = jax.random.PRNGKey(2)
    species_map = {'S1': 0, 'S2': 1}

    kinetics = MLP(
        input_species=('S1',),
        hidden_sizes=(),  # No hidden layers
        key=key,
        # Use identity for final activation to simplify testing
        final_activation=lambda x: x,
        weight_init=jax.nn.initializers.ones,
        bias_init=jax.nn.initializers.zeros,
    )

    kinetics = kinetics._bind_to_network(species_map)

    x = jnp.array([10.0, 5.0])
    volume = 2.0

    # propensity = rate_in_conc * volume
    # rate_in_conc = mlp(x / volume)
    # inputs = [10.0/2.0] = [5.0]
    # rate_in_conc = 1 * 5.0 + 0 = 5.0
    # propensity = 5.0 * 2.0 = 10.0
    propensity = kinetics.propensity_fn(x, reactants=None, volume=volume)
    assert np.isclose(propensity, 10.0)

    # ode_rate = propensity / volume
    ode_rate = kinetics.ode_rate_fn(x, reactants=None, volume=volume)
    assert np.isclose(ode_rate, 5.0)

    # Test with a single batch dimension
    x_single_batch = jnp.ones((5, 2))
    vmap_ode_rate_fn = eqx.filter_vmap(lambda x: kinetics.ode_rate_fn(x, None, 1.0))
    ode_rate_single = vmap_ode_rate_fn(x_single_batch)
    assert ode_rate_single.shape == (5,)


# New tests for _hill2d.py
@pytest.mark.parametrize(
    'kinetics_class_name, logic, competitive_binding, x1_val, x2_val, v0, expected_prop',
    [
        # HillAA
        ('HillAA', 'or', False, 50, 50, 0.0, 75.0),
        ('HillAA', 'or', True, 50, 50, 0.0, 66.66666),
        ('HillAA', 'and', False, 50, 50, 0.0, 25.0),
        ('HillAA', 'or', False, 50, 50, 10.0, 85.0),  # with leakage
        # HillRR - Note: 'and' with competitive binding is blocked by __init__, so not tested here.
        ('HillRR', 'and', False, 50, 50, 0.0, 25.0),
        ('HillRR', 'or', False, 50, 50, 0.0, 75.0),
        ('HillRR', 'and', False, 50, 50, 10.0, 35.0),  # with leakage
        # HillAR
        ('HillAR', 'and', False, 50, 50, 0.0, 25.0),
        ('HillAR', 'and', True, 50, 50, 0.0, 33.33333),
        ('HillAR', 'or', False, 50, 50, 0.0, 75.0),
        ('HillAR', 'or', True, 50, 50, 0.0, 66.66666),
        ('HillAR', 'and', False, 50, 50, 10.0, 35.0),  # with leakage
    ],
)
def test_hill_2d(
    kinetics_class_name, logic, competitive_binding, x1_val, x2_val, v0, expected_prop
):
    """Test 2D Hill kinetics for valid combinations."""
    v = 100.0
    K1 = 50.0
    K2 = 50.0
    n1 = 2.0
    n2 = 2.0

    kinetics_map = {'HillAA': HillAA, 'HillRR': HillRR, 'HillAR': HillAR}
    KineticsClass = kinetics_map[kinetics_class_name]

    if kinetics_class_name == 'HillAR':
        reg1_name, reg2_name = 'Xa', 'Xr'
        k_params = {'Ka': K1, 'Kr': K2, 'na': n1, 'nr': n2}
        reg_args = {'activator': reg1_name, 'repressor': reg2_name}
    else:
        reg1_name, reg2_name = 'X1', 'X2'
        k_params = {'K1': K1, 'K2': K2, 'n1': n1, 'n2': n2}
        if kinetics_class_name == 'HillAA':
            reg_args = {'activator1': reg1_name, 'activator2': reg2_name}
        else:  # HillRR
            reg_args = {'repressor1': reg1_name, 'repressor2': reg2_name}

    kinetics = KineticsClass(
        **reg_args,
        v=v,
        **k_params,
        logic=logic,
        competitive_binding=competitive_binding,
        v0=v0,
    )

    species_map = {reg1_name: 0, reg2_name: 1}
    kinetics = kinetics._bind_to_network(species_map)
    x = jnp.array([x1_val, x2_val], dtype=float)

    propensity = kinetics.propensity_fn(x, reactants=None, volume=1.0)
    assert np.isclose(propensity, expected_prop, rtol=1e-5)

    # Test ODE rate
    ode_rate = kinetics.ode_rate_fn(x, reactants=None, volume=1.0)
    assert np.isclose(ode_rate, expected_prop, rtol=1e-5)


@pytest.mark.parametrize(
    'kinetics_class_name, logic, competitive_binding',
    [
        ('HillAA', 'and', True),
        ('HillRR', 'and', True),
    ],
)
def test_hill_2d_invalid_combo(kinetics_class_name, logic, competitive_binding):
    """Test 2D Hill kinetics with invalid logic/binding combinations."""
    kinetics_map = {'HillAA': HillAA, 'HillRR': HillRR}
    KineticsClass = kinetics_map[kinetics_class_name]

    if kinetics_class_name == 'HillAA':
        reg_args = {'activator1': 'X1', 'activator2': 'X2'}
    else:  # HillRR
        reg_args = {'repressor1': 'X1', 'repressor2': 'X2'}

    k_params = {'K1': 1.0, 'K2': 1.0, 'n1': 1.0, 'n2': 1.0}

    with pytest.raises(ValueError, match='not compatible'):
        KineticsClass(
            **reg_args,
            v=100.0,
            **k_params,
            logic=logic,
            competitive_binding=competitive_binding,
        )


def test_hill_2d_invalid_logic():
    """Test Hill 2D kinetics with invalid logic."""
    with pytest.raises(ValueError, match="`logic` must be either 'and' or 'or'"):
        HillAA('X1', 'X2', 1, 1, 1, 1, 1, logic='xor')
    with pytest.raises(ValueError, match="`logic` must be either 'and' or 'or'"):
        HillRR('X1', 'X2', 1, 1, 1, 1, 1, logic='xor')
    with pytest.raises(ValueError, match="`logic` must be either 'and' or 'or'"):
        HillAR('Xa', 'Xr', 1, 1, 1, 1, 1, logic='xor')


@pytest.mark.filterwarnings('ignore:.* is negative')
@pytest.mark.parametrize(
    'kinetics_class_name, kwargs',
    [
        ('HillAA', {'logic': 'or', 'competitive_binding': False}),
        ('HillRR', {'logic': 'or', 'competitive_binding': False}),
        ('HillAR', {'logic': 'or', 'competitive_binding': False}),
    ],
)
def test_hill2d_transforms(kinetics_class_name, kwargs):
    """Test Hill 2D kinetics with parameter transforms."""
    v = np.log(100.0)
    K1 = np.log(50.0)
    K2 = np.log(50.0)
    n1 = 2.0
    n2 = 2.0
    v0 = np.log(1.0)
    transform = jnp.exp

    kinetics_map = {'HillAA': HillAA, 'HillRR': HillRR, 'HillAR': HillAR}
    KineticsClass = kinetics_map[kinetics_class_name]

    if KineticsClass == HillAR:
        k_args = {'Ka': K1, 'Kr': K2, 'na': n1, 'nr': n2}
        regulators = ['Xa', 'Xr']
        reg_args = {'activator': 'Xa', 'repressor': 'Xr'}
        k_transforms = {
            'transform_Ka': transform,
            'transform_Kr': transform,
            'transform_na': None,
            'transform_nr': None,
        }
    else:
        k_args = {'K1': K1, 'K2': K2, 'n1': n1, 'n2': n2}
        regulators = ['X1', 'X2']
        if KineticsClass == HillAA:
            reg_args = {'activator1': 'X1', 'activator2': 'X2'}
        else:
            reg_args = {'repressor1': 'X1', 'repressor2': 'X2'}
        k_transforms = {
            'transform_K1': transform,
            'transform_K2': transform,
            'transform_n1': None,
            'transform_n2': None,
        }

    kinetics = KineticsClass(
        **reg_args,
        v=v,
        **k_args,
        **kwargs,
        v0=v0,
        transform_v=transform,
        **k_transforms,
        transform_v0=transform,
    )
    species_map = {reg: i for i, reg in enumerate(regulators)}
    kinetics = kinetics._bind_to_network(species_map)
    x = jnp.array([50.0, 50.0])
    propensity = kinetics.propensity_fn(x, reactants=None, volume=1.0)
    assert propensity > 0


def test_check_params_warnings():
    """Test that _check_params issues warnings for negative inputs with transforms."""
    with pytest.warns(UserWarning, match='`v` is negative'):
        HillAA(
            'X1', 'X2', v=-1, K1=1, K2=1, n1=1, n2=1, logic='or', transform_v=jnp.exp
        )
    with pytest.warns(UserWarning, match='`v0` is negative'):
        HillAA(
            'X1',
            'X2',
            v=1,
            v0=-1,
            K1=1,
            K2=1,
            n1=1,
            n2=1,
            logic='or',
            transform_v0=jnp.exp,
        )
    with pytest.warns(UserWarning, match='`K` is negative'):
        HillAA(
            'X1', 'X2', v=1, K1=-1, K2=1, n1=1, n2=1, logic='or', transform_K1=jnp.exp
        )
    with pytest.warns(UserWarning, match='`n` is negative'):
        HillAA(
            'X1', 'X2', v=1, K1=1, K2=1, n1=-1, n2=1, logic='or', transform_n1=jnp.exp
        )


def test_check_params_errors():
    """Test that _check_params raises errors for negative inputs without transforms."""
    with pytest.raises(ValueError, match='`v` must be positive'):
        HillAA('X1', 'X2', v=-1, K1=1, K2=1, n1=1, n2=1, logic='or')
    with pytest.raises(ValueError, match='`v0` must be positive'):
        HillAA('X1', 'X2', v=1, v0=-1, K1=1, K2=1, n1=1, n2=1, logic='or')
    with pytest.raises(ValueError, match='`K` must be positive'):
        HillAA('X1', 'X2', v=1, K1=-1, K2=1, n1=1, n2=1, logic='or')
    with pytest.raises(ValueError, match='`n` must be positive'):
        HillAA('X1', 'X2', v=1, K1=1, K2=1, n1=-1, n2=1, logic='or')
