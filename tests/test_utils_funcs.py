import jax.numpy as jnp
import numpy as np

from stochastix.utils import algebraic_sigmoid, entropy, rate_constant_conc_to_count
from stochastix.utils._utils import entr_safe


def test_entr_safe_basic_zero_handling():
    p = jnp.array([0.0, 0.5, 0.5, 1.0])
    vals = entr_safe(p)
    # Expect elementwise: [0, 0.5, 0.5, 0] and sum = 1.0
    assert np.isclose(float(jnp.sum(vals)), 1.0)
    assert np.isclose(float(vals[0]), 0.0)
    assert np.isclose(float(vals[-1]), 0.0)


def test_algebraic_sigmoid_properties():
    # f(0) = 0.5
    assert np.isclose(float(algebraic_sigmoid(jnp.array(0.0))), 0.5)

    # Symmetry: f(-x) = 1 - f(x)
    x = jnp.array(3.0)
    f_pos = float(algebraic_sigmoid(x))
    f_neg = float(algebraic_sigmoid(-x))
    assert np.isclose(f_neg, 1.0 - f_pos)

    # Formula check at x=1
    x1 = 1.0
    expected = 0.5 + (x1 / (2.0 * np.sqrt(1.0 + x1**2)))
    assert np.isclose(float(algebraic_sigmoid(jnp.array(x1))), expected)


def test_entropy_bits_and_nats():
    # Uniform over 4 outcomes → 2 bits
    p = jnp.array([0.25, 0.25, 0.25, 0.25])
    assert np.isclose(float(entropy(p)), 2.0)

    # Deterministic → 0 bits
    p_det = jnp.array([1.0, 0.0, 0.0, 0.0])
    assert np.isclose(float(entropy(p_det)), 0.0)

    # Nats: H([0.5, 0.5]) = ln 2
    p2 = jnp.array([0.5, 0.5])
    assert np.isclose(float(entropy(p2, base=np.e)), np.log(2.0))


def test_rate_constant_conc_to_count_molar_units():
    NA = 6.02214076e23

    # First order (m=1): c = k
    k = 0.3  # 1/s
    V = 1e-15  # L
    c = rate_constant_conc_to_count(k, reaction_order=1, volume=V, use_molar_units=True)
    assert np.isclose(float(c), k)

    # Second order (m=2): c = k / (NA * V)
    k = 1e6  # M^-1 s^-1
    V = 1e-15  # L
    c_manual = k / (NA * V)
    c = rate_constant_conc_to_count(k, reaction_order=2, volume=V, use_molar_units=True)
    assert np.isclose(float(c), c_manual, rtol=1e-5)

    # Zeroth order (m=0): c = k * (NA * V)
    k = 1e-9  # M s^-1
    V = 1e-15  # L
    c_manual = k * NA * V
    c = rate_constant_conc_to_count(k, reaction_order=0, volume=V, use_molar_units=True)
    assert np.isclose(float(c), c_manual, rtol=1e-5)


def test_rate_constant_conc_to_count_number_density():
    # Number-density units (use_molar_units=False): replace NA*V with V
    V = 2.0  # arbitrary volume units

    # First order (m=1): c = k
    k = 5.0
    c = rate_constant_conc_to_count(
        k, reaction_order=1, volume=V, use_molar_units=False
    )
    assert np.isclose(float(c), k)

    # Second order (m=2): c = k / V
    k = 10.0
    c = rate_constant_conc_to_count(
        k, reaction_order=2, volume=V, use_molar_units=False
    )
    assert np.isclose(float(c), k / V)

    # Zeroth order (m=0): c = k * V
    k = 0.25
    c = rate_constant_conc_to_count(
        k, reaction_order=0, volume=V, use_molar_units=False
    )
    assert np.isclose(float(c), k * V)


def test_rate_constant_conc_to_count_log10_mode():
    NA = 6.02214076e23
    k = 1e6
    V = 1e-15
    c_manual = k / (NA * V)
    log10_c = rate_constant_conc_to_count(
        k, 2, V, use_molar_units=True, return_log=True
    )
    assert np.isclose(float(log10_c), np.log10(c_manual), rtol=1e-5)

    # k == 0 → -inf in log mode
    log10_c_zero = rate_constant_conc_to_count(
        0.0, 1, V, use_molar_units=True, return_log=True
    )
    assert np.isneginf(float(log10_c_zero))


def test_rate_constant_conc_to_count_invalid_volume():
    try:
        rate_constant_conc_to_count(1.0, 1, 0.0, use_molar_units=True)
    except ValueError as e:
        assert 'volume must be positive' in str(e)
    else:
        assert False, 'Expected ValueError for non-positive volume'


def test_rate_constant_conc_to_count_noninteger_order():
    # Sanity check: return_log should be consistent with value via 10**log10
    k = 3.7
    V = 1.23
    m = 1.5
    log10_c = rate_constant_conc_to_count(
        k, m, V, use_molar_units=False, return_log=True
    )
    c = rate_constant_conc_to_count(k, m, V, use_molar_units=False, return_log=False)
    assert np.isclose(float(c), 10 ** float(log10_c))
