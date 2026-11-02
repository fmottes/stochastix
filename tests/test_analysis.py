import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from stochastix import SimulationResults
from stochastix.analysis import (
    autocorrelation,
    cross_correlation,
)
from stochastix.analysis.corr import _cross_corr_1d


# Fixture for creating dummy simulation results
@pytest.fixture
def dummy_results():
    times = jnp.linspace(0, 100, 1001)
    species_counts = jnp.vstack(
        [
            jnp.sin(times / 10),
            jnp.cos(times / 10),
            jnp.ones_like(times) * 5,
        ]
    ).T
    results = SimulationResults(
        t=times,
        x=species_counts,
        propensities=jnp.zeros((1000, 2)),
        reactions=jnp.zeros((1000, 1)),
        species=('sin', 'cos', 'const'),
        time_overflow=False,
    )
    return results


def test_autocorrelation_runs(dummy_results):
    lags, autocorrs = autocorrelation(dummy_results, time_kernel_bw=0)
    assert lags is not None
    assert autocorrs is not None
    assert lags.shape[0] == autocorrs.shape[0]
    assert autocorrs.shape[1] == dummy_results.x.shape[1]


def test_autocorrelation_species_selection(dummy_results):
    _lags, autocorrs = autocorrelation(dummy_results, species='sin', time_kernel_bw=0)
    assert autocorrs.shape[1] == 1

    _lags, autocorrs = autocorrelation(
        dummy_results, species=('sin', 'cos'), time_kernel_bw=0
    )
    assert autocorrs.shape[1] == 2


def test_autocorrelation_constant_signal(dummy_results):
    _lags, autocorrs = autocorrelation(dummy_results, species='const', time_kernel_bw=0)
    # Autocorrelation of a constant signal should be close to 0
    assert jnp.allclose(autocorrs[:, 0], 0.0, atol=1e-6)


def test_cross_correlation_runs(dummy_results):
    lags, cross_corrs = cross_correlation(dummy_results, 'sin', 'cos', time_kernel_bw=0)
    assert lags is not None
    assert cross_corrs is not None
    assert lags.shape[0] == cross_corrs.shape[0]


def test_cross_correlation_with_self(dummy_results):
    lags_cross, cross_corrs = cross_correlation(
        dummy_results, 'sin', 'sin', time_kernel_bw=0
    )
    _lags_auto, auto_corrs = autocorrelation(
        dummy_results, species='sin', time_kernel_bw=0
    )

    # cross-correlation gives for negative and positive lags
    # autocorrelation gives only for positive lags
    # find where lags_cross is 0
    zero_lag_idx = jnp.where(lags_cross == 0)[0][0]

    assert jnp.allclose(cross_corrs[zero_lag_idx:], auto_corrs.flatten(), atol=1e-6)


@pytest.mark.parametrize('constant_side', ['first', 'second', 'both'])
def test_cross_correlation_constant_signals_have_finite_gradients(constant_side):
    x = jnp.ones(5) if constant_side in ('first', 'both') else jnp.arange(5.0)
    y = jnp.ones(5) if constant_side in ('second', 'both') else jnp.arange(5.0)

    def loss(signals):
        corr, _ = _cross_corr_1d(*signals)
        return corr @ jnp.cos(jnp.arange(corr.size))

    value, gradients = eqx.filter_jit(eqx.filter_value_and_grad(loss))((x, y))
    assert value == 0
    np.testing.assert_array_equal(_cross_corr_1d(x, y)[0], jnp.zeros(9))
    assert all(jnp.all(jnp.isfinite(g)) for g in gradients)


def test_cross_correlation_nonconstant_gradient_matches_finite_difference():
    x = jnp.array([0.2, 1.3, 0.7, 2.1, 0.5])
    y = jnp.array([1.2, 0.6, 1.8, 0.3, 0.9])
    direction = jnp.array([0.2, -0.3, 0.7, 0.1, -0.4])
    loss = lambda x: _cross_corr_1d(x, y)[0] @ jnp.cos(jnp.arange(9))
    gradient = eqx.filter_jit(eqx.filter_grad(loss))(x)
    epsilon = 1e-3
    fd = (loss(x + epsilon * direction) - loss(x - epsilon * direction)) / (2 * epsilon)
    np.testing.assert_allclose(gradient @ direction, fd, rtol=2e-3, atol=1e-4)


def test_constant_cross_correlation_with_adaptive_time_kernel_has_finite_gradients():
    results = SimulationResults(
        t=jnp.arange(5.0),
        x=jnp.stack([jnp.ones(5), jnp.array([0.0, 1.0, 3.0, 2.0, 2.0])], axis=1),
        propensities=jnp.ones((4, 1)),
        reactions=jnp.array([0, 0, 0, -1]),
        species=('constant', 'variable'),
    )

    def loss(r):
        corr = cross_correlation(r, 'constant', 'variable', n_points=9)[1]
        return corr @ jnp.cos(jnp.arange(corr.size))

    value, gradient = eqx.filter_jit(eqx.filter_value_and_grad(loss))(results)
    assert value == 0
    assert jnp.all(jnp.isfinite(gradient.x))
    assert jnp.all(jnp.isfinite(gradient.t))
