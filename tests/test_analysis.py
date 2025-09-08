import jax.numpy as jnp
import pytest

from stochastix import SimulationResults
from stochastix.utils.analysis import (
    autocorrelation,
    cross_correlation,
)


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
    lags, autocorrs = autocorrelation(dummy_results)
    assert lags is not None
    assert autocorrs is not None
    assert lags.shape[0] == autocorrs.shape[0]
    assert autocorrs.shape[1] == dummy_results.x.shape[1]


def test_autocorrelation_species_selection(dummy_results):
    lags, autocorrs = autocorrelation(dummy_results, species='sin')
    assert autocorrs.shape[1] == 1

    lags, autocorrs = autocorrelation(dummy_results, species=('sin', 'cos'))
    assert autocorrs.shape[1] == 2


def test_autocorrelation_constant_signal(dummy_results):
    lags, autocorrs = autocorrelation(dummy_results, species='const')
    # Autocorrelation of a constant signal should be close to 0
    assert jnp.allclose(autocorrs[:, 0], 0.0, atol=1e-6)


def test_cross_correlation_runs(dummy_results):
    lags, cross_corrs = cross_correlation(dummy_results, 'sin', 'cos')
    assert lags is not None
    assert cross_corrs is not None
    assert lags.shape[0] == cross_corrs.shape[0]


def test_cross_correlation_with_self(dummy_results):
    lags_cross, cross_corrs = cross_correlation(dummy_results, 'sin', 'sin')
    lags_auto, auto_corrs = autocorrelation(dummy_results, species='sin')

    # cross-correlation gives for negative and positive lags
    # autocorrelation gives only for positive lags
    # find where lags_cross is 0
    zero_lag_idx = jnp.where(lags_cross == 0)[0][0]

    assert jnp.allclose(cross_corrs[zero_lag_idx:], auto_corrs.flatten(), atol=1e-6)
