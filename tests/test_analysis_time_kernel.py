import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from stochastix import SimulationResults
from stochastix.analysis import (
    autocorrelation,
    cross_correlation,
    entropy,
    kde,
    mutual_information,
    state_entropy,
    state_kde,
    state_mutual_info,
)
from stochastix.analysis.corr import _autocorr_1d, _cross_corr_1d


def _results():
    return SimulationResults(
        x=jnp.array(
            [
                [[0, 1], [1, 1], [1, 3], [2, 3], [2, 3]],
                [[1, 0], [3, 0], [3, 2], [3, 4], [3, 4]],
                [[0, 2], [2, 2], [2, 1], [4, 1], [4, 1]],
            ],
            dtype=float,
        ),
        t=jnp.array(
            [[0, 0.7, 1.5, 2.6, 4], [0, 0.9, 1.9, 3, 4], [0, 0.5, 1.4, 2.8, 4]]
        ),
        propensities=jnp.tile(jnp.array([[1.0, 1.0]] * 3 + [[0.0, 0.0]]), (3, 1, 1)),
        reactions=jnp.tile(jnp.array([0, 1, 0, -1]), (3, 1)),
        species=('A', 'B'),
    )


@pytest.mark.parametrize('bandwidth', [None, 0.4, 'adaptive'])
@pytest.mark.parametrize('analysis', ['kde', 'entropy', 'mi', 'auto', 'cross'])
def test_analysis_time_kernel_matches_explicit_interpolation(analysis, bandwidth):
    results = _results()
    if analysis in ('auto', 'cross'):
        results = results[0]
    grid = {'n_grid_points': 6, 'min_max_vals': (0.0, 5.0), 'bw_multiplier': 1.5}

    def evaluate(r, bw, explicit=False):
        kernel_kwargs = {} if bw is None else {'time_kernel_bw': bw}
        bw = 'adaptive' if bw is None else bw
        if analysis == 'kde':
            t = 1.7
            if explicit:
                x = r.interpolate(jnp.array(t), time_kernel_bw=bw).x[:, 0]
                return kde(x, **grid)[1]
            return state_kde(r, 'A', t=t, **kernel_kwargs, **grid)[1][:, 0]
        if analysis == 'entropy':
            if explicit:
                x = r.interpolate(jnp.array(1.7), time_kernel_bw=bw).x[:, 0]
                return entropy(x, **grid)
            return state_entropy(r, ('A', 1.7), **kernel_kwargs, **grid)
        if analysis == 'mi':
            kwargs = {
                'n_grid_points1': 6,
                'n_grid_points2': 6,
                'min_max_vals1': (0.0, 5.0),
                'min_max_vals2': (0.0, 5.0),
                'bw_multiplier': 1.5,
            }
            if explicit:
                x = r.interpolate(jnp.array(1.7), time_kernel_bw=bw).x[:, 0]
                y = r.interpolate(jnp.array(2.4), time_kernel_bw=bw).x[:, 1]
                return mutual_information(x, y, **kwargs)
            return state_mutual_info(
                r, [('A', 1.7), ('B', 2.4)], **kernel_kwargs, **kwargs
            )
        if explicit:
            times = jnp.linspace(r.t[0], r.t[-1], 7)
            x = r.interpolate(times, time_kernel_bw=bw).x
            return (
                _autocorr_1d(x[:, 0])
                if analysis == 'auto'
                else _cross_corr_1d(x[:, 0], x[:, 1])[0]
            )
        if analysis == 'auto':
            return autocorrelation(r, n_points=7, species='A', **kernel_kwargs)[1][:, 0]
        return cross_correlation(r, 'A', 'B', n_points=7, **kernel_kwargs)[1]

    actual = eqx.filter_jit(evaluate)(results, bandwidth)
    # JIT fusion can change float32 reduction rounding in derived statistics.
    # Exact forward-fill values are checked separately in test_time_kernel.py.
    np.testing.assert_allclose(actual, evaluate(results, 0), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(
        actual, evaluate(results, bandwidth, True), rtol=1e-6, atol=1e-7
    )

    def loss(times, explicit=False):
        r = eqx.tree_at(lambda r: r.t, results, times)
        value = jnp.ravel(evaluate(r, bandwidth, explicit))
        return value @ jnp.linspace(0.3, 1.3, value.size)

    gradient = eqx.filter_jit(eqx.filter_grad(loss))(results.t)
    expected = eqx.filter_grad(loss)(results.t, True)
    np.testing.assert_allclose(gradient, expected, rtol=2e-5, atol=1e-7)
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.linalg.norm(gradient) > 0


@pytest.mark.parametrize('analysis', ['kde', 'entropy', 'mi', 'auto', 'cross'])
@pytest.mark.parametrize('missing_field', ['propensities', 'reactions'])
def test_analysis_default_requires_history_and_explains_opt_out(
    analysis, missing_field
):
    results = _results()
    results = eqx.tree_at(lambda r: getattr(r, missing_field), results, None)
    if analysis in ('auto', 'cross'):
        results = results[0]
    grid = {'n_grid_points': 6, 'min_max_vals': (0.0, 5.0)}
    functions = {
        'kde': lambda **kw: state_kde(results, 'A', t=1.7, **grid, **kw)[1],
        'entropy': lambda **kw: state_entropy(results, ('A', 1.7), **grid, **kw),
        'mi': lambda **kw: state_mutual_info(
            results,
            [('A', 1.7), ('B', 2.4)],
            n_grid_points1=6,
            n_grid_points2=6,
            min_max_vals1=(0.0, 5.0),
            min_max_vals2=(0.0, 5.0),
            **kw,
        ),
        'auto': lambda **kw: autocorrelation(results, n_points=7, species='A', **kw)[1],
        'cross': lambda **kw: cross_correlation(results, 'A', 'B', n_points=7, **kw)[1],
    }
    with pytest.raises(ValueError, match='time_kernel_bw=0'):
        functions[analysis]()
    assert jnp.all(jnp.isfinite(functions[analysis](time_kernel_bw=0)))
