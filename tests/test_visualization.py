import jax.numpy as jnp
import matplotlib.pyplot as plt
import pytest

from stochastix import SimulationResults
from stochastix.utils.visualization import plot_abundance_dynamic


def test_plot_abundance_dynamic_runs():
    # Create dummy data
    times = jnp.linspace(0, 10, 100)
    species_counts = jnp.vstack([jnp.sin(times), jnp.cos(times)]).T
    results = SimulationResults(
        t=times,
        x=species_counts,
        propensities=jnp.zeros((99, 2)),
        reactions=jnp.zeros((99, 1)),
        species=('S0', 'S1'),
        time_overflow=False,
    )

    # Test if it runs without error
    try:
        fig, ax = plot_abundance_dynamic(results)
        # Check return types
        assert fig is not None
        assert isinstance(ax, plt.Axes)
        plt.close(fig)  # Close the figure to avoid displaying it
    except Exception as e:
        pytest.fail(f'plot_abundance_dynamic raised an exception: {e}')


def test_plot_abundance_dynamic_with_options():
    # Create dummy data
    times = jnp.linspace(0, 10, 100)
    species_counts = jnp.vstack(
        [
            jnp.sin(times) + 1,  # Add 1 to avoid log(0)
            jnp.cos(times) + 1,
        ]
    ).T
    results = SimulationResults(
        t=times,
        x=species_counts,
        propensities=jnp.zeros((99, 2)),
        reactions=jnp.zeros((99, 1)),
        species=('S0', 'S1'),
        time_overflow=False,
    )

    # Test with various options
    try:
        fig, ax = plot_abundance_dynamic(
            results,
            species=['S0'],
            species_labels=['Species A'],
            log_x_scale=True,
            log_y_scale=True,
            time_unit='m',
        )
        assert fig is not None
        assert isinstance(ax, plt.Axes)
        assert ax.get_xscale() == 'log'
        assert ax.get_yscale() == 'log'
        assert ax.get_xlabel() == 'time [m]'
        assert len(ax.get_lines()) == 1
        assert ax.get_legend().get_texts()[0].get_text() == 'Species A'
        plt.close(fig)
    except Exception as e:
        pytest.fail(f'plot_abundance_dynamic with options raised an exception: {e}')


def test_plot_abundance_dynamic_input_validation():
    # Test with invalid ssa_results
    with pytest.raises(
        ValueError, match="ssa_results must have 't' and 'x' attributes"
    ):
        plot_abundance_dynamic(None)

    class MockResults:
        pass

    with pytest.raises(
        ValueError, match="ssa_results must have 't' and 'x' attributes"
    ):
        plot_abundance_dynamic(MockResults())

    # Test with invalid time_unit
    times = jnp.linspace(0, 10, 100)
    species_counts = jnp.zeros((100, 2))
    results = SimulationResults(
        t=times,
        x=species_counts,
        propensities=jnp.zeros((99, 2)),
        reactions=jnp.zeros((99, 1)),
        species=('S0', 'S1'),
        time_overflow=False,
    )

    with pytest.raises(ValueError, match='time_unit must be one of'):
        plot_abundance_dynamic(results, time_unit='invalid')
