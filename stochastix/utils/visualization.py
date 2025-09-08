"""Plotting functions for simulation results visualization."""

from __future__ import annotations

import itertools
import typing
from typing import Any, Literal

import matplotlib.pyplot as plt

if typing.TYPE_CHECKING:
    from .._simulation_results import SimulationResults


def plot_abundance_dynamic(
    ssa_results: SimulationResults,
    species: str | list[str] | tuple[str, ...] = '*',
    time_unit: Literal['s', 'm', 'h', 'd'] = 's',
    figsize: tuple[float, float] = (9, 6),
    line_alpha: float = 0.5,
    log_x_scale: bool = False,
    log_y_scale: bool = False,
    grid_params: dict[str, Any] | None = None,
    ax: plt.Axes | None = None,
    species_labels: list[str] | None = None,
    legend: bool = True,
    base_time_unit: Literal['s', 'm', 'h', 'd'] = 's',
) -> tuple[plt.Figure | None, plt.Axes]:
    """Plot the time evolution of molecular abundances.

    Args:
        ssa_results: A SimulationResults object containing time points and abundances.
        species: The species to plot. Can be "*" for all species, a single
            species name, or a list/tuple of species names.
        time_unit: The time unit for the x-axis ('s', 'm', 'h', 'd').
        figsize: The figure size (width, height).
        line_alpha: The transparency of the plot lines.
        log_x_scale: Whether to use a logarithmic scale for the x-axis.
        log_y_scale: Whether to use a logarithmic scale for the y-axis.
        grid_params: A dictionary of parameters for grid customization.
        ax: A matplotlib.axes.Axes object to plot on. If `None`, a new
            figure and axes are created.
        species_labels: A list of custom labels for the species. If None,
            the species names from `ssa_results` are used.
        legend: Whether to display the legend.
        base_time_unit: Time unit of ssa_results.t (used for conversion to time_unit).

    Returns:
        A tuple `(fig, ax)`, where `fig` is the `matplotlib.figure.Figure`
        object (or None if `ax` was provided) and `ax` is the
        `matplotlib.axes.Axes` object.

    Raises:
        ValueError: If `ssa_results` is missing required attributes, `time_unit`
            is invalid, or a species name is not found.
    """
    # Input validation
    if not hasattr(ssa_results, 't') or not hasattr(ssa_results, 'x'):
        raise ValueError("ssa_results must have 't' and 'x' attributes")

    valid_time_units = {'ms': 1e-3, 's': 1, 'm': 60, 'h': 3600, 'd': 86400}
    if time_unit not in valid_time_units:
        raise ValueError(f'time_unit must be one of {list(valid_time_units.keys())}')

    if base_time_unit not in valid_time_units:
        raise ValueError(
            f'base_time_unit must be one of {list(valid_time_units.keys())}'
        )

    from .._stochsimsolve import pytree_to_state

    # Handle species selection
    if species == '*':
        indices = list(range(len(ssa_results.species)))
    elif isinstance(species, str):
        try:
            indices = [ssa_results.species.index(species)]
        except ValueError as e:
            raise ValueError(
                f"Species '{species}' not found in ssa_results.species"
            ) from e
    else:
        # Handle iterable of species names
        indices = []
        for sp in species:
            try:
                indices.append(ssa_results.species.index(sp))
            except ValueError as e:
                raise ValueError(
                    f"Species '{sp}' not found in ssa_results.species"
                ) from e

    # Set default grid parameters
    grid_params = grid_params or {'alpha': 0.2}

    # Create figure and axes
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None

    rescale_time = valid_time_units[time_unit] / valid_time_units[base_time_unit]

    # Get color cycle
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    color_cycle = itertools.cycle(colors)

    if ax is not None:
        # Advance the color cycle by the number of lines already on the axes
        num_existing_lines = len(ax.get_lines())
        for _ in range(num_existing_lines):
            next(color_cycle)

    xs = pytree_to_state(ssa_results.x, ssa_results.species)

    # Iterate over selected species
    for label_idx, idx in enumerate(indices):
        label = (
            ssa_results.species[idx]
            if species_labels is None
            else species_labels[label_idx]
        )
        color = next(color_cycle)  # Assign a color to this species

        if xs.ndim <= 2:
            results = ssa_results.clean()
            ts = results.t / rescale_time
            ax.plot(
                ts,
                pytree_to_state(results.x, ssa_results.species)[:, idx],
                alpha=line_alpha,
                label=label,
                color=color,
            )
        else:
            # Handle batched results
            for i in range(xs.shape[0]):
                current_results = ssa_results[i].clean()
                ts = current_results.t / rescale_time
                ax.plot(
                    ts,
                    pytree_to_state(current_results.x, ssa_results.species)[:, idx],
                    alpha=line_alpha,
                    color=color,
                )

            # Plot an invisible line with the label for the legend
            label = label if legend else None
            ax.plot([], [], label=label, color=color, alpha=0.5, lw=2)

    ax.grid(**grid_params)

    if log_x_scale:
        ax.set_xscale('log')

    if log_y_scale:
        ax.set_yscale('log')

    ax.set_xlabel(f'time [{time_unit}]')
    ax.set_ylabel('number of molecules')

    if legend:
        ax.legend()

    return fig, ax
