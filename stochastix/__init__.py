"""JAX-based stochastic simulation algorithms for chemical reaction networks."""

import importlib.metadata

__version__ = importlib.metadata.version('stochastix')

from . import _systems as systems
from . import (
    controllers,
    generators,
    kinetics,
    reaction,
    solvers,
    utils,
)
from ._simulation_results import SimulationResults
from ._state_utils import add_to_state, pytree_to_state, state_to_pytree
from ._stochsimsolve import stochsimsolve
from ._systems import MeanFieldModel, StochasticModel
from .reaction import Reaction, ReactionNetwork
from .solvers import (
    DGA,
    DifferentiableDirect,
    DifferentiableFirstReaction,
    DirectMethod,
    FirstReactionMethod,
    TauLeaping,
)
from .utils.visualization import plot_abundance_dynamic

__all__ = [
    'kinetics',
    'generators',
    'controllers',
    'reaction',
    'solvers',
    'utils',
    'systems',
    'StochasticModel',
    'stochsimsolve',
    'pytree_to_state',
    'state_to_pytree',
    'add_to_state',
    'ReactionNetwork',
    'Reaction',
    'SimulationResults',
    'plot_abundance_dynamic',
    'DirectMethod',
    'FirstReactionMethod',
    'TauLeaping',
    'DGA',
    'DifferentiableDirect',
    'DifferentiableFirstReaction',
    'MeanFieldModel',
]
