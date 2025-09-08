"""Optimization utilities for reinforcement learning and parameter fitting.

.. warning::
    This module is experimental and it may change or be removed in future versions.
"""

import warnings

from . import _grad as grad
from . import _losses as losses
from . import _rewards as rewards
from . import _utils as utils
from ._losses import reinforce_loss
from ._rewards import (
    neg_final_state_distance,
    rewards_from_state_metric,
    steady_state_distance,
)
from ._utils import dataloader, discounted_returns

warnings.warn(
    "The 'stochastix.utils.optimization' module is experimental and it may change or be removed "
    'without notice in future versions.',
    category=FutureWarning,
    stacklevel=2,
)

__all__ = [
    'reinforce_loss',
    'losses',
    'rewards',
    'utils',
    'dataloader',
    'discounted_returns',
    'grad',
    'neg_final_state_distance',
    'steady_state_distance',
    'rewards_from_state_metric',
]
