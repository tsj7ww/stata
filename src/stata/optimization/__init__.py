"""
Optimization components for the auto_ml package.
"""

from .search_spaces import (
    SearchSpace,
    get_search_space,
    combine_search_spaces,
    MODEL_SEARCH_SPACES
)
from .hyperopt import (
    HyperoptOptimizer,
    OptimizationResult
)
from .model_selection import (
    ModelSelector,
    ModelSelectionResult
)

__all__ = [
    # Search Spaces
    'SearchSpace',
    'get_search_space',
    'combine_search_spaces',
    'MODEL_SEARCH_SPACES',
    
    # Hyperparameter Optimization
    'HyperoptOptimizer',
    'OptimizationResult',
    
    # Model Selection
    'ModelSelector',
    'ModelSelectionResult'
]