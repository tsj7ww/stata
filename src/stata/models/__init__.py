"""
auto_ml models package.

This package provides various model implementations for both classification and regression tasks.
"""

from .base import BaseModel
from .tree_models import (
    DecisionTreeModel,
    RandomForestModel,
    GradientBoostingModel,
    XGBoostModel,
    LightGBMModel
)
from .linear_models import (
    LinearModel,
    RidgeModel,
    LassoModel,
    ElasticNetModel,
    SGDModel
)
from .neural_nets import MLPModel
from .ensembles import (
    VotingEnsemble,
    StackingEnsemble,
    BaggingEnsemble
)

__all__ = [
    # Base
    'BaseModel',
    
    # Tree-based models
    'DecisionTreeModel',
    'RandomForestModel',
    'GradientBoostingModel',
    'XGBoostModel',
    'LightGBMModel',
    
    # Linear models
    'LinearModel',
    'RidgeModel',
    'LassoModel',
    'ElasticNetModel',
    'SGDModel',
    
    # Neural networks
    'MLPModel',
    
    # Ensemble models
    'VotingEnsemble',
    'StackingEnsemble',
    'BaggingEnsemble',
]