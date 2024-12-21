"""
AutoML - Automated Machine Learning Package
=========================================

A robust automated machine learning package that handles:
- Automated feature preprocessing
- Model selection
- Hyperparameter optimization
- Ensemble creation
- Model evaluation
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("auto_ml")
except PackageNotFoundError:
    __version__ = "unknown"

# Core components
from auto_ml.pipeline.auto_ml import AutoML
from auto_ml.config.data_config import DataConfig
from auto_ml.config.model_config import ModelConfig

# Preprocessing components
from auto_ml.preprocessing.base import BasePreprocessor
from auto_ml.preprocessing.categorical import CategoricalPreprocessor
from auto_ml.preprocessing.numerical import NumericalPreprocessor

# Model components
from auto_ml.models.base import BaseModel
from auto_ml.models.tree_models import LightGBMModel, XGBoostModel, CatBoostModel

# Optimization components
from auto_ml.optimization.model_selection import ModelSelector
from auto_ml.optimization.hyperopt import HyperparameterOptimizer

__all__ = [
    'AutoML',
    'DataConfig',
    'ModelConfig',
    'BasePreprocessor',
    'CategoricalPreprocessor',
    'NumericalPreprocessor',
    'BaseModel',
    'LightGBMModel',
    'XGBoostModel',
    'CatBoostModel',
    'ModelSelector',
    'HyperparameterOptimizer',
]