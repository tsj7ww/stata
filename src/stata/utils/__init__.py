"""
Utility functions and classes for the auto_ml package.
"""

from .logging import (
    AutoMLLogger,
    LoggerMixin,
    configure_logger,
    default_logger
)
from .validation import (
    validate_dataset,
    validate_hyperparameters,
    validate_feature_names,
    validate_categorical_features,
    validate_metrics,
    validate_cv_params,
    ValidationError
)
from .io import (
    load_data,
    save_model,
    load_model,
    save_predictions,
    save_config,
    load_config,
    save_results,
    IOError
)

__all__ = [
    # Logging
    'AutoMLLogger',
    'LoggerMixin',
    'configure_logger',
    'default_logger',
    
    # Validation
    'validate_dataset',
    'validate_hyperparameters',
    'validate_feature_names',
    'validate_categorical_features',
    'validate_metrics',
    'validate_cv_params',
    'ValidationError',
    
    # I/O
    'load_data',
    'save_model',
    'load_model',
    'save_predictions',
    'save_config',
    'load_config',
    'save_results',
    'IOError'
]