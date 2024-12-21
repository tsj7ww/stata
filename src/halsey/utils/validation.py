"""
Validation module for the auto_ml package.

This module provides utilities for validating inputs, checking parameters,
and ensuring data consistency throughout the auto_ml package.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging

from ..utils.logging import default_logger

logger = default_logger

class ValidationError(Exception):
    """Custom exception for validation errors in auto_ml."""
    pass

def validate_dataset(
    data: Union[pd.DataFrame, np.ndarray],
    target: Optional[Union[str, np.ndarray]] = None,
    task_type: Optional[str] = None
) -> Tuple[Union[pd.DataFrame, np.ndarray], Optional[np.ndarray]]:
    """
    Validate input dataset and target variable.
    
    Args:
        data: Input features as DataFrame or numpy array
        target: Target variable name (if DataFrame) or values
        task_type: Type of ML task ('classification' or 'regression')
    
    Returns:
        Tuple of (validated_features, validated_target)
    
    Raises:
        ValidationError: If validation fails
    """
    # Validate input data
    if isinstance(data, pd.DataFrame):
        if target is not None and isinstance(target, str):
            if target not in data.columns:
                raise ValidationError(f"Target column '{target}' not found in DataFrame")
            y = data[target].values
            X = data.drop(target, axis=1)
        else:
            X = data
            y = target
    elif isinstance(data, np.ndarray):
        X = data
        y = target
    else:
        raise ValidationError(
            f"Data must be pandas DataFrame or numpy array, got {type(data)}"
        )
    
    # Validate shapes
    if y is not None:
        if len(X) != len(y):
            raise ValidationError(
                f"Features and target must have same length. "
                f"Got {len(X)} and {len(y)}"
            )
    
    # Validate task type and target values
    if task_type and y is not None:
        if task_type == 'classification':
            if not np.issubdtype(y.dtype, np.number):
                raise ValidationError(
                    "Classification target must contain numeric values"
                )
        elif task_type == 'regression':
            if not np.issubdtype(y.dtype, np.number):
                raise ValidationError(
                    "Regression target must contain numeric values"
                )
    
    return X, y

def validate_hyperparameters(
    params: Dict[str, Any],
    valid_params: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Validate hyperparameters against their specifications.
    
    Args:
        params: Dictionary of hyperparameters to validate
        valid_params: Dictionary of valid parameter specifications
    
    Returns:
        Validated parameters dictionary
    
    Raises:
        ValidationError: If validation fails
    """
    validated_params = {}
    
    for param_name, param_value in params.items():
        if param_name not in valid_params:
            raise ValidationError(f"Unknown parameter: {param_name}")
        
        spec = valid_params[param_name]
        param_type = spec.get('type')
        param_range = spec.get('range')
        param_choices = spec.get('choices')
        
        # Validate type
        if param_type and not isinstance(param_value, param_type):
            raise ValidationError(
                f"Parameter {param_name} must be of type {param_type}"
            )
        
        # Validate range
        if param_range:
            min_val, max_val = param_range
            if not min_val <= param_value <= max_val:
                raise ValidationError(
                    f"Parameter {param_name} must be between {min_val} and {max_val}"
                )
        
        # Validate choices
        if param_choices and param_value not in param_choices:
            raise ValidationError(
                f"Parameter {param_name} must be one of {param_choices}"
            )
        
        validated_params[param_name] = param_value
    
    return validated_params

def validate_feature_names(
    feature_names: List[str]
) -> List[str]:
    """
    Validate feature names for uniqueness and format.
    
    Args:
        feature_names: List of feature names to validate
    
    Returns:
        Validated list of feature names
    
    Raises:
        ValidationError: If validation fails
    """
    # Check for uniqueness
    if len(feature_names) != len(set(feature_names)):
        raise ValidationError("Feature names must be unique")
    
    # Check for valid string format
    for name in feature_names:
        if not isinstance(name, str):
            raise ValidationError(f"Feature name must be string, got {type(name)}")
        if not name.strip():
            raise ValidationError("Feature names cannot be empty")
    
    return feature_names

def validate_categorical_features(
    data: pd.DataFrame,
    categorical_features: Optional[List[str]] = None
) -> List[str]:
    """
    Validate and identify categorical features in the dataset.
    
    Args:
        data: Input DataFrame
        categorical_features: Optional list of categorical feature names
    
    Returns:
        List of validated categorical feature names
    
    Raises:
        ValidationError: If validation fails
    """
    if categorical_features is None:
        # Automatically detect categorical features
        categorical_features = []
        for col in data.columns:
            if data[col].dtype == 'object' or (
                data[col].dtype.name == 'category'
            ) or (
                data[col].dtype in ['int64', 'int32'] and 
                data[col].nunique() / len(data) < 0.05  # Less than 5% unique values
            ):
                categorical_features.append(col)
    else:
        # Validate provided categorical features
        for col in categorical_features:
            if col not in data.columns:
                raise ValidationError(f"Categorical feature '{col}' not found in data")
    
    return categorical_features

def validate_metrics(
    metrics: Union[str, List[str]],
    task_type: str
) -> List[str]:
    """
    Validate performance metrics for the given task type.
    
    Args:
        metrics: Metric or list of metrics to validate
        task_type: Type of ML task ('classification' or 'regression')
    
    Returns:
        List of validated metric names
    
    Raises:
        ValidationError: If validation fails
    """
    if isinstance(metrics, str):
        metrics = [metrics]
    
    valid_classification_metrics = {
        'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'average_precision'
    }
    valid_regression_metrics = {
        'mse', 'rmse', 'mae', 'r2', 'explained_variance', 'max_error'
    }
    
    valid_metrics = (
        valid_classification_metrics if task_type == 'classification'
        else valid_regression_metrics
    )
    
    for metric in metrics:
        if metric not in valid_metrics:
            raise ValidationError(
                f"Invalid metric '{metric}' for task type '{task_type}'. "
                f"Valid metrics are: {valid_metrics}"
            )
    
    return metrics

def validate_cv_params(
    cv_params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate cross-validation parameters.
    
    Args:
        cv_params: Dictionary of cross-validation parameters
    
    Returns:
        Validated parameters dictionary
    
    Raises:
        ValidationError: If validation fails
    """
    required_params = {'n_splits'}
    optional_params = {'shuffle', 'random_state', 'stratify'}
    
    # Check for required parameters
    missing_params = required_params - set(cv_params.keys())
    if missing_params:
        raise ValidationError(f"Missing required CV parameters: {missing_params}")
    
    # Validate specific parameters
    if cv_params['n_splits'] < 2:
        raise ValidationError("n_splits must be at least 2")
    
    if 'random_state' in cv_params:
        if not isinstance(cv_params['random_state'], (int, type(None))):
            raise ValidationError("random_state must be integer or None")
    
    if 'shuffle' in cv_params:
        if not isinstance(cv_params['shuffle'], bool):
            raise ValidationError("shuffle must be boolean")
    
    # Remove any unexpected parameters
    return {
        k: v for k, v in cv_params.items()
        if k in required_params | optional_params
    }