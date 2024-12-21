"""
I/O module for the auto_ml package.

This module provides utilities for loading and saving data, models,
and configuration files in various formats.
"""

import json
import pickle
import yaml
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

from ..utils.logging import default_logger
from ..utils.validation import validate_dataset

logger = default_logger

class IOError(Exception):
    """Custom exception for I/O operations in auto_ml."""
    pass

def load_data(
    path: Union[str, Path],
    target_column: Optional[str] = None,
    **kwargs
) -> Tuple[Union[pd.DataFrame, np.ndarray], Optional[np.ndarray]]:
    """
    Load data from various file formats (csv, parquet, pickle, etc.).
    
    Args:
        path: Path to the data file
        target_column: Name of the target column (if applicable)
        **kwargs: Additional arguments passed to the underlying reader
    
    Returns:
        Tuple of (features, target)
    
    Raises:
        IOError: If file loading fails
    """
    path = Path(path)
    
    try:
        if path.suffix == '.csv':
            data = pd.read_csv(path, **kwargs)
        elif path.suffix == '.parquet':
            data = pd.read_parquet(path, **kwargs)
        elif path.suffix == '.pickle':
            with open(path, 'rb') as f:
                data = pickle.load(f)
        elif path.suffix == '.feather':
            data = pd.read_feather(path, **kwargs)
        else:
            raise IOError(f"Unsupported file format: {path.suffix}")
        
        logger.info(f"Successfully loaded data from {path}")
        return validate_dataset(data, target_column)
    
    except Exception as e:
        raise IOError(f"Error loading data from {path}: {str(e)}")

def save_model(
    model: Any,
    path: Union[str, Path],
    save_format: str = 'pickle',
    **kwargs
) -> None:
    """
    Save a trained model to disk.
    
    Args:
        model: Trained model instance
        path: Path to save the model
        save_format: Format to save the model ('pickle', 'joblib')
        **kwargs: Additional arguments for the saving method
    
    Raises:
        IOError: If model saving fails
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if save_format == 'pickle':
            with open(path, 'wb') as f:
                pickle.dump(model, f, **kwargs)
        elif save_format == 'joblib':
            joblib.dump(model, path, **kwargs)
        else:
            raise IOError(f"Unsupported save format: {save_format}")
        
        logger.info(f"Successfully saved model to {path}")
    
    except Exception as e:
        raise IOError(f"Error saving model to {path}: {str(e)}")

def load_model(
    path: Union[str, Path],
    load_format: str = 'pickle',
    **kwargs
) -> Any:
    """
    Load a saved model from disk.
    
    Args:
        path: Path to the saved model
        load_format: Format of the saved model ('pickle', 'joblib')
        **kwargs: Additional arguments for the loading method
    
    Returns:
        Loaded model instance
    
    Raises:
        IOError: If model loading fails
    """
    path = Path(path)
    
    try:
        if load_format == 'pickle':
            with open(path, 'rb') as f:
                model = pickle.load(f, **kwargs)
        elif load_format == 'joblib':
            model = joblib.load(path, **kwargs)
        else:
            raise IOError(f"Unsupported load format: {load_format}")
        
        logger.info(f"Successfully loaded model from {path}")
        return model
    
    except Exception as e:
        raise IOError(f"Error loading model from {path}: {str(e)}")

def save_predictions(
    predictions: Union[np.ndarray, pd.Series],
    path: Union[str, Path],
    index: Optional[Union[List, pd.Index]] = None,
    **kwargs
) -> None:
    """
    Save model predictions to disk.
    
    Args:
        predictions: Model predictions
        path: Path to save predictions
        index: Optional index for the predictions
        **kwargs: Additional arguments for the saving method
    
    Raises:
        IOError: If saving predictions fails
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if isinstance(predictions, pd.Series):
            predictions.to_csv(path, **kwargs)
        else:
            pd.Series(predictions, index=index).to_csv(path, **kwargs)
        
        logger.info(f"Successfully saved predictions to {path}")
    
    except Exception as e:
        raise IOError(f"Error saving predictions to {path}: {str(e)}")

def save_config(
    config: Dict[str, Any],
    path: Union[str, Path],
    format: str = 'yaml'
) -> None:
    """
    Save configuration dictionary to disk.
    
    Args:
        config: Configuration dictionary
        path: Path to save configuration
        format: Format to save configuration ('yaml', 'json')
    
    Raises:
        IOError: If saving configuration fails
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(path, 'w') as f:
            if format == 'yaml':
                yaml.dump(config, f, default_flow_style=False)
            elif format == 'json':
                json.dump(config, f, indent=4)
            else:
                raise IOError(f"Unsupported configuration format: {format}")
        
        logger.info(f"Successfully saved configuration to {path}")
    
    except Exception as e:
        raise IOError(f"Error saving configuration to {path}: {str(e)}")

def load_config(
    path: Union[str, Path],
    format: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load configuration from disk.
    
    Args:
        path: Path to configuration file
        format: Format of configuration file (if None, inferred from extension)
    
    Returns:
        Configuration dictionary
    
    Raises:
        IOError: If loading configuration fails
    """
    path = Path(path)
    
    if format is None:
        format = path.suffix.lstrip('.')
    
    try:
        with open(path, 'r') as f:
            if format == 'yaml':
                config = yaml.safe_load(f)
            elif format == 'json':
                config = json.load(f)
            else:
                raise IOError(f"Unsupported configuration format: {format}")
        
        logger.info(f"Successfully loaded configuration from {path}")
        return config
    
    except Exception as e:
        raise IOError(f"Error loading configuration from {path}: {str(e)}")

def save_results(
    results: Dict[str, Any],
    path: Union[str, Path],
    format: str = 'json'
) -> None:
    """
    Save experiment results to disk.
    
    Args:
        results: Dictionary containing experiment results
        path: Path to save results
        format: Format to save results ('json', 'yaml')
    
    Raises:
        IOError: If saving results fails
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy and pandas objects to native Python types
    def convert_to_serializable(obj):
        if isinstance(obj, (np.ndarray, pd.Series)):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        return obj
    
    serializable_results = {
        k: convert_to_serializable(v) for k, v in results.items()
    }
    
    try:
        with open(path, 'w') as f:
            if format == 'json':
                json.dump(serializable_results, f, indent=4)
            elif format == 'yaml':
                yaml.dump(serializable_results, f, default_flow_style=False)
            else:
                raise IOError(f"Unsupported results format: {format}")
        
        logger.info(f"Successfully saved results to {path}")
    
    except Exception as e:
        raise IOError(f"Error saving results to {path}: {str(e)}")