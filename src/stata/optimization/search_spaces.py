"""
Search spaces module for hyperparameter optimization.

This module defines hyperparameter search spaces for different models
and provides utilities for search space manipulation.
"""

from typing import Dict, Any, List, Optional, Union
import numpy as np
from scipy.stats import uniform, loguniform
import pandas as pd

class SearchSpace:
    """
    Class to define and manage hyperparameter search spaces.
    Supports both continuous and discrete parameters with various
    distributions.
    """
    
    def __init__(
        self,
        parameter_space: Dict[str, Dict[str, Any]]
    ):
        """
        Initialize search space.
        
        Args:
            parameter_space: Dictionary defining parameter spaces
                Format: {
                    'param_name': {
                        'type': 'continuous'|'integer'|'categorical',
                        'range': [min, max] or list of values,
                        'distribution': 'uniform'|'log-uniform'|None
                    }
                }
        """
        self.parameter_space = parameter_space
        self._validate_space()
    
    def _validate_space(self) -> None:
        """Validate the parameter space configuration."""
        valid_types = {'continuous', 'integer', 'categorical'}
        valid_distributions = {'uniform', 'log-uniform', None}
        
        for param, config in self.parameter_space.items():
            if 'type' not in config:
                raise ValueError(f"Type not specified for parameter {param}")
            
            if config['type'] not in valid_types:
                raise ValueError(
                    f"Invalid type {config['type']} for parameter {param}"
                )
            
            if 'range' not in config:
                raise ValueError(f"Range not specified for parameter {param}")
            
            if config['type'] in {'continuous', 'integer'}:
                if not isinstance(config['range'], (list, tuple)) or \
                   len(config['range']) != 2:
                    raise ValueError(
                        f"Invalid range format for parameter {param}"
                    )
            
            if 'distribution' in config and \
               config['distribution'] not in valid_distributions:
                raise ValueError(
                    f"Invalid distribution {config['distribution']} "
                    f"for parameter {param}"
                )
    
    def sample(self, n_samples: int = 1) -> Dict[str, np.ndarray]:
        """
        Generate random samples from the parameter space.
        
        Args:
            n_samples: Number of samples to generate
        
        Returns:
            Dictionary of parameter samples
        """
        samples = {}
        
        for param, config in self.parameter_space.items():
            if config['type'] == 'continuous':
                samples[param] = self._sample_continuous(
                    config, n_samples
                )
            elif config['type'] == 'integer':
                samples[param] = self._sample_integer(
                    config, n_samples
                )
            else:  # categorical
                samples[param] = self._sample_categorical(
                    config, n_samples
                )
        
        return samples
    
    def _sample_continuous(
        self,
        config: Dict[str, Any],
        n_samples: int
    ) -> np.ndarray:
        """Sample from continuous parameter space."""
        low, high = config['range']
        if config.get('distribution') == 'log-uniform':
            return loguniform(low, high).rvs(n_samples)
        else:
            return uniform(low, high - low).rvs(n_samples)
    
    def _sample_integer(
        self,
        config: Dict[str, Any],
        n_samples: int
    ) -> np.ndarray:
        """Sample from integer parameter space."""
        low, high = config['range']
        if config.get('distribution') == 'log-uniform':
            samples = loguniform(low, high).rvs(n_samples)
        else:
            samples = uniform(low, high - low).rvs(n_samples)
        return np.round(samples).astype(int)
    
    def _sample_categorical(
        self,
        config: Dict[str, Any],
        n_samples: int
    ) -> np.ndarray:
        """Sample from categorical parameter space."""
        return np.random.choice(config['range'], size=n_samples)

# Define model-specific search spaces
def get_random_forest_space(task_type: str) -> SearchSpace:
    """Get search space for Random Forest models."""
    space = {
        'n_estimators': {
            'type': 'integer',
            'range': [50, 500],
            'distribution': 'log-uniform'
        },
        'max_depth': {
            'type': 'integer',
            'range': [3, 20],
            'distribution': 'uniform'
        },
        'min_samples_split': {
            'type': 'integer',
            'range': [2, 20],
            'distribution': 'uniform'
        },
        'min_samples_leaf': {
            'type': 'integer',
            'range': [1, 10],
            'distribution': 'uniform'
        },
        'max_features': {
            'type': 'categorical',
            'range': ['sqrt', 'log2', None]
        }
    }
    return SearchSpace(space)

def get_gradient_boosting_space(task_type: str) -> SearchSpace:
    """Get search space for Gradient Boosting models."""
    space = {
        'n_estimators': {
            'type': 'integer',
            'range': [50, 500],
            'distribution': 'log-uniform'
        },
        'learning_rate': {
            'type': 'continuous',
            'range': [1e-4, 0.3],
            'distribution': 'log-uniform'
        },
        'max_depth': {
            'type': 'integer',
            'range': [3, 12],
            'distribution': 'uniform'
        },
        'min_samples_split': {
            'type': 'integer',
            'range': [2, 20],
            'distribution': 'uniform'
        },
        'subsample': {
            'type': 'continuous',
            'range': [0.5, 1.0],
            'distribution': 'uniform'
        }
    }
    return SearchSpace(space)

def get_elastic_net_space(task_type: str) -> SearchSpace:
    """Get search space for Elastic Net models."""
    space = {
        'alpha': {
            'type': 'continuous',
            'range': [1e-5, 10.0],
            'distribution': 'log-uniform'
        },
        'l1_ratio': {
            'type': 'continuous',
            'range': [0.0, 1.0],
            'distribution': 'uniform'
        }
    }
    return SearchSpace(space)

def get_neural_net_space(task_type: str) -> SearchSpace:
    """Get search space for Neural Network models."""
    space = {
        'hidden_layer_sizes': {
            'type': 'categorical',
            'range': [
                (50,), (100,), (50, 50),
                (100, 50), (100, 100)
            ]
        },
        'learning_rate_init': {
            'type': 'continuous',
            'range': [1e-4, 0.1],
            'distribution': 'log-uniform'
        },
        'alpha': {
            'type': 'continuous',
            'range': [1e-5, 0.1],
            'distribution': 'log-uniform'
        },
        'batch_size': {
            'type': 'integer',
            'range': [16, 256],
            'distribution': 'log-uniform'
        }
    }
    return SearchSpace(space)

def get_svm_space(task_type: str) -> SearchSpace:
    """Get search space for SVM models."""
    space = {
        'C': {
            'type': 'continuous',
            'range': [1e-3, 100.0],
            'distribution': 'log-uniform'
        },
        'gamma': {
            'type': 'continuous',
            'range': [1e-4, 10.0],
            'distribution': 'log-uniform'
        },
        'kernel': {
            'type': 'categorical',
            'range': ['rbf', 'linear', 'poly']
        }
    }
    return SearchSpace(space)

# Mapping of model names to their search spaces
MODEL_SEARCH_SPACES = {
    'random_forest': get_random_forest_space,
    'gradient_boosting': get_gradient_boosting_space,
    'elastic_net': get_elastic_net_space,
    'neural_net': get_neural_net_space,
    'svm': get_svm_space
}

def get_search_space(
    model_name: str,
    task_type: str
) -> SearchSpace:
    """
    Get the search space for a specific model.
    
    Args:
        model_name: Name of the model
        task_type: Type of ML task
    
    Returns:
        SearchSpace instance for the model
    
    Raises:
        ValueError: If model_name is not supported
    """
    if model_name not in MODEL_SEARCH_SPACES:
        raise ValueError(
            f"Search space not defined for model: {model_name}"
        )
    
    return MODEL_SEARCH_SPACES[model_name](task_type)

def combine_search_spaces(
    search_spaces: List[SearchSpace],
    prefixes: Optional[List[str]] = None
) -> SearchSpace:
    """
    Combine multiple search spaces into one.
    
    Args:
        search_spaces: List of SearchSpace instances
        prefixes: Optional prefixes for parameter names
    
    Returns:
        Combined SearchSpace instance
    """
    if prefixes is None:
        prefixes = [f"model_{i}_" for i in range(len(search_spaces))]
    
    combined_space = {}
    for space, prefix in zip(search_spaces, prefixes):
        for param, config in space.parameter_space.items():
            combined_space[f"{prefix}{param}"] = config
    
    return SearchSpace(combined_space)