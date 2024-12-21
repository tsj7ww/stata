"""
Model Configuration Module
========================

This module provides configuration classes for model training, evaluation,
and optimization settings. It includes hyperparameter search spaces,
cross-validation settings, and model selection criteria.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import json

@dataclass
class ModelConfig:
    """Configuration for model training and optimization.
    
    Attributes:
        problem_type: Type of ML problem ('classification' or 'regression')
        metric: Primary metric for optimization
        models: List of models to try
        optimization_direction: Direction of metric optimization
        cv_settings: Cross-validation configuration
        optimization_settings: Hyperparameter optimization settings
        hardware_settings: Compute resource settings
    """
    
    problem_type: str
    metric: str
    models: List[str] = field(default_factory=lambda: ['lightgbm', 'xgboost', 'catboost'])
    optimization_direction: str = 'maximize'
    
    cv_settings: Dict[str, Any] = field(default_factory=lambda: {
        'n_splits': 5,
        'shuffle': True,
        'stratify': True,  # Only for classification
        'random_state': 42
    })
    
    optimization_settings: Dict[str, Any] = field(default_factory=lambda: {
        'n_trials': 100,
        'timeout': None,  # seconds, None for no timeout
        'n_jobs': -1,
        'early_stopping': True,
        'early_stopping_rounds': 10
    })
    
    hardware_settings: Dict[str, Any] = field(default_factory=lambda: {
        'use_gpu': False,
        'gpu_ids': None,
        'cpu_threads': -1,
        'memory_limit': None  # GB, None for no limit
    })
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
        
    def _validate_config(self):
        """Validate configuration settings."""
        # Validate problem type
        valid_problem_types = {'classification', 'regression'}
        if self.problem_type not in valid_problem_types:
            raise ValueError(f"problem_type must be one of {valid_problem_types}")
        
        # Validate metric based on problem type
        valid_metrics = self._get_valid_metrics()
        if self.metric not in valid_metrics:
            raise ValueError(f"metric must be one of {valid_metrics}")
        
        # Validate optimization direction
        valid_directions = {'maximize', 'minimize'}
        if self.optimization_direction not in valid_directions:
            raise ValueError(f"optimization_direction must be one of {valid_directions}")
        
        # Validate models
        valid_models = {'lightgbm', 'xgboost', 'catboost', 'random_forest', 
                       'extra_trees', 'neural_network'}
        invalid_models = set(self.models) - valid_models
        if invalid_models:
            raise ValueError(f"Invalid models: {invalid_models}. Must be subset of {valid_models}")
    
    def _get_valid_metrics(self) -> set:
        """Get valid metrics based on problem type."""
        if self.problem_type == 'classification':
            return {'accuracy', 'precision', 'recall', 'f1', 'auc', 'log_loss'}
        else:  # regression
            return {'mse', 'rmse', 'mae', 'mape', 'r2'}
    
    def get_metric_function(self):
        """Get the corresponding metric function."""
        from sklearn import metrics
        
        metric_mapping = {
            # Classification metrics
            'accuracy': metrics.accuracy_score,
            'precision': metrics.precision_score,
            'recall': metrics.recall_score,
            'f1': metrics.f1_score,
            'auc': metrics.roc_auc_score,
            'log_loss': metrics.log_loss,
            
            # Regression metrics
            'mse': metrics.mean_squared_error,
            'rmse': lambda y_true, y_pred: metrics.mean_squared_error(y_true, y_pred, squared=False),
            'mae': metrics.mean_absolute_error,
            'mape': metrics.mean_absolute_percentage_error,
            'r2': metrics.r2_score
        }
        
        return metric_mapping[self.metric]
    
    def get_search_space(self, model_name: str) -> Dict:
        """Get hyperparameter search space for specified model."""
        search_spaces = {
            'lightgbm': {
                'learning_rate': ('log_uniform', 1e-4, 1e-1),
                'num_leaves': ('int', 2, 256),
                'max_depth': ('int', 3, 12),
                'min_data_in_leaf': ('int', 2, 100),
                'feature_fraction': ('uniform', 0.5, 1.0),
                'bagging_fraction': ('uniform', 0.5, 1.0),
                'bagging_freq': ('int', 1, 10)
            },
            'xgboost': {
                'learning_rate': ('log_uniform', 1e-4, 1e-1),
                'max_depth': ('int', 3, 12),
                'min_child_weight': ('int', 1, 10),
                'subsample': ('uniform', 0.5, 1.0),
                'colsample_bytree': ('uniform', 0.5, 1.0),
                'gamma': ('log_uniform', 1e-8, 1.0)
            },
            'catboost': {
                'learning_rate': ('log_uniform', 1e-4, 1e-1),
                'depth': ('int', 3, 12),
                'l2_leaf_reg': ('log_uniform', 1e-1, 10),
                'subsample': ('uniform', 0.5, 1.0),
                'colsample_bylevel': ('uniform', 0.5, 1.0)
            }
        }
        
        return search_spaces.get(model_name, {})
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary format."""
        return {
            'problem_type': self.problem_type,
            'metric': self.metric,
            'models': self.models,
            'optimization_direction': self.optimization_direction,
            'cv_settings': self.cv_settings,
            'optimization_settings': self.optimization_settings,
            'hardware_settings': self.hardware_settings
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'ModelConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> 'ModelConfig':
        """Load configuration from JSON file."""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def save_json(self, json_path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)