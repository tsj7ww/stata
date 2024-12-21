"""
Data Configuration Module
=======================

This module provides configuration classes for data preprocessing and handling.
It includes settings for feature types, target variables, and data processing options.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
import numpy as np
from pathlib import Path

@dataclass
class DataConfig:
    """Configuration for data preprocessing and feature handling.
    
    Attributes:
        target_column: Name of the target variable
        features: List of features to use. If None, all columns except target will be used
        categorical_features: List of categorical feature names
        numerical_features: List of numerical feature names
        datetime_features: List of datetime feature names
        text_features: List of text feature names
        id_columns: Columns to ignore during training
        class_weights: Dictionary mapping class labels to weights
        missing_values: Strategy for handling missing values
        feature_selection: Feature selection strategy and parameters
    """
    
    target_column: str
    features: Optional[List[str]] = None
    categorical_features: Optional[List[str]] = None
    numerical_features: Optional[List[str]] = None
    datetime_features: Optional[List[str]] = None
    text_features: Optional[List[str]] = None
    id_columns: Optional[List[str]] = None
    class_weights: Optional[Dict[Union[str, int], float]] = None
    
    # Missing value handling
    missing_values: str = 'auto'  # Options: 'auto', 'drop', 'impute', 'ignore'
    
    # Feature selection settings
    feature_selection: Dict = field(default_factory=lambda: {
        'method': 'none',  # Options: 'none', 'variance', 'correlation', 'importance'
        'max_features': None,
        'threshold': None
    })
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
        
    def _validate_config(self):
        """Validate configuration settings."""
        valid_missing_strategies = {'auto', 'drop', 'impute', 'ignore'}
        if self.missing_values not in valid_missing_strategies:
            raise ValueError(f"missing_values must be one of {valid_missing_strategies}")
            
        valid_feature_selection = {'none', 'variance', 'correlation', 'importance'}
        if self.feature_selection['method'] not in valid_feature_selection:
            raise ValueError(f"feature_selection method must be one of {valid_feature_selection}")
            
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary format."""
        return {
            'target_column': self.target_column,
            'features': self.features,
            'categorical_features': self.categorical_features,
            'numerical_features': self.numerical_features,
            'datetime_features': self.datetime_features,
            'text_features': self.text_features,
            'id_columns': self.id_columns,
            'class_weights': self.class_weights,
            'missing_values': self.missing_values,
            'feature_selection': self.feature_selection
        }
        
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'DataConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
        
    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> 'DataConfig':
        """Load configuration from JSON file."""
        import json
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def save_json(self, json_path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        import json
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
            
    def infer_feature_types(self, df) -> None:
        """Automatically infer feature types from DataFrame.
        
        Args:
            df: pandas DataFrame containing the dataset
        """
        if self.features is None:
            self.features = [col for col in df.columns if col != self.target_column]
            
        if not any([self.categorical_features, self.numerical_features, 
                   self.datetime_features, self.text_features]):
            
            self.categorical_features = []
            self.numerical_features = []
            self.datetime_features = []
            self.text_features = []
            
            for col in self.features:
                if col in (self.id_columns or []):
                    continue
                    
                # Check datetime
                if df[col].dtype.name == 'datetime64[ns]':
                    self.datetime_features.append(col)
                    continue
                
                # Check categorical
                if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                    if df[col].nunique() / len(df) < 0.05:  # Less than 5% unique values
                        self.categorical_features.append(col)
                    else:
                        self.text_features.append(col)
                    continue
                
                # Numerical features
                if np.issubdtype(df[col].dtype, np.number):
                    if df[col].nunique() / len(df) < 0.05:  # Less than 5% unique values
                        self.categorical_features.append(col)
                    else:
                        self.numerical_features.append(col)