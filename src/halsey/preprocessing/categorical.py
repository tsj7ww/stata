"""
Categorical Preprocessor Module
=============================

This module provides preprocessing functionality for categorical features,
including encoding methods, missing value handling, and cardinality management.
"""

from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from category_encoders import TargetEncoder, CatBoostEncoder
import logging

from auto_pred_ml.preprocessing.base import BasePreprocessor

logger = logging.getLogger(__name__)

class CategoricalPreprocessor(BasePreprocessor):
    """Preprocessor for categorical features.
    
    Supports multiple encoding strategies including:
    - Label encoding
    - One-hot encoding
    - Target encoding
    - CatBoost encoding
    """
    
    def __init__(self,
                 features: List[str],
                 encoding_method: str = 'auto',
                 max_cardinality: Optional[int] = None,
                 min_frequency: Optional[float] = None,
                 handle_missing: bool = True,
                 handle_unknown: str = 'impute',
                 n_jobs: int = -1):
        """Initialize the categorical preprocessor.
        
        Args:
            features: List of categorical feature names
            encoding_method: Encoding strategy ('auto', 'label', 'onehot', 'target', 'catboost')
            max_cardinality: Maximum number of unique categories to keep
            min_frequency: Minimum frequency for a category to be kept
            handle_missing: Whether to handle missing values
            handle_unknown: Strategy for handling unknown values
            n_jobs: Number of parallel jobs
        """
        super().__init__(features, handle_missing, handle_unknown, n_jobs)
        
        self.encoding_method = encoding_method
        self.max_cardinality = max_cardinality
        self.min_frequency = min_frequency
        
        # Initialize encoders
        self.encoders: Dict[str, Any] = {}
        self.category_maps: Dict[str, Dict] = {}
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate initialization parameters."""
        super()._validate_parameters()
        
        valid_methods = {'auto', 'label', 'onehot', 'target', 'catboost'}
        if self.encoding_method not in valid_methods:
            raise ValueError(f"encoding_method must be one of: {valid_methods}")
            
        if self.max_cardinality is not None and self.max_cardinality < 2:
            raise ValueError("max_cardinality must be at least 2")
            
        if self.min_frequency is not None:
            if not 0 < self.min_frequency < 1:
                raise ValueError("min_frequency must be between 0 and 1")
    
    def _select_encoding_method(self, feature: str, n_unique: int) -> str:
        """Select appropriate encoding method based on cardinality.
        
        Args:
            feature: Feature name
            n_unique: Number of unique values
            
        Returns:
            Selected encoding method
        """
        if self.encoding_method != 'auto':
            return self.encoding_method
            
        if n_unique <= 2:
            return 'label'
        elif n_unique <= 10:
            return 'onehot'
        else:
            return 'target'
    
    def _handle_high_cardinality(self, X: pd.DataFrame, feature: str) -> pd.Series:
        """Handle high cardinality in categorical features.
        
        Args:
            X: Input DataFrame
            feature: Feature to process
            
        Returns:
            Processed feature series
        """
        value_counts = X[feature].value_counts(normalize=True)
        
        if self.max_cardinality:
            top_categories = value_counts.nlargest(self.max_cardinality).index
            self.category_maps[feature] = {cat: cat for cat in top_categories}
            
        if self.min_frequency:
            frequent_categories = value_counts[value_counts >= self.min_frequency].index
            self.category_maps[feature] = {cat: cat for cat in frequent_categories}
        
        if feature in self.category_maps:
            return X[feature].map(self.category_maps[feature]).fillna('_other_')
        
        return X[feature]
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'CategoricalPreprocessor':
        """Fit categorical preprocessor to the data.
        
        Args:
            X: Input features
            y: Target variable (required for target encoding)
            
        Returns:
            self: Fitted preprocessor
        """
        self._validate_data(X)
        
        for feature in self.features:
            # Handle missing values
            if self.handle_missing:
                X[feature] = X[feature].fillna('_missing_')
            
            # Handle high cardinality
            processed_feature = self._handle_high_cardinality(X, feature)
            
            # Select and fit encoder
            n_unique = processed_feature.nunique()
            method = self._select_encoding_method(feature, n_unique)
            
            if method == 'label':
                encoder = LabelEncoder()
                encoder.fit(processed_feature)
            elif method == 'onehot':
                encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                encoder.fit(processed_feature.values.reshape(-1, 1))
            elif method == 'target':
                if y is None:
                    raise ValueError("Target variable required for target encoding")
                encoder = TargetEncoder()
                encoder.fit(processed_feature.values.reshape(-1, 1), y)
            else:  # catboost
                if y is None:
                    raise ValueError("Target variable required for CatBoost encoding")
                encoder = CatBoostEncoder()
                encoder.fit(processed_feature.values.reshape(-1, 1), y)
            
            self.encoders[feature] = (method, encoder)
            
            # Store feature statistics
            self.feature_statistics_[feature] = {
                'n_unique': n_unique,
                'encoding_method': method,
                'missing_rate': X[feature].isnull().mean(),
                'value_counts': X[feature].value_counts().to_dict()
            }
        
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform categorical features using fitted encoders.
        
        Args:
            X: Input features
            
        Returns:
            Transformed features
        """
        self._check_is_fitted()
        self._validate_data(X)
        
        result = pd.DataFrame()
        
        for feature in self.features:
            # Handle missing values
            if self.handle_missing:
                X[feature] = X[feature].fillna('_missing_')
            
            # Apply category mapping if exists
            if feature in self.category_maps:
                processed_feature = X[feature].map(self.category_maps[feature]).fillna('_other_')
            else:
                processed_feature = X[feature]
            
            # Apply encoding
            method, encoder = self.encoders[feature]
            
            if method == 'label':
                encoded = encoder.transform(processed_feature)
                result[feature] = encoded
            elif method == 'onehot':
                encoded = encoder.transform(processed_feature.values.reshape(-1, 1))
                feature_names = [f"{feature}_{cat}" for cat in encoder.categories_[0]]
                result[feature_names] = encoded
            else:  # target or catboost
                encoded = encoder.transform(processed_feature.values.reshape(-1, 1))
                result[feature] = encoded
        
        return result