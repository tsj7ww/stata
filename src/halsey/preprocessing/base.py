"""
Base Preprocessor Module
======================

This module defines the abstract base class for all preprocessors in the package.
It establishes the common interface and shared functionality for data preprocessing.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Union, Any
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class BasePreprocessor(ABC):
    """Abstract base class for all preprocessors.
    
    This class defines the interface that all preprocessors must implement
    and provides common utility methods for preprocessing operations.
    """
    
    def __init__(self, 
                 features: List[str],
                 handle_missing: bool = True,
                 handle_unknown: str = 'ignore',
                 n_jobs: int = -1):
        """Initialize the preprocessor.
        
        Args:
            features: List of feature names to process
            handle_missing: Whether to handle missing values
            handle_unknown: Strategy for handling unknown values ('ignore', 'error', or 'impute')
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        self.features = features
        self.handle_missing = handle_missing
        self.handle_unknown = handle_unknown
        self.n_jobs = n_jobs
        
        # State tracking
        self.is_fitted = False
        self.feature_statistics_ = {}
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate initialization parameters."""
        if not isinstance(self.features, list):
            raise ValueError("features must be a list of column names")
            
        if not all(isinstance(f, str) for f in self.features):
            raise ValueError("all features must be strings")
            
        if self.handle_unknown not in ['ignore', 'error', 'impute']:
            raise ValueError("handle_unknown must be one of: 'ignore', 'error', 'impute'")
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'BasePreprocessor':
        """Fit the preprocessor to the data.
        
        Args:
            X: Input features
            y: Target variable (optional)
            
        Returns:
            self: The fitted preprocessor
        """
        pass
    
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data using the fitted preprocessor.
        
        Args:
            X: Input features
            
        Returns:
            Transformed features
        """
        pass
    
    def fit_transform(self, 
                     X: pd.DataFrame, 
                     y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit the preprocessor and transform the data.
        
        Args:
            X: Input features
            y: Target variable (optional)
            
        Returns:
            Transformed features
        """
        return self.fit(X, y).transform(X)
    
    def _validate_data(self, X: pd.DataFrame):
        """Validate input data.
        
        Args:
            X: Input DataFrame to validate
        
        Raises:
            ValueError: If validation fails
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
            
        missing_features = set(self.features) - set(X.columns)
        if missing_features:
            raise ValueError(f"Features missing from input data: {missing_features}")
    
    def _check_is_fitted(self):
        """Check if the preprocessor is fitted.
        
        Raises:
            RuntimeError: If the preprocessor is not fitted
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before calling transform")
    
    def get_feature_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics about the processed features.
        
        Returns:
            Dictionary containing feature statistics
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted to get feature statistics")
        return self.feature_statistics_
    
    def save(self, path: Union[str, Path]) -> None:
        """Save the fitted preprocessor to disk.
        
        Args:
            path: Path to save the preprocessor
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted preprocessor")
            
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            joblib.dump(self, path)
            logger.info(f"Saved preprocessor to {path}")
        except Exception as e:
            logger.error(f"Error saving preprocessor: {str(e)}")
            raise
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'BasePreprocessor':
        """Load a fitted preprocessor from disk.
        
        Args:
            path: Path to load the preprocessor from
            
        Returns:
            Loaded preprocessor
        """
        try:
            preprocessor = joblib.load(path)
            if not isinstance(preprocessor, cls):
                raise TypeError(f"Loaded object is not a {cls.__name__}")
            logger.info(f"Loaded preprocessor from {path}")
            return preprocessor
        except Exception as e:
            logger.error(f"Error loading preprocessor: {str(e)}")
            raise
    
    def get_params(self) -> Dict[str, Any]:
        """Get preprocessor parameters.
        
        Returns:
            Dictionary of parameters
        """
        return {
            'features': self.features,
            'handle_missing': self.handle_missing,
            'handle_unknown': self.handle_unknown,
            'n_jobs': self.n_jobs
        }
    
    def set_params(self, **params) -> 'BasePreprocessor':
        """Set preprocessor parameters.
        
        Args:
            **params: Parameters to set
            
        Returns:
            self: The preprocessor with updated parameters
        """
        if self.is_fitted:
            raise RuntimeError("Cannot set parameters on fitted preprocessor")
            
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(f"Invalid parameter {key}")
            setattr(self, key, value)
            
        self._validate_parameters()
        return self