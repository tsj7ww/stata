"""
Numerical Preprocessor Module
===========================

This module provides preprocessing functionality for numerical features,
including scaling, outlier handling, missing value imputation, and feature engineering.
"""

from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
import logging

from auto_pred_ml.preprocessing.base import BasePreprocessor

logger = logging.getLogger(__name__)

class NumericalPreprocessor(BasePreprocessor):
    """Preprocessor for numerical features.
    
    Supports:
    - Multiple scaling methods
    - Outlier detection and handling
    - Missing value imputation
    - Basic feature engineering
    """
    
    def __init__(self,
                 features: List[str],
                 scaling_method: str = 'standard',
                 handle_outliers: bool = True,
                 outlier_method: str = 'iqr',
                 outlier_params: Optional[Dict] = None,
                 handle_missing: bool = True,
                 missing_method: str = 'auto',
                 feature_engineering: bool = False,
                 n_jobs: int = -1):
        """Initialize the numerical preprocessor.
        
        Args:
            features: List of numerical feature names
            scaling_method: Scaling strategy ('standard', 'minmax', 'robust', 'none')
            handle_outliers: Whether to handle outliers
            outlier_method: Method for outlier detection ('iqr', 'zscore', 'isolation_forest')
            outlier_params: Parameters for outlier detection
            handle_missing: Whether to handle missing values
            missing_method: Method for missing value imputation ('mean', 'median', 'knn', 'auto')
            feature_engineering: Whether to create additional features
            n_jobs: Number of parallel jobs
        """
        super().__init__(features, handle_missing, 'impute', n_jobs)
        
        self.scaling_method = scaling_method
        self.handle_outliers = handle_outliers
        self.outlier_method = outlier_method
        self.outlier_params = outlier_params or {}
        self.missing_method = missing_method
        self.feature_engineering = feature_engineering
        
        # Initialize components
        self.scalers: Dict[str, Any] = {}
        self.imputers: Dict[str, Any] = {}
        self.feature_ranges: Dict[str, Dict[str, float]] = {}
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate initialization parameters."""
        super()._validate_parameters()
        
        valid_scaling = {'standard', 'minmax', 'robust', 'none'}
        if self.scaling_method not in valid_scaling:
            raise ValueError(f"scaling_method must be one of: {valid_scaling}")
            
        valid_outlier = {'iqr', 'zscore', 'isolation_forest'}
        if self.outlier_method not in valid_outlier:
            raise ValueError(f"outlier_method must be one of: {valid_outlier}")
            
        valid_missing = {'mean', 'median', 'knn', 'auto'}
        if self.missing_method not in valid_missing:
            raise ValueError(f"missing_method must be one of: {valid_missing}")
    
    def _get_scaler(self, method: str) -> Any:
        """Get scaler instance based on method.
        
        Args:
            method: Scaling method
            
        Returns:
            Scaler instance
        """
        if method == 'standard':
            return StandardScaler()
        elif method == 'minmax':
            return MinMaxScaler()
        elif method == 'robust':
            return RobustScaler()
        else:
            return None
    
    def _handle_outliers(self, X: pd.Series) -> pd.Series:
        """Handle outliers in numerical features.
        
        Args:
            X: Input series
            
        Returns:
            Series with outliers handled
        """
        if not self.handle_outliers:
            return X
            
        if self.outlier_method == 'iqr':
            Q1 = X.quantile(0.25)
            Q3 = X.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            return X.clip(lower_bound, upper_bound)
            
        elif self.outlier_method == 'zscore':
            z_scores = np.abs((X - X.mean()) / X.std())
            threshold = self.outlier_params.get('threshold', 3)
            
            return X.mask(z_scores > threshold, X.median())
            
        elif self.outlier_method == 'isolation_forest':
            from sklearn.ensemble import IsolationForest
            
            iso_forest = IsolationForest(
                n_estimators=self.outlier_params.get('n_estimators', 100),
                contamination=self.outlier_params.get('contamination', 'auto'),
                random_state=42,
                n_jobs=self.n_jobs
            )
            
            outlier_mask = iso_forest.fit_predict(X.values.reshape(-1, 1)) == -1
            X[outlier_mask] = X.median()
            
            return X
    
    def _get_imputer(self, method: str, n_features: int) -> Any:
        """Get imputer instance based on method.
        
        Args:
            method: Imputation method
            n_features: Number of features
            
        Returns:
            Imputer instance
        """
        if method == 'knn':
            return KNNImputer(
                n_neighbors=min(5, n_features),
                weights='uniform'
            )
        else:
            return SimpleImputer(
                strategy=method,
                add_indicator=True
            )
    
    def _select_imputation_method(self, missing_rate: float) -> str:
        """Select appropriate imputation method based on missing rate.
        
        Args:
            missing_rate: Rate of missing values
            
        Returns:
            Selected imputation method
        """
        if self.missing_method != 'auto':
            return self.missing_method
            
        if missing_rate < 0.1:
            return 'median'
        else:
            return 'knn'
    
    def _engineer_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features.
        
        Args:
            X: Input features
            
        Returns:
            DataFrame with additional engineered features
        """
        if not self.feature_engineering:
            return X
            
        result = X.copy()
        
        # Add basic statistical features
        for feature in self.features:
            result[f"{feature}_squared"] = X[feature] ** 2
            result[f"{feature}_cubed"] = X[feature] ** 3
            
        # Add interactions between features
        if len(self.features) > 1:
            for i, feat1 in enumerate(self.features):
                for feat2 in self.features[i+1:]:
                    result[f"{feat1}_{feat2}_interaction"] = X[feat1] * X[feat2]
        
        return result
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'NumericalPreprocessor':
        """Fit numerical preprocessor to the data.
        
        Args:
            X: Input features
            y: Target variable (optional, not used)
            
        Returns:
            self: Fitted preprocessor
        """
        self._validate_data(X)
        
        for feature in self.features:
            # Store feature ranges
            self.feature_ranges[feature] = {
                'min': X[feature].min(),
                'max': X[feature].max(),
                'mean': X[feature].mean(),
                'std': X[feature].std()
            }
            
            # Handle missing values if needed
            if self.handle_missing:
                missing_rate = X[feature].isnull().mean()
                method = self._select_imputation_method(missing_rate)
                imputer = self._get_imputer(method, len(self.features))
                imputer.fit(X[[feature]])
                self.imputers[feature] = imputer
            
            # Handle outliers
            if self.handle_outliers:
                X[feature] = self._handle_outliers(X[feature])
            
            # Fit scaler
            if self.scaling_method != 'none':
                scaler = self._get_scaler(self.scaling_method)
                scaler.fit(X[[feature]])
                self.scalers[feature] = scaler
            
            # Store feature statistics
            self.feature_statistics_[feature] = {
                'missing_rate': X[feature].isnull().mean(),
                'mean': X[feature].mean(),
                'std': X[feature].std(),
                'min': X[feature].min(),
                'max': X[feature].max(),
                'skew': X[feature].skew(),
                'kurtosis': X[feature].kurtosis()
            }
        
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform numerical features using fitted preprocessors.
        
        Args:
            X: Input features
            
        Returns:
            Transformed features
        """
        self._check_is_fitted()
        self._validate_data(X)
        
        result = pd.DataFrame()
        
        for feature in self.features:
            processed = X[[feature]].copy()
            
            # Handle missing values
            if self.handle_missing and feature in self.imputers:
                processed = pd.DataFrame(
                    self.imputers[feature].transform(processed),
                    columns=[feature]
                )
            
            # Handle outliers
            if self.handle_outliers:
                processed[feature] = self._handle_outliers(processed[feature])
            
            # Apply scaling
            if self.scaling_method != 'none' and feature in self.scalers:
                processed = pd.DataFrame(
                    self.scalers[feature].transform(processed),
                    columns=[feature]
                )
            
            result[feature] = processed[feature]
        
        # Apply feature engineering if enabled
        if self.feature_engineering:
            result = self._engineer_features(result)
        
        return result