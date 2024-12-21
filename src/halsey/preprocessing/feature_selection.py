"""
Feature Selection Module
======================

This module provides feature selection functionality using various methods:
- Statistical tests (chi-square, f_test)
- Model-based selection (tree importance, lasso)
- Correlation analysis
- Variance threshold
"""

from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.feature_selection import (
    SelectKBest, f_classif, chi2, f_regression,
    VarianceThreshold, SelectFromModel
)
from sklearn.linear_model import Lasso, LogisticRegression
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class FeatureImportance:
    """Container for feature importance information."""
    feature_name: str
    importance_score: float
    method: str
    additional_info: Optional[Dict] = None

class FeatureSelector:
    """Feature selection using multiple methods.
    
    Supports:
    - Statistical feature selection
    - Model-based feature selection
    - Correlation-based selection
    - Variance-based selection
    """
    
    def __init__(self,
                 method: str = 'auto',
                 n_features: Optional[int] = None,
                 problem_type: str = 'classification',
                 importance_threshold: Optional[float] = None,
                 correlation_threshold: float = 0.95,
                 variance_threshold: float = 0.01,
                 random_state: int = 42):
        """Initialize the feature selector.
        
        Args:
            method: Selection method ('auto', 'statistical', 'model', 'correlation', 'variance')
            n_features: Number of features to select
            problem_type: Type of problem ('classification' or 'regression')
            importance_threshold: Minimum importance score to keep feature
            correlation_threshold: Maximum correlation between features
            variance_threshold: Minimum variance to keep feature
            random_state: Random state for reproducibility
        """
        self.method = method
        self.n_features = n_features
        self.problem_type = problem_type
        self.importance_threshold = importance_threshold
        self.correlation_threshold = correlation_threshold
        self.variance_threshold = variance_threshold
        self.random_state = random_state
        
        # State tracking
        self.is_fitted = False
        self.selected_features_: List[str] = []
        self.feature_importances_: List[FeatureImportance] = []
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate initialization parameters."""
        valid_methods = {'auto', 'statistical', 'model', 'correlation', 'variance'}
        if self.method not in valid_methods:
            raise ValueError(f"method must be one of: {valid_methods}")
            
        valid_problems = {'classification', 'regression'}
        if self.problem_type not in valid_problems:
            raise ValueError(f"problem_type must be one of: {valid_problems}")
            
        if self.n_features is not None and self.n_features < 1:
            raise ValueError("n_features must be positive")
            
        if (self.importance_threshold is not None and 
            not 0 <= self.importance_threshold <= 1):
            raise ValueError("importance_threshold must be between 0 and 1")
            
        if not 0 <= self.correlation_threshold <= 1:
            raise ValueError("correlation_threshold must be between 0 and 1")
            
        if not 0 <= self.variance_threshold <= 1:
            raise ValueError("variance_threshold must be between 0 and 1")
    
    def _select_method(self, X: pd.DataFrame) -> str:
        """Select appropriate feature selection method.
        
        Args:
            X: Input features
            
        Returns:
            Selected method
        """
        if self.method != 'auto':
            return self.method
            
        n_features = X.shape[1]
        n_samples = X.shape[0]
        
        if n_features > 1000:
            return 'variance'
        elif n_samples < 100:
            return 'correlation'
        else:
            return 'model'
    
    def _get_statistical_importance(self, 
                                  X: pd.DataFrame, 
                                  y: pd.Series) -> List[FeatureImportance]:
        """Calculate statistical feature importance.
        
        Args:
            X: Input features
            y: Target variable
            
        Returns:
            List of feature importance objects
        """
        if self.problem_type == 'classification':
            score_func = f_classif
            # Ensure non-negative values for chi2
            X_scaled = X - X.min()
            chi2_scores, _ = chi2(X_scaled, y)
        else:
            score_func = f_regression
            chi2_scores = np.zeros(X.shape[1])
        
        f_scores, _ = score_func(X, y)
        
        importances = []
        for i, feature in enumerate(X.columns):
            # Combine f-test and chi2 scores for classification
            if self.problem_type == 'classification':
                score = (0.5 * f_scores[i] / np.max(f_scores) + 
                        0.5 * chi2_scores[i] / np.max(chi2_scores))
            else:
                score = f_scores[i] / np.max(f_scores)
                
            importances.append(FeatureImportance(
                feature_name=feature,
                importance_score=score,
                method='statistical',
                additional_info={
                    'f_score': f_scores[i],
                    'chi2_score': chi2_scores[i] if self.problem_type == 'classification' else None
                }
            ))
        
        return importances
    
    def _get_model_importance(self, 
                            X: pd.DataFrame, 
                            y: pd.Series) -> List[FeatureImportance]:
        """Calculate model-based feature importance.
        
        Args:
            X: Input features
            y: Target variable
            
        Returns:
            List of feature importance objects
        """
        importances = []
        
        # Tree-based importance
        if self.problem_type == 'classification':
            rf = RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            rf = RandomForestRegressor(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            )
        
        rf.fit(X, y)
        rf_importance = rf.feature_importances_
        
        # L1-based importance
        if self.problem_type == 'classification':
            l1 = LogisticRegression(
                penalty='l1',
                solver='liblinear',
                random_state=self.random_state
            )
        else:
            l1 = Lasso(
                random_state=self.random_state
            )
        
        l1.fit(X, y)
        l1_importance = np.abs(l1.coef_.ravel())
        
        # LightGBM importance
        if self.problem_type == 'classification':
            lgb_model = lgb.LGBMClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            lgb_model = lgb.LGBMRegressor(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            )
        
        lgb_model.fit(X, y)
        lgb_importance = lgb_model.feature_importances_
        
        # Combine importances
        for i, feature in enumerate(X.columns):
            # Normalize and combine scores
            rf_score = rf_importance[i] / np.max(rf_importance)
            l1_score = l1_importance[i] / np.max(l1_importance)
            lgb_score = lgb_importance[i] / np.max(lgb_importance)
            
            combined_score = (rf_score + l1_score + lgb_score) / 3
            
            importances.append(FeatureImportance(
                feature_name=feature,
                importance_score=combined_score,
                method='model',
                additional_info={
                    'rf_importance': rf_score,
                    'l1_importance': l1_score,
                    'lgb_importance': lgb_score
                }
            ))
        
        return importances
    
    def _get_correlation_groups(self, 
                              X: pd.DataFrame) -> List[List[str]]:
        """Group highly correlated features.
        
        Args:
            X: Input features
            
        Returns:
            List of feature groups with high correlation
        """
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        groups = []
        used_features = set()
        
        for feature in X.columns:
            if feature in used_features:
                continue
                
            # Find highly correlated features
            correlated = upper[feature][
                upper[feature] > self.correlation_threshold
            ].index.tolist()
            
            if correlated:
                group = [feature] + correlated
                groups.append(group)
                used_features.update(group)
            else:
                groups.append([feature])
                used_features.add(feature)
        
        return groups
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'FeatureSelector':
        """Fit feature selector to the data.
        
        Args:
            X: Input features
            y: Target variable
            
        Returns:
            self: Fitted selector
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
            
        # Select method if auto
        method = self._select_method(X)
        
        # Calculate feature importances
        if method in {'statistical', 'model'}:
            if method == 'statistical':
                importances = self._get_statistical_importance(X, y)
            else:  # model
                importances = self._get_model_importance(X, y)
                
            # Sort by importance
            importances.sort(key=lambda x: x.importance_score, reverse=True)
            self.feature_importances_ = importances
            
            # Select features
            if self.n_features:
                selected = [imp.feature_name for imp in importances[:self.n_features]]
            elif self.importance_threshold:
                selected = [imp.feature_name for imp in importances 
                          if imp.importance_score >= self.importance_threshold]
            else:
                selected = [imp.feature_name for imp in importances]
                
        elif method == 'correlation':
            # Group correlated features and select representatives
            groups = self._get_correlation_groups(X)
            selected = [group[0] for group in groups]  # Select first feature from each group
            
            if self.n_features and len(selected) > self.n_features:
                selected = selected[:self.n_features]
                
        else:  # variance
            # Remove low variance features
            selector = VarianceThreshold(threshold=self.variance_threshold)
            selector.fit(X)
            mask = selector.get_support()
            selected = X.columns[mask].tolist()
            
            if self.n_features and len(selected) > self.n_features:
                selected = selected[:self.n_features]
        
        self.selected_features_ = selected
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by selecting features.
        
        Args:
            X: Input features
            
        Returns:
            Transformed features
        """
        if not self.is_fitted:
            raise RuntimeError("Feature selector must be fitted before transform")
            
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
            
        missing_features = set(self.selected_features_) - set(X.columns)
        if missing_features:
            raise ValueError(f"Features missing from input: {missing_features}")
        
        return X[self.selected_features_]
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit selector and transform data.
        
        Args:
            X: Input features
            y: Target variable
            
        Returns:
            Transformed features
        """
        return self.fit(X, y).transform(X)