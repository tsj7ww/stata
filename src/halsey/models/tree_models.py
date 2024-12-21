from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb

from .base import BaseModel
from ..config.model_config import ModelConfig

class DecisionTreeModel(BaseModel):
    """
    Decision Tree model implementation supporting both classification and regression.
    """
    
    def __init__(self, model_config: Optional[ModelConfig] = None, task: str = 'classification'):
        """
        Initialize Decision Tree model.
        
        Args:
            model_config (ModelConfig, optional): Model configuration
            task (str): Either 'classification' or 'regression'
        """
        super().__init__(model_config)
        self.task = task
        
    def _build_model(self) -> Any:
        """Build the appropriate decision tree model based on the task."""
        params = self.model_config.get_model_params('decision_tree')
        
        if self.task == 'classification':
            return DecisionTreeClassifier(**params)
        elif self.task == 'regression':
            return DecisionTreeRegressor(**params)
        else:
            raise ValueError(f"Unknown task type: {self.task}")

class RandomForestModel(BaseModel):
    """
    Random Forest model implementation supporting both classification and regression.
    Includes additional methods for feature importance analysis.
    """
    
    def __init__(self, model_config: Optional[ModelConfig] = None, task: str = 'classification'):
        super().__init__(model_config)
        self.task = task
        
    def _build_model(self) -> Any:
        """Build the appropriate random forest model based on the task."""
        params = self.model_config.get_model_params('random_forest')
        
        if self.task == 'classification':
            return RandomForestClassifier(**params)
        elif self.task == 'regression':
            return RandomForestRegressor(**params)
        else:
            raise ValueError(f"Unknown task type: {self.task}")
            
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Returns:
            pd.DataFrame: DataFrame containing feature names and their importance scores
        """
        self._check_is_fitted()
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        return pd.DataFrame({
            'feature': [self.feature_names[i] for i in indices],
            'importance': importances[indices]
        })

class GradientBoostingModel(BaseModel):
    """
    Gradient Boosting model implementation supporting both classification and regression.
    Includes early stopping and learning rate scheduling.
    """
    
    def __init__(self, model_config: Optional[ModelConfig] = None, task: str = 'classification'):
        super().__init__(model_config)
        self.task = task
        
    def _build_model(self) -> Any:
        """Build the appropriate gradient boosting model based on the task."""
        params = self.model_config.get_model_params('gradient_boosting')
        
        if self.task == 'classification':
            return GradientBoostingClassifier(**params)
        elif self.task == 'regression':
            return GradientBoostingRegressor(**params)
        else:
            raise ValueError(f"Unknown task type: {self.task}")
            
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            y: Union[np.ndarray, pd.Series],
            eval_set: Optional[list] = None,
            **kwargs) -> 'GradientBoostingModel':
        """
        Fit the gradient boosting model with optional early stopping.
        
        Args:
            X: Training features
            y: Target values
            eval_set: Optional evaluation set for early stopping
            **kwargs: Additional arguments for the fit method
        """
        return super().fit(X, y, eval_set=eval_set, **kwargs)

class XGBoostModel(BaseModel):
    """
    XGBoost model implementation with advanced features like early stopping,
    custom objective functions, and GPU acceleration support.
    """
    
    def __init__(self, model_config: Optional[ModelConfig] = None, task: str = 'classification'):
        super().__init__(model_config)
        self.task = task
        
    def _build_model(self) -> Any:
        """Build the appropriate XGBoost model based on the task."""
        params = self.model_config.get_model_params('xgboost')
        
        if self.task == 'classification':
            params['objective'] = 'binary:logistic' if not params.get('objective') else params['objective']
            return xgb.XGBClassifier(**params)
        elif self.task == 'regression':
            params['objective'] = 'reg:squarederror' if not params.get('objective') else params['objective']
            return xgb.XGBRegressor(**params)
        else:
            raise ValueError(f"Unknown task type: {self.task}")
            
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            y: Union[np.ndarray, pd.Series],
            eval_set: Optional[list] = None,
            early_stopping_rounds: Optional[int] = None,
            **kwargs) -> 'XGBoostModel':
        """
        Fit the XGBoost model with support for early stopping and evaluation sets.
        
        Args:
            X: Training features
            y: Target values
            eval_set: Optional evaluation set for early stopping
            early_stopping_rounds: Number of rounds for early stopping
            **kwargs: Additional arguments for the fit method
        """
        kwargs['eval_set'] = eval_set if eval_set else [(X, y)]
        kwargs['early_stopping_rounds'] = early_stopping_rounds
        return super().fit(X, y, **kwargs)

class LightGBMModel(BaseModel):
    """
    LightGBM model implementation with support for categorical features,
    early stopping, and GPU acceleration.
    """
    
    def __init__(self, model_config: Optional[ModelConfig] = None, task: str = 'classification'):
        super().__init__(model_config)
        self.task = task
        
    def _build_model(self) -> Any:
        """Build the appropriate LightGBM model based on the task."""
        params = self.model_config.get_model_params('lightgbm')
        
        if self.task == 'classification':
            params['objective'] = 'binary' if not params.get('objective') else params['objective']
            return lgb.LGBMClassifier(**params)
        elif self.task == 'regression':
            params['objective'] = 'regression' if not params.get('objective') else params['objective']
            return lgb.LGBMRegressor(**params)
        else:
            raise ValueError(f"Unknown task type: {self.task}")
            
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            y: Union[np.ndarray, pd.Series],
            eval_set: Optional[list] = None,
            categorical_feature: Optional[list] = None,
            early_stopping_rounds: Optional[int] = None,
            **kwargs) -> 'LightGBMModel':
        """
        Fit the LightGBM model with support for categorical features and early stopping.
        
        Args:
            X: Training features
            y: Target values
            eval_set: Optional evaluation set for early stopping
            categorical_feature: List of categorical feature indices or names
            early_stopping_rounds: Number of rounds for early stopping
            **kwargs: Additional arguments for the fit method
        """
        kwargs['eval_set'] = eval_set if eval_set else [(X, y)]
        kwargs['categorical_feature'] = categorical_feature
        kwargs['early_stopping_rounds'] = early_stopping_rounds
        return super().fit(X, y, **kwargs)