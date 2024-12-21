from typing import Any, Dict, Optional, Union, List
import numpy as np
import pandas as pd
from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    SGDClassifier,
    SGDRegressor
)
from sklearn.preprocessing import StandardScaler
from .base import BaseModel
from ..config.model_config import ModelConfig

class LinearModel(BaseModel):
    """
    Base linear model implementation supporting both regression and classification.
    Includes automatic scaling and sparse data handling.
    """
    
    def __init__(self, model_config: Optional[ModelConfig] = None, task: str = 'regression'):
        """
        Initialize linear model.
        
        Args:
            model_config (ModelConfig, optional): Model configuration
            task (str): Either 'regression' or 'classification'
        """
        super().__init__(model_config)
        self.task = task
        self.scaler = StandardScaler()
        self.scale_data = True  # Can be toggled through model_config
        
    def _build_model(self) -> Any:
        """Build the appropriate linear model based on the task."""
        if self.task == 'regression':
            return LinearRegression(**self.model_config.get_model_params('linear'))
        elif self.task == 'classification':
            return LogisticRegression(**self.model_config.get_model_params('logistic'))
        else:
            raise ValueError(f"Unknown task type: {self.task}")
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            y: Union[np.ndarray, pd.Series], 
            **kwargs) -> 'LinearModel':
        """
        Fit the linear model with optional scaling.
        
        Args:
            X: Training features
            y: Target values
            **kwargs: Additional arguments for fit method
        """
        X_processed = X.copy()
        if self.scale_data:
            X_processed = self.scaler.fit_transform(X)
        return super().fit(X_processed, y, **kwargs)
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions with optional scaling.
        
        Args:
            X: Features to predict
        """
        X_processed = X.copy()
        if self.scale_data:
            X_processed = self.scaler.transform(X)
        return super().predict(X_processed)
    
    def get_coefficients(self) -> pd.DataFrame:
        """
        Get model coefficients with feature names.
        
        Returns:
            pd.DataFrame: DataFrame containing feature names and their coefficients
        """
        self._check_is_fitted()
        coefficients = self.model.coef_
        if self.task == 'classification':
            coefficients = coefficients[0] if len(coefficients.shape) > 1 else coefficients
            
        return pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': coefficients
        })

class RidgeModel(LinearModel):
    """
    Ridge regression model with L2 regularization.
    Includes cross-validation support for alpha selection.
    """
    
    def _build_model(self) -> Any:
        """Build Ridge model with specified parameters."""
        params = self.model_config.get_model_params('ridge')
        if self.task == 'regression':
            return Ridge(**params)
        else:
            # For classification, use LogisticRegression with L2 penalty
            params['penalty'] = 'l2'
            return LogisticRegression(**params)

class LassoModel(LinearModel):
    """
    Lasso regression model with L1 regularization.
    Includes automatic feature selection and sparsity control.
    """
    
    def _build_model(self) -> Any:
        """Build Lasso model with specified parameters."""
        params = self.model_config.get_model_params('lasso')
        if self.task == 'regression':
            return Lasso(**params)
        else:
            # For classification, use LogisticRegression with L1 penalty
            params['penalty'] = 'l1'
            params['solver'] = 'liblinear'  # Better for L1
            return LogisticRegression(**params)
    
    def get_selected_features(self) -> List[str]:
        """
        Get features selected by Lasso (non-zero coefficients).
        
        Returns:
            List[str]: List of selected feature names
        """
        coef_df = self.get_coefficients()
        return coef_df[coef_df['coefficient'] != 0]['feature'].tolist()

class ElasticNetModel(LinearModel):
    """
    ElasticNet model combining L1 and L2 regularization.
    Includes automatic mixing parameter selection.
    """
    
    def _build_model(self) -> Any:
        """Build ElasticNet model with specified parameters."""
        params = self.model_config.get_model_params('elastic_net')
        if self.task == 'regression':
            return ElasticNet(**params)
        else:
            # For classification, use LogisticRegression with elastic net penalty
            params['penalty'] = 'elasticnet'
            params['solver'] = 'saga'  # Required for elastic net
            return LogisticRegression(**params)

class SGDModel(LinearModel):
    """
    Stochastic Gradient Descent model supporting various loss functions
    and penalties. Suitable for large-scale learning.
    """
    
    def __init__(self, model_config: Optional[ModelConfig] = None, 
                 task: str = 'regression',
                 loss: str = 'squared_error'):
        """
        Initialize SGD model.
        
        Args:
            model_config (ModelConfig, optional): Model configuration
            task (str): Either 'regression' or 'classification'
            loss (str): Loss function to be used
        """
        super().__init__(model_config, task)
        self.loss = loss
    
    def _build_model(self) -> Any:
        """Build SGD model with specified parameters."""
        params = self.model_config.get_model_params('sgd')
        params['loss'] = self.loss
        
        if self.task == 'regression':
            return SGDRegressor(**params)
        elif self.task == 'classification':
            return SGDClassifier(**params)
        else:
            raise ValueError(f"Unknown task type: {self.task}")
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            y: Union[np.ndarray, pd.Series],
            **kwargs) -> 'SGDModel':
        """
        Fit the SGD model with support for partial_fit.
        
        Args:
            X: Training features
            y: Target values
            **kwargs: Additional arguments including batch_size for partial_fit
        """
        # Extract batch_size if provided
        batch_size = kwargs.pop('batch_size', None)
        
        if batch_size is not None:
            # Implement partial_fit logic for large datasets
            n_samples = len(X)
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]
                
                if self.scale_data:
                    X_batch = self.scaler.transform(X_batch)
                
                if not self.is_fitted:
                    self.model.partial_fit(X_batch, y_batch, 
                                         classes=np.unique(y) if self.task == 'classification' else None)
                    self.is_fitted = True
                else:
                    self.model.partial_fit(X_batch, y_batch)
        else:
            # Use regular fit for smaller datasets
            super().fit(X, y, **kwargs)
        
        return self