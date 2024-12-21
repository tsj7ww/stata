from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, List
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from ..utils.validation import check_array, check_consistent_length
from ..config.model_config import ModelConfig

class BaseModel(BaseEstimator, ABC):
    """
    Abstract base class for all models in the auto_ml package.
    
    This class defines the interface that all model implementations must follow
    and provides common functionality for model handling.
    
    Attributes:
        model_config (ModelConfig): Configuration object containing model parameters
        model (Any): The underlying model instance
        is_fitted (bool): Flag indicating if the model has been fitted
        feature_names (List[str]): Names of features used during training
    """
    
    def __init__(self, model_config: Optional[ModelConfig] = None):
        """
        Initialize the base model.
        
        Args:
            model_config (ModelConfig, optional): Configuration object for the model.
                If None, default configuration will be used.
        """
        self.model_config = model_config or ModelConfig()
        self.model = None
        self.is_fitted = False
        self.feature_names: List[str] = []
        
    @abstractmethod
    def _build_model(self) -> Any:
        """
        Build the underlying model instance.
        
        This method must be implemented by concrete model classes.
        
        Returns:
            Any: The built model instance
        """
        pass
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            y: Union[np.ndarray, pd.Series],
            **kwargs) -> 'BaseModel':
        """
        Fit the model to the provided training data.
        
        Args:
            X: Training features
            y: Target values
            **kwargs: Additional arguments to pass to the underlying model's fit method
            
        Returns:
            self: The fitted model instance
        """
        # Validate input
        X = self._validate_input(X)
        check_consistent_length(X, y)
        
        # Build model if not already built
        if self.model is None:
            self.model = self._build_model()
            
        # Store feature names if available
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            
        # Fit the model
        self.model.fit(X, y, **kwargs)
        self.is_fitted = True
        
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions using the fitted model.
        
        Args:
            X: Features to make predictions for
            
        Returns:
            np.ndarray: Model predictions
            
        Raises:
            ValueError: If the model hasn't been fitted
        """
        self._check_is_fitted()
        X = self._validate_input(X)
        return self.model.predict(X)
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict class probabilities for X.
        
        Args:
            X: Features to make predictions for
            
        Returns:
            np.ndarray: Predicted class probabilities
            
        Raises:
            ValueError: If the model doesn't support probability predictions
                       or hasn't been fitted
        """
        self._check_is_fitted()
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError(f"{self.__class__.__name__} does not support probability predictions")
        X = self._validate_input(X)
        return self.model.predict_proba(X)
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.
        
        Returns:
            Dict[str, Any]: Dictionary of model parameters
        """
        if self.model is None:
            return {}
        return self.model.get_params()
    
    def set_params(self, **params) -> 'BaseModel':
        """
        Set model parameters.
        
        Args:
            **params: Parameters to set
            
        Returns:
            self: The model instance
        """
        if self.model is not None:
            self.model.set_params(**params)
        return self
    
    def _validate_input(self, X: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """
        Validate input data.
        
        Args:
            X: Input features to validate
            
        Returns:
            Union[np.ndarray, pd.DataFrame]: Validated input features
        """
        # Check array
        X = check_array(X)
        
        # Validate feature names if available
        if isinstance(X, pd.DataFrame) and self.feature_names:
            if not all(col in X.columns for col in self.feature_names):
                raise ValueError("Feature names do not match those from training")
                
        return X
    
    def _check_is_fitted(self):
        """
        Check if the model has been fitted.
        
        Raises:
            ValueError: If the model hasn't been fitted
        """
        if not self.is_fitted:
            raise ValueError(
                f"This {self.__class__.__name__} instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this model."
            )