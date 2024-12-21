from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.base import clone
from scipy import stats

from .base import BaseModel
from ..config.model_config import ModelConfig
from ..utils.validation import check_consistent_length

class VotingEnsemble(BaseModel):
    """
    Voting ensemble that combines predictions from multiple models.
    Supports both hard and soft voting for classification,
    and weighted averaging for regression.
    """
    
    def __init__(self, 
                 models: List[BaseModel],
                 weights: Optional[List[float]] = None,
                 voting: str = 'hard'):
        """
        Initialize voting ensemble.
        
        Args:
            models: List of initialized models
            weights: Optional weights for each model
            voting: 'hard' or 'soft' for classification, ignored for regression
        """
        super().__init__()
        self.models = models
        self.weights = weights
        if weights is not None:
            if len(weights) != len(models):
                raise ValueError("Number of weights must match number of models")
            if not np.isclose(sum(weights), 1.0):
                raise ValueError("Weights must sum to 1.0")
        self.voting = voting
        
    def _build_model(self) -> Any:
        """No model to build for voting ensemble."""
        return None
        
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            y: Union[np.ndarray, pd.Series],
            **kwargs) -> 'VotingEnsemble':
        """
        Fit all models in the ensemble.
        
        Args:
            X: Training features
            y: Target values
            **kwargs: Additional arguments passed to each model's fit method
        """
        for model in self.models:
            model.fit(X, y, **kwargs)
        self.is_fitted = True
        return self
        
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions using the ensemble.
        
        Args:
            X: Features to predict
            
        Returns:
            np.ndarray: Ensemble predictions
        """
        self._check_is_fitted()
        
        # Get predictions from all models
        predictions = np.array([model.predict(X) for model in self.models])
        
        # For regression or soft voting with probabilities
        if (self.voting == 'soft' and hasattr(self.models[0], 'predict_proba')) or \
           not hasattr(self.models[0], 'predict_proba'):
            if self.weights is not None:
                return np.average(predictions, weights=self.weights, axis=0)
            return np.mean(predictions, axis=0)
            
        # For hard voting (classification)
        if self.weights is not None:
            # Weighted vote
            weighted_pred = np.zeros_like(predictions[0])
            for pred, weight in zip(predictions, self.weights):
                weighted_pred[pred == 1] += weight
            return (weighted_pred > 0.5).astype(int)
            
        # Unweighted vote
        return stats.mode(predictions, axis=0)[0]

class StackingEnsemble(BaseModel):
    """
    Stacking ensemble that uses predictions from base models as features
    for a meta-model. Supports both classification and regression.
    """
    
    def __init__(self, 
                 base_models: List[BaseModel],
                 meta_model: BaseModel,
                 n_folds: int = 5,
                 use_proba: bool = False):
        """
        Initialize stacking ensemble.
        
        Args:
            base_models: List of base models
            meta_model: Model to make final predictions
            n_folds: Number of folds for cross-validation
            use_proba: Whether to use probability predictions for classification
        """
        super().__init__()
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
        self.use_proba = use_proba
        
    def _build_model(self) -> Any:
        """No model to build for stacking ensemble."""
        return None
        
    def _get_meta_features(self, X: Union[np.ndarray, pd.DataFrame], 
                          y: Optional[Union[np.ndarray, pd.Series]] = None,
                          is_train: bool = False) -> np.ndarray:
        """
        Generate meta-features from base models.
        
        Args:
            X: Input features
            y: Target values (only needed for training)
            is_train: Whether this is for training
            
        Returns:
            np.ndarray: Meta-features for meta-model
        """
        if is_train:
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            meta_features = np.zeros((X.shape[0], len(self.base_models)))
            
            # Generate meta-features using cross-validation
            for i, model in enumerate(self.base_models):
                for train_idx, val_idx in kf.split(X):
                    # Clone model for each fold
                    clone_model = clone(model)
                    
                    # Fit on training fold
                    clone_model.fit(X[train_idx], y[train_idx])
                    
                    # Predict on validation fold
                    if self.use_proba and hasattr(clone_model, 'predict_proba'):
                        fold_pred = clone_model.predict_proba(X[val_idx])[:, 1]
                    else:
                        fold_pred = clone_model.predict(X[val_idx])
                    
                    meta_features[val_idx, i] = fold_pred
                    
            # Fit base models on full dataset
            for model in self.base_models:
                model.fit(X, y)
                
        else:
            # For prediction, just use fitted base models
            meta_features = np.column_stack([
                model.predict_proba(X)[:, 1] if self.use_proba and hasattr(model, 'predict_proba')
                else model.predict(X)
                for model in self.base_models
            ])
            
        return meta_features
        
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            y: Union[np.ndarray, pd.Series],
            **kwargs) -> 'StackingEnsemble':
        """
        Fit the stacking ensemble.
        
        Args:
            X: Training features
            y: Target values
            **kwargs: Additional arguments passed to models' fit methods
        """
        # Generate meta-features
        meta_features = self._get_meta_features(X, y, is_train=True)
        
        # Fit meta-model
        self.meta_model.fit(meta_features, y, **kwargs)
        
        self.is_fitted = True
        return self
        
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions using the stacking ensemble.
        
        Args:
            X: Features to predict
            
        Returns:
            np.ndarray: Ensemble predictions
        """
        self._check_is_fitted()
        
        # Generate meta-features
        meta_features = self._get_meta_features(X, is_train=False)
        
        # Make final predictions using meta-model
        return self.meta_model.predict(meta_features)

class BaggingEnsemble(BaseModel):
    """
    Bagging ensemble that trains multiple instances of the same model
    on different bootstrap samples of the dataset.
    """
    
    def __init__(self, 
                 base_model: BaseModel,
                 n_estimators: int = 10,
                 bootstrap_fraction: float = 1.0,
                 random_state: Optional[int] = None):
        """
        Initialize bagging ensemble.
        
        Args:
            base_model: Base model to create ensemble from
            n_estimators: Number of models in ensemble
            bootstrap_fraction: Fraction of dataset to use for each bootstrap
            random_state: Random seed for reproducibility
        """
        super().__init__()
        self.base_model = base_model
        self.n_estimators = n_estimators
        self.bootstrap_fraction = bootstrap_fraction
        self.random_state = random_state
        self.models: List[BaseModel] = []
        
    def _build_model(self) -> Any:
        """No model to build for bagging ensemble."""
        return None
        
    def _bootstrap_sample(self, X: np.ndarray, y: np.ndarray, 
                         random_state: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate bootstrap sample."""
        np.random.seed(random_state)
        n_samples = int(X.shape[0] * self.bootstrap_fraction)
        indices = np.random.choice(X.shape[0], size=n_samples, replace=True)
        return X[indices], y[indices]
        
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            y: Union[np.ndarray, pd.Series],
            **kwargs) -> 'BaggingEnsemble':
        """
        Fit the bagging ensemble.
        
        Args:
            X: Training features
            y: Target values
            **kwargs: Additional arguments passed to models' fit methods
        """
        X = np.array(X)
        y = np.array(y)
        
        self.models = []
        for i in range(self.n_estimators):
            # Create bootstrap sample
            X_bootstrap, y_bootstrap = self._bootstrap_sample(
                X, y, random_state=self.random_state + i if self.random_state else None
            )
            
            # Clone and fit model
            model = clone(self.base_model)
            model.fit(X_bootstrap, y_bootstrap, **kwargs)
            self.models.append(model)
            
        self.is_fitted = True
        return self
        
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions using the bagging ensemble.
        
        Args:
            X: Features to predict
            
        Returns:
            np.ndarray: Ensemble predictions
        """
        self._check_is_fitted()
        
        # Get predictions from all models
        predictions = np.array([model.predict(X) for model in self.models])
        
        # For regression or if models don't support predict_proba
        if not hasattr(self.models[0], 'predict_proba'):
            return np.mean(predictions, axis=0)
            
        # For classification
        return stats.mode(predictions, axis=0)[0]
        
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict class probabilities for classification tasks.
        
        Args:
            X: Features to predict
            
        Returns:
            np.ndarray: Class probabilities
        """
        self._check_is_fitted()
        
        if not hasattr(self.models[0], 'predict_proba'):
            raise ValueError("Base model doesn't support probability predictions")
            
        # Get probability predictions from all models
        probas = np.array([model.predict_proba(X) for model in self.models])
        
        # Average probabilities
        return np.mean(probas, axis=0)