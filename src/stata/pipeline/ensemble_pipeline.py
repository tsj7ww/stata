"""
Ensemble pipeline module for the auto_ml package.

This module implements various ensemble methods including stacking,
weighted averaging, and voting, allowing combination of multiple
base models for improved performance.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from pathlib import Path

from ..models.base import BaseModel
from ..utils.logging import LoggerMixin
from ..utils.validation import (
    validate_dataset,
    validate_metrics,
    validate_cv_params
)
from ..evaluation.metrics import calculate_metrics
from ..optimization.hyperopt import HyperoptOptimizer

class EnsemblePipeline(LoggerMixin):
    """
    Advanced ensemble pipeline that combines multiple models using
    stacking, weighted averaging, or voting methods.
    """
    
    def __init__(
        self,
        task_type: str,
        base_models: List[BaseModel],
        meta_model: Optional[BaseModel] = None,
        ensemble_method: str = 'stacking',
        optimization_metric: str = 'accuracy',
        cv_params: Optional[Dict[str, Any]] = None,
        weights: Optional[List[float]] = None,
        use_proba: bool = True,
        random_state: Optional[int] = None
    ):
        """
        Initialize ensemble pipeline.
        
        Args:
            task_type: Type of ML task ('classification' or 'regression')
            base_models: List of base models for the ensemble
            meta_model: Model for meta-learning in stacking
            ensemble_method: Method to combine predictions
                           ('stacking', 'weighted', 'voting')
            optimization_metric: Metric to optimize during training
            cv_params: Cross-validation parameters
            weights: Model weights for weighted averaging
            use_proba: Use probability predictions for classification
            random_state: Random seed for reproducibility
        """
        super().__init__()
        
        self.task_type = task_type
        self.base_models = base_models
        self.meta_model = meta_model
        self.ensemble_method = ensemble_method
        self.optimization_metric = optimization_metric
        self.cv_params = cv_params or {'n_splits': 5, 'shuffle': True}
        self.weights = self._validate_weights(weights)
        self.use_proba = use_proba and task_type == 'classification'
        self.random_state = random_state
        
        # Validate parameters
        self.cv_params = validate_cv_params(self.cv_params)
        validate_metrics(optimization_metric, task_type)
        
        # Initialize storage for model predictions
        self.base_predictions_ = None
        self.meta_features_ = None
        
        self.info(f"Initialized ensemble pipeline with {ensemble_method} method")
    
    def _validate_weights(
        self,
        weights: Optional[List[float]]
    ) -> Optional[np.ndarray]:
        """Validate and normalize model weights."""
        if weights is None:
            return None
        
        if len(weights) != len(self.base_models):
            raise ValueError(
                f"Number of weights ({len(weights)}) must match "
                f"number of models ({len(self.base_models)})"
            )
        
        weights = np.array(weights)
        if np.any(weights < 0):
            raise ValueError("Weights must be non-negative")
        
        return weights / weights.sum()
    
    def _get_cv_splitter(self) -> Union[KFold, StratifiedKFold]:
        """Get appropriate cross-validation splitter."""
        cv_class = (
            StratifiedKFold if self.task_type == 'classification'
            else KFold
        )
        return cv_class(
            random_state=self.random_state,
            **self.cv_params
        )
    
    def _generate_meta_features(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> np.ndarray:
        """
        Generate meta-features through cross-validation predictions.
        
        Args:
            X: Training features
            y: Target variable
        
        Returns:
            Array of meta-features
        """
        n_samples = len(X)
        cv = self._get_cv_splitter()
        
        if self.use_proba:
            n_classes = len(np.unique(y))
            meta_features = np.zeros((n_samples, len(self.base_models) * n_classes))
        else:
            meta_features = np.zeros((n_samples, len(self.base_models)))
        
        # Generate out-of-fold predictions
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train = y[train_idx]
            
            for model_idx, model in enumerate(self.base_models):
                model.fit(X_train, y_train)
                if self.use_proba:
                    probs = model.predict_proba(X_val)
                    start_idx = model_idx * n_classes
                    end_idx = start_idx + n_classes
                    meta_features[val_idx, start_idx:end_idx] = probs
                else:
                    preds = model.predict(X_val)
                    meta_features[val_idx, model_idx] = preds
        
        return meta_features
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> 'EnsemblePipeline':
        """
        Fit the ensemble pipeline.
        
        Args:
            X: Training features
            y: Target variable
        
        Returns:
            self: Fitted ensemble pipeline
        """
        self.info("Starting ensemble pipeline fitting")
        
        # Validate and convert input data
        X, y = validate_dataset(X, y, self.task_type)
        
        if self.ensemble_method == 'stacking':
            # Generate meta-features
            self.meta_features_ = self._generate_meta_features(X, y)
            
            # Train base models on full dataset
            for model in self.base_models:
                model.fit(X, y)
            
            # Train meta-model
            if self.meta_model is None:
                # Use hyperparameter optimization to select meta-model
                optimizer = HyperoptOptimizer(
                    task_type=self.task_type,
                    optimization_metric=self.optimization_metric,
                    cv_params=self.cv_params,
                    random_state=self.random_state
                )
                self.meta_model, _ = optimizer.optimize(
                    self.meta_features_, y
                )
            
            self.meta_model.fit(self.meta_features_, y)
            
        else:  # weighted or voting
            # Train all base models
            for model in self.base_models:
                model.fit(X, y)
        
        self.info("Ensemble pipeline fitting completed")
        return self
    
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        return_base_predictions: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, List[np.ndarray]]]:
        """
        Generate ensemble predictions for new data.
        
        Args:
            X: Features to generate predictions for
            return_base_predictions: Whether to return individual
                                   model predictions
        
        Returns:
            Model predictions and optionally base model predictions
        """
        X, _ = validate_dataset(X, None, self.task_type)
        
        # Get predictions from base models
        base_predictions = []
        for model in self.base_models:
            if self.use_proba:
                preds = model.predict_proba(X)
            else:
                preds = model.predict(X)
            base_predictions.append(preds)
        
        # Combine predictions based on ensemble method
        if self.ensemble_method == 'stacking':
            # Create meta-features
            if self.use_proba:
                meta_features = np.column_stack(base_predictions)
            else:
                meta_features = np.column_stack(base_predictions)
            
            # Generate final predictions
            ensemble_predictions = self.meta_model.predict(meta_features)
            
        elif self.ensemble_method == 'weighted':
            # Weighted average of predictions
            weights = (
                self.weights if self.weights is not None
                else np.ones(len(self.base_models)) / len(self.base_models)
            )
            
            ensemble_predictions = np.zeros_like(base_predictions[0])
            for pred, weight in zip(base_predictions, weights):
                ensemble_predictions += weight * pred
            
            if not self.use_proba:
                if self.task_type == 'classification':
                    ensemble_predictions = np.round(ensemble_predictions)
                    
        else:  # voting
            if self.task_type == 'classification':
                # Majority voting
                predictions = np.array([
                    model.predict(X) for model in self.base_models
                ])
                ensemble_predictions = np.apply_along_axis(
                    lambda x: np.bincount(x).argmax(),
                    axis=0,
                    arr=predictions
                )
            else:
                # Mean prediction for regression
                predictions = np.array([
                    model.predict(X) for model in self.base_models
                ])
                ensemble_predictions = np.mean(predictions, axis=0)
        
        if return_base_predictions:
            return ensemble_predictions, base_predictions
        return ensemble_predictions
    
    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Evaluate ensemble performance on test data.
        
        Args:
            X: Test features
            y: True target values
            metrics: List of metrics to calculate
        
        Returns:
            Dictionary of evaluation metrics
        """
        # Validate metrics
        metrics = metrics or [self.optimization_metric]
        validate_metrics(metrics, self.task_type)
        
        # Generate predictions
        predictions = self.predict(X)
        if self.task_type == 'classification' and self.use_proba:
            proba_predictions = self.predict(X)
        else:
            proba_predictions = None
        
        # Calculate metrics
        evaluation_results = calculate_metrics(
            y_true=y,
            y_pred=predictions,
            y_proba=proba_predictions,
            metrics=metrics,
            task_type=self.task_type
        )
        
        self.info("Ensemble evaluation completed")
        return evaluation_results
    
    def get_feature_importance(
        self,
        feature_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get feature importance scores from base models.
        
        Args:
            feature_names: Names of features
        
        Returns:
            DataFrame with aggregated feature importance scores
        """
        if not all(hasattr(model, 'feature_importances_')
                  for model in self.base_models):
            raise AttributeError(
                "Not all base models support feature importance calculation"
            )
        
        feature_names = (
            feature_names if feature_names is not None
            else [f"feature_{i}" for i in range(
                len(self.base_models[0].feature_importances_)
            )]
        )
        
        # Calculate weighted average of feature importances
        importances = np.zeros_like(self.base_models[0].feature_importances_)
        weights = (
            self.weights if self.weights is not None
            else np.ones(len(self.base_models)) / len(self.base_models)
        )
        
        for model, weight in zip(self.base_models, weights):
            importances += weight * model.feature_importances_
        
        return pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)