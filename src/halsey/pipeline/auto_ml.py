"""
Main AutoML pipeline module that orchestrates the entire machine learning workflow.

This module provides the core AutoML class that handles data preprocessing,
model selection, hyperparameter optimization, and evaluation.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import time
from datetime import datetime

from ..preprocessing.base import BasePreprocessor
from ..models.base import BaseModel
from ..optimization.hyperopt import HyperoptOptimizer
from ..evaluation.metrics import calculate_metrics
from ..utils.logging import LoggerMixin
from ..utils.validation import (
    validate_dataset,
    validate_metrics,
    validate_cv_params
)
from ..utils.io import save_model, save_results

class AutoML(LoggerMixin):
    """
    Main class for automated machine learning pipeline.
    
    This class coordinates the entire ML workflow, including:
    - Data preprocessing and validation
    - Feature engineering
    - Model selection
    - Hyperparameter optimization
    - Model evaluation
    - Results logging and model persistence
    """
    
    def __init__(
        self,
        task_type: str,
        optimization_metric: str,
        models: Optional[List[str]] = None,
        preprocessors: Optional[List[str]] = None,
        cv_params: Optional[Dict[str, Any]] = None,
        optimization_params: Optional[Dict[str, Any]] = None,
        random_state: Optional[int] = None,
        output_dir: Optional[Union[str, Path]] = None
    ):
        """
        Initialize AutoML pipeline.
        
        Args:
            task_type: Type of ML task ('classification' or 'regression')
            optimization_metric: Metric to optimize during model selection
            models: List of model names to consider
            preprocessors: List of preprocessor names to apply
            cv_params: Cross-validation parameters
            optimization_params: Hyperparameter optimization parameters
            random_state: Random seed for reproducibility
            output_dir: Directory for saving models and results
        """
        super().__init__()
        
        self.task_type = task_type
        self.optimization_metric = optimization_metric
        self.models = models or self._get_default_models()
        self.preprocessors = preprocessors or self._get_default_preprocessors()
        self.cv_params = cv_params or {'n_splits': 5, 'shuffle': True}
        self.optimization_params = optimization_params or {}
        self.random_state = random_state
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        
        # Validate parameters
        self.cv_params = validate_cv_params(self.cv_params)
        validate_metrics(self.optimization_metric, self.task_type)
        
        # Initialize components
        self.preprocessor = None
        self.best_model = None
        self.best_params = None
        self.results = {}
        
        self.info(f"Initialized AutoML pipeline for {task_type} task")
    
    def _get_default_models(self) -> List[str]:
        """Get default list of models based on task type."""
        if self.task_type == 'classification':
            return ['random_forest', 'gradient_boosting', 'extra_trees']
        else:
            return ['random_forest', 'gradient_boosting', 'elastic_net']
    
    def _get_default_preprocessors(self) -> List[str]:
        """Get default list of preprocessors."""
        return ['standard_scaler', 'categorical_encoder']
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        categorical_features: Optional[List[str]] = None,
        groups: Optional[np.ndarray] = None
    ) -> 'AutoML':
        """
        Fit the AutoML pipeline to training data.
        
        Args:
            X: Training features
            y: Target variable
            categorical_features: List of categorical feature names
            groups: Optional group labels for cross-validation
        
        Returns:
            self: Fitted AutoML instance
        """
        start_time = time.time()
        self.info("Starting AutoML pipeline fitting")
        
        # Validate input data
        X, y = validate_dataset(X, y, self.task_type)
        
        # Initialize preprocessor pipeline
        self.preprocessor = BasePreprocessor(
            preprocessors=self.preprocessors,
            categorical_features=categorical_features
        )
        
        # Preprocess training data
        X_processed = self.preprocessor.fit_transform(X)
        self.info("Completed data preprocessing")
        
        # Initialize optimizer
        optimizer = HyperoptOptimizer(
            task_type=self.task_type,
            models=self.models,
            optimization_metric=self.optimization_metric,
            cv_params=self.cv_params,
            random_state=self.random_state,
            **self.optimization_params
        )
        
        # Perform hyperparameter optimization
        self.best_model, self.best_params = optimizer.optimize(
            X_processed, y, groups=groups
        )
        
        # Fit best model on entire dataset
        self.best_model.fit(X_processed, y)
        
        # Calculate and store results
        duration = time.time() - start_time
        self.results = {
            'task_type': self.task_type,
            'best_model': self.best_model.__class__.__name__,
            'best_params': self.best_params,
            'optimization_metric': self.optimization_metric,
            'training_duration': duration,
            'timestamp': datetime.now().isoformat()
        }
        
        self.info(f"AutoML pipeline fitting completed in {duration:.2f} seconds")
        return self
    
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        return_proba: bool = False
    ) -> np.ndarray:
        """
        Generate predictions for new data.
        
        Args:
            X: Features to generate predictions for
            return_proba: Whether to return probability estimates
                        (classification only)
        
        Returns:
            Model predictions
        """
        if self.best_model is None:
            raise RuntimeError("AutoML pipeline has not been fitted")
        
        # Preprocess input data
        X_processed = self.preprocessor.transform(X)
        
        # Generate predictions
        if return_proba and self.task_type == 'classification':
            predictions = self.best_model.predict_proba(X_processed)
        else:
            predictions = self.best_model.predict(X_processed)
        
        return predictions
    
    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            X: Test features
            y: True target values
            metrics: List of metrics to calculate
        
        Returns:
            Dictionary of evaluation metrics
        """
        if self.best_model is None:
            raise RuntimeError("AutoML pipeline has not been fitted")
        
        # Validate metrics
        metrics = metrics or [self.optimization_metric]
        validate_metrics(metrics, self.task_type)
        
        # Generate predictions
        predictions = self.predict(X)
        if self.task_type == 'classification':
            proba_predictions = self.predict(X, return_proba=True)
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
        
        self.info("Model evaluation completed")
        return evaluation_results
    
    def save(
        self,
        path: Optional[Union[str, Path]] = None,
        save_preprocessor: bool = True
    ) -> None:
        """
        Save the fitted AutoML pipeline.
        
        Args:
            path: Path to save the pipeline (directory)
            save_preprocessor: Whether to save preprocessor separately
        """
        if self.best_model is None:
            raise RuntimeError("AutoML pipeline has not been fitted")
        
        # Create save directory
        save_dir = Path(path) if path else self.output_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = save_dir / 'best_model.pkl'
        save_model(self.best_model, model_path)
        
        # Save preprocessor
        if save_preprocessor:
            preprocessor_path = save_dir / 'preprocessor.pkl'
            save_model(self.preprocessor, preprocessor_path)
        
        # Save results and metadata
        save_results(self.results, save_dir / 'results.json')
        
        self.info(f"AutoML pipeline saved to {save_dir}")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores if supported by the model.
        
        Returns:
            DataFrame with feature importance scores
        """
        if self.best_model is None:
            raise RuntimeError("AutoML pipeline has not been fitted")
        
        if not hasattr(self.best_model, 'feature_importances_'):
            raise AttributeError(
                "Best model does not support feature importance calculation"
            )
        
        feature_names = self.preprocessor.get_feature_names()
        importance_scores = self.best_model.feature_importances_
        
        return pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)