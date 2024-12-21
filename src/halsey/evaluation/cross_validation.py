from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    GroupKFold,
    train_test_split
)
from sklearn.base import clone

from ..models.base import BaseModel
from .metrics import MetricsCalculator

class CrossValidator:
    """
    Comprehensive cross-validation implementation supporting various splitting strategies
    and evaluation schemes.
    """
    
    def __init__(self,
                 cv_type: str = 'kfold',
                 n_splits: int = 5,
                 shuffle: bool = True,
                 random_state: Optional[int] = None,
                 stratify: bool = False,
                 group_column: Optional[str] = None):
        """
        Initialize cross-validator.
        
        Args:
            cv_type: Type of CV ('kfold', 'stratified', 'timeseries', 'group')
            n_splits: Number of splits
            shuffle: Whether to shuffle data
            random_state: Random seed
            stratify: Whether to preserve label distribution
            group_column: Column name for group-based splitting
        """
        self.cv_type = cv_type
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.stratify = stratify
        self.group_column = group_column
        
        # Initialize metrics calculator
        self.metrics_calculator = MetricsCalculator()
        
    def _get_splitter(self) -> Any:
        """Get the appropriate cross-validation splitter."""
        if self.cv_type == 'kfold':
            return KFold(n_splits=self.n_splits,
                        shuffle=self.shuffle,
                        random_state=self.random_state)
        elif self.cv_type == 'stratified':
            return StratifiedKFold(n_splits=self.n_splits,
                                 shuffle=self.shuffle,
                                 random_state=self.random_state)
        elif self.cv_type == 'timeseries':
            return TimeSeriesSplit(n_splits=self.n_splits)
        elif self.cv_type == 'group':
            return GroupKFold(n_splits=self.n_splits)
        else:
            raise ValueError(f"Unknown cv_type: {self.cv_type}")
    
    def validate(self,
                model: BaseModel,
                X: Union[np.ndarray, pd.DataFrame],
                y: Union[np.ndarray, pd.Series],
                groups: Optional[Union[np.ndarray, pd.Series]] = None,
                task_type: str = 'classification',
                scoring: Optional[Union[str, List[str], Dict[str, Callable]]] = None,
                return_predictions: bool = False) -> Dict[str, Any]:
        """
        Perform cross-validation on the given model.
        
        Args:
            model: Model to validate
            X: Features
            y: Target values
            groups: Groups for group-based CV
            task_type: Type of task ('classification' or 'regression')
            scoring: Metrics to compute
            return_predictions: Whether to return OOF predictions
            
        Returns:
            Dict containing cross-validation results
        """
        # Convert inputs to numpy arrays
        X = np.array(X)
        y = np.array(y)
        if groups is not None:
            groups = np.array(groups)
            
        # Initialize splitter
        splitter = self._get_splitter()
        
        # Initialize results storage
        fold_scores = []
        all_metrics = []
        oof_predictions = np.zeros_like(y, dtype=float)
        oof_probabilities = (np.zeros((len(y), len(np.unique(y))))
                           if task_type == 'classification' else None)
        
        # Perform cross-validation
        for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X, y, groups)):
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Clone and fit model
            fold_model = clone(model)
            fold_model.fit(X_train, y_train)
            
            # Make predictions
            val_pred = fold_model.predict(X_val)
            oof_predictions[val_idx] = val_pred
            
            # Get probabilities for classification
            val_prob = None
            if task_type == 'classification' and hasattr(fold_model, 'predict_proba'):
                val_prob = fold_model.predict_proba(X_val)
                oof_probabilities[val_idx] = val_prob
            
            # Calculate metrics
            if task_type == 'classification':
                fold_metrics = self.metrics_calculator.get_classification_metrics(
                    y_val, val_pred, val_prob
                )
            else:
                fold_metrics = self.metrics_calculator.get_regression_metrics(
                    y_val, val_pred
                )
            
            # Add custom metrics if provided
            if isinstance(scoring, dict):
                for metric_name, metric_fn in scoring.items():
                    fold_metrics[metric_name] = self.metrics_calculator.get_custom_metric(
                        metric_fn, y_val, val_pred
                    )
            
            all_metrics.append(fold_metrics)
            
            # Store basic score for quick reference
            primary_metric = ('accuracy' if task_type == 'classification' else 'r2')
            fold_scores.append(fold_metrics[primary_metric])
        
        # Calculate aggregate metrics
        cv_results = {
            'fold_scores': fold_scores,
            'mean_score': float(np.mean(fold_scores)),
            'std_score': float(np.std(fold_scores)),
            'all_metrics': all_metrics,
            'aggregate_metrics': self._aggregate_metrics(all_metrics)
        }
        
        if return_predictions:
            cv_results['oof_predictions'] = oof_predictions
            if oof_probabilities is not None:
                cv_results['oof_probabilities'] = oof_probabilities
        
        return cv_results
    
    def _aggregate_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Aggregate metrics across folds.
        
        Args:
            metrics_list: List of metric dictionaries from each fold
            
        Returns:
            Dict containing aggregated metrics
        """
        aggregated = {}
        
        # Get all metric names (excluding confusion matrix and per-class metrics)
        metric_names = {k for metrics in metrics_list 
                       for k, v in metrics.items() 
                       if isinstance(v, (int, float))}
        
        for metric in metric_names:
            values = [m[metric] for m in metrics_list if metric in m]
            aggregated[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
        
        return aggregated
    
    @staticmethod
    def create_holdout(X: Union[np.ndarray, pd.DataFrame],
                      y: Union[np.ndarray, pd.Series],
                      test_size: float = 0.2,
                      stratify: bool = False,
                      random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create a holdout set for final evaluation.
        
        Args:
            X: Features
            y: Target values
            test_size: Fraction of data to use for holdout
            stratify: Whether to preserve label distribution
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        stratify_arg = y if stratify else None
        return train_test_split(X, y,
                              test_size=test_size,
                              stratify=stratify_arg,
                              random_state=random_state)
    
    def validate_with_holdout(self,
                            model: BaseModel,
                            X: Union[np.ndarray, pd.DataFrame],
                            y: Union[np.ndarray, pd.Series],
                            holdout_size: float = 0.2,
                            task_type: str = 'classification',
                            scoring: Optional[Union[str, List[str], Dict[str, Callable]]] = None) -> Dict[str, Any]:
        """
        Perform cross-validation with a final holdout set.
        
        Args:
            model: Model to validate
            X: Features
            y: Target values
            holdout_size: Size of holdout set
            task_type: Type of task
            scoring: Metrics to compute
            
        Returns:
            Dict containing CV and holdout results
        """
        # Create holdout set
        X_train, X_holdout, y_train, y_holdout = self.create_holdout(
            X, y,
            test_size=holdout_size,
            stratify=self.stratify,
            random_state=self.random_state
        )
        
        # Perform cross-validation on training data
        cv_results = self.validate(
            model, X_train, y_train,
            task_type=task_type,
            scoring=scoring
        )
        
        # Evaluate on holdout set
        model.fit(X_train, y_train)
        holdout_pred = model.predict(X_holdout)
        
        holdout_prob = None
        if task_type == 'classification' and hasattr(model, 'predict_proba'):
            holdout_prob = model.predict_proba(X_holdout)
        
        # Calculate holdout metrics
        if task_type == 'classification':
            holdout_metrics = self.metrics_calculator.get_classification_metrics(
                y_holdout, holdout_pred, holdout_prob
            )
        else:
            holdout_metrics = self.metrics_calculator.get_regression_metrics(
                y_holdout, holdout_pred
            )
        
        # Add custom metrics for holdout if provided
        if isinstance(scoring, dict):
            for metric_name, metric_fn in scoring.items():
                holdout_metrics[metric_name] = self.metrics_calculator.get_custom_metric(
                    metric_fn, y_holdout, holdout_pred
                )
        
        # Combine results
        return {
            'cv_results': cv_results,
            'holdout_metrics': holdout_metrics
        }