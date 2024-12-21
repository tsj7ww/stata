from typing import Dict, List, Optional, Union, Callable
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    log_loss
)

class MetricsCalculator:
    """
    Class for calculating and aggregating various evaluation metrics.
    Supports both classification and regression metrics with customizable options.
    """
    
    @staticmethod
    def get_classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        average: str = 'binary',
        class_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Calculate classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional)
            average: Averaging strategy for multiclass ('binary', 'micro', 'macro', 'weighted')
            class_names: List of class names for labeling
            
        Returns:
            Dict containing various classification metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, average=average, zero_division=0)
        
        # Probability-based metrics if available
        if y_prob is not None:
            # Handle binary vs multiclass for probabilities
            if y_prob.ndim > 1 and y_prob.shape[1] > 2:
                metrics['log_loss'] = log_loss(y_true, y_prob)
                # For multiclass, use one-vs-rest ROC AUC
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob, average=average, multi_class='ovr')
                metrics['average_precision'] = average_precision_score(
                    y_true, y_prob, average=average
                )
            else:
                # For binary classification
                prob_positive = y_prob if y_prob.ndim == 1 else y_prob[:, 1]
                metrics['log_loss'] = log_loss(y_true, prob_positive)
                metrics['roc_auc'] = roc_auc_score(y_true, prob_positive)
                metrics['average_precision'] = average_precision_score(y_true, prob_positive)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        
        # Per-class metrics if class names are provided
        if class_names is not None:
            per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
            per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
            per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
            
            metrics['per_class'] = {
                class_name: {
                    'precision': prec,
                    'recall': rec,
                    'f1': f1,
                    'support': np.sum(y_true == i)
                }
                for i, (class_name, prec, rec, f1) in enumerate(zip(
                    class_names, per_class_precision, per_class_recall, per_class_f1
                ))
            }
        
        return metrics
    
    @staticmethod
    def get_regression_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weights: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate regression metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            sample_weights: Optional weights for samples
            
        Returns:
            Dict containing various regression metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['mse'] = mean_squared_error(y_true, y_pred, sample_weight=sample_weights)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred, sample_weight=sample_weights)
        metrics['r2'] = r2_score(y_true, y_pred, sample_weight=sample_weights)
        
        # Additional metrics
        metrics['explained_variance'] = float(np.var(y_true - y_pred) / np.var(y_true))
        metrics['median_absolute_error'] = float(np.median(np.abs(y_true - y_pred)))
        
        # Calculate MAPE if no zeros in y_true
        if not np.any(y_true == 0):
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            metrics['mape'] = float(mape)
        
        return metrics
    
    @staticmethod
    def get_custom_metric(
        metric_fn: Callable,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        **kwargs
    ) -> float:
        """
        Calculate a custom metric using provided function.
        
        Args:
            metric_fn: Custom metric function
            y_true: True values
            y_pred: Predicted values
            **kwargs: Additional arguments for metric function
            
        Returns:
            Custom metric value
        """
        return float(metric_fn(y_true, y_pred, **kwargs))
    
    @staticmethod
    def get_threshold_metrics(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        thresholds: Optional[List[float]] = None
    ) -> Dict[str, Dict[float, Dict[str, float]]]:
        """
        Calculate metrics at different probability thresholds.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            thresholds: List of thresholds to evaluate
            
        Returns:
            Dict containing metrics at each threshold
        """
        if thresholds is None:
            thresholds = np.linspace(0, 1, 11)  # Default to 0.0 to 1.0 in 0.1 steps
            
        # Get probability of positive class
        if y_prob.ndim > 1:
            y_prob = y_prob[:, 1]
            
        threshold_metrics = {}
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            
            threshold_metrics[threshold] = {
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1': f1_score(y_true, y_pred, zero_division=0),
                'accuracy': accuracy_score(y_true, y_pred)
            }
            
        return {'threshold_metrics': threshold_metrics}
    
    @staticmethod
    def get_prediction_intervals(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        confidence: float = 0.95
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        Calculate prediction intervals for regression problems.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            confidence: Confidence level for intervals
            
        Returns:
            Dict containing prediction interval metrics
        """
        # Calculate residuals
        residuals = y_true - y_pred
        
        # Calculate standard error
        std_error = np.std(residuals)
        
        # Calculate z-score for given confidence level
        z_score = float(abs(np.percentile(residuals, (1 - confidence) * 100)))
        
        # Calculate intervals
        lower_bound = y_pred - z_score * std_error
        upper_bound = y_pred + z_score * std_error
        
        # Calculate coverage probability
        in_interval = (y_true >= lower_bound) & (y_true <= upper_bound)
        coverage_prob = float(np.mean(in_interval))
        
        return {
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'confidence_level': confidence,
            'coverage_probability': coverage_prob,
            'interval_width': float(np.mean(upper_bound - lower_bound))
        }