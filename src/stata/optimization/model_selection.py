"""
Model selection module for auto_ml package.

This module implements model selection strategies including performance-based
ranking, multi-metric optimization, and ensemble selection.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from dataclasses import dataclass
import time
from pathlib import Path

from ..utils.logging import LoggerMixin
from ..models.base import BaseModel
from ..evaluation.metrics import calculate_metrics
from ..utils.validation import validate_metrics

@dataclass
class ModelSelectionResult:
    """Container for model selection results."""
    model_name: str
    model_params: Dict[str, Any]
    cv_scores: Dict[str, List[float]]
    mean_scores: Dict[str, float]
    std_scores: Dict[str, float]
    fit_times: List[float]
    score_times: List[float]
    model_size: int  # Memory footprint in bytes

class ModelSelector(LoggerMixin):
    """
    Model selection and ranking based on multiple criteria including
    performance metrics, training time, and model complexity.
    """
    
    def __init__(
        self,
        task_type: str,
        metrics: Union[str, List[str]],
        cv_params: Optional[Dict[str, Any]] = None,
        selection_criteria: Optional[Dict[str, float]] = None,
        n_jobs: int = -1,
        random_state: Optional[int] = None
    ):
        """
        Initialize model selector.
        
        Args:
            task_type: Type of ML task ('classification' or 'regression')
            metrics: Metrics to evaluate models
            cv_params: Cross-validation parameters
            selection_criteria: Weights for different selection criteria
                (e.g., {'performance': 0.7, 'speed': 0.2, 'complexity': 0.1})
            n_jobs: Number of parallel jobs
            random_state: Random seed
        """
        super().__init__()
        
        self.task_type = task_type
        self.metrics = [metrics] if isinstance(metrics, str) else metrics
        self.cv_params = cv_params or {'n_splits': 5}
        self.selection_criteria = selection_criteria or {'performance': 1.0}
        self.n_jobs = n_jobs
        self.random_state = random_state
        
        # Validate metrics
        validate_metrics(self.metrics, task_type)
        
        # Initialize storage for results
        self.results: Dict[str, ModelSelectionResult] = {}
        
        self.info("Initialized model selector")
    
    def evaluate_model(
        self,
        model: BaseModel,
        X: np.ndarray,
        y: np.ndarray,
        model_name: Optional[str] = None
    ) -> ModelSelectionResult:
        """
        Evaluate a single model using cross-validation.
        
        Args:
            model: Model to evaluate
            X: Training features
            y: Target variable
            model_name: Optional name for the model
        
        Returns:
            ModelSelectionResult with evaluation metrics
        """
        model_name = model_name or model.__class__.__name__
        self.info(f"Evaluating model: {model_name}")
        
        # Perform cross-validation with multiple metrics
        cv_results = cross_validate(
            model,
            X,
            y,
            scoring=self.metrics,
            cv=self.cv_params['n_splits'],
            n_jobs=self.n_jobs,
            return_train_score=True,
            error_score='raise'
        )
        
        # Extract and organize scores
        cv_scores = {}
        mean_scores = {}
        std_scores = {}
        
        for metric in self.metrics:
            test_scores = cv_results[f'test_{metric}']
            cv_scores[metric] = test_scores.tolist()
            mean_scores[metric] = np.mean(test_scores)
            std_scores[metric] = np.std(test_scores)
        
        # Get model parameters and size
        model_params = model.get_params()
        model_size = self._estimate_model_size(model)
        
        result = ModelSelectionResult(
            model_name=model_name,
            model_params=model_params,
            cv_scores=cv_scores,
            mean_scores=mean_scores,
            std_scores=std_scores,
            fit_times=cv_results['fit_time'].tolist(),
            score_times=cv_results['score_time'].tolist(),
            model_size=model_size
        )
        
        self.results[model_name] = result
        return result
    
    def evaluate_models(
        self,
        models: List[Tuple[str, BaseModel]],
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, ModelSelectionResult]:
        """
        Evaluate multiple models using cross-validation.
        
        Args:
            models: List of (name, model) tuples
            X: Training features
            y: Target variable
        
        Returns:
            Dictionary of evaluation results
        """
        for model_name, model in models:
            self.evaluate_model(model, X, y, model_name)
        
        return self.results
    
    def rank_models(
        self,
        primary_metric: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Rank models based on selection criteria.
        
        Args:
            primary_metric: Primary metric for ranking
                (default: first metric in self.metrics)
        
        Returns:
            DataFrame with model rankings
        """
        if not self.results:
            raise ValueError("No models have been evaluated")
        
        primary_metric = primary_metric or self.metrics[0]
        rankings = []
        
        for model_name, result in self.results.items():
            # Calculate composite score based on selection criteria
            scores = {
                'model_name': model_name,
                'primary_score': result.mean_scores[primary_metric],
                'primary_std': result.std_scores[primary_metric]
            }
            
            # Add all metrics
            for metric in self.metrics:
                scores[f'{metric}_mean'] = result.mean_scores[metric]
                scores[f'{metric}_std'] = result.std_scores[metric]
            
            # Add other criteria
            scores['mean_fit_time'] = np.mean(result.fit_times)
            scores['mean_score_time'] = np.mean(result.score_times)
            scores['model_size'] = result.model_size
            
            # Calculate weighted score
            composite_score = self._calculate_composite_score(
                result, primary_metric
            )
            scores['composite_score'] = composite_score
            
            rankings.append(scores)
        
        # Create DataFrame and sort by composite score
        rankings_df = pd.DataFrame(rankings)
        return rankings_df.sort_values(
            'composite_score', ascending=False
        ).reset_index(drop=True)
    
    def _calculate_composite_score(
        self,
        result: ModelSelectionResult,
        primary_metric: str
    ) -> float:
        """Calculate weighted composite score based on selection criteria."""
        scores = {}
        
        # Performance score (normalized by all models)
        max_score = max(
            r.mean_scores[primary_metric]
            for r in self.results.values()
        )
        min_score = min(
            r.mean_scores[primary_metric]
            for r in self.results.values()
        )
        score_range = max_score - min_score
        
        if score_range > 0:
            scores['performance'] = (
                result.mean_scores[primary_metric] - min_score
            ) / score_range
        else:
            scores['performance'] = 1.0
        
        # Speed score (inverse of time, normalized)
        if 'speed' in self.selection_criteria:
            max_time = max(
                np.mean(r.fit_times) for r in self.results.values()
            )
            time_score = 1 - (np.mean(result.fit_times) / max_time)
            scores['speed'] = time_score
        
        # Complexity score (inverse of model size, normalized)
        if 'complexity' in self.selection_criteria:
            max_size = max(
                r.model_size for r in self.results.values()
            )
            size_score = 1 - (result.model_size / max_size)
            scores['complexity'] = size_score
        
        # Calculate weighted sum
        composite_score = sum(
            scores[criterion] * weight
            for criterion, weight in self.selection_criteria.items()
            if criterion in scores
        )
        
        return composite_score
    
    def _estimate_model_size(self, model: BaseModel) -> int:
        """Estimate model size in bytes."""
        import sys
        import pickle
        
        # Serialize model and get size
        return sys.getsizeof(pickle.dumps(model))
    
    def select_best_models(
        self,
        n_models: int = 1,
        diversity_threshold: Optional[float] = None
    ) -> List[Tuple[str, BaseModel]]:
        """
        Select top N models considering performance and diversity.
        
        Args:
            n_models: Number of models to select
            diversity_threshold: Minimum prediction diversity required
        
        Returns:
            List of (name, model) tuples
        """
        if n_models > len(self.results):
            n_models = len(self.results)
        
        rankings = self.rank_models()
        selected_models = []
        
        for _, row in rankings.iterrows():
            model_name = row['model_name']
            if len(selected_models) >= n_models:
                break
                
            # Check prediction diversity if threshold is set
            if diversity_threshold and selected_models:
                # This would require predictions to be stored
                # Implementation depends on specific diversity metric
                pass
            
            selected_models.append(
                (model_name, self.results[model_name])
            )
        
        return selected_models
    
    def get_selection_summary(self) -> pd.DataFrame:
        """
        Get detailed summary of model selection results.
        
        Returns:
            DataFrame with comprehensive selection metrics
        """
        if not self.results:
            raise ValueError("No models have been evaluated")
        
        summary_data = []
        
        for model_name, result in self.results.items():
            summary = {
                'model_name': model_name,
                'params': str(result.model_params)
            }
            
            # Add all metrics
            for metric in self.metrics:
                summary[f'{metric}_mean'] = result.mean_scores[metric]
                summary[f'{metric}_std'] = result.std_scores[metric]
            
            # Add timing and resource metrics
            summary['mean_fit_time'] = np.mean(result.fit_times)
            summary['std_fit_time'] = np.std(result.fit_times)
            summary['mean_score_time'] = np.mean(result.score_times)
            summary['std_score_time'] = np.std(result.score_times)
            summary['model_size_bytes'] = result.model_size
            
            summary_data.append(summary)
        
        return pd.DataFrame(summary_data)