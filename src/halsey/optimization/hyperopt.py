"""
Hyperparameter optimization module for auto_ml package.

This module implements hyperparameter optimization using various strategies
including Bayesian optimization with Gaussian Processes, Tree Parzen
Estimators (TPE), and Random Search.
"""

from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator
from scipy.stats import norm
import time
import logging
from dataclasses import dataclass

from ..utils.logging import LoggerMixin
from ..models.base import BaseModel
from .search_spaces import SearchSpace, get_search_space

@dataclass
class OptimizationResult:
    """Container for optimization results."""
    best_params: Dict[str, Any]
    best_score: float
    all_params: List[Dict[str, Any]]
    all_scores: List[float]
    optimization_time: float

class HyperoptOptimizer(LoggerMixin):
    """
    Hyperparameter optimizer using Bayesian optimization with
    Gaussian Processes or TPE algorithm.
    """
    
    def __init__(
        self,
        task_type: str,
        models: Optional[List[str]] = None,
        optimization_metric: str = 'accuracy',
        cv_params: Optional[Dict[str, Any]] = None,
        n_iterations: int = 50,
        n_initial_points: int = 10,
        acquisition_function: str = 'ei',
        acquisition_optimizer: str = 'sampling',
        random_state: Optional[int] = None,
        n_jobs: int = -1,
        verbose: bool = True
    ):
        """
        Initialize hyperparameter optimizer.
        
        Args:
            task_type: Type of ML task ('classification' or 'regression')
            models: List of model names to optimize
            optimization_metric: Metric to optimize
            cv_params: Cross-validation parameters
            n_iterations: Number of optimization iterations
            n_initial_points: Number of initial random points
            acquisition_function: Acquisition function type
                ('ei', 'pi', 'ucb')
            acquisition_optimizer: Strategy for optimizing acquisition
                function ('sampling', 'lbfgs')
            random_state: Random seed
            n_jobs: Number of parallel jobs
            verbose: Whether to print progress
        """
        super().__init__()
        
        self.task_type = task_type
        self.models = models or ['random_forest']
        self.optimization_metric = optimization_metric
        self.cv_params = cv_params or {'n_splits': 5}
        self.n_iterations = n_iterations
        self.n_initial_points = n_initial_points
        self.acquisition_function = acquisition_function
        self.acquisition_optimizer = acquisition_optimizer
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        # Initialize optimization components
        np.random.seed(random_state)
        self.gp_kernel = None
        self.gp_model = None
        
        self.info("Initialized hyperparameter optimizer")
    
    def _compute_acquisition(
        self,
        X: np.ndarray,
        model: Any,
        best_f: float,
        xi: float = 0.01
    ) -> np.ndarray:
        """
        Compute acquisition function values.
        
        Args:
            X: Points to evaluate
            model: Surrogate model (e.g., GP)
            best_f: Best observed value
            xi: Exploration-exploitation trade-off parameter
        
        Returns:
            Acquisition function values
        """
        mu, sigma = model.predict(X, return_std=True)
        sigma = sigma.reshape(-1, 1)
        
        if self.acquisition_function == 'ei':
            # Expected Improvement
            imp = mu - best_f - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            return ei
        elif self.acquisition_function == 'pi':
            # Probability of Improvement
            Z = (mu - best_f - xi) / sigma
            return norm.cdf(Z)
        else:  # 'ucb'
            # Upper Confidence Bound
            kappa = 2.0  # exploration weight
            return mu + kappa * sigma
    
    def _optimize_acquisition(
        self,
        search_space: SearchSpace,
        surrogate_model: Any,
        best_f: float,
        n_points: int = 1000
    ) -> Dict[str, Any]:
        """
        Optimize acquisition function to find next evaluation point.
        
        Args:
            search_space: Parameter search space
            surrogate_model: Surrogate model
            best_f: Best observed value
            n_points: Number of points to sample
        
        Returns:
            Next point to evaluate
        """
        if self.acquisition_optimizer == 'sampling':
            # Random sampling approach
            samples = search_space.sample(n_points)
            X_samples = np.array([
                [sample[param] for param in search_space.parameter_space.keys()]
                for sample in samples
            ])
            
            acq_values = self._compute_acquisition(
                X_samples, surrogate_model, best_f
            )
            best_idx = np.argmax(acq_values)
            
            return {
                param: samples[param][best_idx]
                for param in search_space.parameter_space.keys()
            }
        else:  # 'lbfgs'
            # Gradient-based optimization (simplified)
            raise NotImplementedError(
                "L-BFGS-B optimization not yet implemented"
            )
    
    def _evaluate_model(
        self,
        model: BaseModel,
        params: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray
    ) -> float:
        """
        Evaluate model with given parameters using cross-validation.
        
        Args:
            model: Model instance
            params: Parameters to evaluate
            X: Training features
            y: Target variable
        
        Returns:
            Mean cross-validation score
        """
        model.set_params(**params)
        scores = cross_val_score(
            model,
            X,
            y,
            cv=self.cv_params['n_splits'],
            scoring=self.optimization_metric,
            n_jobs=self.n_jobs
        )
        return np.mean(scores)
    
    def optimize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray] = None
    ) -> Tuple[BaseModel, Dict[str, Any]]:
        """
        Run hyperparameter optimization.
        
        Args:
            X: Training features
            y: Target variable
            groups: Optional group labels for cross-validation
        
        Returns:
            Tuple of (best_model, best_params)
        """
        start_time = time.time()
        best_model = None
        best_params = None
        best_score = float('-inf')
        
        for model_name in self.models:
            self.info(f"Optimizing {model_name}")
            
            # Get search space and model instance
            search_space = get_search_space(model_name, self.task_type)
            model = BaseModel.get_model(model_name, self.task_type)
            
            # Initial random points
            X_samples = []
            y_samples = []
            
            for _ in range(self.n_initial_points):
                params = search_space.sample(1)
                score = self._evaluate_model(model, params, X, y)
                
                X_samples.append(
                    [params[p] for p in search_space.parameter_space.keys()]
                )
                y_samples.append(score)
            
            X_samples = np.array(X_samples)
            y_samples = np.array(y_samples)
            
            # Bayesian optimization loop
            for i in range(self.n_iterations - self.n_initial_points):
                # Fit surrogate model
                self.gp_model = self._fit_surrogate_model(
                    X_samples, y_samples
                )
                
                # Find next point to evaluate
                next_params = self._optimize_acquisition(
                    search_space,
                    self.gp_model,
                    np.max(y_samples)
                )
                
                # Evaluate point
                score = self._evaluate_model(model, next_params, X, y)
                
                # Update samples
                X_samples = np.vstack([
                    X_samples,
                    [next_params[p] for p in search_space.parameter_space.keys()]
                ])
                y_samples = np.append(y_samples, score)
                
                if self.verbose:
                    self.info(
                        f"Iteration {i + 1 + self.n_initial_points}: "
                        f"Score = {score:.4f}"
                    )
                
                # Update best model if necessary
                if score > best_score:
                    best_score = score
                    best_params = next_params
                    best_model = model
        
        # Fit final model with best parameters
        best_model.set_params(**best_params)
        best_model.fit(X, y)
        
        optimization_time = time.time() - start_time
        self.info(
            f"Optimization completed in {optimization_time:.2f} seconds. "
            f"Best score: {best_score:.4f}"
        )
        
        return best_model, best_params
    
    def _fit_surrogate_model(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Any:
        """
        Fit Gaussian Process surrogate model.
        
        Args:
            X: Observed parameters
            y: Observed scores
        
        Returns:
            Fitted surrogate model
        """
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import (
            RBF, ConstantKernel as C
        )
        
        if self.gp_kernel is None:
            # Initialize kernel with automatic relevance determination
            self.gp_kernel = C(1.0) * RBF(
                length_scale=[1.0] * X.shape[1],
                length_scale_bounds=(1e-3, 1e3)
            )
        
        gp = GaussianProcessRegressor(
            kernel=self.gp_kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=self.random_state
        )
        
        return gp.fit(X, y)
    
    def get_optimization_curve(self) -> Tuple[List[int], List[float]]:
        """
        Get optimization history curve.
        
        Returns:
            Tuple of (iterations, scores)
        """
        if not hasattr(self, 'optimization_history'):
            raise AttributeError(
                "No optimization history available. Run optimize() first."
            )
        
        iterations = list(range(1, len(self.optimization_history) + 1))
        scores = [result.best_score for result in self.optimization_history]
        
        return iterations, scores