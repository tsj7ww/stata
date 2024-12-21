from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
import shap
import warnings

class ModelAnalyzer:
    """
    Comprehensive model analysis toolkit providing various interpretability
    and analysis methods for machine learning models.
    """
    
    def __init__(self, random_state: Optional[int] = None):
        """
        Initialize model analyzer.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        
    def feature_importance(self,
                         model: Any,
                         X: Union[np.ndarray, pd.DataFrame],
                         y: Union[np.ndarray, pd.Series],
                         method: str = 'auto',
                         n_repeats: int = 10) -> pd.DataFrame:
        """
        Calculate feature importance using various methods.
        
        Args:
            model: Trained model
            X: Feature data
            y: Target values
            method: Method to use ('auto', 'permutation', 'shap', 'built_in')
            n_repeats: Number of repeats for permutation importance
            
        Returns:
            DataFrame containing feature importance scores
        """
        feature_names = (X.columns if isinstance(X, pd.DataFrame)
                        else [f'feature_{i}' for i in range(X.shape[1])])
        
        if method == 'auto':
            # Try built-in feature importance first
            if hasattr(model, 'feature_importances_'):
                method = 'built_in'
            else:
                method = 'permutation'
        
        importance_scores: Optional[np.ndarray] = None
        
        if method == 'built_in':
            if not hasattr(model, 'feature_importances_'):
                raise ValueError("Model doesn't have built-in feature importance")
            importance_scores = model.feature_importances_
            
        elif method == 'permutation':
            result = permutation_importance(
                model, X, y,
                n_repeats=n_repeats,
                random_state=self.random_state
            )
            importance_scores = result.importances_mean
            
        elif method == 'shap':
            # Handle different model types for SHAP
            try:
                explainer = shap.Explainer(model, X)
                shap_values = explainer(X)
                importance_scores = np.mean(np.abs(shap_values.values), axis=0)
            except Exception as e:
                warnings.warn(f"SHAP analysis failed: {str(e)}. Falling back to permutation importance.")
                result = permutation_importance(
                    model, X, y,
                    n_repeats=n_repeats,
                    random_state=self.random_state
                )
                importance_scores = result.importances_mean
        
        else:
            raise ValueError(f"Unknown importance method: {method}")
        
        # Create and sort importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        })
        return importance_df.sort_values('importance', ascending=False)
    
    def partial_dependence(self,
                         model: Any,
                         X: Union[np.ndarray, pd.DataFrame],
                         feature_idx: Union[int, str],
                         num_points: int = 50,
                         ice_lines: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate partial dependence plots for a feature.
        
        Args:
            model: Trained model
            X: Feature data
            feature_idx: Index or name of feature
            num_points: Number of points to evaluate
            ice_lines: Whether to return individual conditional expectation lines
            
        Returns:
            Tuple of (feature_values, predictions)
        """
        if isinstance(X, pd.DataFrame):
            feature_name = (feature_idx if isinstance(feature_idx, str)
                          else X.columns[feature_idx])
            feature_idx = X.columns.get_loc(feature_name)
            X = X.values
        
        # Get feature range
        feature_min = X[:, feature_idx].min()
        feature_max = X[:, feature_idx].max()
        feature_values = np.linspace(feature_min, feature_max, num_points)
        
        # Calculate predictions
        predictions = []
        for value in feature_values:
            X_temp = X.copy()
            X_temp[:, feature_idx] = value
            pred = model.predict(X_temp)
            predictions.append(pred)
        
        if ice_lines:
            return feature_values, np.array(predictions)
        else:
            return feature_values, np.mean(predictions, axis=1)
    
    def feature_interactions(self,
                           model: Any,
                           X: Union[np.ndarray, pd.DataFrame],
                           top_k: int = 10) -> pd.DataFrame:
        """
        Analyze feature interactions using SHAP interaction values.
        
        Args:
            model: Trained model
            X: Feature data
            top_k: Number of top interactions to return
            
        Returns:
            DataFrame containing interaction scores
        """
        feature_names = (X.columns if isinstance(X, pd.DataFrame)
                        else [f'feature_{i}' for i in range(X.shape[1])])
        
        try:
            # Calculate SHAP interaction values
            explainer = shap.Explainer(model, X)
            shap_interaction = explainer(X, interactions=True)
            
            # Calculate interaction strength
            interaction_matrix = np.abs(shap_interaction.values).mean(0)
            
            # Create interaction DataFrame
            interactions = []
            for i in range(len(feature_names)):
                for j in range(i+1, len(feature_names)):
                    interactions.append({
                        'feature1': feature_names[i],
                        'feature2': feature_names[j],
                        'interaction_strength': float(interaction_matrix[i, j])
                    })
            
            interaction_df = pd.DataFrame(interactions)
            return interaction_df.nlargest(top_k, 'interaction_strength')
            
        except Exception as e:
            warnings.warn(f"SHAP interaction analysis failed: {str(e)}")
            return pd.DataFrame(columns=['feature1', 'feature2', 'interaction_strength'])
    
    def model_diagnostics(self,
                        model: Any,
                        X: Union[np.ndarray, pd.DataFrame],
                        y: Union[np.ndarray, pd.Series]) -> Dict[str, Any]:
        """
        Perform diagnostic analysis on model predictions.
        
        Args:
            model: Trained model
            X: Feature data
            y: True target values
            
        Returns:
            Dict containing various diagnostic metrics
        """
        # Get predictions
        y_pred = model.predict(X)
        
        # Basic residual analysis
        residuals = y - y_pred
        standardized_residuals = residuals / np.std(residuals)
        
        # Calculate diagnostic metrics
        diagnostics = {
            'residual_stats': {
                'mean': float(np.mean(residuals)),
                'std': float(np.std(residuals)),
                'skewness': float(self._calculate_skewness(residuals)),
                'kurtosis': float(self._calculate_kurtosis(residuals))
            },
            'outlier_info': {
                'num_outliers': int(np.sum(np.abs(standardized_residuals) > 3)),
                'outlier_indices': np.where(np.abs(standardized_residuals) > 3)[0]
            }
        }
        
        # Add probability calibration for classifiers
        if hasattr(model, 'predict_proba'):
            try:
                prob_pred = model.predict_proba(X)
                diagnostics['probability_stats'] = {
                    'mean_confidence': float(np.mean(np.max(prob_pred, axis=1))),
                    'min_confidence': float(np.min(np.max(prob_pred, axis=1))),
                    'max_confidence': float(np.max(np.max(prob_pred, axis=1)))
                }
            except Exception:
                pass
        
        return diagnostics
    
    def learning_curve_analysis(self,
                              model: Any,
                              X: Union[np.ndarray, pd.DataFrame],
                              y: Union[np.ndarray, pd.Series],
                              cv_splits: int = 5,
                              train_sizes: Optional[List[float]] = None) -> Dict[str, np.ndarray]:
        """
        Generate learning curves to analyze model performance vs training size.
        
        Args:
            model: Model to analyze
            X: Feature data
            y: Target values
            cv_splits: Number of cross-validation splits
            train_sizes: List of training set sizes to evaluate
            
        Returns:
            Dict containing training and validation scores
        """
        from sklearn.model_selection import learning_curve
        
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
            
        # Calculate learning curve
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X, y,
            train_sizes=train_sizes,
            cv=cv_splits,
            n_jobs=-1,
            random_state=self.random_state
        )
        
        return {
            'train_sizes': train_sizes_abs,
            'train_scores': train_scores.mean(axis=1),
            'train_scores_std': train_scores.std(axis=1),
            'val_scores': val_scores.mean(axis=1),
            'val_scores_std': val_scores.std(axis=1)
        }
    
    @staticmethod
    def _calculate_skewness(x: np.ndarray) -> float:
        """Calculate skewness of a distribution."""
        n = len(x)
        mean = np.mean(x)
        std = np.std(x, ddof=1)
        return (n * np.sum((x - mean) ** 3) / 
                ((n - 1) * (n - 2) * std ** 3))
    
    @staticmethod
    def _calculate_kurtosis(x: np.ndarray) -> float:
        """Calculate kurtosis of a distribution."""
        n = len(x)
        mean = np.mean(x)
        std = np.std(x, ddof=1)
        return (n * (n + 1) * np.sum((x - mean) ** 4) / 
                ((n - 1) * (n - 2) * (n - 3) * std ** 4) -
                3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))