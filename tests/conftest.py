import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

from predml.models import ClassificationModel, RegressionModel
from predml.preprocessing import FeatureEngineer


@pytest.fixture
def sample_classification_data():
    """Generate sample classification data."""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    return df

@pytest.fixture
def sample_regression_data():
    """Generate sample regression data."""
    X, y = make_regression(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        random_state=42
    )
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    return df

@pytest.fixture
def classification_model():
    """Create a classification model instance."""
    return ClassificationModel(
        model_type="random_forest",
        model_params={
            "n_estimators": 100,
            "max_depth": 5,
            "random_state": 42
        }
    )

@pytest.fixture
def regression_model():
    """Create a regression model instance."""
    return RegressionModel(
        model_type="random_forest",
        model_params={
            "n_estimators": 100,
            "max_depth": 5,
            "random_state": 42
        }
    )

@pytest.fixture
def feature_engineer():
    """Create a feature engineer instance."""
    return FeatureEngineer(
        numeric_features=[f"feature_{i}" for i in range(10)],
        categorical_features=[f"feature_{i}" for i in range(10, 20)],
        scaling_method="standard",
        handle_missing=True
    )