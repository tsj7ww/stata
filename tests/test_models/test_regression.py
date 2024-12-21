import pytest
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError

from predml.models import RegressionModel


def test_regression_model_initialization():
    """Test model initialization."""
    model = RegressionModel(
        model_type="random_forest",
        model_params={"n_estimators": 100}
    )
    assert model.model_type == "random_forest"
    assert model.model_params["n_estimators"] == 100

def test_invalid_model_type():
    """Test initialization with invalid model type."""
    with pytest.raises(ValueError):
        RegressionModel(model_type="invalid_model")

def test_model_training(regression_model, sample_regression_data):
    """Test model training."""
    X = sample_regression_data.drop('target', axis=1)
    y = sample_regression_data['target']
    
    metrics = regression_model.train(X, y)
    
    assert isinstance(metrics, dict)
    assert 'train_mse' in metrics
    assert 'train_r2' in metrics
    assert regression_model.model is not None

def test_model_prediction(regression_model, sample_regression_data):
    """Test model prediction."""
    X = sample_regression_data.drop('target', axis=1)
    y = sample_regression_data['target']
    
    regression_model.train(X, y)
    predictions = regression_model.predict(X)
    
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape[0] == len(X)
    assert predictions.dtype == np.float64

def test_prediction_without_training(regression_model, sample_regression_data):
    """Test prediction without training."""
    X = sample_regression_data.drop('target', axis=1)
    
    with pytest.raises(ValueError):
        regression_model.predict(X)

def test_model_save_load(regression_model, sample_regression_data, tmp_path):
    """Test model saving and loading."""
    X = sample_regression_data.drop('target', axis=1)
    y = sample_regression_data['target']
    
    # Train and save model
    regression_model.train(X, y)
    save_path = tmp_path / "model.joblib"
    regression_model.save(save_path)
    
    # Load model and make predictions
    new_model = RegressionModel(model_type="random_forest")
    new_model.load(save_path)
    
    assert np.allclose(
        regression_model.predict(X),
        new_model.predict(X)
    )

def test_feature_importance(regression_model, sample_regression_data):
    """Test feature importance calculation."""
    X = sample_regression_data.drop('target', axis=1)
    y = sample_regression_data['target']
    
    regression_model.train(X, y)
    feature_importance = regression_model.model.feature_importances_
    
    assert isinstance(feature_importance, np.ndarray)
    assert len(feature_importance) == X.shape[1]
    assert np.all(feature_importance >= 0)
    assert np.isclose(np.sum(feature_importance), 1.0)