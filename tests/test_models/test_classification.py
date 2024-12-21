import pytest
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError

from predml.models import ClassificationModel


def test_classification_model_initialization():
    """Test model initialization."""
    model = ClassificationModel(
        model_type="random_forest",
        model_params={"n_estimators": 100}
    )
    assert model.model_type == "random_forest"
    assert model.model_params["n_estimators"] == 100

def test_invalid_model_type():
    """Test initialization with invalid model type."""
    with pytest.raises(ValueError):
        ClassificationModel(model_type="invalid_model")

def test_model_training(classification_model, sample_classification_data):
    """Test model training."""
    X = sample_classification_data.drop('target', axis=1)
    y = sample_classification_data['target']
    
    metrics = classification_model.train(X, y)
    
    assert isinstance(metrics, dict)
    assert 'train_accuracy' in metrics
    assert 'train_f1' in metrics
    assert classification_model.model is not None

def test_model_prediction(classification_model, sample_classification_data):
    """Test model prediction."""
    X = sample_classification_data.drop('target', axis=1)
    y = sample_classification_data['target']
    
    classification_model.train(X, y)
    predictions = classification_model.predict(X)
    
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape[0] == len(X)
    assert len(np.unique(predictions)) == len(np.unique(y))

def test_prediction_without_training(classification_model, sample_classification_data):
    """Test prediction without training."""
    X = sample_classification_data.drop('target', axis=1)
    
    with pytest.raises(ValueError):
        classification_model.predict(X)

def test_model_save_load(classification_model, sample_classification_data, tmp_path):
    """Test model saving and loading."""
    X = sample_classification_data.drop('target', axis=1)
    y = sample_classification_data['target']
    
    # Train and save model
    classification_model.train(X, y)
    save_path = tmp_path / "model.joblib"
    classification_model.save(save_path)
    
    # Load model and make predictions
    new_model = ClassificationModel(model_type="random_forest")
    new_model.load(save_path)
    
    assert np.array_equal(
        classification_model.predict(X),
        new_model.predict(X)
    )

def test_predict_proba(classification_model, sample_classification_data):
    """Test probability predictions."""
    X = sample_classification_data.drop('target', axis=1)
    y = sample_classification_data['target']
    
    classification_model.train(X, y)
    probabilities = classification_model.predict_proba(X)
    
    assert isinstance(probabilities, np.ndarray)
    assert probabilities.shape[0] == len(X)
    assert probabilities.shape[1] == len(np.unique(y))
    assert np.allclose(probabilities.sum(axis=1), 1.0)