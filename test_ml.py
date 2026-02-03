import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, inference, compute_model_metrics
# TODO: add necessary import

@pytest.fixture
def data():
    """ Simple fixture to provide dummy data for testing. """
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])
    return X, y

def test_train_model_type(data):
    """
    Test 1: Verify the training function returns the correct model type.
    Ensures the ML model uses the expected algorithm (RandomForest).
    """
    X, y = data
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)
    pass


# TODO: implement the second test. Change the function name and input as needed
def test_inference_output_shape(data):
    """
    Test 2: Verify the inference function returns the expected shape and type.
    Ensures that for N input rows, we get N predictions.
    """
    X, y = data
    model = train_model(X, y)
    preds = inference(model, X)
    
    assert isinstance(preds, np.ndarray), "Inference should return a numpy array"
    assert len(preds) == len(X), "Number of predictions must match number of input rows"
    assert set(preds).issubset({0, 1}), "Predictions should be binary (0 or 1)"
    pass


# TODO: implement the third test. Change the function name and input as needed
def test_compute_metrics_logic():
    """
    Test 3: Verify the computing metrics function returns expected values.
    Uses a perfect prediction scenario to check if precision/recall/F1 return 1.0.
    """
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    
    precision, recall, f1 = compute_model_metrics(y_true, y_pred)
    
    # Check if metrics are floats and within the valid range [0, 1]
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(f1, float)
    assert precision == 1.0, f"Expected precision 1.0, got {precision}"
    assert recall == 1.0, f"Expected recall 1.0, got {recall}"
    assert f1 == 1.0, f"Expected F1 1.0, got {f1}"
    pass
