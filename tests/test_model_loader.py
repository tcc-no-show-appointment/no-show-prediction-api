
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
from app.services.model_loader import get_model


def test_get_model_loads():
    if hasattr(get_model, "_model"):
        delattr(get_model, "_model")
    
    model1 = get_model()
    model2 = get_model()
    
    assert model1 is not None
    assert model1 is model2
    assert hasattr(model1, 'predict')
    assert hasattr(model1, 'predict_proba')
