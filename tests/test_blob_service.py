
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
from app.services.blob_service import load_joblib_from_url
from app.config import Config


def test_load_joblib_from_url_success():
    result = load_joblib_from_url(Config.MODEL_URL)
    
    assert result is not None
    assert hasattr(result, 'predict')
    assert hasattr(result, 'predict_proba')


def test_load_joblib_from_url_failure():
    invalid_url = "https://invalid-url-that-does-not-exist.blob.core.windows.net/model.pkl"
    
    with pytest.raises(Exception, match="Failed to download file from URL"):
        load_joblib_from_url(invalid_url)
