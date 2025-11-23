import numpy as np
import __main__
from app.services.blob_service import load_joblib_from_url
from app.config import Config
from app.utils.logger import get_logger

logger = get_logger(__name__)

def to_float32(X):
    return X.astype(np.float32)

__main__.to_float32 = to_float32


def load_model():
    """Load trained sklearn Pipeline from Azure Blob Storage."""
    logger.info(f"Loading model from: {Config.MODEL_URL}")
    
    try:
        model = load_joblib_from_url(Config.MODEL_URL)
        logger.info(f"Model loaded successfully. Type: {type(model).__name__}")
        
        if not hasattr(model, 'predict') or not hasattr(model, 'predict_proba'):
            raise ValueError("Loaded model does not have required predict/predict_proba methods")
        
        logger.info("Model validation passed")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise


def get_model():
    """Get cached model instance (singleton pattern)."""
    if not hasattr(get_model, "_model"):
        logger.info("Model not in cache, loading from blob storage")
        get_model._model = load_model()
    return get_model._model
