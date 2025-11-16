import numpy as np
import __main__
from app.services.blob_service import load_pickle_from_url
from app.config import Config
from app.utils.logger import get_logger

logger = get_logger(__name__)


def to_float32(X):
    return X.astype(np.float32)


# Register to_float32 in __main__ for pickle compatibility
__main__.to_float32 = to_float32


def load_model():
    logger.info(f"Loading model from: {Config.MODEL_URL}")
    model = load_pickle_from_url(Config.MODEL_URL)
    logger.info("Model loaded successfully")
    return model


def get_model():
    if not hasattr(get_model, "_model"):
        get_model._model = load_model()
    return get_model._model
