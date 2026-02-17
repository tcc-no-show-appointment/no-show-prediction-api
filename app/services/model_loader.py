import numpy as np
import io
import joblib
import __main__
from app.services.blob_service import load_joblib_from_url, BlobStorageClient
from app.config import config
from app.utils.logger import get_logger

logger = get_logger(__name__)

def to_float32(X):
    return X.astype(np.float32)

__main__.to_float32 = to_float32


def load_model():
    
    logger.info(f"Loading model from {config.ENVIRONMENT} environment")
    
    try:
        blob_client = BlobStorageClient(
            connection_string=config.AZURE_STORAGE_CONNECTION_STRING,
            account_name=config.AZURE_STORAGE_ACCOUNT_NAME,
            account_key=config.AZURE_STORAGE_ACCOUNT_KEY,
            container_name=config.AZURE_BLOB_CONTAINER_NAME
        )
        
        environment = config.ENVIRONMENT
        
        logger.info("Attempting to download latest model...")
        model_bytes = blob_client.download_latest_model(
            environment=environment,
            base_name="model"
        )
        
        if not model_bytes:
            logger.warning("Latest model not found, falling back to versioned models...")
            model_bytes = blob_client.download_newest_versioned_model(
                environment=environment,
                base_name="model"
            )
        
        if not model_bytes:
            raise Exception(f"Failed to download model from {environment} environment")
        
        model_data = io.BytesIO(model_bytes)
        model = joblib.load(model_data)
        
        logger.info(f"Model loaded successfully. Type: {type(model).__name__}")
        
        if not hasattr(model, 'predict') or not hasattr(model, 'predict_proba'):
            raise ValueError("Loaded model does not have required predict/predict_proba methods")
        
        logger.info("Model validation passed")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise


def get_model():
    if not hasattr(get_model, "_model"):
        logger.info("Model not in cache, loading from blob storage")
        get_model._model = load_model()
    return get_model._model
