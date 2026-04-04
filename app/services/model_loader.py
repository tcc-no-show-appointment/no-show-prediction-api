import numpy as np
import io
import joblib
import __main__
from typing import Dict, Any, Optional
from app.services.blob_service import BlobStorageClient
from app.config import config
from app.constants import KNOWN_SPECIALTY_GROUPS
from app.utils.logger import get_logger

logger = get_logger(__name__)

def to_float32(X):
    return X.astype(np.float32)

__main__.to_float32 = to_float32


def load_models() -> Dict[str, Any]:
    """
    Load effective models dict from blob storage.

    Downloads:
    - Per-specialty models: {env}/{specialty}/model_latest.joblib
    - Default fallback:     {env}/model_latest.joblib (if exists)

    Returns:
        Dict covering all KNOWN_SPECIALTY_GROUPS: dedicated model where
        available, default model otherwise.
    """
    logger.info(f"Loading models from {config.ENVIRONMENT} environment")
    
    try:
        blob_client = BlobStorageClient(
            connection_string=config.AZURE_STORAGE_CONNECTION_STRING,
            account_name=config.AZURE_STORAGE_ACCOUNT_NAME,
            account_key=config.AZURE_STORAGE_ACCOUNT_KEY,
            container_name=config.AZURE_BLOB_CONTAINER_NAME
        )
        
        # Specialty-specific models
        raw_models = blob_client.download_specialty_models(environment=config.ENVIRONMENT)
        specialty_models: Dict[str, Any] = {}
        for specialty, (model_bytes, blob_name) in raw_models.items():
            model = joblib.load(io.BytesIO(model_bytes))
            if hasattr(model, 'predict') and hasattr(model, 'predict_proba'):
                specialty_models[specialty] = model
                logger.info(f"[{specialty}] Model loaded: {type(model).__name__}")
            else:
                logger.warning(f"[{specialty}] Missing predict methods. Skipping.")
        
        # Fallback = OUTRAS_ESPECIALIDADES model
        default_model: Optional[Any] = specialty_models.get("OUTRAS_ESPECIALIDADES")
        if default_model is not None:
            logger.info("Fallback set to OUTRAS_ESPECIALIDADES")
        else:
            logger.warning(
                "OUTRAS_ESPECIALIDADES model not found — specialties without a "
                "dedicated model will be skipped."
            )

        # Build effective models dict
        effective: Dict[str, Any] = {}
        for specialty in KNOWN_SPECIALTY_GROUPS:
            if specialty in specialty_models:
                effective[specialty] = specialty_models[specialty]
            elif default_model is not None:
                effective[specialty] = default_model
                logger.info(f"[{specialty}] Using OUTRAS_ESPECIALIDADES fallback")
        
        # Include any extra specialty models not in KNOWN_SPECIALTY_GROUPS
        for specialty, model in specialty_models.items():
            if specialty not in effective:
                effective[specialty] = model
        
        if not effective:
            raise Exception("No models available (no specialty models, no default)")
        
        logger.info(f"{len(effective)} effective model(s) ready: {list(effective.keys())}")
        return effective
        
    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}")
        raise


def get_models() -> Dict[str, Any]:
    """Cached accessor — loads once, returns same dict on subsequent calls."""
    if not hasattr(get_models, "_models"):
        logger.info("Models not in cache, loading from blob storage")
        get_models._models = load_models()
    return get_models._models
