"""
Model Manager - Handles model lifecycle (load on startup, keep in memory)
This implements Option C: Download na Inicialização (Load on Startup)

The model is downloaded ONCE when the container starts and kept in memory.
All subsequent requests use the cached model from RAM.
"""
import io
import joblib
import numpy as np
import yaml
from typing import Optional, Dict, Any
import __main__
from app.services.blob_service import BlobStorageClient
from app.config import config
from app.utils.logger import get_logger

logger = get_logger(__name__)


def to_float32(X):
    """Custom function required for model deserialization"""
    return X.astype(np.float32)


__main__.to_float32 = to_float32


class ModelManager:
    """
    Singleton class to manage the ML model lifecycle.
    
    The model is loaded once at startup and kept in memory for the entire
    container lifetime. This avoids downloading the model on every request.
    """
    
    _instance: Optional['ModelManager'] = None
    _model: Optional[Any] = None
    _config: Optional[Dict[str, Any]] = None
    _model_name: Optional[str] = None  # Track which blob file was loaded
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    async def load_model_and_config(self) -> None:
        """
        Load the model and configuration from Azure Blob Storage.
        This should be called once during application startup.
        """
        logger.info("=" * 80)
        logger.info("INITIALIZING MODEL MANAGER - Load on Startup")
        logger.info("=" * 80)
        
        try:
            # Initialize blob client
            blob_client = BlobStorageClient(
                connection_string=config.AZURE_STORAGE_CONNECTION_STRING,
                account_name=config.AZURE_STORAGE_ACCOUNT_NAME,
                account_key=config.AZURE_STORAGE_ACCOUNT_KEY,
                container_name=config.AZURE_BLOB_CONTAINER_NAME
            )
            
            environment = config.ENVIRONMENT
            logger.info(f"Environment: {environment}")
            
            # Load model
            logger.info("Step 1/2: Downloading model from blob storage...")
            model_result = blob_client.download_latest_model(
                environment=environment,
                base_name="model"
            )
            
            if not model_result:
                logger.warning("Latest model not found, trying versioned models...")
                model_result = blob_client.download_newest_versioned_model(
                    environment=environment,
                    base_name="model"
                )
            
            if not model_result:
                raise Exception(f"Failed to download model from {environment} environment")
            
            # Unpack tuple (bytes, blob_name)
            model_bytes, blob_name = model_result
            self._model_name = blob_name  # Store the blob name for traceability
            
            logger.info(f"Model downloaded: {len(model_bytes) / (1024*1024):.2f} MB from {blob_name}")
            
            # Deserialize model
            logger.info("Deserializing model...")
            model_data = io.BytesIO(model_bytes)
            self._model = joblib.load(model_data)
            
            if not hasattr(self._model, 'predict') or not hasattr(self._model, 'predict_proba'):
                raise ValueError("Model does not have required predict/predict_proba methods")
            
            logger.info(f"✓ Model loaded successfully. Type: {type(self._model).__name__}")
            
            # Load configuration
            logger.info("Step 2/2: Downloading configuration from blob storage...")
            config_content = blob_client.download_config_file(
                folder="model_configuration",
                filename="prod.yaml"
            )
            
            if not config_content:
                raise Exception("Configuration file not available")
            
            self._config = yaml.safe_load(config_content)
            logger.info(f"✓ Configuration loaded with keys: {list(self._config.keys())}")
            
            logger.info("=" * 80)
            logger.info("MODEL MANAGER READY - Model and config cached in memory")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"CRITICAL ERROR during model/config loading: {str(e)}", exc_info=True)
            raise
    
    def get_model(self) -> Any:
        """
        Get the cached model instance.
        
        Returns:
            The loaded ML model
            
        Raises:
            RuntimeError: If model hasn't been loaded yet
        """
        if self._model is None:
            raise RuntimeError(
                "Model not loaded. Make sure load_model_and_config() was called during startup."
            )
        return self._model
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the cached configuration.
        
        Returns:
            The configuration dictionary
            
        Raises:
            RuntimeError: If config hasn't been loaded yet
        """
        if self._config is None:
            raise RuntimeError(
                "Config not loaded. Make sure load_model_and_config() was called during startup."
            )
        return self._config
    
    def get_model_name(self) -> str:
        """
        Get the loaded model's blob name (e.g., 'homolog/model_20240215_123456.joblib').
        
        Returns:
            The blob name of the loaded model
            
        Raises:
            RuntimeError: If model hasn't been loaded yet
        """
        if self._model_name is None:
            raise RuntimeError(
                "Model name not available. Make sure load_model_and_config() was called during startup."
            )
        return self._model_name
    
    async def cleanup(self) -> None:
        """
        Cleanup resources on shutdown.
        """
        logger.info("Cleaning up model manager resources...")
        self._model = None
        self._config = None
        self._model_name = None
        logger.info("Model manager cleanup complete")


# Singleton instance
model_manager = ModelManager()
