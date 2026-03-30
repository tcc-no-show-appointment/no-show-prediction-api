"""
Model Manager - Handles model lifecycle (load on startup, keep in memory)
This implements Option C: Download na Inicialização (Load on Startup)

At startup all per-specialty models (one per specialty_group) are downloaded
and kept in memory.  If a specialty has no dedicated model, the
OUTRAS_ESPECIALIDADES model is used as fallback.  All models are pre-wired
into _effective_models at startup so predictions require no per-request
model lookup logic.
"""
import io
import joblib
import numpy as np
import yaml
from typing import Optional, Dict, Any
import __main__
from app.services.blob_service import BlobStorageClient
from app.config import config
from app.constants import KNOWN_SPECIALTY_GROUPS
from app.utils.logger import get_logger

logger = get_logger(__name__)


def to_float32(X):
    """Custom function required for model deserialization"""
    return X.astype(np.float32)


__main__.to_float32 = to_float32


class ModelManager:
    """
    Singleton that manages per-specialty ML models.

    At startup downloads:
    1. All {env}/{specialty}/model_latest.joblib  → _models
    2. {env}/thresholds_latest.json               → _thresholds
    3. model_configuration/prod.yaml              → _config

    _effective_models is the merged dict used for inference:
    - Each KNOWN_SPECIALTY_GROUP maps to its dedicated model if available,
      or to the OUTRAS_ESPECIALIDADES model as fallback.
    - _default_model is an alias for _models["OUTRAS_ESPECIALIDADES"].
    """
    
    _instance: Optional['ModelManager'] = None
    _models: Optional[Dict[str, Any]] = None
    _default_model: Optional[Any] = None
    _effective_models: Optional[Dict[str, Any]] = None
    _thresholds: Optional[Dict[str, float]] = None
    _config: Optional[Dict[str, Any]] = None
    _model_names: Optional[Dict[str, str]] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    async def load_model_and_config(self) -> None:
        """
        Load all specialty models, optional default model, thresholds, and
        configuration from Azure Blob Storage. Called once at startup.
        """
        logger.info("=" * 80)
        logger.info("INITIALIZING MODEL MANAGER - Loading per-specialty models")
        logger.info("=" * 80)
        
        try:
            blob_client = BlobStorageClient(
                connection_string=config.AZURE_STORAGE_CONNECTION_STRING,
                account_name=config.AZURE_STORAGE_ACCOUNT_NAME,
                account_key=config.AZURE_STORAGE_ACCOUNT_KEY,
                container_name=config.AZURE_BLOB_CONTAINER_NAME
            )
            
            environment = config.ENVIRONMENT
            logger.info(f"Environment: {environment}")
            
            # ── Step 1: specialty-specific models ──────────────────────────
            logger.info("Step 1/4: Downloading specialty models from blob storage...")
            raw_models = blob_client.download_specialty_models(environment=environment)
            
            self._models = {}
            self._model_names = {}
            
            for specialty, (model_bytes, blob_name) in raw_models.items():
                model_data = io.BytesIO(model_bytes)
                model = joblib.load(model_data)
                
                if not hasattr(model, 'predict') or not hasattr(model, 'predict_proba'):
                    logger.warning(
                        f"[{specialty}] Model from {blob_name} lacks predict/predict_proba. Skipping."
                    )
                    continue
                
                self._models[specialty] = model
                self._model_names[specialty] = blob_name
                logger.info(
                    f"[{specialty}] Model loaded: {type(model).__name__} from {blob_name}"
                )
            
            logger.info(
                f"✓ {len(self._models)} specialty model(s) loaded: {list(self._models.keys())}"
            )

            # ── Step 2: resolve fallback = OUTRAS_ESPECIALIDADES ───────────
            self._default_model = self._models.get("OUTRAS_ESPECIALIDADES")
            if self._default_model is not None:
                logger.info(
                    "✓ Fallback model set to OUTRAS_ESPECIALIDADES"
                )
            else:
                logger.warning(
                    "OUTRAS_ESPECIALIDADES model not found — specialties without a "
                    "dedicated model will be skipped during prediction."
                )

            # ── Step 3: build effective models dict ────────────────────────
            # Pre-wire every known specialty to its model (dedicated > fallback).
            # This avoids any per-request routing logic.
            self._effective_models = {}

            for specialty in KNOWN_SPECIALTY_GROUPS:
                if specialty in self._models:
                    self._effective_models[specialty] = self._models[specialty]
                elif self._default_model is not None:
                    self._effective_models[specialty] = self._default_model
                    logger.info(f"[{specialty}] No dedicated model — using OUTRAS_ESPECIALIDADES fallback")

            # Include any specialty models that aren't in KNOWN_SPECIALTY_GROUPS
            for specialty, model in self._models.items():
                if specialty not in self._effective_models:
                    self._effective_models[specialty] = model

            if not self._effective_models:
                raise Exception(
                    "No models available. Make sure training has been run."
                )

            logger.info(
                f"✓ Effective models dict covers {len(self._effective_models)} specialties: "
                f"{list(self._effective_models.keys())}"
            )

            # ── Step 4: thresholds ─────────────────────────────────────────
            logger.info("Step 3/4: Downloading thresholds from blob storage...")
            self._thresholds = blob_client.download_thresholds(environment=environment)
            if not self._thresholds:
                logger.warning(
                    "No thresholds file found — will use 0.5 as fallback for all specialties"
                )
                self._thresholds = {}
            else:
                logger.info(f"✓ Thresholds loaded: {self._thresholds}")
            
            # ── Step 5: configuration ──────────────────────────────────────
            logger.info("Step 4/4: Downloading configuration from blob storage...")
            config_content = blob_client.download_config_file(
                folder="model_configuration",
                filename="prod.yaml"
            )
            
            if not config_content:
                raise Exception("Configuration file not available")
            
            self._config = yaml.safe_load(config_content)
            logger.info(f"✓ Configuration loaded with keys: {list(self._config.keys())}")
            
            logger.info("=" * 80)
            logger.info(
                f"MODEL MANAGER READY — {len(self._effective_models)} effective model(s) in memory"
            )
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"CRITICAL ERROR during model/config loading: {str(e)}", exc_info=True)
            raise
    
    def get_models(self) -> Dict[str, Any]:
        """Return specialty-specific models only (no default fallback applied)."""
        if self._models is None:
            raise RuntimeError(
                "Models not loaded. Make sure load_model_and_config() was called during startup."
            )
        return self._models
    
    def get_effective_models(self) -> Dict[str, Any]:
        """
        Return the merged dict of models to use for inference.

        Every KNOWN_SPECIALTY_GROUP is represented: either with its dedicated
        model or with the default fallback.  Use this for all predict() calls.
        """
        if self._effective_models is None:
            raise RuntimeError(
                "Effective models not built. Make sure load_model_and_config() was called."
            )
        return self._effective_models
    
    def get_default_model(self) -> Optional[Any]:
        """Return the fallback model, or None if not available."""
        return self._default_model
    
    def get_thresholds(self) -> Dict[str, float]:
        """Return optimal thresholds per specialty."""
        if self._thresholds is None:
            raise RuntimeError(
                "Thresholds not loaded. Make sure load_model_and_config() was called during startup."
            )
        return self._thresholds
    
    def get_config(self) -> Dict[str, Any]:
        """Return the cached configuration."""
        if self._config is None:
            raise RuntimeError(
                "Config not loaded. Make sure load_model_and_config() was called during startup."
            )
        return self._config
    
    def get_model_names(self) -> Dict[str, str]:
        """Return blob names of loaded specialty models."""
        if self._model_names is None:
            raise RuntimeError(
                "Model names not available. Make sure load_model_and_config() was called."
            )
        return self._model_names
    
    async def cleanup(self) -> None:
        """Cleanup resources on shutdown."""
        logger.info("Cleaning up model manager resources...")
        self._models = None
        self._default_model = None
        self._effective_models = None
        self._thresholds = None
        self._config = None
        self._model_names = None
        logger.info("Model manager cleanup complete")


# Singleton instance
model_manager = ModelManager()
