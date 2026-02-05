import pandas as pd
import yaml
from typing import Dict, Any, Optional
from functools import lru_cache
from app.services.model_loader import get_model
from app.services.blob_service import BlobStorageClient
from app.constants import PREDICTION_LABEL_SHOW, PREDICTION_LABEL_NO_SHOW
from app.utils.logger import get_logger
from app.config import Config
from noshow_lib.feature_engineering import build_features

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def _get_config() -> Dict[str, Any]:
    """
    Download and parse the configuration file from blob storage.
    Uses caching to avoid repeated downloads.
    
    Returns:
        Parsed YAML configuration as dictionary
    """
    logger.info("Loading configuration from blob storage")
    
    try:
        blob_client = BlobStorageClient(
            connection_string=Config.AZURE_STORAGE_CONNECTION_STRING,
            account_name=Config.AZURE_STORAGE_ACCOUNT_NAME,
            account_key=Config.AZURE_STORAGE_ACCOUNT_KEY,
            container_name=Config.AZURE_BLOB_CONTAINER_NAME
        )
        
        config_content = blob_client.download_config_file(
            folder="model_configuration",
            filename="prod.yaml"
        )
        
        if not config_content:
            logger.error("Failed to download configuration file from blob storage")
            raise Exception("Configuration file not available")
        
        config = yaml.safe_load(config_content)
        logger.info(f"Configuration loaded successfully with keys: {list(config.keys())}")
        return config
        
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}", exc_info=True)
        raise

async def predict(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict patient no-show using noshow_lib feature engineering and trained model.
    
    Args:
        raw_data: Dictionary or list of dictionaries with appointment data
                 (using Portuguese column names from config.yaml)
        
    Returns:
        Dictionary with prediction results including probabilities
    """
    logger.info("Starting prediction pipeline")
    
    try:
        # Load configuration from blob storage
        config = _get_config()
        
        # Prepare input dataframe with Portuguese column names
        df = pd.DataFrame([raw_data]) if isinstance(raw_data, dict) else pd.DataFrame(raw_data)
        logger.info(f"Input data shape: {df.shape}, columns: {list(df.columns)}")
        
        # Validate required columns from config
        required_columns = config.get("schema", {}).get("required_columns", [])
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.warning(f"Missing columns: {missing_columns}. These will be handled by feature engineering.")
        
        # Build features using noshow_lib with config
        # The build_features function will handle column mapping and transformations
        logger.info("Engineering features with noshow_lib.build_features")
        features = build_features(df, config)
        logger.info(f"Features engineered successfully. Shape: {features.shape}, columns: {list(features.columns)[:10]}...")
        
        # Remove target column if present
        if 'no_show' in features.columns:
            features = features.drop(columns=['no_show'])
            logger.info("Dropped target column 'no_show' from features")
        
        # Get expected features from config and filter
        expected_features = config.get("model", {}).get("features", [])
        if expected_features:
            available_features = [f for f in expected_features if f in features.columns]
            missing_features = [f for f in expected_features if f not in features.columns]
            
            if missing_features:
                logger.warning(f"Missing {len(missing_features)} expected features: {missing_features[:5]}...")
            
            features = features[available_features]
            logger.info(f"Filtered to {len(available_features)} model features")
        
        # Load model and make prediction
        logger.info("Loading model and making prediction")
        model = get_model()
        
        prediction = model.predict(features)
        probability = model.predict_proba(features)
        
        logger.info(f"Raw prediction: {prediction}, probabilities: {probability}")
        
        result = {
            "prediction": int(prediction[0]),
            "prediction_label": PREDICTION_LABEL_NO_SHOW if prediction[0] == 1 else PREDICTION_LABEL_SHOW,
            "probability_show": float(probability[0][0]),
            "probability_no_show": float(probability[0][1])
        }
        
        logger.info(f"Prediction completed: {result['prediction_label']} (confidence: {max(result['probability_show'], result['probability_no_show']):.2%})")
        return result
        
    except Exception as e:
        logger.error(f"Error in prediction pipeline: {str(e)}", exc_info=True)
        raise
