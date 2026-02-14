import pandas as pd
from typing import Dict, Any
from app.services.model_manager import model_manager
from app.constants import PREDICTION_LABEL_SHOW, PREDICTION_LABEL_NO_SHOW
from app.utils.logger import get_logger
from noshow_lib.model_inference import predict as noshow_predict

logger = get_logger(__name__)


async def predict(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict patient no-show using noshow_lib inference pipeline.
    
    The model and config are already loaded in memory (loaded at startup).
    This function just performs the prediction using the cached resources.
    
    Args:
        raw_data: Dictionary or list of dictionaries with appointment data
                 (using Portuguese column names from config.yaml)
        
    Returns:
        Dictionary with prediction results including probabilities
    """
    logger.info("Starting prediction pipeline")
    
    try:
        # Get model and config from memory (already loaded at startup)
        config = model_manager.get_config()
        model = model_manager.get_model()
        logger.info("Using cached model and config from memory")
        df = pd.DataFrame([raw_data]) if isinstance(raw_data, dict) else pd.DataFrame(raw_data)
        logger.info(f"Input data shape: {df.shape}, columns: {list(df.columns)}")
        
        # Use noshow_lib's predict function (handles feature engineering, categorical features, etc.)
        logger.info("Using noshow_lib.model_inference.predict for inference")
        result_df = noshow_predict(
            model=model,
            input_data=df,
            config=config,
            output_path=None,  # Don't save to file
            threshold=0.5
        )
        
        logger.info(f"Prediction completed. Result shape: {result_df.shape}")
        
        # Extract results from the first row
        probability_no_show = float(result_df['probability'].iloc[0])
        prediction_value = int(result_df['prediction'].iloc[0])
        probability_show = 1.0 - probability_no_show
        
        result = {
            "prediction": prediction_value,
            "prediction_label": PREDICTION_LABEL_NO_SHOW if prediction_value == 1 else PREDICTION_LABEL_SHOW,
            "probability_show": probability_show,
            "probability_no_show": probability_no_show
        }
        
        logger.info(f"Prediction: {result['prediction_label']} (confidence: {max(probability_show, probability_no_show):.2%})")
        return result
        
    except Exception as e:
        logger.error(f"Error in prediction pipeline: {str(e)}", exc_info=True)
        raise
