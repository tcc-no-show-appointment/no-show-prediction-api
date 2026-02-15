import pandas as pd
from typing import Dict, Any, List
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


async def predict_batch(appointments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Predict patient no-show for multiple appointments in batch.
    
    This function processes all appointments in a single vectorized operation
    for maximum performance and scalability.
    
    Args:
        appointments: List of dictionaries with appointment data
                     (using Portuguese column names from config.yaml)
        
    Returns:
        Dictionary with:
        - total: Total number of predictions
        - predicted_show: Count of Show predictions
        - predicted_no_show: Count of No-Show predictions
        - results: List of individual results with original data + predictions
    """
    logger.info(f"Starting batch prediction for {len(appointments)} appointments")
    
    try:
        # Get model and config from memory (already loaded at startup)
        config = model_manager.get_config()
        model = model_manager.get_model()
        logger.info("Using cached model and config from memory")
        
        # Convert all appointments to a single DataFrame for vectorized processing
        df = pd.DataFrame(appointments)
        logger.info(f"Batch input shape: {df.shape}, columns: {list(df.columns)}")
        
        # Use noshow_lib's predict function (handles all feature engineering in one go)
        logger.info("Running batch inference using noshow_lib.model_inference.predict")
        result_df = noshow_predict(
            model=model,
            input_data=df,
            config=config,
            output_path=None,  # Don't save to file
            threshold=0.5
        )
        
        logger.info(f"Batch prediction completed. Result shape: {result_df.shape}")
        
        # Build results list with original data + predictions
        results = []
        predicted_show_count = 0
        predicted_no_show_count = 0
        
        for idx in range(len(result_df)):
            probability_no_show = float(result_df['probability'].iloc[idx])
            prediction_value = int(result_df['prediction'].iloc[idx])
            probability_show = 1.0 - probability_no_show
            
            # Count predictions
            if prediction_value == 1:
                predicted_no_show_count += 1
            else:
                predicted_show_count += 1
            
            # Build result with original appointment data + predictions
            result = {
                "appointment": appointments[idx],
                "prediction": prediction_value,
                "prediction_label": PREDICTION_LABEL_NO_SHOW if prediction_value == 1 else PREDICTION_LABEL_SHOW,
                "probability_show": probability_show,
                "probability_no_show": probability_no_show
            }
            results.append(result)
        
        batch_result = {
            "total": len(appointments),
            "predicted_show": predicted_show_count,
            "predicted_no_show": predicted_no_show_count,
            "results": results
        }
        
        logger.info(
            f"Batch prediction complete: {predicted_show_count} Show, "
            f"{predicted_no_show_count} No-Show (total: {len(appointments)})"
        )
        return batch_result
        
    except Exception as e:
        logger.error(f"Error in batch prediction pipeline: {str(e)}", exc_info=True)
        raise
