import pandas as pd
from typing import Dict, Any
from app.services.model_loader import get_model
from app.constants import PREDICTION_LABEL_SHOW, PREDICTION_LABEL_NO_SHOW
from app.utils.logger import get_logger
from noshow_lib import load_and_process_data, build_features

logger = get_logger(__name__)


async def predict(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """Predict patient no-show using noshow_lib pipeline and trained model."""
    logger.info("Starting prediction pipeline")
    
    try:
        df = pd.DataFrame([raw_data]) if isinstance(raw_data, dict) else pd.DataFrame(raw_data)
        logger.info(f"Input data shape: {df.shape}")
        
        if 'AppointmentID' not in df.columns:
            df['AppointmentID'] = range(len(df))
        if 'No-show' not in df.columns:
            df['No-show'] = 'No'
        
        logger.info("Processing data with noshow_lib")
        processed_data = load_and_process_data(df)
        logger.info(f"Processed data shape: {processed_data.shape}")
        
        logger.info("Engineering features with noshow_lib")
        features = build_features(processed_data, {"target_column": "No-show"})
        logger.info(f"Features shape: {features.shape}, columns: {list(features.columns)}")
        
        if 'No-show' in features.columns:
            features = features.drop(columns=['No-show'])
            logger.info("Dropped target column from features")
        
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
