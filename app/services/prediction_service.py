import pandas as pd
from typing import Dict, Any
from app.services.model_loader import get_model
from app.constants import PREDICTION_LABEL_SHOW, PREDICTION_LABEL_NO_SHOW
from app.utils.logger import get_logger
from noshow_lib import load_and_process_data, build_features

logger = get_logger(__name__)


async def predict(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """Predict patient no-show using noshow_lib pipeline"""
    logger.info("Starting prediction pipeline")
    
    try:
        df = pd.DataFrame([raw_data]) if isinstance(raw_data, dict) else pd.DataFrame(raw_data)
        
        if 'AppointmentID' not in df.columns:
            df['AppointmentID'] = range(len(df))
        if 'No-show' not in df.columns:
            df['No-show'] = 'No'
        
        processed_data = load_and_process_data(df)
        features = build_features(processed_data, {"target_column": "No-show"})
        
        if 'No-show' in features.columns:
            features = features.drop(columns=['No-show'])
        
        model = get_model()
        prediction = model.predict(features)
        probability = model.predict_proba(features)
        
        result = {
            "prediction": int(prediction[0]),
            "prediction_label": PREDICTION_LABEL_NO_SHOW if prediction[0] == 1 else PREDICTION_LABEL_SHOW,
            "probability_show": float(probability[0][0]),
            "probability_no_show": float(probability[0][1])
        }
        
        logger.info(f"Prediction completed: {result['prediction_label']}")
        return result
        
    except Exception as e:
        logger.error(f"Error in prediction pipeline: {str(e)}")
        raise
