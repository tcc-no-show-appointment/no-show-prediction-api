import pandas as pd
import numpy as np
from app.utils.preprocessing import extract_features_from_row
from app.services.blob_service import load_csv_from_url
from app.services.model_loader import get_model
from app.config import Config
from app.constants import PREDICTION_LABEL_SHOW, PREDICTION_LABEL_NO_SHOW
from app.utils.logger import get_logger

logger = get_logger(__name__)

df_appointments = load_csv_from_url(Config.CSV_URL)

def convert_to_native_types(data):
    if isinstance(data, dict):
        return {key: convert_to_native_types(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_native_types(item) for item in data]
    elif isinstance(data, np.generic):
        return data.item()
    return data


async def predict(patient_id: int) -> dict:
    logger.info(f"Making prediction for patient_id: {patient_id}")
    
    matches = df_appointments[df_appointments['PatientId'] == patient_id]
    if matches.empty:
        logger.warning(f"PatientId {patient_id} not found in the dataset")
        raise ValueError(f"PatientId {patient_id} not found in the dataset.")
    
    row = matches.iloc[0]
    row_index = matches.index[0]
    
    features = extract_features_from_row(row)
    df = pd.DataFrame(features)
    
    model = get_model()
    prediction = model.predict(df)
    probability = model.predict_proba(df)
    
    result = {
        "row_index": row_index,
        "patient_id": row['PatientId'],
        "appointment_id": row['AppointmentID'],
        "actual_outcome": row['No-show'],
        "input_features": features,
        "prediction": int(prediction[0]),
        "prediction_label": PREDICTION_LABEL_NO_SHOW if prediction[0] == 1 else PREDICTION_LABEL_SHOW,
        "probability_show": float(probability[0][0]),
        "probability_no_show": float(probability[0][1])
    }
    
    logger.info(f"Prediction completed for patient_id {patient_id}: {result['prediction_label']}")
    
    return result
