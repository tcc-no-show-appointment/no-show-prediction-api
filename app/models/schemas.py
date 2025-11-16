from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class PredictionResponse(BaseModel):
    row_index: int = Field(..., description="Index of the row in the dataset")
    patient_id: int = Field(..., description="Unique patient identifier")
    appointment_id: int = Field(..., description="Unique appointment identifier")
    actual_outcome: str = Field(..., description="Actual outcome from the dataset")
    input_features: Dict[str, List] = Field(..., description="Extracted features used for prediction")
    prediction: int = Field(..., description="Predicted class (0=Show, 1=No-show)")
    prediction_label: str = Field(..., description="Human-readable prediction label")
    probability_show: float = Field(..., description="Probability of patient showing up")
    probability_no_show: float = Field(..., description="Probability of patient not showing up")

    class Config:
        schema_extra = {
            "example": {
                "row_index": 0,
                "patient_id": 7542951368435,
                "appointment_id": 5642903,
                "actual_outcome": "No",
                "input_features": {
                    "Gender": ["F"],
                    "Age": [62],
                    "Neighbourhood": ["JARDIM CAMBURI"],
                    "Scholarship": [0],
                    "Hipertension": [1],
                    "Diabetes": [0],
                    "Alcoholism": [0],
                    "Handcap": [0],
                    "SMS_received": [0],
                    "waiting_days": [7],
                    "scheduled_hour": [8],
                    "appointment_weekday": [1]
                },
                "prediction": 0,
                "prediction_label": "Show",
                "probability_show": 0.75,
                "probability_no_show": 0.25
            }
        }


class PredictionRequest(BaseModel):
    patient_id: Optional[int] = Field(None, description="Patient ID for prediction")

    class Config:
        schema_extra = {
            "example": {
                "patient_id": 7542951368435
            }
        }


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Error message describing what went wrong")

    class Config:
        schema_extra = {
            "example": {
                "detail": "PatientId 7542951368435 not found in the dataset."
            }
        }
