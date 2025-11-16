from pydantic import BaseModel, Field
from typing import Optional


class PredictionRequest(BaseModel):
    """Request model for prediction endpoint"""
    patient_id: Optional[int] = Field(None, description="Patient ID")
    age: int = Field(..., description="Patient age")
    gender: str = Field(..., description="Patient gender (M/F)")
    scheduled_day: str = Field(..., description="Day when appointment was scheduled (ISO format)")
    appointment_day: str = Field(..., description="Day of the appointment (ISO format)")
    neighbourhood: str = Field(..., description="Location of the appointment")
    scholarship: int = Field(..., description="Whether patient has scholarship (0/1)")
    hipertension: int = Field(..., description="Whether patient has hypertension (0/1)")
    diabetes: int = Field(..., description="Whether patient has diabetes (0/1)")
    alcoholism: int = Field(..., description="Whether patient has alcoholism (0/1)")
    handcap: int = Field(..., description="Handicap level (0-4)")
    sms_received: int = Field(..., description="Whether SMS reminder was received (0/1)")

    class Config:
        schema_extra = {
            "example": {
                "patient_id": 7542951368435,
                "age": 62,
                "gender": "F",
                "scheduled_day": "2024-11-16T08:00:00",
                "appointment_day": "2024-11-23T14:00:00",
                "neighbourhood": "JARDIM CAMBURI",
                "scholarship": 0,
                "hipertension": 1,
                "diabetes": 0,
                "alcoholism": 0,
                "handcap": 0,
                "sms_received": 1
            }
        }


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint"""
    prediction: int = Field(..., description="Predicted class (0=Show, 1=No-show)")
    prediction_label: str = Field(..., description="Human-readable prediction label (Show/No-show)")
    probability_show: float = Field(..., description="Probability of patient showing up")
    probability_no_show: float = Field(..., description="Probability of patient not showing up")

    class Config:
        schema_extra = {
            "example": {
                "prediction": 0,
                "prediction_label": "Show",
                "probability_show": 0.75,
                "probability_no_show": 0.25
            }
        }


class ErrorResponse(BaseModel):
    """Error response model"""
    detail: str = Field(..., description="Error message describing what went wrong")

    class Config:
        schema_extra = {
            "example": {
                "detail": "Invalid input data"
            }
        }
