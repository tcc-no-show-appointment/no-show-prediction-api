from pydantic import BaseModel, Field, ConfigDict
from typing import Optional


class PredictionRequest(BaseModel):
    """Request model for prediction endpoint"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "Age": 62,
                "Gender": "F",
                "ScheduledDay": "2024-11-16T08:00:00",
                "AppointmentDay": "2024-11-23T14:00:00",
                "Neighbourhood": "JARDIM CAMBURI",
                "Scholarship": 0,
                "Hipertension": 1,
                "Diabetes": 0,
                "Alcoholism": 0,
                "Handcap": 0,
                "SMS_received": 1
            }
        }
    )
    
    AppointmentID: Optional[int] = None
    PatientId: Optional[int] = None
    Age: int
    Gender: str
    ScheduledDay: str
    AppointmentDay: str
    Neighbourhood: str
    Scholarship: int
    Hipertension: int
    Diabetes: int
    Alcoholism: int
    Handcap: int
    SMS_received: int


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prediction": 0,
                "prediction_label": "Show",
                "probability_show": 0.75,
                "probability_no_show": 0.25
            }
        }
    )
    
    prediction: int
    prediction_label: str
    probability_show: float
    probability_no_show: float


class ErrorResponse(BaseModel):
    """Error response model"""
    detail: str
