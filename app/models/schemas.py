from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Any, Dict
from datetime import datetime


class PredictionRequest(BaseModel):
    """
    Request model for prediction endpoint.
    Uses Portuguese column names matching config.yaml required_columns.
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": 5642903,
                "Marcacao": "2024-11-16T08:00:00",
                "DataHoraConsulta": "2024-11-23T14:00:00",
                "Idade": 62,
                "Sexo": "F",
                "CidadePaciente": "SAO PAULO",
                "BairroPaciente": "BELA VISTA",
                "TipoConvenio": "Enfermaria",
                "idUnicoPaciente": "ID369425000",
                "UnidadeAtendimento": "CAMPO BELO",
                "EnderecoUnidadeAtendimento": "RUA VIEIRA DE MORAES",
                "CEPUnidadeAtendimento": "04617-015",
                "Especialidade": "CARDIOLOGIA"
            }
        }
    )
    
    # Required columns based on config.yaml schema
    id: Optional[int] = Field(None, description="Appointment ID")
    Marcacao: str = Field(..., description="Scheduled date and time")
    DataHoraConsulta: str = Field(..., description="Appointment date and time")
    Idade: int = Field(..., ge=0, le=120, description="Patient age")
    Sexo: str = Field(..., description="Patient gender (M/F)")
    CidadePaciente: str = Field(..., description="Patient city")
    BairroPaciente: str = Field(..., description="Patient neighborhood")
    TipoConvenio: str = Field(..., description="Insurance type")
    idUnicoPaciente: Optional[str] = Field(None, description="Unique patient ID")
    UnidadeAtendimento: str = Field(..., description="Healthcare unit name")
    EnderecoUnidadeAtendimento: str = Field(..., description="Healthcare unit address")
    CEPUnidadeAtendimento: str = Field(..., description="Healthcare unit postal code")
    Especialidade: str = Field(..., description="Medical specialty")


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


class BatchPredictionRequest(BaseModel):
    """Request model for batch prediction endpoint"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "appointments": [
                    {
                        "id": 5642903,
                        "Marcacao": "2024-11-16T08:00:00",
                        "DataHoraConsulta": "2024-11-23T14:00:00",
                        "Idade": 62,
                        "Sexo": "F",
                        "CidadePaciente": "SAO PAULO",
                        "BairroPaciente": "BELA VISTA",
                        "TipoConvenio": "Enfermaria",
                        "idUnicoPaciente": "ID369425000",
                        "UnidadeAtendimento": "CAMPO BELO",
                        "EnderecoUnidadeAtendimento": "RUA VIEIRA DE MORAES",
                        "CEPUnidadeAtendimento": "04617-015",
                        "Especialidade": "CARDIOLOGIA"
                    },
                    {
                        "id": 5642904,
                        "Marcacao": "2024-11-17T09:00:00",
                        "DataHoraConsulta": "2024-11-24T15:00:00",
                        "Idade": 35,
                        "Sexo": "M",
                        "CidadePaciente": "SAO PAULO",
                        "BairroPaciente": "JARDINS",
                        "TipoConvenio": "Particular",
                        "idUnicoPaciente": "ID369425001",
                        "UnidadeAtendimento": "CAMPO BELO",
                        "EnderecoUnidadeAtendimento": "RUA VIEIRA DE MORAES",
                        "CEPUnidadeAtendimento": "04617-015",
                        "Especialidade": "CLINICA MEDICA"
                    }
                ]
            }
        }
    )
    
    appointments: List[PredictionRequest] = Field(
        ..., 
        min_length=1,
        max_length=1000,  # Reasonable limit for batch processing
        description="List of appointments to predict"
    )


class AppointmentPredictionResult(BaseModel):
    """Individual appointment with prediction results"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "appointment": {
                    "id": 5642903,
                    "Marcacao": "2024-11-16T08:00:00",
                    "DataHoraConsulta": "2024-11-23T14:00:00",
                    "Idade": 62,
                    "Sexo": "F",
                    "CidadePaciente": "SAO PAULO",
                    "BairroPaciente": "BELA VISTA",
                    "TipoConvenio": "Enfermaria",
                    "idUnicoPaciente": "ID369425000",
                    "UnidadeAtendimento": "CAMPO BELO",
                    "EnderecoUnidadeAtendimento": "RUA VIEIRA DE MORAES",
                    "CEPUnidadeAtendimento": "04617-015",
                    "Especialidade": "CARDIOLOGIA"
                },
                "prediction": 0,
                "prediction_label": "Show",
                "probability_show": 0.75,
                "probability_no_show": 0.25
            }
        }
    )
    
    appointment: Dict[str, Any] = Field(..., description="Original appointment data")
    prediction: int = Field(..., description="Prediction value (0=Show, 1=No-Show)")
    prediction_label: str = Field(..., description="Human-readable prediction label")
    probability_show: float = Field(..., description="Probability of patient showing up")
    probability_no_show: float = Field(..., description="Probability of patient not showing up")


class BatchPredictionResponse(BaseModel):
    """Response model for batch prediction endpoint"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total": 2,
                "predicted_show": 1,
                "predicted_no_show": 1,
                "results": [
                    {
                        "appointment": {"id": 5642903},
                        "prediction": 0,
                        "prediction_label": "Show",
                        "probability_show": 0.75,
                        "probability_no_show": 0.25
                    }
                ]
            }
        }
    )
    
    total: int = Field(..., description="Total number of predictions")
    predicted_show: int = Field(..., description="Number of predictions showing 'Show'")
    predicted_no_show: int = Field(..., description="Number of predictions showing 'No-Show'")
    results: List[AppointmentPredictionResult] = Field(..., description="Individual prediction results")


class RangePredictionRequest(BaseModel):
    """Request model for range prediction endpoint"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "appointment": {
                    "id": 5642903,
                    "Marcacao": "2024-11-16T08:00:00",
                    "DataHoraConsulta": "2024-11-23T14:00:00",
                    "Idade": 62,
                    "Sexo": "F",
                    "CidadePaciente": "SAO PAULO",
                    "BairroPaciente": "BELA VISTA",
                    "TipoConvenio": "Enfermaria",
                    "idUnicoPaciente": "ID369425000",
                    "UnidadeAtendimento": "CAMPO BELO",
                    "EnderecoUnidadeAtendimento": "RUA VIEIRA DE MORAES",
                    "CEPUnidadeAtendimento": "04617-015",
                    "Especialidade": "CARDIOLOGIA"
                },
                "range_days": 5
            }
        }
    )
    
    appointment: PredictionRequest = Field(..., description="Appointment data to predict across date range")
    range_days: int = Field(..., ge=3, le=5, description="Number of days to predict (3-5 days from appointment date)")


class DatePrediction(BaseModel):
    """Prediction for a specific date"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "date": "2024-11-23",
                "prediction": 1,
                "prediction_label": "No-Show",
                "probability_no_show": 0.65,
                "probability_show": 0.35
            }
        }
    )
    
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    prediction: int = Field(..., description="Prediction value (0=Show, 1=No-Show)")
    prediction_label: str = Field(..., description="Human-readable prediction label")
    probability_no_show: float = Field(..., description="Probability of patient not showing up (%)")
    probability_show: float = Field(..., description="Probability of patient showing up (%)")


class RangePredictionResponse(BaseModel):
    """Response model for range prediction endpoint"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "original_appointment_date": "2024-11-23",
                "original_appointment_time": "14:00:00",
                "patient_id": "ID369425000",
                "predictions": [
                    {
                        "date": "2024-11-23",
                        "prediction": 0,
                        "prediction_label": "Show",
                        "probability_no_show": 0.25,
                        "probability_show": 0.75
                    }
                ],
                "range_days": 5,
                "summary": {
                    "avg_probability_no_show": 0.45,
                    "min_probability_no_show": 0.25,
                    "max_probability_no_show": 0.65,
                    "best_date": "2024-11-23",
                    "worst_date": "2024-11-25"
                }
            }
        }
    )
    
    original_appointment_date: str = Field(..., description="Original appointment date (YYYY-MM-DD)")
    original_appointment_time: str = Field(..., description="Original appointment time (HH:MM:SS)")
    patient_id: Optional[str] = Field(None, description="Patient unique ID")
    predictions: List[DatePrediction] = Field(..., description="Predictions for each date in range")
    range_days: int = Field(..., description="Number of days analyzed")
    summary: Dict[str, Any] = Field(..., description="Summary statistics for the date range")


# ============================================================================
# APPOINTMENT SCHEMAS (for appointment_predictions table)
# ============================================================================

class AppointmentCreate(BaseModel):
    """Schema for creating a new appointment with prediction"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "patient_id": "ID369425000",
                "appointment_status": "Realizado",
                "scheduled_at": "2024-11-16T08:00:00",
                "appointment_at": "2024-11-23T14:00:00",
                "patient_age": 62,
                "patient_sex": "F",
                "patient_city": "SAO PAULO",
                "patient_neighborhood": "BELA VISTA",
                "insurance_type": "Enfermaria",
                "unit_name": "CAMPO BELO",
                "unit_address": "RUA VIEIRA DE MORAES",
                "unit_zipcode": "04617-015",
                "specialty": "CARDIOLOGIA",
                "prediction_class": 0,
                "prediction_label": "Show",
                "probability_show": 0.75,
                "probability_no_show": 0.25
            }
        }
    )
    
    # Model name (which model generated this prediction)
    model_name: Optional[str] = Field(None, description="Model blob name that generated the prediction (e.g., 'homolog/model_20240215_123456.joblib')")
    
    # Context fields
    patient_id: Optional[str] = Field(None, description="Patient unique ID")
    appointment_status: Optional[str] = Field(None, description="Appointment status")
    scheduled_at: Optional[str] = Field(None, description="When appointment was scheduled")
    appointment_at: Optional[str] = Field(None, description="When appointment is scheduled for")
    patient_age: Optional[int] = Field(None, ge=0, le=120, description="Patient age")
    patient_sex: Optional[str] = Field(None, description="Patient gender (M/F)")
    patient_city: Optional[str] = Field(None, description="Patient city")
    patient_neighborhood: Optional[str] = Field(None, description="Patient neighborhood")
    insurance_type: Optional[str] = Field(None, description="Insurance type")
    unit_name: Optional[str] = Field(None, description="Healthcare unit name")
    unit_address: Optional[str] = Field(None, description="Healthcare unit address")
    unit_zipcode: Optional[str] = Field(None, description="Healthcare unit postal code")
    specialty: Optional[str] = Field(None, description="Medical specialty")
    
    # Prediction output fields
    prediction_class: Optional[int] = Field(None, description="Prediction class (0=Show, 1=No-Show)")
    prediction_label: Optional[str] = Field(None, description="Human-readable prediction label")
    probability_show: Optional[float] = Field(None, description="Probability of showing up")
    probability_no_show: Optional[float] = Field(None, description="Probability of not showing up")


class AppointmentStatusUpdate(BaseModel):
    """Schema for updating appointment status after the appointment"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "appointment_status": "Realizado"
            }
        }
    )
    
    appointment_status: str = Field(..., description="Appointment status: Realizado (showed), Falta (no-show), Cancelado (canceled)")


class AppointmentResponse(BaseModel):
    """Schema for appointment response"""
    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "appointment_prediction_id": 1,
                "model_name": "homolog/model_20240215_123456.joblib",
                "patient_id": "ID369425000",
                "appointment_status": "Realizado",
                "scheduled_at": "2024-11-16T08:00:00",
                "appointment_at": "2024-11-23T14:00:00",
                "patient_age": 62,
                "patient_sex": "F",
                "patient_city": "SAO PAULO",
                "patient_neighborhood": "BELA VISTA",
                "insurance_type": "Enfermaria",
                "unit_name": "CAMPO BELO",
                "unit_address": "RUA VIEIRA DE MORAES",
                "unit_zipcode": "04617-015",
                "specialty": "CARDIOLOGIA",
                "prediction_class": 0,
                "prediction_label": "Show",
                "probability_show": 0.75,
                "probability_no_show": 0.25,
                "created_at": "2024-11-16T08:00:00",
                "updated_at": "2024-11-16T08:00:00"
            }
        }
    )
    
    appointment_prediction_id: int
    model_name: Optional[str] = None
    
    # Context fields
    patient_id: Optional[str] = None
    appointment_status: Optional[str] = None
    scheduled_at: Optional[datetime] = None
    appointment_at: Optional[datetime] = None
    patient_age: Optional[int] = None
    patient_sex: Optional[str] = None
    patient_city: Optional[str] = None
    patient_neighborhood: Optional[str] = None
    insurance_type: Optional[str] = None
    unit_name: Optional[str] = None
    unit_address: Optional[str] = None
    unit_zipcode: Optional[str] = None
    specialty: Optional[str] = None
    
    # Prediction fields
    prediction_class: Optional[int] = None
    prediction_label: Optional[str] = None
    probability_show: Optional[float] = None
    probability_no_show: Optional[float] = None
    
    # Timestamps
    created_at: datetime
    updated_at: Optional[datetime] = None


class AppointmentListResponse(BaseModel):
    """Schema for list of appointments"""
    total: int = Field(..., description="Total number of appointments")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Items per page")
    appointments: List[AppointmentResponse] = Field(..., description="List of appointments")
