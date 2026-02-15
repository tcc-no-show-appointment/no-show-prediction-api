from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Any, Dict


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
